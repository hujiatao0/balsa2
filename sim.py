import balsa
from balsa import envs
from balsa import hyperparams
from balsa import search
from balsa.util import plans_lib
import numpy as np
import torch
import pickle
import pytorch_lightning as pl
from absl import logging
from balsa import experience
from balsa import costing
import train_utils
from balsa import models
import torch.nn.functional as F
from balsa.util import postgres
import time
import hashlib


class SimModel(pl.LightningModule):
    
    def __init__(self,
                 use_tree_conv,
                 query_feat_dims,
                 plan_feat_dims,
                 mlp_hiddens,
                 tree_conv_version=None,
                 loss_type=None,
                 torch_invert_cost=None,
                 query_featurizer=None,
                 perturb_query_features=False):
        super().__init__()
        assert loss_type in [None, 'mean_qerror'], loss_type
        self.save_hyperparameters()
        self.use_tree_conv = use_tree_conv
        if use_tree_conv:
            self.tree_conv = models.treeconv.TreeConvolution(
                feature_size=query_feat_dims,
                plan_size=plan_feat_dims,
                label_size=1,
                version=tree_conv_version)
        else:
            self.mlp = balsa.models.MakeMlp(input_size=query_feat_dims +
                                            plan_feat_dims,
                                            num_outputs=1,
                                            hiddens=mlp_hiddens,
                                            activation='relu')
        self.loss_type = loss_type
        self.torch_invert_cost = torch_invert_cost
        self.query_featurizer = query_featurizer
        self.perturb_query_features = perturb_query_features

    def forward(self, query_feat, plan_feat, indexes=None):
        if self.use_tree_conv:
            return self.tree_conv(query_feat, plan_feat, indexes)
        return self.mlp(torch.cat([query_feat, plan_feat], -1))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._ComputeLoss(batch)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        val_loss = self._ComputeLoss(batch)
        result = pl.EvalResult(checkpoint_on=val_loss, early_stop_on=val_loss)
        result.log('val_loss', val_loss, prog_bar=True)
        return result

    def _ComputeLoss(self, batch):
        query_feat, plan_feat, *rest = batch
        target = rest[-1]
        if self.training and self.perturb_query_features is not None:
            # No-op for non-enabled featurizers.
            query_feat = self.query_featurizer.PerturbQueryFeatures(
                query_feat, distribution=self.perturb_query_features)
        if self.use_tree_conv:
            assert len(rest) == 2
            output = self.forward(query_feat, plan_feat, rest[0])
        else:
            assert len(rest) == 1
            output = self.forward(query_feat, plan_feat)
        if self.loss_type == 'mean_qerror':
            output_inverted = self.torch_invert_cost(output.reshape(-1,))
            target_inverted = self.torch_invert_cost(target.reshape(-1,))
            return train_utils.QErrorLoss(output_inverted, target_inverted)
        return F.mse_loss(output.reshape(-1,), target.reshape(-1,))

    def on_after_backward(self):
        if self.global_step % 50 == 0:
            norm_dict = self.grad_norm(norm_type=2)
            total_norm = norm_dict['grad_2.0_norm_total']
            self.logger.log_metrics({'total_grad_norm': total_norm},
                                    step=self.global_step)

class SimPlanFeaturizer(plans_lib.Featurizer):
    """Implements the plan featurizer.

        plan node -> [ multi-hot of tables on LHS ] [ same for RHS ]
    """

    def __init__(self, workload_info):
        self.workload_info = workload_info

    def __call__(self, node):
        vec = np.zeros(self.dims, dtype=np.float32)

        # Tables on LHS.
        for rel_id in node.children[0].leaf_ids():
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx] = 1.0

        # Tables on RHS.
        for rel_id in node.children[1].leaf_ids():
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx + len(self.workload_info.rel_ids)] = 1.0

        return vec

    @property
    def dims(self):
        return len(self.workload_info.rel_ids) * 2
    

class SimQueryFeaturizer(plans_lib.Featurizer):
    """Implements the query featurizer.

        Query node -> [ multi-hot of what tables are present ]
                    * [ each-table's selectivities ]
    """

    def __init__(self, workload_info):
        self.workload_info = workload_info

    def __call__(self, node):
        vec = np.zeros(self.dims, dtype=np.float32)

        # Joined tables: [table: 1].
        joined = node.leaf_ids()
        for rel_id in joined:
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx] = 1.0

        # Filtered tables.
        table_id_to_name = lambda table_id: table_id.split(' ')[0]  # Hack.

        for rel_id, est_rows in node.info['all_filters_est_rows'].items():
            if rel_id not in joined:
                # Due to the way we copy Nodes and populate this info field,
                # leaf_ids() might be a subset of info['all_filters_est_rows'].
                continue

            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            total_rows = self.workload_info.table_num_rows[table_id_to_name(
                rel_id)]

            # NOTE: without ANALYZE, for some reason this predicate is
            # estimated to have 703 rows, whereas the table only has 4 rows:
            #   (kind IS NOT NULL) AND ((kind)::text <> 'production
            #   companies'::text)
            # With ANALYZE run, this assert passes.
            assert est_rows >= 0 and est_rows <= total_rows, (node.info,
                                                              est_rows,
                                                              total_rows)
            vec[idx] = est_rows / total_rows
        return vec
    
    @property
    def dims(self):
        return len(self.workload_info.rel_ids)
    

class SimQueryFeaturizerV2(SimQueryFeaturizer):
    """Concat SimQueryFeaturizer's output with indicators of filtered columns.

    Query feature vec
    = [each table: selectivity (0 if non-joined)]
      concat [bools of filtered cols].
    """

    def __call__(self, node):
        parent_vec = super().__call__(node)
        num_tables = len(self.workload_info.rel_ids)
        filtered_attrs = node.GetFilteredAttributes()
        for attr in filtered_attrs:
            idx = np.where(self.workload_info.all_attributes == attr)[0][0]
            parent_vec[num_tables + idx] = 1.0
        return parent_vec

    @property
    def dims(self):
        return len(self.workload_info.rel_ids) + len(
            self.workload_info.all_attributes)


class SimQueryFeaturizerV3(SimQueryFeaturizer):
    """[table->bool] concat [filtered col->selectivity]."""

    def __call__(self, node):
        vec = np.zeros(self.dims, dtype=np.float32)
        # Joined tables: [table: 1].
        joined = node.leaf_ids()
        for rel_id in joined:
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx] = 1.0
        num_tables = len(self.workload_info.rel_ids)

        # Filtered cols.
        rel_id_to_est_rows = node.info['all_filters_est_rows']
        leaves = node.GetLeaves()
        for leaf in leaves:
            leaf_filters = leaf.GetFilters()
            if not leaf_filters:
                continue
            # PG's parser groups all pushed-down filters by table.
            assert len(leaf_filters) == 1, leaf_filters
            leaf_filter = leaf_filters[0]

            # Get the overall selectivity of this expr.
            table_id = leaf.get_table_id()
            expr_est_rows = rel_id_to_est_rows[table_id]
            table_name = leaf.get_table_id(with_alias=False)
            total_rows = self.workload_info.table_num_rows[table_name]
            assert expr_est_rows >= 0 and expr_est_rows <= total_rows, (
                node.info, expr_est_rows, total_rows)
            table_expr_selectivity = expr_est_rows / total_rows

            # Assign this selectivity to all filtered columns in this expr.
            # Note that the expr may contain multiple cols & OR, in which case
            # we make a simplification to assign the same sel. to all cols.
            filtered_attrs = leaf.GetFilteredAttributes()
            for attr in filtered_attrs:
                idx = np.where(self.workload_info.all_attributes == attr)[0][0]
                vec[num_tables + idx] = table_expr_selectivity
        return vec

    @property
    def dims(self):
        return len(self.workload_info.rel_ids) + len(
            self.workload_info.all_attributes)


class SimQueryFeaturizerV4(plans_lib.Featurizer):
    """Raw estimated rows per table -> log(1+x) -> min_max scaling."""

    def __init__(self, workload_info):
        self.workload_info = workload_info
        self._min = None
        self._max = None
        self._range = None
        self._min_torch = None
        self._max_torch = None
        self._range_torch = None

    def __call__(self, node):
        vec = self._FeaturizePreScaling(node)
        return (vec - self._min) / self._range

    def PerturbQueryFeatures(self, query_feat, distribution):
        """Randomly perturbs a query feature vec returned by __call__()."""
        _min = self._min_torch.to(query_feat.device)
        _max = self._max_torch.to(query_feat.device)
        _range = self._range_torch.to(query_feat.device)
        pre_scaling = query_feat * _range + _min
        est_rows = torch.exp(pre_scaling) - 1.0
        # Chance of each joined table being perturbed.
        #   0.5: ~3% original; mean # tables scaled 3.6
        #   0.25: ~16.6% original; mean # tables scaled 1.8
        #   0.3: ~10.5% original; mean # tables scaled 2.1
        #
        # % kept original:
        #   ((multipliers > 1).sum(1) == 0).sum().float() / len(multipliers)
        # Mean # tables scaled:
        #   (multipliers > 1).sum(1).float().mean()
        #
        # "Default": chance = 0.25, unif = [0.5, 2].
        chance, unif = distribution
        should_scale = torch.rand(est_rows.shape,
                                  device=est_rows.device) < chance
        # The non-zero entries are joined tables.
        should_scale *= (est_rows > 0)
        # Sample multipliers ~ Unif[l, r].
        multipliers = torch.rand(est_rows.shape, device=est_rows.device) * (
            unif[1] - unif[0]) + unif[0]
        multipliers *= should_scale
        # Now, the 0 entries mean "should not scale", which needs to be
        # translated into using a multiplier of 1.
        multipliers[multipliers == 0] = 1
        # Perturb.
        new_est_rows = est_rows * multipliers
        # Re-perform transforms.
        logged = torch.log(1.0 + new_est_rows)
        logged_clamped = torch.min(logged, _max)
        new_query_feat_transformed = (logged_clamped - _min) / _range
        return new_query_feat_transformed

    def _FeaturizePreScaling(self, node):
        vec = np.zeros(self.dims, dtype=np.float32)
        table_id_to_name = lambda table_id: table_id.split(' ')[0]  # Hack.
        joined = node.leaf_ids()
        # Joined tables: [table: rows of table].
        for rel_id in joined:
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            total_rows = self.workload_info.table_num_rows[table_id_to_name(
                rel_id)]
            vec[idx] = total_rows
        # Filtered tables: [table: estimated rows of table].
        for rel_id, est_rows in node.info['all_filters_est_rows'].items():
            if rel_id not in joined:
                # Due to the way we copy Nodes and populate this info field,
                # leaf_ids() might be a subset of info['all_filters_est_rows'].
                continue
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            total_rows = self.workload_info.table_num_rows[table_id_to_name(
                rel_id)]
            assert est_rows >= 0 and est_rows <= total_rows, (node.info,
                                                              est_rows,
                                                              total_rows)
            vec[idx] = est_rows
        # log1p.
        return np.log(1.0 + vec)

    def Fit(self, nodes):
        assert self._min is None and self._max is None, (self._min, self._max)
        pre_scaling = np.asarray(
            [self._FeaturizePreScaling(node) for node in nodes])
        self._min = np.min(pre_scaling, 0)
        self._max = np.max(pre_scaling, 0)
        self._range = self._max - self._min
        # For PerturbQueryFeatures().
        self._min_torch = torch.from_numpy(self._min)
        self._max_torch = torch.from_numpy(self._max)
        self._range_torch = torch.from_numpy(self._range)
        logging.info('log(1+est_rows): min {}\nmax {}'.format(
            self._min, self._max))

    @property
    def dims(self):
        return len(self.workload_info.rel_ids)


class Sim(object):
    """Balsa simulation."""

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        # Train.
        p.Define('epochs', 100, 'Maximum training epochs.  '\
                 'Early-stopping may kick in.')
        p.Define('gradient_clip_val', 0, 'Clip the gradient norm computed over'\
                 ' all model parameters together. 0 means no clipping.')
        p.Define('bs', 2048, 'Batch size.')
        # Validation.
        p.Define('validate_fraction', 0.1,
                 'Sample this fraction of the dataset as the validation set.  '\
                 '0 to disable validation.')
        # Search, train-time.
        p.Define('search', search.DynamicProgramming.Params(),
                 'Params of the enumeration routine to use for training data.')
        # Search space.
        p.Define('plan_physical', False,
                 'Learn and plan physical scans/joins, or just join orders?')
        # Infer, test-time.
        p.Define('infer_search_method', 'beam_bk', 'Options: beam_bk.')
        p.Define('infer_beam_size', 10, 'Beam size.')
        p.Define('infer_search_until_n_complete_plans', 1,
                 'Search until how many complete plans?')
        # Workload.
        p.Define('workload', envs.JoinOrderBenchmark.Params(),
                 'Params of the Workload, i.e., a set of queries.')
        # Data collection.
        p.Define('skip_data_collection_geq_num_rels', None,
                 'If specified, do not collect data for queries with at '\
                 'least this many number of relations.')
        p.Define(
            'generic_ops_only_for_min_card_cost', False,
            'If using MinCardCost, whether to enumerate generic ops only.')
        p.Define('sim_data_collection_intermediate_goals', True,
                 'For each query, also collect sim data with intermediate '\
                 'query goals?')
        # Featurizations.
        p.Define('plan_featurizer_cls', SimPlanFeaturizer,
                 'Featurizer to use for plans.')
        p.Define('query_featurizer_cls', SimQueryFeaturizer,
                 'Featurizer to use for queries.')
        p.Define('label_transforms', ['log1p', 'standardize'],
                 'Transforms for labels.')
        p.Define('perturb_query_features', None, 'See experiments.')
        # Eval.
        p.Define('eval_output_path', 'eval-cost.csv',
                 'Path to write evaluation output into.')
        p.Define('eval_latency_output_path', 'eval-latency.csv',
                 'Path to write evaluation latency output into.')
        # Model/loss.
        p.Define('tree_conv_version', None, 'Options: None, V2.')
        p.Define('loss_type', None, 'Options: None (MSE), mean_qerror.')
        return p
    
    @classmethod
    def HashOfSimData(cls, p):
        """Gets the hash that should determine the simulation data."""
        # Use (a few attributes inside Params, Postgres configs) as hash key.
        # Using PG configs is necessary because things like PG version / PG
        # optimizer settings affect collected costs.
        # NOTE: in theory, other stateful effects such as whether ANALYZE has
        # been called on a PG database also affects the collected costs.
        _RELEVANT_HPARAMS = [
            'search',
            'workload',
            'skip_data_collection_geq_num_rels',
            'generic_ops_only_for_min_card_cost',
            'plan_physical',
        ]
        param_vals = [p.Get(hparam) for hparam in _RELEVANT_HPARAMS]
        param_vals = [
            v.ToText() if isinstance(v, hyperparams.Params) else str(v)
            for v in param_vals
        ]
        spec = '\n'.join(param_vals)
        if p.search.cost_model.cls is costing.PostgresCost:
            # Only PostgresCost would depend on PG configs.
            pg_configs = map(str, postgres.GetServerConfigs())
            spec += '\n'.join(pg_configs)
        hash_sim = hashlib.sha1(spec.encode()).hexdigest()[:8]
        return hash_sim
    
    def __init__(self, params):
        self.params = params.Copy()
        p = self.params
        # Plumb through same flags.
        p.search.plan_physical_ops = p.plan_physical
        p.search.cost_model.cost_physical_ops = p.plan_physical
        logging.info(p)

        # Instantiate search.
        self.search = p.search.cls(p.search)

        # Instantiate workload.
        self.workload = p.workload.cls(p.workload)
        wi = self.workload.workload_info
        generic_join = np.array(['Join'])
        generic_scan = np.array(['Scan'])
        if not p.plan_physical:
            # These are used in optimizer.py (for planning).
            wi.join_types = generic_join
            wi.scan_types = generic_scan
        else:
            self.search.SetPhysicalOps(join_ops=wi.join_types,
                                       scan_ops=wi.scan_types)
        if self.IsPlanPhysicalButUseGenericOps():
            self.search.SetPhysicalOps(join_ops=generic_join,
                                       scan_ops=generic_scan)

        # A list of SubplanGoalCost.
        self.simulation_data = []

        self.planner = None
        self.query_featurizer = None

        self.all_nodes = self.workload.Queries(split='all')
        self.train_nodes = self.workload.Queries(split='train')
        self.test_nodes = self.workload.Queries(split='test')
        logging.info('{} train queries: {}'.format(
            len(self.train_nodes),
            [node.info['query_name'] for node in self.train_nodes]))
        logging.info('{} test queries: {}'.format(
            len(self.test_nodes),
            [node.info['query_name'] for node in self.test_nodes]))

        plans_lib.RewriteAsGenericJoinsScans(self.all_nodes)

        # This call ensures that node.info['all_filters_est_rows'] is written,
        # which is used by the query featurizer.
        experience.SimpleReplayBuffer(self.all_nodes)

    def IsPlanPhysicalButUseGenericOps(self):
        p = self.params
        # This is a logical-only cost model.  Let's only enumerate generic ops.
        return (p.plan_physical and p.generic_ops_only_for_min_card_cost and
                isinstance(self.search.cost_model, costing.MinCardCost))
    
    def _SimulationDataPath(self):
        p = self.params
        hash_key = Sim.HashOfSimData(p)
        return 'data/sim-data-{}.pkl'.format(hash_key)

    def _LoadSimulationData(self):
        path = self._SimulationDataPath()
        try:
            with open(path, 'rb') as f:
                self.simulation_data = pickle.load(f)
        except Exception as e:
            return False
        logging.info('Loaded simulation data (len {}) from: {}'.format(
            len(self.simulation_data), path))
        logging.info('Training data (first 50, total {}):'.format(
            len(self.simulation_data)))
        logging.info('\n'.join(map(str, self.simulation_data[:50])))
        return True
    
    def CollectSimulationData(self, try_load=True):
        p = self.params
        if try_load:
            done = self._LoadSimulationData()
            if done:
                return

        start = time.time()
        num_collected = 0
        for query_node in self.train_nodes:
            # TODO: can parallelize this loop.  Take care of the hooks.
            num_rels = len(query_node.leaf_ids())
            logging.info('query={} num_rels={}\n{}'.format(
                query_node.info['query_name'], num_rels,
                query_node.info['sql_str']))
            if p.skip_data_collection_geq_num_rels is not None:
                if num_rels >= p.skip_data_collection_geq_num_rels:
                    continue
            num_collected += 1

            # Accumulate data points from this query.
            accum = []
            info_to_attach = {
                'overall_join_graph': query_node.info['parsed_join_graph'],
                'overall_join_conds': query_node.info['parsed_join_conds'],
                'path': query_node.info['path'],
            }
            self.search.PushOnEnumeratedHook(
                self._MakeOnEnumeratedHook(accum, info_to_attach, num_rels))

            # Enumerate plans.
            self.search.Run(query_node, query_node.info['sql_str'])

            self.search.PopOnEnumeratedHook()

            # Dedup accumulated data points.
            accum = self._DedupDataPoints(accum)

            self.simulation_data.extend(accum)

        simulation_time = time.time() - start

        logging.info('Collection done, stats:')
        logging.info('  num_queries={} num_collected_queries={} num_points={}'\
                     ' latency_s={:.1f}'.format(
            len(self.train_nodes), num_collected, len(self.simulation_data),
            simulation_time))

        if try_load:
            self._SaveSimulationData()

        return simulation_time, len(self.simulation_data)
    
    def Train(self, train_data=None, load_from_checkpoint=None, loggers=None):
        p = self.params
        # Pre-process and featurize data.
        data = train_data
        if data is None:
            data = self._FeaturizeTrainingData()

        # Make the DataLoader.
        logging.info('_MakeDatasetAndLoader()')
        self.train_dataset, self.train_loader, _, self.val_loader = \
            self._MakeDatasetAndLoader(data)
        batch = next(iter(self.train_loader))
        logging.info(
            'Example batch (query,plan,indexes,cost):\n{}'.format(batch))

        # Initialize model.
        _, query_feat_dims = batch[0].shape
        if issubclass(p.plan_featurizer_cls, plans_lib.TreeNodeFeaturizer):
            # make_and_featurize_trees() tranposes the latter 2 dims.
            unused_bs, plan_feat_dims, unused_max_tree_nodes = batch[1].shape
            logging.info(
                'unused_bs, plan_feat_dims, unused_max_tree_nodes {}'.format(
                    (unused_bs, plan_feat_dims, unused_max_tree_nodes)))
        else:
            unused_bs, plan_feat_dims = batch[1].shape
        self.model = self._MakeModel(query_feat_dims=query_feat_dims,
                                     plan_feat_dims=plan_feat_dims)
        balsa.models.ReportModel(self.model)

        # Train or load.
        self.trainer = self._MakeTrainer(loggers=loggers)
        if load_from_checkpoint:
            self.model = SimModel.load_from_checkpoint(load_from_checkpoint)
            logging.info(
                'Loaded pretrained checkpoint: {}'.format(load_from_checkpoint))
        else:
            self.trainer.fit(self.model, self.train_loader, self.val_loader)
        return data

    