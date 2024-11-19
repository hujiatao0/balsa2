from absl import app
from absl import flags
from balsa import envs
import logging
import ray
import os
import pickle
import balsa
import numpy as np
import experiments

from balsa.util import plans_lib
import train_utils
from pytorch_lightning import loggers as pl_loggers
from torch.utils.tensorboard import SummaryWriter
import signal
import sim as sim_lib
from balsa import costing
from balsa.experience import Experience


FLAGS = flags.FLAGS
flags.DEFINE_string('run', 'Balsa_JOBRandSplit', 'Experiment config to run.')
flags.DEFINE_boolean('local', False,
                     'Whether to use local engine for query execution.')


class BalsaAgent(object):
    
    def __init__(self, params):
        self.params = params.Copy()
        p = self.params
        print('BalsaAgent params:\n{}'.format(p))
        self.sim = None

        if p.use_local_execution:
            ray.init(resources={'pg': 1})
        else:
            # Cluster access: make sure the cluster has been launched.
            import uuid
            ray.init(address='auto',
                     namespace=f'{uuid.uuid4().hex[:4]}',
                     logging_level=logging.ERROR)
        try:
            print('Connected to ray!  Resources:', ray.available_resources())
        except RuntimeError as e:
            if 'dictionary changed size during iteration' not in str(e):
                raise e
            print('Connected to ray but ray.available_resources() failed, '
                  'likely indicating issues with the cluster.\nTry running '
                  '1 run only and see if tasks go through or get stuck.'
                  '  Exception:\n   {}'.format(e))
            
        self.workload = self._MakeWorkload()
        self.all_nodes = self.workload.Queries(split='all')
        self.train_nodes = self.workload.Queries(split='train')
        self.test_nodes = self.workload.Queries(split='test')
        print(len(self.train_nodes), 'train queries:',
              [node.info['query_name'] for node in self.train_nodes])
        print(len(self.test_nodes), 'test queries:',
              [node.info['query_name'] for node in self.test_nodes])
        if p.test_query_glob is None:
            print('Consider all queries as training nodes.')
        # Rewrite ops if physical plan is not used.
        if not p.plan_physical:
            plans_lib.RewriteAsGenericJoinsScans(self.all_nodes)
        # If the target engine has a dialect != Postgres, overwrite
        # node.info['sql_str'] with the dialected SQL.
        if p.engine_dialect_query_dir is not None:
            self.workload.UseDialectSql(p)

        # Unused.
        assert p.use_adaptive_lr is None
        self.adaptive_lr_schedule = None
        if p.linear_decay_to_zero:
            self.adaptive_lr_schedule = (
                train_utils.AdaptiveMetricPiecewiseDecayToZero(
                    [(0, p.lr)],
                    metric_max_value=0,  # Does not matter.
                    total_steps=p.val_iters))
            
        # Logging.
        self._InitLogging()
        self.timer = train_utils.Timer()
        # Experience (replay) buffer.
        self.exp, self.exp_val = self._MakeExperienceBuffer()
        self._latest_replay_buffer_path = None

        # Cleanup handlers.  Ensures that the Ray cluster state remains healthy
        # even if this driver program is killed.
        signal.signal(signal.SIGTERM, self.Cleanup)
        signal.signal(signal.SIGINT, self.Cleanup)

    def Cleanup(self, signum, frame):
        """Calls ray.shutdown() on cleanup."""
        print('Received signal {}; calling ray.shutdown().'.format(
            signal.Signals(signum).name))
        ray.shutdown()

    def _MakeWorkload(self):
        p = self.params
        if os.path.isfile(p.init_experience):
            # Load the expert optimizer experience.
            with open(p.init_experience, 'rb') as f:
                workload = pickle.load(f)
            # Filter queries based on the current query_glob.
            workload.FilterQueries(p.query_dir, p.query_glob, p.test_query_glob)
        else:
            wp = envs.JoinOrderBenchmark.Params()
            wp.query_dir = p.query_dir
            wp.query_glob = p.query_glob
            wp.test_query_glob = None
            workload = wp.cls(wp)
            # Requires baseline to run in this scenario.
            p.run_baseline = True
        return workload
    
    def _InitLogging(self):
        p = self.params
        self.loggers = [
            pl_loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                         version=None,
                                         name='tensorboard_logs'),
            pl_loggers.WandbLogger(save_dir=os.getcwd(), project='balsa'),
        ]
        self.summary_writer = SummaryWriter()
        self.wandb_logger = self.loggers[-1]
        p_dict = balsa.utils.SanitizeToText(dict(p))
        for logger in self.loggers:
            logger.log_hyperparams(p_dict)
        with open(os.path.join(self.wandb_logger.experiment.dir, 'params.txt'),
                  'w') as f:
            # Files saved to wandb's rundir are auto-uploaded.
            f.write(p.ToText())
        if not p.run_baseline:
            self.LogExpertExperience(self.train_nodes, self.test_nodes)

    def _MakeExperienceBuffer(self):
        p = self.params
        if not p.run_baseline and p.sim:
            wi = self.GetOrTrainSim().training_workload_info
        else:
            # E.g., if sim is disabled, we just use the overall workload info
            # (thus, this covers both train & test queries).
            wi = self.workload.workload_info
        if p.tree_conv:
            plan_feat_cls = plans_lib.TreeNodeFeaturizer
            if p.plan_physical:
                # Physical-aware plan featurizer.
                plan_feat_cls = plans_lib.PhysicalTreeNodeFeaturizer
        else:
            plan_feat_cls = plans_lib.PreOrderSequenceFeaturizer
        query_featurizer_cls = _GetQueryFeaturizerClass(p)
        if self.sim is not None:
            # Use the already instantiated query featurizer, which may contain
            # computed normalization stats.
            query_featurizer_cls = self.GetOrTrainSim().query_featurizer
        exp = Experience(self.train_nodes,
                         p.tree_conv,
                         workload_info=wi,
                         query_featurizer_cls=query_featurizer_cls,
                         plan_featurizer_cls=plan_feat_cls)
        if p.prev_replay_buffers_glob is not None:
            exp.Load(p.prev_replay_buffers_glob,
                     p.prev_replay_keep_last_fraction)
            pa = plan_analysis.PlanAnalysis.Build(exp.nodes[exp.initial_size:])
            pa.Print()

        if p.prev_replay_buffers_glob_val is not None:
            print('Building validation experience buffer...')
            exp_val = Experience(self.train_nodes,
                                 p.tree_conv,
                                 workload_info=wi,
                                 query_featurizer_cls=query_featurizer_cls,
                                 plan_featurizer_cls=plan_feat_cls)
            exp_val.Load(p.prev_replay_buffers_glob_val)
            pa = plan_analysis.PlanAnalysis.Build(
                exp_val.nodes[exp_val.initial_size:])
            pa.Print()
        else:
            exp_val = None

        return exp, exp_val

    def LogScalars(self, metrics):
        if not isinstance(metrics, list):
            assert len(metrics) == 3, 'Expected (tag, val, global_step)'
            metrics = [metrics]
        for tag, val, global_step in metrics:
            self.summary_writer.add_scalar(tag, val, global_step=global_step)
        d = dict([(tag, val) for tag, val, _ in metrics])
        assert len(set([gs for _, _, gs in metrics])) == 1, metrics
        self.wandb_logger.log_metrics(d)

    def LogExpertExperience(self, expert_train_nodes, expert_test_nodes):
        p = self.params
        total_s = 0
        data_to_log = []
        num_joins = []
        for node in expert_train_nodes:
            # Real latency in ms was assigned to node.cost as impl convenience.
            data_to_log.append(
                ('latency_expert/q{}'.format(node.info['query_name']),
                 node.cost / 1e3, 0))
            total_s += node.cost / 1e3
            num_joins.append(len(node.leaf_ids()) - 1)
        data_to_log.append(('latency_expert/workload', total_s, 0))
        print('latency_expert/workload (seconds): {:.2f} ({} queries)'.format(
            total_s, len(expert_train_nodes)))

        if p.test_query_glob is not None:
            total_s_test = 0
            for node in expert_test_nodes:
                data_to_log.append(
                    ('latency_expert_test/q{}'.format(node.info['query_name']),
                     node.cost / 1e3, 0))
                total_s_test += node.cost / 1e3
                num_joins.append(len(node.leaf_ids()) - 1)
            data_to_log.append(
                ('latency_expert_test/workload', total_s_test, 0))
            print('latency_expert_test/workload (seconds): {:.2f} ({} queries)'.
                  format(total_s_test, len(expert_test_nodes)))
        data_to_log.append(('curr_value_iter', 0, 0))
        self.LogScalars(data_to_log)
        print('Number of joins [{}, {}], avg {:.1f}'.format(
            np.min(num_joins), np.max(num_joins), np.mean(num_joins)))
        
    def GetOrTrainSim(self):
        p = self.params
        if self.sim is None:
            self.sim = TrainSim(p, self.loggers)
        return self.sim
      
    def Run(self):
        pass


def _GetQueryFeaturizerClass(p):
    return {
        True: sim_lib.SimQueryFeaturizer,
        False: plans_lib.QueryFeaturizer,
        'SimQueryFeaturizerV2': sim_lib.SimQueryFeaturizerV2,
        'SimQueryFeaturizerV3': sim_lib.SimQueryFeaturizerV3,
        'SimQueryFeaturizerV4': sim_lib.SimQueryFeaturizerV4,
    }[p.sim_query_featurizer]


def TrainSim(p, loggers=None):
    sim_p = sim_lib.Sim.Params()
    # Copy over relevant params.
    sim_p.workload.query_dir = p.query_dir
    sim_p.workload.query_glob = p.query_glob
    sim_p.workload.test_query_glob = p.test_query_glob
    sim_p.workload.search_space_join_ops = p.search_space_join_ops
    sim_p.workload.search_space_scan_ops = p.search_space_scan_ops
    sim_p.skip_data_collection_geq_num_rels = 12
    if p.cost_model == 'mincardcost':
        sim_p.search.cost_model = costing.MinCardCost.Params()
    else:
        sim_p.search.cost_model = costing.PostgresCost.Params()
    sim_p.query_featurizer_cls = _GetQueryFeaturizerClass(p)
    sim_p.plan_featurizer_cls = plans_lib.TreeNodeFeaturizer
    sim_p.infer_search_method = p.search_method
    sim_p.infer_beam_size = p.beam
    sim_p.infer_search_until_n_complete_plans = p.search_until_n_complete_plans
    if p.plan_physical:
        sim_p.plan_physical = True
        # Use a physical-aware plan featurizer.
        sim_p.plan_featurizer_cls = plans_lib.PhysicalTreeNodeFeaturizer
    sim_p.generic_ops_only_for_min_card_cost = \
        p.generic_ops_only_for_min_card_cost
    sim_p.label_transforms = p.label_transforms
    sim_p.tree_conv_version = p.tree_conv_version
    sim_p.loss_type = p.loss_type
    sim_p.gradient_clip_val = p.gradient_clip_val
    sim_p.bs = p.bs
    sim_p.epochs = p.epochs
    sim_p.perturb_query_features = p.perturb_query_features
    sim_p.validate_fraction = p.validate_fraction

    # Instantiate.
    sim = sim_lib.Sim(sim_p)
    if p.sim_checkpoint is None:
        sim.CollectSimulationData()
    sim.Train(load_from_checkpoint=p.sim_checkpoint, loggers=loggers)
    sim.model.freeze()
    sim.EvaluateCost()
    sim.FreeData()
    return sim


def Main(argv):
    del argv  # Unused.
    name = FLAGS.run
    print('Looking up params by name:', name)
    p = balsa.params_registry.Get(name)

    p.use_local_execution = FLAGS.local
    
    # Override params here for quick debugging.
    # p.sim_checkpoint = None
    # p.epochs = 1
    # p.val_iters = 0
    # p.query_glob = ['7*.sql']
    # p.test_query_glob = ['7c.sql']
    # p.search_until_n_complete_plans = 1
    
    agent = BalsaAgent(p)
    agent.Run()
    

if __name__ == '__main__':
    app.run(Main)