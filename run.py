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
import pprint

from balsa.util import plans_lib
import train_utils
from pytorch_lightning import loggers as pl_loggers
from torch.utils.tensorboard import SummaryWriter
import signal
import sim as sim_lib
from balsa import costing
from balsa.experience import Experience
from balsa.util import postgres
import pg_executor


FLAGS = flags.FLAGS
flags.DEFINE_string('run', 'Balsa_JOBRandSplit', 'Experiment config to run.')
flags.DEFINE_boolean('local', False,
                     'Whether to use local engine for query execution.')

def Save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path


@ray.remote
def ExecuteSql(query_name,
               sql_str,
               hint_str,
               hinted_plan,
               query_node,
               predicted_latency,
               curr_timeout_ms=None,
               found_plans=None,
               predicted_costs=None,
               silent=False,
               is_test=False,
               use_local_execution=True,
               plan_physical=True,
               repeat=1,
               engine='postgres'):
    """Executes a query.

    Returns:
      If use_local_execution:
        A (pg_executor, dbmsx_executor).Result.
      Else:
        A ray.ObjectRef of the above.
    """
    # Unused args.
    del query_name, hinted_plan, query_node, predicted_latency, found_plans,\
        predicted_costs, silent, is_test, plan_physical

    assert engine in ('postgres', 'dbmsx'), engine
    if engine == 'postgres':
        return postgres.ExplainAnalyzeSql(sql_str,
                                          comment=hint_str,
                                          verbose=False,
                                          geqo_off=True,
                                          timeout_ms=curr_timeout_ms,
                                          remote=not use_local_execution)
    

def HintStr(node, with_physical_hints, engine):
    if engine == 'postgres':
        return node.hint_str(with_physical_hints=with_physical_hints)


def ParseExecutionResult(result_tup,
                         query_name,
                         sql_str,
                         hint_str,
                         hinted_plan,
                         query_node,
                         predicted_latency,
                         curr_timeout_ms=None,
                         found_plans=None,
                         predicted_costs=None,
                         silent=False,
                         is_test=False,
                         use_local_execution=True,
                         plan_physical=True,
                         repeat=None,
                         engine='postgres'):
    del repeat  # Unused.
    messages = []
    result = result_tup.result
    has_timeout = result_tup.has_timeout
    server_ip = result_tup.server_ip
    if has_timeout:
        assert not result, result
    if engine == 'dbmsx':
        real_cost = -1 if has_timeout else result_tup.latency
    else:
        if has_timeout:
            real_cost = -1
        else:
            json_dict = result[0][0][0]
            real_cost = json_dict['Execution Time']
    if hint_str is not None:
        # Check that the hint has been respected.  No need to check if running
        # baseline.
        do_hint_check = True
        if engine == 'dbmsx':
            raise NotImplementedError
        else:
            if not has_timeout:
                executed_node = postgres.ParsePostgresPlanJson(json_dict)
            else:
                # Timeout has occurred & 'result' is empty.  Fallback to
                # checking against local Postgres.
                print('Timeout occurred; checking the hint against local PG.')
                executed_node, _ = postgres.SqlToPlanNode(sql_str,
                                                          comment=hint_str,
                                                          verbose=False)
            executed_node = plans_lib.FilterScansOrJoins(executed_node)
            executed_hint_str = executed_node.hint_str(
                with_physical_hints=plan_physical)
        if do_hint_check and hint_str != executed_hint_str:
            print('initial\n', hint_str)
            print('after\n', executed_hint_str)
            msg = 'Hint not respected for {}; server_ip={}'.format(
                query_name, server_ip)
            try:
                assert False, msg
            except Exception as e:
                print(e, flush=True)
                import ipdb
                ipdb.set_trace()

    if not silent:
        messages.append('{}Running {}: hinted plan\n{}'.format(
            '[Test set] ' if is_test else '', query_name, hinted_plan))
        messages.append('filters')
        messages.append(pprint.pformat(query_node.info['all_filters']))
        messages.append('')
        messages.append('q{},{:.1f},{}'.format(query_node.info['query_name'],
                                               real_cost, hint_str))
        messages.append(
            '{} Execution time: {:.1f} (predicted {:.1f}) curr_timeout_ms={}'.
            format(query_name, real_cost, predicted_latency, curr_timeout_ms))

    if hint_str is None or silent:
        # Running baseline: don't print debug messages below.
        return result_tup, real_cost, server_ip, '\n'.join(messages)

    messages.append('Expert plan: latency, predicted, hint')
    expert_hint_str = query_node.hint_str()
    expert_hint_str_physical = query_node.hint_str(with_physical_hints=True)
    messages.append('  {:.1f} (predicted {:.1f})  {}'.format(
        query_node.cost, query_node.info['curr_predicted_latency'],
        expert_hint_str))
    if found_plans:
        if predicted_costs is None:
            predicted_costs = [None] * len(found_plans)
        messages.append('SIM-predicted costs, predicted latency, plan: ')
        min_p_latency = np.min([p_latency for p_latency, _ in found_plans])
        for p_cost, found in zip(predicted_costs, found_plans):
            p_latency, found_plan = found
            found_hint_str = found_plan.hint_str()
            found_hint_str_physical = HintStr(found_plan,
                                              with_physical_hints=True,
                                              engine=engine)
            extras = [
                'cheapest' if p_latency == min_p_latency else '',
                '[expert plan]'
                if found_hint_str_physical == expert_hint_str_physical else '',
                '[picked]' if found_hint_str_physical == hint_str else ''
            ]
            extras = ' '.join(filter(lambda s: s, extras)).strip()
            if extras:
                extras = '<-- {}'.format(extras)
            if p_cost:
                messages.append('  {:.1f}  {:.1f}  {}  {}'.format(
                    p_cost, p_latency, found_hint_str, extras))
            else:
                messages.append('          {:.1f}  {}  {}'.format(
                    p_latency, found_hint_str, extras))
    messages.append('-' * 80)
    return result_tup, real_cost, server_ip, '\n'.join(messages)


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
    
    def RunBaseline(self):
        p = self.params
        print('Dropping buffer cache.')
        postgres.DropBufferCache()
        print('Running queries as-is (baseline PG performance)...')

        def Args(node):
            return {
                'query_name': node.info['query_name'],
                'sql_str': node.info['sql_str'],
                'hint_str': None,
                'hinted_plan': None,
                'query_node': node,
                'predicted_latency': 0,
                'silent': True,
                'use_local_execution': p.use_local_execution,
                'engine': p.engine,
            }

        tasks = []
        for node in self.all_nodes:
            # Run the query.
            tasks.append(
                ExecuteSql.options(resources={
                    f'node:{ray.util.get_node_ip_address()}': 1,
                }).remote(**Args(node)))
        if not p.use_local_execution:
            refs = ray.get(tasks)
        else:
            refs = tasks
        for i, node in enumerate(self.all_nodes):
            result_tup = ray.get(refs[i])
            assert isinstance(
                result_tup,
                pg_executor.Result), result_tup
            result, real_cost, _, message = ParseExecutionResult(
                result_tup, **Args(node))
            # Save real cost (execution latency) to actual.
            node.cost = real_cost
            print('---------------------------------------')
            if p.engine == 'postgres':
                node.info['explain_json'] = result[0][0][0]
                # 'node' is a PG plan; doesn't make sense to print if executed
                # on a different engine.
                print(node)
            print(message)
            print('q{},{:.1f} (baseline)'.format(node.info['query_name'],
                                                 real_cost))
            print('Execution time: {}'.format(real_cost))
        # NOTE: if engine != pg, we're still saving PG plans but with target
        # engine's latencies.  This mainly affects debug strings.
        Save(self.workload, './data/initial_policy_data.pkl')
        self.LogExpertExperience(self.train_nodes, self.test_nodes)
      
    def Run(self):
        p = self.params
        if p.run_baseline:
            return self.RunBaseline()
        else:
            self.curr_value_iter = 0
            self.num_query_execs = 0
            self.num_total_timeouts = 0
            self.overall_best_train_latency = np.inf
            self.overall_best_test_latency = np.inf
            self.overall_best_test_swa_latency = np.inf
            self.overall_best_test_ema_latency = np.inf
            # For reporting cleaner hint strings for expert plans, remove their
            # unary ops (e.g., Aggregates).  These calls return copies, so
            # self.{all,train,test}_nodes no longer share any references.
            self.train_nodes = plans_lib.FilterScansOrJoins(self.train_nodes)
            self.test_nodes = plans_lib.FilterScansOrJoins(self.test_nodes)

        while self.curr_value_iter < p.val_iters:
            has_timeouts = self.RunOneIter()
            self.LogTimings()

            if (p.early_stop_on_skip_fraction is not None and
                    self.curr_iter_skipped_queries >=
                    p.early_stop_on_skip_fraction * len(self.train_nodes)):
                break

            if p.drop_cache and p.use_local_execution:
                print('Dropping buffer cache.')
                postgres.DropBufferCache()

            if p.increment_iter_despite_timeouts:
                # Always increment the iteration counter.  This makes it fairer
                # to compare runs with & without the timeout mechanism (or even
                # between timeout runs).
                self.curr_value_iter += 1
                self.lr_schedule.Step()
                if self.adaptive_lr_schedule is not None:
                    self.adaptive_lr_schedule.Step()
            else:
                if has_timeouts:
                    # Don't count this value iter.
                    # NOTE: it is possible for runs with use_timeout=False to
                    # have timeout events.  This can happen due to pg_executor
                    # encountering an out-of-memory / internal error and
                    # treating an execution as a timeout.
                    pass
                else:
                    self.curr_value_iter += 1
                    self.lr_schedule.Step()
                    if self.adaptive_lr_schedule is not None:
                        self.adaptive_lr_schedule.Step()


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