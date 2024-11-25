import balsa
from balsa import hyperparams

class BalsaParams(object):
      
    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        p.Define('db', 'imdbload', 'Name of the Postgres database.')
        p.Define('query_dir', 'queries/join-order-benchmark',
                 'Directory of the .sql queries.')
        p.Define('query_glob', '*.sql',
            'If supplied, glob for this pattern. Otherwise, use all queries. Example: 29*.sql.')
        p.Define('test_query_glob', None,
            'Similar usage as query_glob. If None, treat all queries as training nodes.')
        p.Define('engine', 'postgres',
                 'The execution engine.  Options: postgres.')
        p.Define('engine_dialect_query_dir', None,
                 'Directory of the .sql queries in target engine\'s dialect.')
        p.Define('run_baseline', False,
                 'If true, just load the queries and run them.')
        p.Define(
            'drop_cache', True,
            'If true, drop the buffer cache at the end of each value iteration.'
        )
        p.Define(
            'plan_physical', True,
            'If true, plans physical scan/join operators.  '\
            'Otherwise, just join ordering.'
        )
        p.Define('use_local_execution', False,
                 'For query executions, connect to local engine or the remote'\
                 ' cluster?  Non-execution EXPLAINs are always issued to'\
                 ' local.')
        p.Define('search_space_join_ops',
                 ['Hash Join', 'Merge Join', 'Nested Loop'],
                 'Action space: join operators to learn and use.')
        p.Define('search_space_scan_ops',
                 ['Index Scan', 'Index Only Scan', 'Seq Scan'],
                 'Action space: scan operators to learn and use.')
        
        #LR
        p.Define('lr', 1e-3, 'Learning rate.')
        p.Define('use_adaptive_lr', None, 'Experimental.')
        p.Define('linear_decay_to_zero', False,
                 'Linearly decay from lr to 0 in val_iters.')

        # Training.
        p.Define('val_iters', 500, '# of value iterations.')

        # Pre-training.
        p.Define('sim', True, 'Initialize from a pre-trained SIM model?')

        p.Define(
                  'init_experience', 'data/initial_policy_data.pkl',
                  'Initial data set of query plans to learn from. By default, this'\
                  ' is the expert optimizer experience collected when baseline'\
                  ' performance is evaluated.'
            )
        
        # Offline replay.
        p.Define(
            'prev_replay_buffers_glob', None,
            'If specified, load previous replay buffers and merge them as training purpose.'
        )
        p.Define(
            'prev_replay_buffers_glob_val', None,
            'If specified, load previous replay buffers and merge them as validation purpose.'
        )
        
        # Modeling: tree convolution (suggested).
        p.Define('tree_conv', True,
                 'If true, use tree convolutional neural net.')
        p.Define('tree_conv_version', None, 'Options: None.')
        p.Define('sim_query_featurizer', True,
                 'If true, use SimQueryFeaturizer to produce query features.')
        return p
            

@balsa.params_registry.Register
class Baseline(BalsaParams):

    def Params(self):
        p = super().Params()
        p.run_baseline = True
        return p