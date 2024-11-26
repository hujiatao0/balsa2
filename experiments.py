import balsa
from balsa import hyperparams


SLOW_TEST_QUERIES = [
    '16b.sql', '17a.sql', '17e.sql', '17f.sql', '17b.sql', '19d.sql', '17d.sql',
    '17c.sql', '10c.sql', '26c.sql', '25c.sql', '6d.sql', '6f.sql', '8c.sql',
    '18c.sql', '9d.sql', '30a.sql', '19c.sql', '20a.sql'
]

# A random split using seed 52.  Test latency is chosen to be close to the
# bootstrapped mean.
RAND_52_TEST_QUERIES = [
    '8a.sql', '16a.sql', '2a.sql', '30c.sql', '17e.sql', '20a.sql', '26b.sql',
    '12b.sql', '15b.sql', '15d.sql', '10b.sql', '15a.sql', '4c.sql', '4b.sql',
    '22b.sql', '17c.sql', '24b.sql', '10a.sql', '22c.sql'
]

LR_SCHEDULES = {
    'C': {
        'lr_piecewise': [
            (0, 0.001),
            (50, 0.0005),
            (100, 0.00025),
            (150, 0.000125),
            (200, 0.0001),
        ]
    },
    # Delay C's decay by 10 iters.
    'C10': {
        'lr_piecewise': [
            (0, 0.001),
            (50 + 10, 0.0005),
            (100 + 10, 0.00025),
            (150 + 10, 0.000125),
            (200 + 10, 0.0001),
        ]
    },
}

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
        p.Define('cost_model', 'postgrescost',
                 'A choice of postgrescost, mincardcost.')
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
        p.Define('lr_piecewise', None, 'If supplied, use Piecewise.  Example:'\
                 '[(0, 1e-3), (200, 1e-4)].')
        p.Define('use_adaptive_lr', None, 'Experimental.')
        p.Define('linear_decay_to_zero', False,
                 'Linearly decay from lr to 0 in val_iters.')

        # Training.
        p.Define('epochs', 100, 'Num epochs to train.')
        p.Define('bs', 1024, 'Batch size.')
        p.Define('val_iters', 500, '# of value iterations.')
        p.Define('increment_iter_despite_timeouts', False,
                 'Increment the iteration counter even if timeouts occurred?')
        
        p.Define('label_transforms', ['log1p', 'standardize'],
                 'Transforms for labels.')
        p.Define('loss_type', None, 'Options: None (MSE), mean_qerror.')
        p.Define('gradient_clip_val', 0, 'Clip the gradient norm computed over'\
                 ' all model parameters together. 0 means no clipping.')
        
        # Validation.
        p.Define('validate_fraction', 0.1,
                 'Sample this fraction of the dataset as the validation set.  '\
                 '0 to disable validation.')

        # Pre-training.
        p.Define('sim', True, 'Initialize from a pre-trained SIM model?')
        p.Define(
            'sim_checkpoint', None,
            'Path to a pretrained SIM checkpoint.  Load it instead '
            'of retraining.')

        p.Define(
                  'init_experience', 'data/initial_policy_data.pkl',
                  'Initial data set of query plans to learn from. By default, this'\
                  ' is the expert optimizer experience collected when baseline'\
                  ' performance is evaluated.'
            )
        p.Define(
            'generic_ops_only_for_min_card_cost', False,
            'This affects sim model training and only if MinCardCost is used. '\
            'See sim.py for documentation.')
        
        # Training data / replay buffer.
        p.Define('on_policy', False,
                 'Whether to train on only data from the latest iteration.')
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
        
        # Featurization.
        p.Define('perturb_query_features', None,
                 'If not None, randomly perturb query features on each forward'\
                 ' pass, and this flag specifies '\
                 '(perturb_prob_per_table, [scale_min, scale_max]).  '\
                 'A multiplicative scale is drawn from '\
                 'Unif[scale_min, scale_max].  Only performed when training '\
                 'and using a query featurizer with perturbation implemented.')
        
        # Inference.
        p.Define('beam', 20, 'Beam size.')
        p.Define(
            'search_method', 'beam_bk',
            'Algorithm used to search for execution plans with cost model.')
        p.Define(
            'search_until_n_complete_plans', 10,
            'Keep doing plan search for each query until this many complete'\
            ' plans have been found.  Returns the predicted cheapest one out'\
            ' of them.  Recommended: 10.')
        
        # Exploration during inference.
        p.Define('explore_visit_counts', False, 'Explores using a visit count?')
        p.Define('explore_visit_counts_sort', False,
                 'Explores by executing the plan with the smallest '\
                 '(visit count, predicted latency) out of k-best plans?')
        return p
            

@balsa.params_registry.Register
class Baseline(BalsaParams):

    def Params(self):
        p = super().Params()
        p.run_baseline = True
        return p
    

@balsa.params_registry.Register
class MinCardCost(BalsaParams):

    def Params(self):
        p = super().Params()
        p.cost_model = 'mincardcost'
        p.sim_checkpoint = None
        # Exploration schemes.
        p.explore_visit_counts = True
        return p


@balsa.params_registry.Register
class MinCardCostSortCnts(MinCardCost):

    def Params(self):
        return super().Params().Set(
            explore_visit_counts=False,
            explore_visit_counts_sort=True,
        )


@balsa.params_registry.Register
class MinCardCostOnPol(MinCardCostSortCnts):

    def Params(self):
        p = super().Params()

        from_p = BalsaParams().Params()
        from_p.cost_model = 'mincardcost'
        from_p.query_glob = ['*.sql']
        from_p.test_query_glob = 'TODO: Subclasses should fill this.'
        from_p.sim_checkpoint = None
        # Exploration schemes.
        from_p.explore_visit_counts = False
        from_p.explore_visit_counts_sort = True

        p = hyperparams.CopyFieldsTo(from_p, p)
        return p.Set(on_policy=True)


@balsa.params_registry.Register
class Rand52MinCardCostOnPol(MinCardCostOnPol):

    def Params(self):
        p = super().Params()
        p.test_query_glob = RAND_52_TEST_QUERIES
        p.sim_checkpoint = 'checkpoints/sim-MinCardCost-rand52split-680secs.ckpt'
        return p


@balsa.params_registry.Register
class Rand52MinCardCostOnPolLrC(Rand52MinCardCostOnPol):

    def Params(self):
        return super().Params().Set(**LR_SCHEDULES['C'])


@balsa.params_registry.Register  # keep
class Balsa_JOBRandSplit(Rand52MinCardCostOnPolLrC):

    def Params(self):
        p = super().Params()
        p.increment_iter_despite_timeouts = True
        p = p.Set(**LR_SCHEDULES['C10'])
        return p