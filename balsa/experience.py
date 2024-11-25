import numpy as np
from balsa.util import plans_lib, postgres
import os
import pickle
import pprint
import time
import glob
import gc
import collections

class Experience(object):

    def __init__(
        self,
        data,
        tree_conv=False,
        keep_scans_joins_only=True,
        plan_featurizer_cls=plans_lib.PreOrderSequenceFeaturizer,
        query_featurizer_cls=plans_lib.QueryFeaturizer,
        workload_info=None,
    ):
        self.tree_conv = tree_conv
        if keep_scans_joins_only:
            print('plans_lib.FilterScansOrJoins()')
            self.nodes = plans_lib.FilterScansOrJoins(data)
        else:
            self.nodes = data
        self.initial_size = len(self.nodes)

        self.plan_featurizer_cls = plan_featurizer_cls
        self.query_featurizer_cls = query_featurizer_cls
        self.query_featurizer = None
        self.workload_info = workload_info

        #### Affects query featurization.
        print('plans_lib.GatherUnaryFiltersInfo()')
        # Requires:
        #   leaf.info['filter'] for leaf under node.
        # Writes:
        #   node.info['all_filters'], which is used by EstimateFilterRows()
        #   below.
        plans_lib.GatherUnaryFiltersInfo(self.nodes)

        print('postgres.EstimateFilterRows()')
        # Get histogram-estimated # rows of filters from PG.
        # Requires:
        #   node.info['all_filters']
        # Writes:
        #   node.info['all_filters_est_rows'], which is used by, e.g.,
        #   plans_lib.QueryFeaturizer.
        #
        # TODO: check that first N nodes don't change.
        postgres.EstimateFilterRows(self.nodes)

    def Save(self, path):
        """Saves all Nodes in the current replay buffer to a file."""
        if os.path.exists(path):
            old_path = path
            path = '{}-{}'.format(old_path, time.time())
            print('Path {} exists, appending current time: {}'.format(
                old_path, path))
            assert not os.path.exists(path), path
        to_save = (self.initial_size, self.nodes)
        with open(path, 'wb') as f:
            pickle.dump(to_save, f)
        print('Saved Experience to:', path)

    def Load(self, path_glob, keep_last_fraction=1):
        """Loads multiple serialized Experience buffers into a single one.

        The 'initial_size' Nodes from self would be kept, while those from the
        loaded buffers would be dropped.  Internally, checked that all buffers
        and self have the same 'initial_size' field.
        """
        paths = glob.glob(os.path.expanduser(path_glob))
        if not paths:
            raise ValueError('No replay buffer files found')
        assert 0 <= keep_last_fraction <= 1, keep_last_fraction
        # query name -> set(plan string)
        total_unique_plans_table = collections.defaultdict(set)
        total_num_unique_plans = 0
        initial_nodes_len = len(self.nodes)
        for path in paths:
            t1 = time.time()
            print('Loading replay buffer', path)
            # np.load() is much faster than pickle.load; disabling gc provides
            # further speedups.
            gc.disable()
            loaded = np.load(path, allow_pickle=True)
            gc.enable()
            print('  ...took {:.1f} seconds'.format(time.time() - t1))
            initial_size, nodes = loaded
            # Sanity checks & some invariant checks.  A more stringent check
            # would be to check that:
            #   buffer 1: qname_0 qname_1 ...
            #   ...
            #   buffer N: qname_0 qname_1, ...
            # I.e., query names all correspond.
            assert type(initial_size) is int and type(nodes) is list, path
            assert initial_size == self.initial_size, (path, initial_size,
                                                       self.initial_size)
            assert len(nodes) >= initial_size and len(
                nodes) % initial_size == 0, (len(nodes), path)
            nodes_executed = nodes[initial_size:]
            if keep_last_fraction < 1:
                assert len(nodes_executed) % initial_size == 0
                num_iters = len(nodes_executed) // initial_size
                keep_num_iters = int(num_iters * keep_last_fraction)
                print('  orig len {} keeping the last fraction {} ({} iters)'.
                      format(len(nodes_executed), keep_last_fraction,
                             keep_num_iters))
                nodes_executed = nodes_executed[-(keep_num_iters *
                                                  initial_size):]
            self.nodes.extend(nodes_executed)
            # Analysis.
            num_unique_plans, unique_plans_table = Experience.CountUniquePlans(
                self.initial_size, nodes_executed)
            total_num_unique_plans_prev = total_num_unique_plans
            total_num_unique_plans = Experience.MergeUniquePlansInto(
                unique_plans_table, total_unique_plans_table)
            print('  num_unique_plans from loaded buffer {}; actually '\
                  'new unique plans contributed (after merging) {}'.
                  format(num_unique_plans,
                         total_num_unique_plans - total_num_unique_plans_prev))
        print('Loaded {} nodes from {} buffers; glob={}, paths:\n{}'.format(
            len(self.nodes) - initial_nodes_len, len(paths), path_glob,
            '\n'.join(paths)))
        print('Total unique plans (num_query_execs):', total_num_unique_plans)
