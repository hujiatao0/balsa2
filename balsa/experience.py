import numpy as np
from balsa.util import plans_lib, postgres
import os
import pickle
import pprint
import time
import glob
import gc
import collections

from balsa.models import treeconv


def TreeConvFeaturize(plan_featurizer, subplans):
    """Returns (featurized plans, tree conv indexes) tensors."""
    assert len(subplans) > 0
    # This class currently requires batch-featurizing, due to internal
    # padding.  This is different from our other per-node Featurizers.
    print('Calling make_and_featurize_trees()...')
    t1 = time.time()
    trees, indexes = treeconv.make_and_featurize_trees(subplans,
                                                       plan_featurizer)
    print('took {:.1f}s'.format(time.time() - t1))
    return trees, indexes

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


class SimpleReplayBuffer(Experience):
    """A simple replay buffer.

    It featurizes each element in 'self.nodes' independently, without
    performing subtree extraction, deduplication, take-the-minimum-cost, or
    assuming relationships among query nodes.

    Usage:

       nodes = [ <list of subplans from a single query, dedup'd> ]
       buffer = SimpleReplayBuffer(nodes)

       nodes = [ <N dedup'd subplans from Q1>; <M from Q2> ]
       buffer = SimpleReplayBuffer(nodes)

       # Simply featurizes each element independently.
       data = buffer.featurize()
    """

    def featurize(self, rewrite_generic=False, verbose=False):
        self.featurize_with_subplans(self.nodes, rewrite_generic, verbose)

    def featurize_with_subplans(self,
                                subplans,
                                rewrite_generic=False,
                                verbose=False):
        t1 = time.time()
        assert len(subplans) == len(self.nodes), (len(subplans),
                                                  len(self.nodes))
        self.prepare(rewrite_generic, verbose)
        all_query_vecs = [None] * len(self.nodes)
        all_feat_vecs = [None] * len(self.nodes)
        all_pa_pos_vecs = [None] * len(self.nodes)
        all_costs = [None] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            all_query_vecs[i] = self.query_featurizer(node)
            all_costs[i] = node.cost

        print('Spent {:.1f}s'.format(time.time() - t1))
        if isinstance(self.featurizer, plans_lib.TreeNodeFeaturizer):
            all_feat_vecs, all_pa_pos_vecs = TreeConvFeaturize(
                self.featurizer, subplans)
        else:
            for i, node in enumerate(self.nodes):
                all_feat_vecs[i] = self.featurizer(subplans[i])

        # Debug print: check if query vectors are different/same.
        for i in range(min(len(self.nodes), 10)):
            print('query={} plan={} cost={}'.format(
                (all_query_vecs[i] *
                 np.arange(1, 1 + len(all_query_vecs[i]))).sum(),
                all_feat_vecs[i], all_costs[i]))

        return all_query_vecs, all_feat_vecs, all_pa_pos_vecs, all_costs