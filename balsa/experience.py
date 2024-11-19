import numpy as np
from balsa.util import plans_lib, postgres

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
