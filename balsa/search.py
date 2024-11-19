import collections
import copy

import numpy as np
from balsa import hyperparams
from balsa import costing


class DynamicProgramming(object):
    """Bottom-up dynamic programming plan search."""

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        p.Define('cost_model', costing.NullCost.Params(),
                 'Params of the cost model to use.')
        p.Define('search_space', 'bushy',
                 'Options: bushy, dbmsx, bushy_norestrict.')

        # Physical planning.
        p.Define('plan_physical_ops', False, 'Do we plan physical joins/scans?')

        # On enumeration hook.
        p.Define(
            'collect_data_include_suboptimal', True, 'Call on enumeration'
            ' hooks on suboptimal plans for each k-relation?')
        return p

    def __init__(self, params):
        self.params = params.Copy()
        p = self.params
        self.cost_model = p.cost_model.cls(p.cost_model)
        self.on_enumerated_hooks = []

        assert p.search_space in ('bushy', 'dbmsx',
                                  'bushy_norestrict'), 'Not implemented.'

        self.join_ops = ['Join']
        self.scan_ops = ['Scan']
        self.use_plan_restrictions = (p.search_space != 'bushy_norestrict')
        