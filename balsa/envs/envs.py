import os
import glob

import numpy as np
import balsa
from balsa import hyperparams
from balsa.util import postgres
from balsa.util import plans_lib


def ParseSqlToNode(path):
    base = os.path.basename(path)
    query_name = os.path.splitext(base)[0]
    with open(path, 'r') as f:
        sql_string = f.read()
    node, json_dict = postgres.SqlToPlanNode(sql_string)
    node.info['path'] = path
    node.info['sql_str'] = sql_string
    node.info['query_name'] = query_name
    node.info['explain_json'] = json_dict
    node.GetOrParseSql()
    return node


class Workload(object):

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        p.Define('query_dir', None, 'Directory to workload queries.')
        p.Define(
            'query_glob', '*.sql',
            'If supplied, glob for this pattern.  Otherwise, use all queries.'\
            '  Example: 29*.sql.'
        )
        p.Define(
            'loop_through_queries', False,
            'Loop through a random permutation of queries? '
            'Desirable for evaluation.')
        p.Define(
            'test_query_glob', None,
            'Similar usage as query_glob. If None, treating all queries'\
            ' as training nodes.'
        )
        p.Define('search_space_join_ops',
                 ['Hash Join', 'Merge Join', 'Nested Loop'],
                 'Join operators to learn.')
        p.Define('search_space_scan_ops',
                 ['Index Scan', 'Index Only Scan', 'Seq Scan'],
                 'Scan operators to learn.')
        return p

    def __init__(self, params):
        self.params = params.Copy()
        p = self.params
        # Subclasses should populate these fields.
        self.query_nodes = None
        self.workload_info = None
        self.train_nodes = None
        self.test_nodes = None

        if p.loop_through_queries:
            self.queries_permuted = False
            self.queries_ptr = 0

    def _get_sql_set(self, query_dir, query_glob):
        if query_glob is None:
            return set()
        else:
            globs = query_glob
            if type(query_glob) is str:
                globs = [query_glob]
            sql_files = np.concatenate([
                glob.glob('{}/{}'.format(query_dir, pattern))
                for pattern in globs
            ]).ravel()
        sql_files = set(sql_files)
        return sql_files
    
    def Queries(self, split='all'):
        """Returns all queries as balsa.Node objects."""
        assert split in ['all', 'train', 'test'], split
        if split == 'all':
            return self.query_nodes
        elif split == 'train':
            return self.train_nodes
        elif split == 'test':
            return self.test_nodes
        
    def UseDialectSql(self, p):
        dialect_sql_dir = p.engine_dialect_query_dir
        for node in self.query_nodes:
            assert 'sql_str' in node.info and 'query_name' in node.info
            path = os.path.join(dialect_sql_dir,
                                node.info['query_name'] + '.sql')
            assert os.path.isfile(path), '{} does not exist'.format(path)
            with open(path, 'r') as f:
                dialect_sql_string = f.read()
            node.info['sql_str'] = dialect_sql_string
            

class JoinOrderBenchmark(Workload):

    @classmethod
    def Params(cls):
        p = super().Params()
        # Needs to be an absolute path for rllib.
        module_dir = os.path.abspath(os.path.dirname(balsa.__file__) + '/../')
        p.query_dir = os.path.join(module_dir, 'queries/join-order-benchmark')
        return p

    def __init__(self, params):
        super().__init__(params)
        p = params
        self.query_nodes, self.train_nodes, self.test_nodes = \
            self._LoadQueries()
        self.workload_info = plans_lib.WorkloadInfo(self.query_nodes)
        self.workload_info.SetPhysicalOps(p.search_space_join_ops,
                                          p.search_space_scan_ops)
        
    def _LoadQueries(self):
        """Loads all queries into balsa.Node objects."""
        p = self.params
        all_sql_set = self._get_sql_set(p.query_dir, p.query_glob)
        test_sql_set = self._get_sql_set(p.query_dir, p.test_query_glob)
        assert test_sql_set.issubset(all_sql_set)
        # sorted by query id for easy debugging
        all_sql_list = sorted(all_sql_set)
        all_nodes = [ParseSqlToNode(sqlfile) for sqlfile in all_sql_list]

        train_nodes = [
            n for n in all_nodes
            if p.test_query_glob is None or n.info['path'] not in test_sql_set
        ]
        test_nodes = [n for n in all_nodes if n.info['path'] in test_sql_set]
        assert len(train_nodes) > 0

        return all_nodes, train_nodes, test_nodes