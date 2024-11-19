import pg_executor
import re
from balsa.util import plans_lib
import pprint


def SqlToPlanNode(sql,
                  comment=None,
                  verbose=False,
                  keep_scans_joins_only=False,
                  cursor=None):
    """Issues EXPLAIN(format json) on a SQL string; parse into our AST node."""
    # Use of 'verbose' would alias-qualify all column names in pushed-down
    # filters, which are beneficial for us (e.g., this ensures that
    # Node.to_sql() returns a non-ambiguous SQL string).
    # Ref: https://www.postgresql.org/docs/11/sql-explain.html
    geqo_off = comment is not None and len(comment) > 0
    result = _run_explain('explain(verbose, format json)',
                          sql,
                          comment,
                          verbose,
                          geqo_off=geqo_off,
                          cursor=cursor).result
    json_dict = result[0][0][0]
    node = ParsePostgresPlanJson(json_dict)
    if not keep_scans_joins_only:
        return node, json_dict
    return plans_lib.FilterScansOrJoins(node), json_dict


def _run_explain(explain_str,
                 sql,
                 comment,
                 verbose,
                 geqo_off=False,
                 timeout_ms=None,
                 cursor=None,
                 is_test=False,
                 remote=False):
    """
    Run the given SQL statement with appropriate EXPLAIN commands.

    timeout_ms is for both setting the timeout for PG execution and for the PG
    cluster manager, which will release the server after timeout expires.
    """
    # if is_test:
    #     assert remote, "testing queries must run on remote Postgres servers"
    if cursor is None and not remote:
        with pg_executor.Cursor() as cursor:
            return _run_explain(explain_str, sql, comment, verbose, geqo_off,
                                timeout_ms, cursor, remote)

    end_of_comment_idx = sql.find('*/')
    if end_of_comment_idx == -1:
        existing_comment = None
    else:
        split_idx = end_of_comment_idx + len('*/\n')
        existing_comment = sql[:split_idx]
        sql = sql[split_idx:]

    # Fuse hint comments.
    if comment:
        assert comment.startswith('/*+') and comment.endswith('*/'), (
            'Don\'t know what to do with these', sql, existing_comment, comment)
        if existing_comment is None:
            fused_comment = comment
        else:
            comment_body = comment[len('/*+ '):-len(' */')].rstrip()
            existing_comment_body_and_tail = existing_comment[len('/*+'):]
            fused_comment = '/*+\n' + comment_body + '\n' + existing_comment_body_and_tail
    else:
        fused_comment = existing_comment

    if fused_comment:
        s = fused_comment + '\n' + str(explain_str).rstrip() + '\n' + sql
    else:
        s = str(explain_str).rstrip() + '\n' + sql

    if remote:
        assert cursor is None
        return pg_executor.ExecuteRemote(s, verbose, geqo_off, timeout_ms)
    else:
        return pg_executor.Execute(s, verbose, geqo_off, timeout_ms, cursor)
    

def _FilterExprsByAlias(exprs, table_alias):
    # Look for <table_alias>.<stuff>.
    pattern = re.compile('.*\(?\\b{}\\b\..*\)?'.format(table_alias))
    return list(filter(pattern.match, exprs))
    

def ParsePostgresPlanJson(json_dict):
    """Takes JSON dict, parses into a Node."""
    curr = json_dict['Plan']

    def _parse_pg(json_dict, select_exprs=None, indent=0):
        op = json_dict['Node Type']
        cost = json_dict['Total Cost']
        if op == 'Aggregate':
            op = json_dict['Partial Mode'] + op
            if select_exprs is None:
                # Record the SELECT <select_exprs> at the topmost Aggregate.
                # E.g., ['min(mi.info)', 'min(miidx.info)', 'min(t.title)'].
                select_exprs = json_dict['Output']

        # Record relevant info.
        curr_node = plans_lib.Node(op)
        curr_node.cost = cost
        # Only available if 'analyze' is set (actual execution).
        curr_node.actual_time_ms = json_dict.get('Actual Total Time')
        if 'Relation Name' in json_dict:
            curr_node.table_name = json_dict['Relation Name']
            curr_node.table_alias = json_dict['Alias']

        # Unary predicate on a table.
        if 'Filter' in json_dict:
            assert 'Scan' in op, json_dict
            assert 'Relation Name' in json_dict, json_dict
            curr_node.info['filter'] = json_dict['Filter']

        if 'Scan' in op and select_exprs:
            # Record select exprs that belong to this leaf.
            # Assume: SELECT <exprs> are all expressed in terms of aliases.
            filtered = _FilterExprsByAlias(select_exprs, json_dict['Alias'])
            if filtered:
                curr_node.info['select_exprs'] = filtered

        # Recurse.
        if 'Plans' in json_dict:
            for n in json_dict['Plans']:
                curr_node.children.append(
                    _parse_pg(n, select_exprs=select_exprs, indent=indent + 2))

        # Special case.
        if op == 'Bitmap Heap Scan':
            for c in curr_node.children:
                if c.node_type == 'Bitmap Index Scan':
                    # 'Bitmap Index Scan' doesn't have the field 'Relation Name'.
                    c.table_name = curr_node.table_name
                    c.table_alias = curr_node.table_alias

        return curr_node

    return _parse_pg(curr)


def EstimateFilterRows(nodes):
    """For each node, issues an EXPLAIN to estimates #rows of unary preds.

    Writes result back into node.info['all_filters_est_rows'], as { relation
    id: num rows }.
    """
    if isinstance(nodes, plans_lib.Node):
        nodes = [nodes]
    cache = {}
    with pg_executor.Cursor() as cursor:
        for node in nodes:
            for table_id, pred in node.info['all_filters'].items():
                key = (table_id, pred)
                if key not in cache:
                    sql = 'EXPLAIN(format json) SELECT * FROM {} WHERE {};'.format(
                        table_id, pred)
                    cursor.execute(sql)
                    json_dict = cursor.fetchall()[0][0][0]
                    num_rows = json_dict['Plan']['Plan Rows']
                    cache[key] = num_rows
    print('{} unique filters'.format(len(cache)))
    pprint.pprint(cache)
    for node in nodes:
        d = {}
        for table_id, pred in node.info['all_filters'].items():
            d[table_id] = cache[(table_id, pred)]
        node.info['all_filters_est_rows'] = d