import argparse
import os
import csv
from sqlglot import exp, parse_one
from collections import OrderedDict
import argparse
import os
import re
from sqlglot import exp, parse_one
from collections import OrderedDict

def clean_val(val_str):
    """剔除字符串中的不可见字符、换行符等乱码源"""
    # 仅保留可打印字符
    return re.sub(r'[^\x20-\x7E]', '', val_str).strip()

def parse_sql_to_csv_row(line):
    line = line.strip()
    if not line:
        return None

    parts = line.split("||")
    sql_str = parts[0].strip()
    true_cardinality = clean_val(parts[1])

    if sql_str.startswith("/*"):
        sql_str = sql_str.split("*/", 1)[-1].strip()

    try:
        expression = parse_one(sql_str, read="postgres")
        
        # 1. Tables & Aliases
        table_alias_list = []
        table_map = OrderedDict() 
        for table in expression.find_all(exp.Table):
            table_name = table.this.sql()
            alias = table.alias if table.alias else table_name
            table_map[alias] = table_name
            table_alias_list.append(f"{table_name} {alias}")
        
        tables_part = ",".join(table_alias_list)

        # 2. Join & Predicates
        where_clause = expression.find(exp.Where)
        join_conditions = []
        predicates = []

        if where_clause:
            conjuncts = list(where_clause.this.flatten()) if isinstance(where_clause.this, exp.And) else [where_clause.this]
            
            for condition in conjuncts:
                cols = list(condition.find_all(exp.Column))
                involved_aliases = set(col.table for col in cols if col.table)
                
                if len(involved_aliases) >= 2:
                    # 去掉最外层可能的括号
                    condition_sql = condition.sql().replace(" ", "")
                    condition_sql = condition_sql.strip("()")
                    join_conditions.append(condition_sql)
                else:
                    # 针对 Predicate 的精细化处理
                    if isinstance(condition, (exp.Binary, exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ)):
                        left = condition.left.sql().replace(" ", "")
                        
                        # 获取操作符
                        op_map = {"EQ": "=", "GT": ">", "LT": "<", "GTE": ">=", "LTE": "<=", "NEQ": "!="}
                        op_str = op_map.get(condition.key.upper(), condition.key)
                        
                        # 右值处理：区分数字和字符串
                        right_node = condition.right
                        raw_val = right_node.sql(dialect="postgres").strip()
                        cleaned_val = clean_val(raw_val)
                        
                        # 如果原本就是字符串（带引号的），保留单引号
                        # 或者检查节点类型是否为 Literal 且是 string
                        if isinstance(right_node, exp.Literal) and right_node.is_string:
                            # 确保是 'val' 格式，而不是双引号或无引号
                            val_content = cleaned_val.replace("'", "").replace('"', "")
                            final_right = f"'{val_content}'"
                        else:
                            final_right = cleaned_val
                            
                        predicates.append(f"{left},{op_str},{final_right}")
                    else:
                        predicates.append(clean_val(condition.sql().replace(" ", "")))

        return f"{tables_part}#{','.join(join_conditions)}#{','.join(predicates)}#{true_cardinality}"
        # return f"{tables_part}#{','.join(join_conditions)}#{','.join(predicates)}#"

    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert SQL with cardinalities to MSCN-style CSV.")
    parser.add_argument("--input", required=True, help="Input .sql file path")
    parser.add_argument("--output", required=True, help="Output .csv file path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found.")
        return

    rows_count = 0
    with open(args.input, "r") as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            csv_row = parse_sql_to_csv_row(line)
            if csv_row:
                f_out.write(csv_row + "\n")
                rows_count += 1

    print(f"Conversion complete. Processed {rows_count} queries. Saved to {args.output}")

if __name__ == "__main__":
    main()