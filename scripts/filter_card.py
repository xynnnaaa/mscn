# #!/usr/bin/env python3
# """
# 过滤查询并拼接基数结果。
# 用法：
#     python filter_queries.py <输入.sql> <输入.csv> <基数.txt> <输出.sql> <输出.csv>
# """

# import sys
# import argparse

# def is_valid_cardinality(card_str):
#     """检查基数字符串是否为有效数字且非零"""
#     try:
#         val = float(card_str)
#         return val != 0.0
#     except ValueError:
#         return False

# def main():

#     input_sql = "/data2/xuyining/learnedcardinalities/data/tpch-skew/ori_query/test-all.sql"
#     input_csv = "/data2/xuyining/learnedcardinalities/data/tpch-skew/ori_query/test-all.csv"
#     input_txt = "/data2/xuyining/learnedcardinalities/data/tpch-skew/ori_query/test_card.txt"
#     output_sql = "/data2/xuyining/learnedcardinalities/data/tpch-skew/test.sql"
#     output_csv = "/data2/xuyining/learnedcardinalities/data/tpch-skew/test.csv"
    

#     # 读取所有文件行
#     with open(input_sql, 'r', encoding='utf-8') as f:
#         sql_lines = f.readlines()
#     with open(input_csv, 'r', encoding='utf-8') as f:
#         csv_lines = f.readlines()
#     with open(input_txt, 'r', encoding='utf-8') as f:
#         card_lines = f.readlines()

#     # 行数校验
#     n = len(sql_lines)
#     if len(csv_lines) != n or len(card_lines) != n:
#         print(f"错误：三个文件的行数不一致。SQL: {n}, CSV: {len(csv_lines)}, TXT: {len(card_lines)}", file=sys.stderr)
#         sys.exit(1)

#     # 去除基数行的换行符，用于检查
#     card_values = [line.strip() for line in card_lines]

#     keep_indices = set()
#     i = 0
#     while i < n:
#         # 完整查询行不应该以 "/*" 开头
#         if sql_lines[i].lstrip().startswith("/*"):
#             print(f"警告：第 {i+1} 行以 '/*' 开头，但预期是完整查询行。跳过该行并视为新块。", file=sys.stderr)
#             # 为安全起见，仍然将其视为一个独立块（完整查询）
#             block = [i]
#             i += 1
#         else:
#             # 收集当前完整查询及其子查询的索引
#             block = [i]
#             i += 1
#             while i < n and sql_lines[i].lstrip().startswith("/*"):
#                 block.append(i)
#                 i += 1

#         # 检查整个块的所有基数是否有效且非零
#         block_valid = all(is_valid_cardinality(card_values[idx]) for idx in block)
#         if block_valid:
#             for idx in block:
#                 keep_indices.add(idx)

#     if not keep_indices:
#         print("警告：没有查询通过过滤，输出文件将为空。", file=sys.stderr)

#     # 输出过滤后的SQL和CSV
#     with open(output_sql, 'w', encoding='utf-8') as f_sql, \
#          open(output_csv, 'w', encoding='utf-8') as f_csv:
#         for idx in range(n):
#             if idx not in keep_indices:
#                 continue

#             # 处理SQL行：去除原换行，添加 "||基数"
#             sql_base = sql_lines[idx].rstrip('\n').rstrip('\r')
#             f_sql.write(f"{sql_base}||{card_values[idx]}\n")

#             # 处理CSV行：去除原换行，追加逗号和基数
#             csv_base = csv_lines[idx].rstrip('\n').rstrip('\r')
#             f_csv.write(f"{csv_base}{card_values[idx]}\n")

#     print(f"过滤完成。保留 {len(keep_indices)} 行（共 {n} 行）。")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
过滤查询行并拼接基数结果（仅根据基数是否合法，逐行独立判断）。
用法：
    python filter_queries.py <输入.sql> <输入.csv> <基数.txt> <输出.sql> <输出.csv>
"""

import sys
import argparse

def is_valid_cardinality(card_str):
    """检查基数字符串是否能转换为数字（TIMEOUT/ERROR等无效）"""
    try:
        val = float(card_str)
        return val != 0.0
    except ValueError:
        return False

def main():
    
    input_sql = "/data2/xuyining/learnedcardinalities/data/tpch-skew/ori_query/train.sql"
    input_csv = "/data2/xuyining/learnedcardinalities/data/tpch-skew/ori_query/train.csv"
    input_txt = "/data2/xuyining/learnedcardinalities/data/tpch-skew/ori_query/train_card.txt"
    output_sql = "/data2/xuyining/learnedcardinalities/data/tpch-skew/train.sql"
    output_csv = "/data2/xuyining/learnedcardinalities/data/tpch-skew/train.csv"

    # 读取所有文件行
    with open(input_sql, 'r', encoding='utf-8') as f:
        sql_lines = f.readlines()
    with open(input_csv, 'r', encoding='utf-8') as f:
        csv_lines = f.readlines()
    with open(input_txt, 'r', encoding='utf-8') as f:
        card_lines = f.readlines()

    # 行数校验
    n = len(sql_lines)
    if len(csv_lines) != n or len(card_lines) != n:
        print(f"错误：三个文件的行数不一致。SQL: {n}, CSV: {len(csv_lines)}, TXT: {len(card_lines)}", file=sys.stderr)
        sys.exit(1)

    # 去除基数行的换行符，用于检查
    card_values = [line.strip() for line in card_lines]

    # 逐行判断基数是否合法，合法的行保留
    keep_indices = set()
    for i in range(n):
        if is_valid_cardinality(card_values[i]):
            keep_indices.add(i)

    if not keep_indices:
        print("警告：没有查询通过过滤，输出文件将为空。", file=sys.stderr)

    # 输出过滤后的SQL和CSV
    with open(output_sql, 'w', encoding='utf-8') as f_sql, \
         open(output_csv, 'w', encoding='utf-8') as f_csv:
        for idx in range(n):
            if idx not in keep_indices:
                continue

            # 处理SQL行：去除原换行，用新的基数替换原行||后面的数字
            sql_base = sql_lines[idx].rstrip('\n').rstrip('\r')
            sql_base = sql_base.split('||')[0]  # 去掉原有的基数部分
            f_sql.write(f"{sql_base}||{card_values[idx]}\n")

            # 处理CSV行：去除原换行，追加基数
            csv_base = csv_lines[idx].rstrip('\n').rstrip('\r')
            f_csv.write(f"{csv_base}{card_values[idx]}\n")

    print(f"过滤完成。保留 {len(keep_indices)} 行（共 {n} 行）。")

if __name__ == "__main__":
    main()