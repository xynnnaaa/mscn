import psycopg2
import numpy as np
import csv
import os
import struct
import argparse

# ======== 配置区域 ========
DB_CONFIG = {
    "host": "localhost",
    "database": "tpch-skew-10",
    "user": "xuyining",
    "port": 5433
}

INPUT_CSV = "/data2/xuyining/QuaSF/join_complexity/tpch-skew/test.csv"
OUTPUT_BITMAP = "/data2/xuyining/QuaSF/join_complexity/tpch-skew/bitmap/test.bitmaps"
SAMPLE_SIZE = 1000  # 对应位图 1000 bits
# =========================

import re
import datetime

def timestamp_to_string(timestamp_str):
    # 将时间戳字符串转换为整数
    unix_timestamp = int(timestamp_str)
    
    # 将 Unix 时间戳转换为 datetime 对象
    dt = datetime.datetime.fromtimestamp(unix_timestamp)
    
    # 将 datetime 对象格式化为时间字符串
    timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return timestamp_str

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

TABLE_EXISTENCE_CACHE = {}


def table_exists(cursor, table_name):
    if table_name not in TABLE_EXISTENCE_CACHE:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = %s
            );
            """,
            (table_name,),
        )
        TABLE_EXISTENCE_CACHE[table_name] = cursor.fetchone()[0]
    return TABLE_EXISTENCE_CACHE[table_name]


def get_bitmap(cursor, table_name, predicates, sample_mode="qa", qa_ratio=None):
    """
    执行查询并返回 {SAMPLE_SIZE} 位的位图
    """
    if sample_mode == "qa":
        ratio_suffix = f"_{qa_ratio}" if qa_ratio is not None else ""
        sample_table = (
            f"{table_name}_s{SAMPLE_SIZE}_join_drift_train_2_qa{ratio_suffix}"
        )
    elif sample_mode == "rs":
        sample_table = f"{table_name}_s{SAMPLE_SIZE}"
    else:
        raise ValueError(f"Unsupported sample mode: {sample_mode}")
    bitmap = np.zeros(SAMPLE_SIZE, dtype=np.uint8)

    # QA 专用样本不存在时，回退到同表的 random sample。
    if sample_mode == "qa" and not table_exists(cursor, sample_table):
        sample_table = f"{table_name}_s{SAMPLE_SIZE}"

    if not table_exists(cursor, sample_table):
        raise RuntimeError(f"Sample table does not exist: {sample_table}")
    
    # 基础查询语句：直接查满足条件的 id
    sql = f"SELECT id FROM {sample_table}"
    
    if predicates:
        sql_preds = []
        for p in predicates:
            # p[0] 是 t.production_year，需要去掉别名部分转为数据库列名
            col = p[0].split('.')[-1]
            op = p[1]
            # val = p[2].replace("'", "''") # 简单的 SQL 注入预防

            # if DB_CONFIG['database'] == 'stats':
            #     if col.endswith('date') or col.endswith('Date'):
            #         val = timestamp_to_string(val)
            #         sql_preds.append(f"{col} {op} '{val}'::timestamp")
            #     else:
            #         # 判断是否为数值，非数值则加单引号
            #         try:
            #             float(val)
            #             sql_preds.append(f"{col} {op} {val}")
            #         except ValueError:
            #             sql_preds.append(f"{col} {op} '{val}'")
            # else:
            #     try:
            #         float(val)
            #         sql_preds.append(f"{col} {op} {val}")
            #     except ValueError:
            #         sql_preds.append(f"{col} {op} '{val}'")


            # 🌟 核心修复：如果 CSV 中的值本身已经自带单引号（如 'FOB' 或 '1'），直接按原样拼接
            val = p[2]
            if val.startswith("'") and val.endswith("'"):
                sql_preds.append(f"{col} {op} {val}")
            else:
                # 只有不带引号的值（数字或 stats 的时间戳），才走原有的逻辑
                val = val.replace("'", "''") # 简单的 SQL 注入预防
                if DB_CONFIG['database'] == 'stats' and (col.endswith('date') or col.endswith('Date')):
                    val = timestamp_to_string(val)
                    sql_preds.append(f"{col} {op} '{val}'::timestamp")
                else:
                    try:
                        float(val)
                        sql_preds.append(f"{col} {op} {val}")
                    except ValueError:
                        sql_preds.append(f"{col} {op} '{val}'")
        
        sql += " WHERE " + " AND ".join(sql_preds)

    try:
        # print(f"\nExecuting SQL: {sql}")
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            id = int(row[0])
            # id 对SAMPLE_SIZE取模得到位图索引
            idx = (id - 1) % SAMPLE_SIZE
            bitmap[idx] = 1
    except Exception as e:
        print(f"\n[Warning] SQL Failed: {sql}")
        print(f"Error: {e}")
    
    return bitmap

def main(
    input_csv=INPUT_CSV,
    output_bitmap=OUTPUT_BITMAP,
    sample_mode="qa",
    qa_ratio=None,
):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    output_dir = os.path.dirname(output_bitmap)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 统计总行数用于进度显示
    with open(input_csv, 'r') as f:
        total_queries = sum(1 for _ in f)
    
    print(
        f"Starting bitmap generation for {total_queries} queries "
        f"with sample mode '{sample_mode}'"
        f"{f' and QA ratio {qa_ratio}' if qa_ratio is not None else ''}..."
    )

    with open(input_csv, 'r') as f_in, open(output_bitmap, 'wb') as f_out:
        reader = csv.reader(f_in, delimiter='#')

        has_zero_bitmap_cnt = 0
        
        for row_idx, row in enumerate(reader):
            # 1. 解析表信息 (title t, cast_info ci)
            tables_raw = row[0].split(',')
            # 建立别名到真实表名的映射
            alias_to_table = {}
            tables_order = [] # 记录查询中表的出现顺序
            for t_entry in tables_raw:
                parts = t_entry.strip().split(' ')
                real_name = parts[0]
                alias = parts[1] if len(parts) > 1 else parts[0]
                alias_to_table[alias] = real_name
                tables_order.append(alias)

            # 2. 解析谓词并按别名分组
            pred_groups = {alias: [] for alias in tables_order}
            if row[2]: # 如果有谓词
                pred_raw = '#'.join(row[2:-1])
                pred_raw = pred_raw.split(',')
                pred_list = list(chunks(pred_raw, 3))
                for p in pred_list:
                    if len(p) == 3:
                        alias = p[0].split('.')[0]
                        if alias in pred_groups:
                            pred_groups[alias].append(p)

            # 3. 写入当前查询包含的位图数量 (4字节)
            f_out.write(struct.pack('<I', len(tables_order)))

            has_zero_bitmap = False
            
            # 4. 生成并写入每个表的位图
            for alias in tables_order:
                real_table = alias_to_table[alias]
                # 执行数据库查询
                bitmap_arr = get_bitmap(
                    cursor,
                    real_table,
                    pred_groups[alias],
                    sample_mode=sample_mode,
                    qa_ratio=qa_ratio,
                )

                #检查是否为全零
                if np.all(bitmap_arr == 0):
                    has_zero_bitmap = True

                # 如果长度不是 8 的倍数，补 0 到下一个字节边界再打包
                pad_len = (8 - bitmap_arr.size % 8) % 8
                if pad_len:
                    bitmap_arr = np.pad(bitmap_arr, (0, pad_len), mode='constant', constant_values=0)

                packed = np.packbits(bitmap_arr)
                f_out.write(packed.tobytes())

            if has_zero_bitmap:
                has_zero_bitmap_cnt += 1

                # print(f"\n[Warning] Query {row_idx + 1} has a zero bitmap. Total so far: {has_zero_bitmap_cnt}")

            if (row_idx + 1) % 500 == 0:
                print(f"Processed {row_idx + 1}/{total_queries} queries...", end='\r')

    cursor.close()
    conn.close()
    print(f"\nSuccessfully generated: {output_bitmap}")
    print(f"Queries with zero bitmaps: {has_zero_bitmap_cnt}")

    return has_zero_bitmap_cnt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MSCN bitmaps for a workload CSV.")
    parser.add_argument("--input_csv", default=INPUT_CSV)
    parser.add_argument("--output_bitmap", default=OUTPUT_BITMAP)
    parser.add_argument(
        "--sample_mode",
        choices=("qa", "rs"),
        default="qa",
        help=(
            "Use *_join_drift_qa[_RATIO] tables (with RS fallback) or "
            "*_s1000 tables."
        ),
    )
    parser.add_argument(
        "--qa_ratio",
        type=int,
        default=None,
        help=(
            "Optional QA mix ratio appended to the sample table name, e.g. "
            "25 selects *_s1000_join_drift_qa_25."
        ),
    )
    args = parser.parse_args()
    if args.qa_ratio is not None and not 0 <= args.qa_ratio <= 100:
        parser.error("--qa_ratio must be between 0 and 100")
    if args.sample_mode == "rs" and args.qa_ratio is not None:
        parser.error("--qa_ratio can only be used with --sample_mode qa")
    main(args.input_csv, args.output_bitmap, args.sample_mode, args.qa_ratio)
