import psycopg2
import numpy as np
import csv
import os
import struct

# ======== 配置区域 ========
DB_CONFIG = {
    "host": "localhost",
    "database": "stats",
    "user": "xuyining",
    "port": 5433
}

INPUT_CSV = "/data2/xuyining/learnedcardinalities/data/stats/finetune.csv"
OUTPUT_BITMAP = "/data2/xuyining/learnedcardinalities/data/stats/finetune.bitmaps"
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

def get_bitmap(cursor, table_name, predicates):
    """
    执行查询并返回 1000 位的位图
    """
    # 假设你的采样表命名为 原表名_s1000
    sample_table = f"{table_name}_s1000"
    bitmap = np.zeros(SAMPLE_SIZE, dtype=np.uint8)
    
    # 基础查询语句：直接查满足条件的 id
    sql = f"SELECT id FROM {sample_table}"
    
    if predicates:
        sql_preds = []
        for p in predicates:
            # p[0] 是 t.production_year，需要去掉别名部分转为数据库列名
            col = p[0].split('.')[-1]
            op = p[1]
            val = p[2].replace("'", "''") # 简单的 SQL 注入预防

            if DB_CONFIG['database'] == 'stats':
                if col.endswith('date') or col.endswith('Date'):
                    val = timestamp_to_string(val)
                    sql_preds.append(f"{col} {op} '{val}'::timestamp")
                else:
                    # 判断是否为数值，非数值则加单引号
                    try:
                        float(val)
                        sql_preds.append(f"{col} {op} {val}")
                    except ValueError:
                        sql_preds.append(f"{col} {op} '{val}'")
            else:
                try:
                    float(val)
                    sql_preds.append(f"{col} {op} {val}")
                except ValueError:
                    sql_preds.append(f"{col} {op} '{val}'")
        
        sql += " WHERE " + " AND ".join(sql_preds)

    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            id = int(row[0])
            # id 对1000取模得到位图索引
            idx = (id - 1) % SAMPLE_SIZE
            bitmap[idx] = 1
    except Exception as e:
        print(f"\n[Warning] SQL Failed: {sql}")
        print(f"Error: {e}")
    
    return bitmap

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # 统计总行数用于进度显示
    with open(INPUT_CSV, 'r') as f:
        total_queries = sum(1 for _ in f)
    
    print(f"Starting bitmap generation for {total_queries} queries...")

    with open(INPUT_CSV, 'r') as f_in, open(OUTPUT_BITMAP, 'wb') as f_out:
        reader = csv.reader(f_in, delimiter='#')
        
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
                pred_raw = row[2].split(',')
                pred_list = list(chunks(pred_raw, 3))
                for p in pred_list:
                    if len(p) == 3:
                        alias = p[0].split('.')[0]
                        if alias in pred_groups:
                            pred_groups[alias].append(p)

            # 3. 写入当前查询包含的位图数量 (4字节)
            f_out.write(struct.pack('<I', len(tables_order)))
            
            # 4. 生成并写入每个表的位图
            for alias in tables_order:
                real_table = alias_to_table[alias]
                # 执行数据库查询
                bitmap_arr = get_bitmap(cursor, real_table, pred_groups[alias])
                
                # 压缩：1000位 -> 125字节
                # packbits 将 8 个 uint8 压缩进 1 个 byte
                packed = np.packbits(bitmap_arr)
                # 1000/8 = 125 字节
                f_out.write(packed[:13].tobytes())

            if (row_idx + 1) % 50 == 0:
                print(f"Processed {row_idx + 1}/{total_queries} queries...", end='\r')

    cursor.close()
    conn.close()
    print(f"\nSuccessfully generated: {OUTPUT_BITMAP}")

if __name__ == "__main__":
    main()