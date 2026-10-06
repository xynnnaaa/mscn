import os
import sys
import argparse
from multiprocessing import Pool, cpu_count
import psycopg2

# 数据库连接配置
DB_CONFIG = {
    "user": "xuyining",
    "password": "123",  # 如果有密码请在此填写
    "host": "localhost",
    "port": 5433,
    "database": "tpch-skew-5"
}

# 2小时超时时间映射为毫秒 (2 * 60 * 60 * 1000)
TIMEOUT_MS = 7200000 

def get_actual_cardinality(conn, sql_str):
    """
    执行 EXPLAIN ANALYZE，并设置会话级别的超时时间
    """
    with conn.cursor() as cur:
        # 1. 设置当前连接的单条 SQL 超时阈值（2小时）
        cur.execute(f"SET statement_timeout = {TIMEOUT_MS};")
        
        cur.execute(sql_str)
        
        return cur.fetchone()[0], "SUCCESS"
                    
    return -1, "PARSE_ERROR"

def worker_process(task):
    """
    子进程独立单元
    """
    line_idx, sql_str = task
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        card, status = get_actual_cardinality(conn, sql_str)
        return line_idx, card, status
    except psycopg2.errors.QueryCanceled as te:
        # 捕获 Postgres 自带的超时取消异常 (57014 错误码)
        return line_idx, -1, "TIMEOUT"
    except Exception as e:
        # 捕获其他数据库或网络异常
        return line_idx, -1, f"ERROR: {str(e)}"
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="Parallel Truth Cardinality Collector with 2h Timeout")
    parser.add_argument("--input", type=str, required=True, help="Path to input SQL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output text file")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel processes")
    args = parser.parse_args()

    # 1. 解析加载 SQL
    tasks = []
    print(f"[*] Loading workload: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split("||")
            sql_str = parts[0].strip()

            if sql_str.startswith("/*"):
                sql_str = sql_str.split("*/", 1)[-1].strip()
            
            if sql_str:
                tasks.append((idx, sql_str))

    total_queries = len(tasks)
    print(f"[+] Loaded {total_queries} queries.")

    # 2. 多进程并发跑数
    num_workers = args.workers if args.workers > 0 else cpu_count()
    print(f"[*] Running with {num_workers} processes. Timeout set to 2 hours per query.")
    
    results = []
    with Pool(processes=num_workers) as pool:
        for i, res in enumerate(pool.imap(worker_process, tasks), 1):
            results.append(res)
            if i % 100 == 0 or i == total_queries:
                sys.stdout.write(f"\r[>] Progress: {i}/{total_queries} processed.")
                sys.stdout.flush()
    print("\n[+] Execution finished. Sorting...")

    # 3. 按序排列
    results.sort(key=lambda x: x[0])

    # 4. 按序输出到结果文件
    print(f"[*] Saving results to: {args.output}")
    stats = {"SUCCESS": 0, "TIMEOUT": 0, "ERROR": 0}
    
    with open(args.output, "w", encoding="utf-8") as f_out:
        for line_idx, card, status in results:
            if status == "SUCCESS":
                f_out.write(f"{card}\n")
                stats["SUCCESS"] += 1
            elif status == "TIMEOUT":
                f_out.write("TIMEOUT\n")
                stats["TIMEOUT"] += 1
            else:
                f_out.write(f"ERROR\n")
                stats["ERROR"] += 1

    print(f"[+] Job Done! Summary -> Success: {stats['SUCCESS']}, Timeout: {stats['TIMEOUT']}, Other Errors: {stats['ERROR']}")

if __name__ == "__main__":
    main()