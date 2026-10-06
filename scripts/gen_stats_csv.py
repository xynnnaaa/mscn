import psycopg2
import csv
from concurrent.futures import ThreadPoolExecutor

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "uniform_v3",
    "user": "xuyining",
    "password": "123"
}

OUTPUT_FILE = "/data2/xuyining/learnedcardinalities/data/uniform/column_min_max_vals.csv"

# TABLE_ALIAS = {
#     "title": "t",
#     "movie_companies": "mc",
#     "cast_info": "ci",
#     "movie_info": "mi",
#     "movie_info_idx": "mii",
#     "movie_keyword": "mk",
#     "complete_cast": "cc",
#     "aka_title": "at",
#     "aka_name": "an",
#     "name": "n"
# }

# TABLE_ALIAS = {
#     "img_obj": "gno_io",
#     "img_rel": "gno_ir",
#     "img_obj_att": "gno_ioa"
# }

TABLE_ALIAS = {
    "main_table": "t"
}

# TABLE_ALIAS = {
#     "part":      "tpch_p",
#     "supplier":  "tpch_s",
#     "partsupp":  "tpch_ps",
#     "customer":  "tpch_c",
#     "orders":    "tpch_o",
#     "lineitem":  "tpch_li",
#     "nation":    "tpch_n",
#     "region":    "tpch_r"
# }

def get_numeric_columns(conn):
    cur = conn.cursor()

    cur.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name IN (
            'main_table'
        )
        AND data_type IN (
            'integer', 'bigint', 'smallint',
            'numeric', 'real', 'double precision',
            'decimal', 'float'
        )
        ORDER BY table_name;
    """)

    res = cur.fetchall()
    cur.close()
    return res


def compute_stats(task):
    table, column = task
    alias = TABLE_ALIAS[table]

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        # 1️⃣ 先算 min / max / count（这个还是必须扫表）
        query_main = f"""
            SELECT
                MIN({column}),
                MAX({column})
            FROM {table};
        """
        cur.execute(query_main)
        min_val, max_val = cur.fetchone()

        name = f"{alias}.{column}".lower()

        print(f"[OK] {name}")

        return [name, min_val, max_val]

    except Exception as e:
        print(f"[ERROR] {table}.{column}: {e}")
        return None

    finally:
        cur.close()
        conn.close()


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    tasks = get_numeric_columns(conn)
    conn.close()

    results = []

    # ⚠️ 保守一点，避免打爆数据库
    with ThreadPoolExecutor(max_workers=4) as executor:
        for res in executor.map(compute_stats, tasks):
            if res:
                results.append(res)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "min", "max"])
        writer.writerows(results)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()