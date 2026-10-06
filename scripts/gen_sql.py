import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path(
    "/data2/xuyining/Workload-Drift-Benchmark/my_workloads/imdb/"
    "join_drift/relations_3_train_all.csv"
)


def normalize_input_path(file_name):
    """Accept either a CSV path or the historical path-without-suffix form."""
    path = Path(file_name)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    return path


def parse_predicates(predicate_field, line_number):
    if not predicate_field:
        return []

    tokens = predicate_field.split(",")
    if len(tokens) % 3 != 0:
        raise ValueError(
            f"line {line_number}: predicate field does not contain complete "
            f"column/operator/value triples"
        )

    predicates = []
    for index in range(0, len(tokens), 3):
        column, operator, value = tokens[index : index + 3]
        if not column or not operator or not value:
            raise ValueError(f"line {line_number}: empty predicate component")
        predicates.append(f"{column} {operator} {value}")
    return predicates


def row_to_sql(row, line_number, append_cardinality=False):
    if len(row) < 4:
        raise ValueError(
            f"line {line_number}: expected tables#joins#predicates#cardinality"
        )

    tables = [table.strip() for table in row[0].split(",") if table.strip()]
    if not tables:
        raise ValueError(f"line {line_number}: query has no tables")

    joins = [join.strip() for join in row[1].split(",") if join.strip()]

    # TPC-H literals may contain '#', for example 'Brand#35'. The CSV reader
    # splits such rows into more than four fields, so reconstruct predicates
    # from every field between the join field and the final cardinality field.
    predicate_field = "#".join(row[2:-1])
    predicates = parse_predicates(predicate_field, line_number)

    conditions = joins + predicates
    sql = f"SELECT COUNT(*) FROM {', '.join(tables)}"
    if conditions:
        sql += f" WHERE {' AND '.join(conditions)}"
    sql += ";"

    if append_cardinality:
        cardinality = row[-1].strip()
        if not cardinality:
            raise ValueError(f"line {line_number}: empty cardinality")
        sql += f"||{cardinality}||"

    return sql


def convert_csv_to_sql(input_csv, output_sql=None, append_cardinality=False):
    input_path = normalize_input_path(input_csv)
    output_path = Path(output_sql) if output_sql else input_path.with_suffix(".sql")

    converted = 0
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        with output_path.open("w", encoding="utf-8", newline="\n") as sql_file:
            reader = csv.reader(csv_file, delimiter="#")
            for line_number, row in enumerate(reader, start=1):
                if not row or all(not field for field in row):
                    continue
                sql_file.write(
                    row_to_sql(row, line_number, append_cardinality=append_cardinality)
                )
                sql_file.write("\n")
                converted += 1

    print(f"Converted {converted} queries: {input_path} -> {output_path}")
    return converted


def load_data(file_name):
    """Backward-compatible entry point used by the previous script."""
    convert_csv_to_sql(file_name)
    return [], [], [], [], []


def main():
    parser = argparse.ArgumentParser(
        description="Convert an MSCN workload CSV into executable COUNT SQL."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input MSCN CSV path; the .csv suffix may be omitted.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output SQL path; defaults to the input path with a .sql suffix.",
    )
    parser.add_argument(
        "--append-cardinality",
        action="store_true",
        help="Append ||cardinality|| after each SQL statement.",
    )
    args = parser.parse_args()
    convert_csv_to_sql(args.input, args.output, args.append_cardinality)


if __name__ == "__main__":
    main()
