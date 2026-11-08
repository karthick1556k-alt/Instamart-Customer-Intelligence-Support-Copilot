
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    sql_text = SQL_PATH.read_text(encoding="utf-8")
    queries = [
        statement.strip()
        for statement in sql_text.split(";")
        if statement.strip()
    ]

    if len(queries) != 7:
        raise ValueError(
            f"Expected 7 SQL queries, but found {len(queries)}. "
            "Check sql/queries.sql for missing or extra semicolons."
        )

    with sqlite3.connect(DB_PATH) as conn:
        for index, query in enumerate(queries, start=1):
            # Each query is executed using pandas as required.
            result = pd.read_sql_query(query, conn)

            output_path = OUTPUT_DIR / f"part_a_q{index}.csv"
            result.to_csv(output_path, index=False)

            print(f"\n{'=' * 60}")
            print(f"Q{index}: {len(result)} rows")
            print(result.head(5).to_string(index=False))
            print(f"Saved: {output_path.relative_to(ROOT)}")

    print("\nAll seven SQL queries executed successfully.")


if __name__ == "__main__":
    main()

