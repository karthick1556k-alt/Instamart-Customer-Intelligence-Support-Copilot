
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

CLEAN.mkdir(parents=True, exist_ok=True)


def audit_data(name, df):
    print(f"\n{'=' * 65}")
    print(f"AUDIT: {name}")
    print(f"Shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    })
    print(missing.to_string())


def main():
    # Load raw inputs. Keep identifiers and raw values available for auditing.
    customers = pd.read_csv(RAW / "customers.csv", dtype={"age": "string"})
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")

    audit_data("customers.csv", customers)
    audit_data("products.csv", products)
    audit_data("orders.csv", orders)

    # ---------------- Customer cleaning ----------------
    customers["city"] = customers["city"].astype("string").str.strip().str.title()

    customers["gender"] = (
        customers["gender"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({
            "m": "male",
            "male": "male",
            "f": "female",
            "female": "female",
            "": "unknown",
            "nan": "unknown",
            "<na>": "unknown",
        })
        .fillna("unknown")
    )


    customers["age"] = pd.to_numeric(
        customers["age"], errors="coerce"
    ).astype("float64")

    customers.loc[
        ~customers["age"].between(10, 100), "age"
    ] = float("nan")

    # Impute missing ages with the median for the customer's city.
    city_medians = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_medians)


    if customers["age"].isna().any():
        raise AssertionError("Some customer ages remain missing after city-median imputation.")

    customers["age"] = customers["age"].round().astype(int)
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    # ---------------- Product cleaning ----------------
    products["price_inr"] = pd.to_numeric(
        products["price_inr"], errors="raise"
    )

    # ---------------- Order cleaning ----------------
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="raise"
    )

    raw_order_rows = len(orders)
    exact_duplicates = int(orders.duplicated().sum())
    pending_rows = int(orders["status"].eq("pending").sum())
    distinct_order_ids = int(orders["order_id"].nunique())

    print("\nRAW ORDER AUDIT")
    print(f"Raw order rows: {raw_order_rows}")
    print(f"Exact duplicate rows: {exact_duplicates}")
    print(f"Pending rows: {pending_rows}")
    print(f"Distinct order IDs: {distinct_order_ids}")

    # Remove exact duplicates first.
    orders = orders.drop_duplicates()

    # Keep the latest record for each order ID.
    orders = orders.sort_values(
        ["order_id", "order_ts"]
    ).drop_duplicates("order_id", keep="last")

    orders["quantity"] = pd.to_numeric(
        orders["quantity"], errors="raise"
    ).astype(int)

    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="raise"
    )

    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    # Replace invalid or missing order amounts using catalogue price.
    orders = orders.merge(
        products[["product_id", "category", "price_inr"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    missing_amount = orders["order_amount"].isna()
    orders.loc[missing_amount, "order_amount"] = (
        orders.loc[missing_amount, "price_inr"]
        * orders.loc[missing_amount, "quantity"]
        * (1 - orders.loc[missing_amount, "discount_pct"] / 100)
    ).round(2)

    # Negative delivery durations are invalid.
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

    completed = orders["status"].eq("completed")

    # Fill missing delivery times using completed-order category medians.
    category_medians = (
        orders.loc[completed]
        .groupby("category")["delivery_minutes"]
        .median()
    )

    missing_completed_delivery = (
        completed & orders["delivery_minutes"].isna()
    )

    orders.loc[missing_completed_delivery, "delivery_minutes"] = (
        orders.loc[missing_completed_delivery, "category"]
        .map(category_medians)
    )

    orders["delivery_minutes"] = orders["delivery_minutes"].round().astype("Int64")

    # Store timestamps in the required SQLite-compatible text format.
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Keep only the required clean order columns.
    orders = orders[
        [
            "order_id", "customer_id", "product_id", "quantity",
            "order_ts", "status", "discount_pct", "order_amount",
            "delivery_minutes",
        ]
    ].copy()

    # ---------------- Save clean CSVs ----------------
    customers.to_csv(CLEAN / "customers.csv", index=False)
    products.to_csv(CLEAN / "products.csv", index=False)
    orders.to_csv(CLEAN / "orders.csv", index=False)

    # ---------------- Create SQLite database ----------------
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        customers.to_sql("customers", conn, index=False, if_exists="replace")
        products.to_sql("products", conn, index=False, if_exists="replace")
        orders.to_sql("orders", conn, index=False, if_exists="replace")

    # ---------------- Validation checkpoints ----------------
    checks = {
        "customer_count": len(customers) == 1200,
        "product_count": len(products) == 40,
        "clean_order_count": len(orders) == 12309,
        "no_pending_orders": not orders["status"].eq("pending").any(),
        "only_final_order_statuses": set(orders["status"].unique()).issubset(
            {"completed", "cancelled"}
        ),
        "no_missing_order_amount": orders["order_amount"].notna().all(),
        "no_missing_completed_delivery": orders.loc[
            orders["status"].eq("completed"), "delivery_minutes"
        ].notna().all(),
        "cancelled_delivery_missing": orders.loc[
            orders["status"].eq("cancelled"), "delivery_minutes"
        ].isna().all(),
        "customer_ages_valid": customers["age"].between(18, 60).all(),
        "customer_ages_complete": customers["age"].notna().all(),
        "clean_files_exist": all(
            (CLEAN / filename).exists()
            for filename in ("customers.csv", "products.csv", "orders.csv")
        ),
        "database_tables_exist": False,
    }

    with sqlite3.connect(DB_PATH) as conn:
        table_names = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        checks["database_tables_exist"] = {
            "customers", "products", "orders"
        }.issubset(table_names)

    print("\nCLEANING SUMMARY")
    print(f"Customers: {len(customers)}")
    print(f"Products: {len(products)}")
    print(f"Orders: {len(orders)}")
    print(f"Completed: {(orders['status'] == 'completed').sum()}")
    print(f"Cancelled: {(orders['status'] == 'cancelled').sum()}")
    print(
        "Completed revenue: INR "
        f"{orders.loc[orders['status'] == 'completed', 'order_amount'].sum():,.2f}"
    )

    print("\nVALIDATION CHECKPOINTS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Validation failed: {failed}")

    print("\nALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
