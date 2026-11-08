
import sqlite3
from pathlib import Path

db_path = Path(__file__).resolve().parents[1] / "data" / "instamart.db"

with sqlite3.connect(db_path) as conn:
    statuses = conn.execute(
        "SELECT status, COUNT(*) FROM orders GROUP BY status"
    ).fetchall()

    revenue = conn.execute(
        "SELECT ROUND(SUM(order_amount), 2) "
        "FROM orders WHERE status = ?",
        ("completed",),
    ).fetchone()[0]

print("Order status counts:", statuses)
print(f"Completed revenue: INR {revenue:,.2f}")
