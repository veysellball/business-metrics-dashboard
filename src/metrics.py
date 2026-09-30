from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MARTS = ROOT / "data" / "marts"

START, END = "2017-01-01", "2018-08-31"

MATURE_UNTIL = "2018-08-31"

TABLES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def connect_raw() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the raw Olist CSVs loaded as tables."""
    con = duckdb.connect()
    for name, file in TABLES.items():
        path = (RAW / file).as_posix()
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv_auto('{path}')")
    return con


ORDER_FACTS_SQL = """
WITH items AS (
    SELECT
        oi.order_id,
        SUM(oi.price)                AS item_value,
        SUM(oi.freight_value)        AS freight_value,
        COUNT(*)                     AS n_items,
        COUNT(DISTINCT oi.seller_id) AS n_sellers,
        -- the category of the most expensive item represents the order
        arg_max(COALESCE(t.product_category_name_english,
                         p.product_category_name, 'unknown'), oi.price) AS main_category
    FROM order_items oi
    LEFT JOIN products p             ON oi.product_id = p.product_id
    LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
    GROUP BY oi.order_id
),
latest_review AS (
    -- some orders have several reviews: keep the most recent answer
    SELECT order_id, arg_max(review_score, review_answer_timestamp) AS review_score
    FROM reviews
    GROUP BY order_id
),
base AS (
    SELECT
        o.order_id,
        c.customer_unique_id,
        c.customer_state,
        o.order_status,
        CAST(o.order_purchase_timestamp AS DATE)                      AS purchase_date,
        CAST(DATE_TRUNC('month', o.order_purchase_timestamp) AS DATE) AS purchase_month,
        o.order_status NOT IN ('canceled', 'unavailable')             AS is_valid_purchase,
        o.order_approved_at IS NOT NULL                               AS is_approved,
        o.order_delivered_carrier_date IS NOT NULL                    AS is_shipped,
        o.order_status = 'delivered'
            AND o.order_delivered_customer_date IS NOT NULL           AS is_delivered,
        -- day-level comparison: the promise is a DATE, not a timestamp (Project 1 fix)
        COALESCE(o.order_status = 'delivered'
                 AND CAST(o.order_delivered_customer_date AS DATE)
                     <= CAST(o.order_estimated_delivery_date AS DATE), FALSE) AS is_on_time,
        DATE_DIFF('day', CAST(o.order_purchase_timestamp AS DATE),
                         CAST(o.order_estimated_delivery_date AS DATE))       AS promised_days,
        COALESCE(i.item_value, 0)             AS item_value,
        COALESCE(i.freight_value, 0)          AS freight_value,
        i.n_items,
        i.n_sellers,
        COALESCE(i.main_category, 'unknown')  AS main_category,
        r.review_score
    FROM orders o
    JOIN customers c          ON o.customer_id = c.customer_id
    LEFT JOIN items i         ON o.order_id = i.order_id
    LEFT JOIN latest_review r ON o.order_id = r.order_id
),
with_first AS (
    SELECT
        *,
        is_on_time AND COALESCE(review_score >= 4, FALSE) AS is_good_order,
        -- computed over ALL history, before the window filter below
        MIN(CASE WHEN is_valid_purchase THEN purchase_date END)
            OVER (PARTITION BY customer_unique_id)        AS first_purchase_date
    FROM base
)
SELECT
    *,
    -- a customer's first purchase DAY, not first order_id (multi-seller baskets, Project 1 fix)
    is_valid_purchase AND purchase_date = first_purchase_date AS is_first_purchase
FROM with_first
WHERE purchase_date BETWEEN DATE '{start}' AND DATE '{end}'
"""


def build_order_facts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    facts = con.execute(ORDER_FACTS_SQL.format(start=START, end=END)).df()
    date_cols = ["purchase_date", "purchase_month", "first_purchase_date"]
    facts[date_cols] = facts[date_cols].apply(pd.to_datetime)
    return facts