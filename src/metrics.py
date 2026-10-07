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


def monthly_kpis(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per purchase month with the core KPIs and their component rates."""
    f = facts[facts["is_valid_purchase"]]
    m = f.groupby("purchase_month").agg(
        orders=("order_id", "size"),
        gmv=("item_value", "sum"),
        delivered=("is_delivered", "sum"),
        on_time=("is_on_time", "sum"),
        good_orders=("is_good_order", "sum"),
        promised_days=("promised_days", "mean"),
    )
    # customers, not orders: a multi-seller basket is several order_ids (Project 1 fix)
    m["new_customers"] = (f[f["is_first_purchase"]]
                          .groupby("purchase_month")["customer_unique_id"].nunique())
    m["delivered_rate"] = m["delivered"] / m["orders"]
    m["on_time_rate"] = m["on_time"] / m["delivered"]
    m["good_review_rate"] = m["good_orders"] / m["on_time"]
    return m


def first_two_purchases(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: first and second purchase DAY (second is NaT if none)."""
    days = (facts.loc[facts["is_valid_purchase"], ["customer_unique_id", "purchase_date"]]
            .drop_duplicates()
            .sort_values(["customer_unique_id", "purchase_date"]))
    days["n"] = days.groupby("customer_unique_id").cumcount() + 1
    out = days[days["n"] <= 2].pivot(index="customer_unique_id", columns="n",
                                      values="purchase_date")
    out.columns = ["first_date", "second_date"]
    out["days_to_second"] = (out["second_date"] - out["first_date"]).dt.days
    return out.reset_index()


def repeat_rate_within(cust: pd.DataFrame, days: int, data_end) -> tuple[float, int]:
    """Share of customers who bought again within `days` of their first purchase,
    among customers observed for at least `days` (censoring-aware)."""
    eligible = cust[cust["first_date"] <= data_end - pd.Timedelta(days=days)]
    return (eligible["days_to_second"] <= days).mean(), len(eligible)


def cohort_matrix(events: pd.DataFrame, id_col: str, month_col: str) -> pd.DataFrame:
    """events: one row per (id, active month). Returns cohort × months-since-first
    table with the share of the cohort active in each month."""
    e = events[[id_col, month_col]].drop_duplicates().copy()
    month = pd.to_datetime(e[month_col])
    e["m_idx"] = month.dt.year * 12 + month.dt.month
    e["offset"] = e["m_idx"] - e.groupby(id_col)["m_idx"].transform("min")
    e["cohort"] = month.groupby(e[id_col]).transform("min").dt.strftime("%Y-%m")
    counts = e.pivot_table(index="cohort", columns="offset", values=id_col, aggfunc="nunique")
    return counts.div(counts[0], axis=0)