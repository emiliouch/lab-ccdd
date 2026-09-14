"""Productos analíticos Gold de Pudubella."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl


def build_rfm_exclusions(
    orders: pl.DataFrame, payments: pl.DataFrame
) -> pl.DataFrame:
    """Registra órdenes entregadas sin pago para excluirlas de RFM."""
    delivered = orders.filter(pl.col("order_status") == "delivered")
    pagos_distintos = payments.select("order_id").unique()
    sin_pago = delivered.join(pagos_distintos, on="order_id", how="anti")
    return sin_pago.select(
        "order_id", "customer_id", "order_purchase_timestamp"
    ).with_columns(pl.lit("delivered_order_without_payment").alias("reason"))


def build_sales_daily(
    orders: pl.DataFrame, items: pl.DataFrame
) -> pl.DataFrame:
    """Construye ventas de ítems por fecha de compra y órdenes entregadas."""
    delivered = orders.filter(pl.col("order_status") == "delivered")
    unido = delivered.join(items, on="order_id", how="inner").with_columns(
        pl.col("order_purchase_timestamp").dt.date().alias("sale_date")
    )
    return (
        unido.group_by("sale_date")
        .agg(
            pl.col("price").sum().alias("items_sold_value"),
            pl.col("order_id").n_unique().alias("delivered_orders"),
        )
        .sort("sale_date")
    )


def build_customer_rfm(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    payments: pl.DataFrame,
    segments: dict[str, Any],
) -> pl.DataFrame:
    """Calcula RFM de compras entregadas y aplica reglas congeladas."""
    delivered = orders.filter(pl.col("order_status") == "delivered")

    pagos_por_orden = payments.group_by("order_id").agg(
        pl.col("payment_value").sum().alias("order_payment_value")
    )

    elegibles = delivered.join(
        pagos_por_orden, on="order_id", how="inner"
    ).join(
        customers.select("customer_id", "customer_unique_id"),
        on="customer_id",
        how="inner",
    )

    fecha_referencia = elegibles[
        "order_purchase_timestamp"
    ].max().date() + timedelta(days=1)

    resultado = (
        elegibles.group_by("customer_unique_id")
        .agg(
            pl.col("order_purchase_timestamp").max().alias("last_purchase"),
            pl.col("order_id").n_unique().alias("frequency"),
            pl.col("order_payment_value").sum().alias("monetary"),
        )
        .with_columns(
            (pl.lit(fecha_referencia) - pl.col("last_purchase").dt.date())
            .dt.total_days()
            .cast(pl.Int64)
            .alias("recency_days")
        )
    )

    reglas = segments["segments"]
    champions = reglas["champions"]
    loyal = reglas["loyal"]
    new = reglas["new"]
    lost = reglas["lost"]

    resultado = resultado.with_columns(
        pl.when(
            (pl.col("recency_days") <= champions["max_recency_days"])
            & (pl.col("frequency") >= champions["min_frequency"])
            & (pl.col("monetary") >= champions["min_monetary"])
        )
        .then(pl.lit("Champions"))
        .when(pl.col("frequency") >= loyal["min_frequency"])
        .then(pl.lit("Loyal"))
        .when(pl.col("recency_days") <= new["max_recency_days"])
        .then(pl.lit("New"))
        .when(pl.col("recency_days") >= lost["min_recency_days"])
        .then(pl.lit("Lost"))
        .otherwise(pl.lit("Regular"))
        .alias("segment")
    )

    return resultado.select(
        "customer_unique_id",
        "last_purchase",
        "frequency",
        "monetary",
        "recency_days",
        "segment",
    ).sort("customer_unique_id")
