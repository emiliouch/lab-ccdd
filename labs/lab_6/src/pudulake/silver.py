"""Transformaciones y reglas críticas de las entidades Silver."""

from __future__ import annotations

import polars as pl

from src.pudulake.contracts import ContractViolation

_DATE_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)

_ALLOWED_STATUSES = {
    "approved",
    "canceled",
    "created",
    "delivered",
    "invoiced",
    "processing",
    "shipped",
    "unavailable",
}

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_orders(orders: pl.DataFrame) -> pl.DataFrame:
    """Tipa fechas de órdenes y comprueba su secuencia temporal."""
    result = orders

    # Cast a String primero: una columna de fechas puede llegar con dtype
    # Null cuando todos sus valores son nulos, y .str.strptime exige String.
    parsed_frame = result.with_columns(
        pl.col(column)
        .cast(pl.String)
        .str.strptime(pl.Datetime, format=_DATETIME_FORMAT, strict=False)
        .alias(column)
        for column in _DATE_COLUMNS
    )

    for column in _DATE_COLUMNS:
        original_nulo = result[column].is_null()
        parseado_nulo = parsed_frame[column].is_null()
        no_interpretable = (~original_nulo) & parseado_nulo
        if no_interpretable.any():
            raise ContractViolation(
                f"{column} contiene una fecha no interpretable."
            )

    result = parsed_frame

    status_desconocidos = (
        set(result["order_status"].drop_nulls().unique().to_list())
        - _ALLOWED_STATUSES
    )
    if status_desconocidos:
        raise ContractViolation(
            "order_status contiene valores no reconocidos: "
            f"{', '.join(sorted(status_desconocidos))}."
        )

    result = result.with_columns(
        (
            (pl.col("order_status") == "delivered")
            & pl.col("order_delivered_customer_date").is_null()
        ).alias("delivery_timestamp_missing")
    )

    entregas_invalidas = result.filter(
        (pl.col("order_status") == "delivered")
        & pl.col("order_delivered_customer_date").is_not_null()
        & (
            pl.col("order_delivered_customer_date")
            < pl.col("order_purchase_timestamp")
        )
    )
    if entregas_invalidas.height > 0:
        raise ContractViolation(
            "Hay órdenes delivered con fecha de entrega anterior a la "
            "fecha de compra."
        )

    return result


def build_customers(customers: pl.DataFrame) -> pl.DataFrame:
    """Conserva clientes y verifica la relación uno a uno con customer_id."""
    if customers["customer_id"].n_unique() != customers.height:
        raise ContractViolation(
            "customer_id no identifica de forma única a customers."
        )
    return customers


def build_order_items(items: pl.DataFrame) -> pl.DataFrame:
    """Comprueba que los ítems no tengan precios ni fletes negativos."""
    columns = ["price", "freight_value"]
    table = "order_items"

    for column in columns:
        series = items[column].drop_nulls()
        if series.is_empty():
            continue
        if (series < 0).any():
            raise ContractViolation(
                f"{table}.{column} contiene valores negativos."
            )
        if not series.is_finite().all():
            raise ContractViolation(
                f"{table}.{column} contiene valores no finitos."
            )

    return items


def build_payments(payments: pl.DataFrame) -> pl.DataFrame:
    """Comprueba que los pagos no tengan montos negativos."""
    columns = ["payment_value"]
    table = "payments"

    for column in columns:
        series = payments[column].drop_nulls()
        if series.is_empty():
            continue
        if (series < 0).any():
            raise ContractViolation(
                f"{table}.{column} contiene valores negativos."
            )
        if not series.is_finite().all():
            raise ContractViolation(
                f"{table}.{column} contiene valores no finitos."
            )

    return payments


def validate_relationships(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    items: pl.DataFrame,
    payments: pl.DataFrame,
) -> None:
    """Verifica las claves foráneas antes de construir productos Gold."""
    huerfanos_orders = orders.join(
        customers.select("customer_id"), on="customer_id", how="anti"
    )
    if huerfanos_orders.height > 0:
        raise ContractViolation(
            "orders tiene customer_id huérfana respecto de customers."
        )

    huerfanos_items = items.join(
        orders.select("order_id"), on="order_id", how="anti"
    )
    if huerfanos_items.height > 0:
        raise ContractViolation(
            "order_items tiene order_id huérfana respecto de orders."
        )

    huerfanos_payments = payments.join(
        orders.select("order_id"), on="order_id", how="anti"
    )
    if huerfanos_payments.height > 0:
        raise ContractViolation(
            "payments tiene order_id huérfana respecto de orders."
        )
