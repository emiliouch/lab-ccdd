"""Funciones para revisar nulos y claves temporales."""

from __future__ import annotations

import polars as pl

from src.meteolab.constantes import PERIODOS_MENSUALES, Tabla


def resumen_de_nulos(temperaturas: pl.DataFrame) -> pl.DataFrame:
    """Devuelve conteos y porcentajes de nulos por columna."""
    total_filas = temperaturas.height

    resumen = temperaturas.select(
        [pl.col(col).null_count().alias(col) for col in temperaturas.columns]
    ).unpivot(
        variable_name="columna",
        value_name="nulos",
    )

    if total_filas == 0:
        return resumen.with_columns(pl.lit(0.0).alias("porcentaje"))

    return resumen.with_columns(
        ((pl.col("nulos") / total_filas) * 100.0).alias("porcentaje")
    )


def claves_repetidas(temperaturas: Tabla) -> Tabla:
    """Cuenta repeticiones de país, año y periodo."""
    return (
        temperaturas.group_by(["country", "year", "period"])
        .len()
        .filter(pl.col("len") > 1)
    )


def limpiar_temperaturas(temperaturas: Tabla) -> Tabla:
    """Conserva el contrato de periodos y los nulos válidos."""
    return temperaturas.filter(
        pl.col("period").is_in(PERIODOS_MENSUALES)
        & pl.col("temperature_c").is_not_null()
    )
