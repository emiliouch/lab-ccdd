"""Agregaciones sobre las temperaturas medias mensuales."""

from __future__ import annotations

import polars as pl


def resumen_mensual(
    mensuales: pl.DataFrame | pl.LazyFrame,
    paises: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Calcula la climatología mensual por país."""
    tabla = mensuales
    if paises is not None:
        tabla = tabla.filter(pl.col("iso_alpha3").is_in(list(paises)))
    return (
        tabla.group_by("iso_alpha3", "country", "month")
        .agg(
            pl.len().alias("observaciones"),
            pl.col("temperature_c").mean().round(2).alias("temperature_mean"),
        )
        .sort("iso_alpha3", "month")
    )


def resumen_anual_desde_mensuales(
    mensuales: pl.DataFrame | pl.LazyFrame,
    paises: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Calcula medias anuales usando únicamente filas mensuales."""
    tabla = mensuales
    if paises is not None:
        tabla = tabla.filter(pl.col("iso_alpha3").is_in(list(paises)))
    return (
        tabla.group_by("iso_alpha3", "country", "year")
        .agg(
            pl.col("month").n_unique().alias("meses_disponibles"),
            pl.col("temperature_c").mean().round(2).alias("temperature_mean"),
        )
        .sort("iso_alpha3", "year")
    )


def anomalias_mensuales(
    mensuales: pl.DataFrame | pl.LazyFrame,
    umbral: float = 2.0,
) -> pl.DataFrame | pl.LazyFrame:
    """Marca anomalías usando una ventana por país y mes."""
    media = pl.col("temperature_c").mean().over("iso_alpha3", "month")
    desviacion = pl.col("temperature_c").std().over("iso_alpha3", "month")
    anomalia = (pl.col("temperature_c") - media) / desviacion
    return mensuales.with_columns(
        media.alias("temperature_mean_month"),
        anomalia.alias("standardized_anomaly"),
    ).with_columns(
        (pl.col("standardized_anomaly").abs() > umbral)
        .fill_null(False)
        .alias("is_anomaly")
    )
