"""Funciones para declarar y validar el esquema CRU."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaErrors

from src.meteolab.constantes import PERIODOS_VALIDOS

# Esquema esperado base según la carga del CSV
ESQUEMA_ESPERADO: dict[str, pl.DataType] = {
    "country": pl.String(),
    "iso_alpha2": pl.String(),
    "iso_alpha3": pl.String(),
    "year": pl.Int64(),
    "period": pl.String(),
    "temperature_c": pl.Float64(),
    "parameter": pl.String(),
    "units": pl.String(),
    "source_file": pl.String(),
}

# Validación con Pandera sobre tipos y restricciones de contenido
ESQUEMA_TEMPERATURAS = pa.DataFrameSchema(
    columns={
        "country": pa.Column(pl.String, nullable=False),
        "iso_alpha2": pa.Column(pl.String, nullable=False),
        "iso_alpha3": pa.Column(pl.String, nullable=False),
        "year": pa.Column(
            pl.Int64,
            checks=[pa.Check.in_range(1901, 2025)],
            nullable=False,
        ),
        "period": pa.Column(
            pl.String,
            checks=[pa.Check.isin(list(PERIODOS_VALIDOS))],
            nullable=False,
        ),
        "temperature_c": pa.Column(pl.Float64, nullable=True),
        "parameter": pa.Column(
            pl.String,
            checks=[pa.Check.equal_to("Mean Temperature")],
            nullable=False,
        ),
        "units": pa.Column(
            pl.String,
            checks=[pa.Check.equal_to("degrees Celsius")],
            nullable=False,
        ),
        "source_file": pa.Column(pl.String, nullable=False),
    },
    strict=True,
)


def comparar_esquema(temperaturas: pl.DataFrame) -> list[str]:
    """Devuelve diferencias entre el esquema real y el esperado."""
    diferencias: list[str] = []
    esquema_real = temperaturas.schema

    for col, tipo_esperado in ESQUEMA_ESPERADO.items():
        if col not in esquema_real:
            diferencias.append(f"Columna faltante: '{col}'")
        elif esquema_real[col] != tipo_esperado:
            diferencias.append(
                f"Tipo incorrecto en '{col}': esperado {tipo_esperado}, obtenido {esquema_real[col]}"
            )

    for col in esquema_real:
        if col not in ESQUEMA_ESPERADO:
            diferencias.append(f"Columna no esperada: '{col}'")

    return diferencias


def validar_esquema(temperaturas: pl.DataFrame) -> None:
    """Comprueba los nombres y tipos de las columnas."""
    diferencias = comparar_esquema(temperaturas)
    if diferencias:
        raise ValueError(
            "El esquema del DataFrame no coincide con el esperado:\n"
            + "\n".join(f"- {d}" for d in diferencias)
        )


def validar_datos(temperaturas: pl.DataFrame) -> pl.DataFrame:
    """Valida tipos, periodos, unidades y valores faltantes."""
    return ESQUEMA_TEMPERATURAS.validate(temperaturas, lazy=True)


def casos_que_fallan(temperaturas: pl.DataFrame) -> pl.DataFrame:
    """Devuelve los incumplimientos sin ocultar sus columnas."""
    try:
        ESQUEMA_TEMPERATURAS.validate(temperaturas, lazy=True)
        # Si no hay errores, retorna un DataFrame vacío con el esquema de casos fallidos
        return pl.DataFrame(
            schema={
                "schema_context": pl.String,
                "column": pl.String,
                "check": pl.String,
                "check_index": pl.Int64,
                "failure_case": pl.String,
                "index": pl.Int64,
            }
        )
    except SchemaErrors as err:
        return pl.DataFrame(err.failure_cases)
