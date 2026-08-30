"""Funciones de lectura del CSV que deben completar ustedes."""

from pathlib import Path

import polars as pl

from src.meteolab.constantes import ESQUEMA_CRU, RUTA_CSV, ruta_existente


def leer_temperaturas(ruta: Path = RUTA_CSV) -> pl.DataFrame:
    """Lee el CSV CRU con sus tipos y valores faltantes."""
    ruta = ruta_existente(ruta)
    return pl.read_csv(ruta, schema_overrides=ESQUEMA_CRU)


def escanear_temperaturas(ruta: Path = RUTA_CSV) -> pl.LazyFrame:
    """Construye una consulta lazy sobre el CSV."""
    ruta = ruta_existente(ruta)
    return pl.scan_csv(ruta, schema_overrides=ESQUEMA_CRU)
