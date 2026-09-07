"""Conversión entre tablas anchas y largas."""

from __future__ import annotations

import polars as pl

from src.gradeslab.constants import Tabla


def pasar_a_formato_largo(tabla: Tabla) -> Tabla:
    """Convierte las cinco asignaturas a las columnas `asignatura` y `puntaje`."""
    raise NotImplementedError(
        "Completen pasar_a_formato_largo antes de ejecutar esta celda."
    )


def pasar_a_formato_ancho(tabla: pl.DataFrame) -> pl.DataFrame:
    """Reconstruye una tabla ancha a partir de una tabla larga."""
    raise NotImplementedError(
        "Completen pasar_a_formato_ancho antes de ejecutar esta celda."
    )
