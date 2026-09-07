"""Uniones y auditoría de claves entre las fuentes."""

from __future__ import annotations

import polars as pl


def unir_registro(
    principal: pl.DataFrame, adicional: pl.DataFrame
) -> pl.DataFrame:
    """Agrega las notas adicionales al registro principal, sin perder alumnos."""
    raise NotImplementedError(
        "Completen unir_registro antes de ejecutar esta celda."
    )


def claves_solo_en_adicional(
    principal: pl.DataFrame, adicional: pl.DataFrame
) -> pl.DataFrame:
    """Devuelve las filas del CSV cuya clave no está en el registro principal."""
    raise NotImplementedError(
        "Completen claves_solo_en_adicional antes de ejecutar esta celda."
    )
