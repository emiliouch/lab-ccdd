"""Lectura de las fuentes del laboratorio."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl


def leer_fragmentos_json(rutas: Sequence[Path]) -> pl.DataFrame:
    """Lee los dumps JSON y los concatena verticalmente."""
    raise NotImplementedError(
        "Completen leer_fragmentos_json antes de ejecutar esta celda."
    )


def leer_notas_adicionales(ruta: Path) -> pl.DataFrame:
    """Lee el CSV adicional con los tipos numéricos de sus dos notas."""
    raise NotImplementedError(
        "Completen leer_notas_adicionales antes de ejecutar esta celda."
    )
