"""Ingesta reproducible de las fuentes Parquet hacia Bronze."""

from __future__ import annotations

from pathlib import Path

import polars as pl

_SOURCE_FILES = {
    "orders": "orders.parquet",
    "customers": "customers.parquet",
    "order_items": "order_items.parquet",
    "payments": "payments.parquet",
}


def read_sources(raw_dir: Path) -> dict[str, pl.DataFrame]:
    """Lee las cuatro fuentes Parquet desde el directorio raw.

    Conserva la estructura original sin transformaciones.
    Levanta FileNotFoundError si falta algún archivo.
    """
    files_map = _SOURCE_FILES

    data: dict[str, pl.DataFrame] = {}
    for name, filename in files_map.items():
        file_path = raw_dir / filename
        if not file_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el archivo de la fuente '{name}': {file_path}"
            )
        data[name] = pl.read_parquet(file_path)

    return data
