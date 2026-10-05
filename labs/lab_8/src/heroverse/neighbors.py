"""Búsqueda de vecinos más cercanos sobre una representación."""

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors


def vecinos_mas_cercanos(
    X: np.ndarray,
    nombres: list[str],
    consulta: str,
    k: int = 5,
    metrica: str = "euclidean",
) -> pl.DataFrame:
    """Devuelve los `k` personajes más cercanos a `consulta`.

    `X` tiene una fila por personaje, en el mismo orden que `nombres`; puede
    ser un arreglo de NumPy o una matriz dispersa, como la de TF-IDF. El
    resultado tiene las columnas `name` y `distancia`, ordenadas de menor a
    mayor distancia, y no incluye a la consulta. Si `consulta` no está en
    `nombres`, levanta `KeyError`.
    """
    if consulta not in nombres:
        raise KeyError(consulta)
    i = nombres.index(consulta)
    modelo = NearestNeighbors(n_neighbors=k + 1, metric=metrica).fit(X)
    distancias, indices = modelo.kneighbors(X[i : i + 1])
    resultado = pl.DataFrame(
        {
            "name": [nombres[j] for j in indices[0]],
            "distancia": distancias[0],
        }
    )
    return resultado.filter(pl.col("name") != consulta).head(k)
