"""Clustering sin etiquetas: elección de k, estabilidad, perfiles y equivalentes."""

from itertools import combinations

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


def elegir_k(X: np.ndarray, ks: list[int], semilla: int = 0) -> pl.DataFrame:
    """Ajusta K-Means para cada `k` y reporta inercia y silhouette.

    Usa `KMeans(n_clusters=k, n_init=10, random_state=semilla)`. Devuelve una
    fila por `k`, en el orden de `ks`, con las columnas `k`, `inercia` y
    `silhouette`. Cada `k` debe estar entre 2 y el número de filas menos uno,
    porque el silhouette no está definido fuera de ese rango.
    """
    filas = []
    for k in ks:
        if not 2 <= k <= X.shape[0] - 1:
            raise ValueError(f"k={k} fuera de rango para silhouette.")
        modelo = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(X)
        filas.append(
            {
                "k": k,
                "inercia": float(modelo.inertia_),
                "silhouette": float(silhouette_score(X, modelo.labels_)),
            }
        )
    return pl.DataFrame(filas)


def estabilidad(X: np.ndarray, k: int, semillas: list[int]) -> float:
    """Promedio del ARI entre las particiones de K-Means con cada semilla.

    Ajusta `KMeans(n_clusters=k, n_init=1, random_state=s)` para cada `s` de
    `semillas` y promedia el ARI de todos los pares de particiones. Vale 1 si
    todas coinciden, aunque numeren los grupos distinto, y cerca de 0 si
    coinciden como lo harían al azar. `n_init=1` hace que cada semilla
    muestre su propio resultado.
    """
    particiones = [
        KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(X)
        for s in semillas
    ]
    aris = [adjusted_rand_score(a, b) for a, b in combinations(particiones, 2)]
    return float(np.mean(aris))


def perfil_clusters(
    features: pl.DataFrame, etiquetas: np.ndarray, columnas: list[str]
) -> pl.DataFrame:
    """Una fila por grupo con su tamaño y el promedio de `columnas`.

    `etiquetas` trae el grupo de cada fila de `features`. Columnas: `cluster`,
    `n` y una por cada elemento de `columnas`, ordenadas por `cluster`.
    """
    return (
        features.with_columns(pl.Series("cluster", etiquetas))
        .group_by("cluster")
        .agg(pl.len().alias("n"), pl.col(columnas).mean())
        .sort("cluster")
    )


def equivalentes(
    X: np.ndarray,
    personajes: pl.DataFrame,
    consulta: str,
    k: int = 5,
    excluir_creator: str = "Marvel Comics",
    metrica: str = "cosine",
) -> pl.DataFrame:
    """Los `k` personajes más cercanos a `consulta` fuera de `excluir_creator`.

    `personajes` tiene las columnas `name` y `creator` en el mismo orden que
    las filas de `X`. Un `creator` nulo no se excluye: no hay evidencia de que
    sea de esa editorial. El resultado tiene `name`, `creator` y `distancia`,
    en distancia creciente.
    """
    nombres = personajes["name"].to_list()
    if consulta not in nombres:
        raise KeyError(consulta)
    i = nombres.index(consulta)
    modelo = NearestNeighbors(n_neighbors=X.shape[0], metric=metrica).fit(X)
    distancias, indices = modelo.kneighbors(X[i : i + 1])
    return (
        personajes.select("name", "creator")[indices[0].tolist()]
        .with_columns(pl.Series("distancia", distancias[0]))
        .filter(
            (pl.col("name") != consulta)
            & (
                pl.col("creator").is_null()
                | (pl.col("creator") != excluir_creator)
            )
        )
        .head(k)
    )
