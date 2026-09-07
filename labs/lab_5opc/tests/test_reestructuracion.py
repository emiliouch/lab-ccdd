import polars as pl
import pytest

from src.gradeslab.reshaping import (
    pasar_a_formato_ancho,
    pasar_a_formato_largo,
)

pytestmark = pytest.mark.etapa3


def test_formato_largo_conserva_las_caracteristicas(registro_unido):
    largo = pasar_a_formato_largo(registro_unido)

    assert largo.height == 4375
    assert {"asignatura", "puntaje", "escala"} <= set(largo.columns)
    conteos = largo["escala"].value_counts().sort("escala")
    assert dict(zip(conteos["escala"], conteos["count"], strict=True)) == {
        "escala chilena": 3500,
        "porcentaje": 875,
    }


def test_formato_largo_no_mezcla_las_escalas(registro_unido):
    largo = pasar_a_formato_largo(registro_unido)

    assert largo.filter(pl.col("escala") == "porcentaje")[
        "asignatura"
    ].unique().to_list() == ["science score"]
    assert set(
        largo.filter(pl.col("escala") == "escala chilena")[
            "asignatura"
        ].unique()
    ) == {"math score", "reading score", "writing score", "history score"}


def test_pivot_reconstruye_las_cinco_asignaturas(registro_unido):
    largo = pasar_a_formato_largo(registro_unido)
    ancho = pasar_a_formato_ancho(largo)

    assert ancho.height == 875
    assert set(ancho.columns[-5:]) == {
        "math score",
        "reading score",
        "writing score",
        "history score",
        "science score",
    }


def test_pivot_rechaza_puntajes_repetidos(registro_unido):
    largo = pasar_a_formato_largo(registro_unido)
    duplicado = pl.concat([largo, largo.head(1)])

    with pytest.raises(pl.exceptions.ComputeError):
        pasar_a_formato_ancho(duplicado)
