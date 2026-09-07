from pathlib import Path

import polars as pl
import pytest

DATA = Path(__file__).parents[1] / "data" / "raw"
JSON_1 = DATA / "students_grades_1.json"
JSON_2 = DATA / "students_grades_2.json"
CSV = DATA / "other_grades.csv"


@pytest.fixture
def registro_principal() -> pl.DataFrame:
    from src.gradeslab.io import leer_fragmentos_json

    return leer_fragmentos_json([JSON_1, JSON_2])


@pytest.fixture
def notas_adicionales() -> pl.DataFrame:
    from src.gradeslab.io import leer_notas_adicionales

    return leer_notas_adicionales(CSV)


@pytest.fixture
def registro_unido(
    registro_principal: pl.DataFrame,
    notas_adicionales: pl.DataFrame,
) -> pl.DataFrame:
    from src.gradeslab.joins import unir_registro

    return unir_registro(registro_principal, notas_adicionales)
