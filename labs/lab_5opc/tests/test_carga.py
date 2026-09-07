from pathlib import Path

import polars as pl
import pytest

from src.gradeslab.io import leer_fragmentos_json, leer_notas_adicionales
from tests.conftest import CSV, JSON_1, JSON_2

pytestmark = pytest.mark.etapa1


def test_reconstruye_los_dos_fragmentos(registro_principal):
    assert registro_principal.height == 875
    assert registro_principal.width == 9
    assert registro_principal["names"].n_unique() == 875


def test_los_fragmentos_tienen_las_dimensiones_de_origen():
    assert leer_fragmentos_json([JSON_1]).height == 400
    assert leer_fragmentos_json([JSON_2]).height == 475


def test_lee_las_notas_adicionales_con_tipos_explicitos():
    notas_adicionales = leer_notas_adicionales(CSV)

    assert notas_adicionales.height == 1000
    assert notas_adicionales.schema["science score"] == pl.Int64
    assert notas_adicionales.schema["history score"] == pl.Float64


def test_reporta_una_ruta_inexistente(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no_existe"):
        leer_fragmentos_json([tmp_path / "no_existe.json"])
