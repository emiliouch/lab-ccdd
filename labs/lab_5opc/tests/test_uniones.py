import polars as pl
import pytest

from src.gradeslab.joins import claves_solo_en_adicional, unir_registro

pytestmark = pytest.mark.etapa2


def test_left_join_conserva_la_matricula_principal(registro_unido):
    assert registro_unido.height == 875
    assert registro_unido.width == 11
    assert registro_unido.columns[-2:] == ["science score", "history score"]


def test_anti_join_encuentra_los_125_registros_extra(
    registro_principal, notas_adicionales
):
    extras = claves_solo_en_adicional(registro_principal, notas_adicionales)

    assert extras.height == 125
    assert extras["names"].n_unique() == 125


def test_validate_11_falla_si_se_repite_una_clave(
    registro_principal, notas_adicionales
):
    repetido = pl.concat([notas_adicionales, notas_adicionales.head(1)])

    with pytest.raises(pl.exceptions.ComputeError):
        unir_registro(registro_principal, repetido)
