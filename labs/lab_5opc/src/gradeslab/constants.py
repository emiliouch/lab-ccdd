"""Constantes y tipos compartidos del Laboratorio 5."""

from __future__ import annotations

import polars as pl

type Tabla = pl.DataFrame | pl.LazyFrame

ASIGNATURAS_CHILENAS = (
    "math score",
    "reading score",
    "writing score",
    "history score",
)
ASIGNATURAS = (*ASIGNATURAS_CHILENAS, "science score")
COLUMNAS_ESTUDIANTE = (
    "names",
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
)
NIVELES_EDUCATIVOS = {
    "some high school": "school",
    "high school": "school",
    "some college": "college",
    "associate's degree": "college",
    "bachelor's degree": "postgraduate",
    "master's degree": "postgraduate",
}
