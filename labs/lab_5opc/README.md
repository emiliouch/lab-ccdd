# Laboratorio 5 — Ordenar las notas de la Universidad de la Cachaña

Trabajen desde esta carpeta. Instalen las dependencias con `uv sync` y ejecuten
las pruebas por etapa con `uv run pytest -m etapaN`.

Si necesitan preparar el ambiente desde cero, ejecuten:

```bash
uv add polars plotly numpy
uv add --dev jupyterlab ipykernel jupytext nbformat pytest ruff
uv sync
```

Después de completar cada etapa, ejecuten la marca correspondiente:

```bash
uv run pytest -m etapa1
uv run pytest -m etapa2
uv run pytest -m etapa3
uv run pytest -m etapa4
uv run pytest -m etapa5
uv run pytest -m etapa6
uv run pytest
```
