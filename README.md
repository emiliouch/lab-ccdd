# MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos

Repositorio del curso MDS7202 (Otoño 2026), Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile.

## Integrantes

| Nombre | GitHub |
|--------|--------|
| Alexander Sacchetti | [@usuario1](https://github.com/AlexanderSV25) |
| Emilio Torres | [@usuario2](https://github.com/emiliouch) |

## Estructura del repositorio

.
├── .github/
│   ├── workflows/
│   │   └── lint.yml
│   └── pull_request_template.md
├── labs/
│   ├── lab_1/
│   └── ...
├── pyproject.toml
├── .github/
├── .pre-commit-config.yaml
└── README.md

## Configuración del entorno

uv sync --locked --all-groups
uv run pre-commit install
