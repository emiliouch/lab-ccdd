"""Kernels de convolución que deben definir para la Etapa 6."""

import numpy as np

KERNELS: list[tuple[str, np.ndarray]] = [
    (
        "identidad",
        # 1 en el centro, 0 en el resto: cada pixel de salida es igual
        # al pixel de entrada, no hay ningún efecto visual.
        np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
    ),
    (
        "laplaciano",
        # centro positivo fuerte rodeado de -1: mide la diferencia entre
        # un pixel y sus 4 vecinos directos. En zonas uniformes da ~0
        # (negro); en bordes (cambios bruscos de color) da valores altos.
        np.array(
            [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ]
        ),
    ),
    (
        "enfoque",
        # identidad + laplaciano: le suma de vuelta a la imagen original
        # los bordes detectados, acentuándolos y dando sensación de nitidez.
        np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ]
        ),
    ),
    (
        "desenfoque",
        # todos los valores iguales y suman 1 (promedio de 3x3): cada
        # pixel se mezcla con sus 8 vecinos, suavizando cambios bruscos.
        np.array(
            [
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9],
            ]
        ),
    ),
    (
        "relieve",
        # valores negativos de un lado, positivos del otro (diagonal):
        # resalta la diferencia direccional entre vecinos, dando un
        # efecto de relieve/3D como si la luz viniera de una esquina.
        np.array(
            [
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2],
            ]
        ),
    ),
]
