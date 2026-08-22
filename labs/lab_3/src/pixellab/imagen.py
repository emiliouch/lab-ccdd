"""Clase ``Imagen``: contenedor de imágenes sobre el que se opera con NumPy."""

from __future__ import annotations

import numpy as np


class Imagen:
    """Contenedor de imágenes RGB.

    Completen el constructor y los operadores de esta clase siguiendo el
    contrato del enunciado y los tests de ``tests/test_imagen.py``.
    """

    def __init__(self, img: np.ndarray) -> None:
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Debes entregar un arreglo de numpy como argumento del "
                "constructor de Imagen"
            )
        if img.ndim != 3:
            raise ValueError(
                "El arreglo debe tener 3 dimensiones (alto, ancho, canales)."
            )
        if img.shape[-1] != 3:
            raise ValueError(
                "El arreglo debe tener 3 canales (RGB) en su última dimensión."
            )
        self.imagen = img

    def __add__(self, other: int | float | np.ndarray | Imagen) -> Imagen:

        if isinstance(other, Imagen):
            # other puede ser Imagen entonces hay que sacar su .imagen y
            # validar que el shape calce antes de sumar
            if other.imagen.shape != self.imagen.shape:
                raise ValueError(
                    "Las dimensiones de la imagen a operar (alto x ancho x "
                    "canales) no calzan con las de la imagen original "
                    "(alto x ancho x canales)"
                )
            operando = other.imagen
        else:
            operando = other  # int/float/ndarray se usan directo

        resultado = self.imagen + operando
        resultado = resultado.astype(int)

        # que no se reinicien los valores
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0

        return Imagen(np.copy(resultado))

    def __radd__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # llamamos a add
        return self.__add__(other)

    def __sub__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        if isinstance(other, Imagen):
            # other puede ser Imagen entonces hay que sacar su .imagen y
            # validar que el shape calce antes de restar
            if other.imagen.shape != self.imagen.shape:
                raise ValueError(
                    "Las dimensiones de la imagen a operar (alto x ancho x "
                    "canales) no calzan con las de la imagen original "
                    "(alto x ancho x canales)"
                )
            operando = other.imagen
        else:
            operando = other  # int/float/ndarray se usan directo

        resultado = self.imagen - operando
        resultado = resultado.astype(int)

        # que no se reinicien los valores
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0

        return Imagen(np.copy(resultado))

    def __rsub__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        if isinstance(other, Imagen):
            # other puede ser Imagen entonces hay que sacar su .imagen y
            # validar que el shape calce antes de restar
            if other.imagen.shape != self.imagen.shape:
                raise ValueError(
                    "Las dimensiones de la imagen a operar (alto x ancho x "
                    "canales) no calzan con las de la imagen original "
                    "(alto x ancho x canales)"
                )
            operando = other.imagen
        else:
            operando = other  # int/float/ndarray se usan directo

        resultado = operando - self.imagen
        resultado = resultado.astype(int)

        # que no se reinicien los valores
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0

        return Imagen(np.copy(resultado))

    def __mul__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        if isinstance(other, Imagen):
            # other puede ser Imagen entonces hay que sacar su .imagen y
            # validar que el shape calce antes de restar
            if other.imagen.shape != self.imagen.shape:
                raise ValueError(
                    "Las dimensiones de la imagen a operar (alto x ancho x "
                    "canales) no calzan con las de la imagen original "
                    "(alto x ancho x canales)"
                )
            operando = other.imagen
        else:
            operando = other  # int/float/ndarray se usan directo

        resultado = self.imagen * operando
        resultado = resultado.astype(int)

        # que no se reinicien los valores
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0

        return Imagen(np.copy(resultado))

    def __rmul__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # llamamos a mul
        return self.__mul__(other)
