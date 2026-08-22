"""Operaciones de procesamiento de imágenes para completar."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from src.pixellab.imagen import Imagen


class LibImagen:
    """Filtros y transformaciones que reciben y retornan ``Imagen``."""

    def to_negative(self, img_in: Imagen) -> Imagen:
        return 255 - img_in

    def to_gray(self, img_in: Imagen) -> Imagen:

        arreglo = img_in.imagen
        R = arreglo[:, :, 0]
        G = arreglo[:, :, 1]
        B = arreglo[:, :, 2]
        gris = 0.299 * R + 0.587 * G + 0.114 * B  # shape (alto, ancho) — 2D
        resultado = np.stack([gris, gris, gris], axis=2).astype(
            int
        )  # shape (alto, ancho, 3canales)
        return Imagen(resultado)

    def get_channel(self, img_in: Imagen, channel: str) -> Imagen:

        arreglo = img_in.imagen
        if channel == "r":
            resultado = np.stack(
                [
                    arreglo[:, :, 0],
                    np.zeros_like(arreglo[:, :, 0]),
                    np.zeros_like(arreglo[:, :, 0]),
                ],
                axis=2,
            ).astype(int)
        elif channel == "g":
            resultado = np.stack(
                [
                    np.zeros_like(arreglo[:, :, 1]),
                    arreglo[:, :, 1],
                    np.zeros_like(arreglo[:, :, 1]),
                ],
                axis=2,
            ).astype(int)
        elif channel == "b":
            resultado = np.stack(
                [
                    np.zeros_like(arreglo[:, :, 2]),
                    np.zeros_like(arreglo[:, :, 2]),
                    arreglo[:, :, 2],
                ],
                axis=2,
            ).astype(int)
        else:
            raise ValueError(
                f"Canal '{channel}' no válido. Valores posibles: 'r', 'g' o 'b'."
            )
        return Imagen(resultado)

    def flip(self, img_in: Imagen, axis: str) -> Imagen:
        arreglo = img_in.imagen
        if axis == "h":
            resultado = np.flip(arreglo, axis=1)
        elif axis == "v":
            resultado = np.flip(arreglo, axis=0)
        else:
            raise ValueError(
                f"Eje '{axis}' no válido. Valores posibles: 'h' (horizontal) o 'v' (vertical)."
            )
        resultado = resultado.astype(int)
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0
        return Imagen(resultado)

    def set_saturation(self, img_in: Imagen, C: float) -> Imagen:
        # diferencia (arreglo - gris) es "cuánto color tiene el pixel además del gris
        arreglo = img_in.imagen
        gris = self.to_gray(img_in).imagen

        resultado = gris + C * (arreglo - gris)

        resultado = resultado.astype(int)
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0

        return Imagen(resultado)

    def set_contrast(self, img_in: Imagen, C: float) -> Imagen:
        # `F = 259 * (C + 255) / (255 * (259 - C))` y `R = F * (img - 128) + 128`.
        arreglo = img_in.imagen
        F = 259 * (C + 255) / (255 * (259 - C))
        resultado = F * (arreglo - 128) + 128

        resultado = resultado.astype(int)
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0

        return Imagen(resultado)

    def conv_channel(self, img_in: Imagen, kernel: np.ndarray) -> Imagen:
        """La convolución desliza el kernel sobre la imagen para cada pixel,
        centrando el kernel, multiplica sus valores por los pixeles vecinos
        correspondientes y suma esos productos para obtener el nuevo valor
        del pixel. El resultado depende del kernel, puede desenfocar,
        enfocar, etc. Se aplica por separado a cada canal r,g,b y se reensambla
        con np.stack, manteniendo el tamaño original y se satura a [0, 255].."""
        # El cuerpo de este método lo entrega el curso.
        img = img_in.imagen
        img_out = []
        for i in range(img.shape[-1]):
            img_channel = convolve2d(
                img[:, :, i], kernel, mode="same", boundary="symm"
            )
            img_out.append(img_channel)
        new_image = np.stack(img_out, axis=2)
        new_image[new_image > 255], new_image[new_image < 0] = 255, 0
        return Imagen(new_image.astype(int))
