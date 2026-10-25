from collections.abc import Sequence

import numpy as np

from src.radio_fcfm.contratos import FRANJAS


def a_franja(estado: str) -> str:
    if estado not in FRANJAS:
        raise ValueError(f"Estado de ánimo desconocido: {estado!r}")
    return FRANJAS[estado]


def acierto_por_franja(y_real: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_real) != len(y_pred):
        raise ValueError(
            f"y_real tiene {len(y_real)} canciones e y_pred tiene {len(y_pred)}"
        )
    aciertos = [
        a_franja(real) == a_franja(pred)
        for real, pred in zip(y_real, y_pred, strict=True)
    ]
    return float(np.mean(aciertos))
