from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.radio_fcfm.data import limpiar_letra
from src.radio_fcfm.evaluation import a_franja


def predecir_animo(modelo: Pipeline, letras: Sequence[str]) -> pd.DataFrame:
    limpias = [limpiar_letra(letra) for letra in letras]
    probabilidades = modelo.predict_proba(limpias)
    # classes_ viene en orden alfabético, no en el orden de ESTADOS.
    mejores = probabilidades.argmax(axis=1)
    animos = [str(animo) for animo in modelo.classes_[mejores]]
    return pd.DataFrame(
        {
            "animo": animos,
            "franja": [a_franja(animo) for animo in animos],
            "confianza": probabilidades[np.arange(len(limpias)), mejores],
        }
    )
