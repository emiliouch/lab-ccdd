from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def construir_pipeline(k_palabras: int | None = None) -> Pipeline:
    pasos: list[tuple[str, BaseEstimator]] = [
        ("tfidf", TfidfVectorizer(stop_words="english", min_df=2))
    ]
    if k_palabras is not None:
        pasos.append(("seleccion", SelectKBest(chi2, k=k_palabras)))
    pasos.append(("clasificador", LogisticRegression(max_iter=1000)))
    return Pipeline(pasos)
