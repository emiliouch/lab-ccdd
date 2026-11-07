"""Particiones temporales, baselines, métricas y backtesting."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import lightgbm
import numpy as np
import polars as pl
from sklearn.base import clone


def _como_fecha(corte: str | datetime, hora: pl.Series) -> datetime:
    """Convierte un corte a `datetime` en la zona horaria de `hora`."""
    if isinstance(corte, str):
        corte = datetime.fromisoformat(corte)
    zona = getattr(hora.dtype, "time_zone", None)
    if corte.tzinfo is None and zona is not None:
        corte = corte.replace(tzinfo=ZoneInfo(zona))
    return corte


def _validar_hora(X: pl.DataFrame) -> None:
    """Exige una columna `hora` de tipo fecha."""
    if "hora" not in X.columns or not isinstance(X.schema["hora"], pl.Datetime):
        raise ValueError("X debe incluir una columna hora de tipo fecha")


def make_model(**params) -> lightgbm.LGBMRegressor:
    """Devuelve un LightGBM sin ajustar con los parámetros recibidos.

    Los valores default sirven para una serie horaria: árboles poco
    profundos, aprendizaje lento y `n_jobs=1` para que el resultado no dependa
    del número de núcleos. `verbose=-1` evita que cada ajuste escriba en la
    salida del notebook.
    """
    return lightgbm.LGBMRegressor(
        n_estimators=params.get("n_estimators", 300),
        num_leaves=params.get("num_leaves", 31),
        learning_rate=params.get("learning_rate", 0.05),
        min_child_samples=params.get("min_child_samples", 20),
        subsample=params.get("subsample", 0.8),
        subsample_freq=1,
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_lambda=params.get("reg_lambda", 1.0),
        random_state=params.get("random_state", 42),
        n_jobs=1,
        verbose=-1,
    )


def temporal_split(
    X: pl.DataFrame,
    y: pl.Series,
    inicio_validation: str | datetime,
    inicio_test: str | datetime,
    horizonte: int = 12,
) -> dict[str, tuple[pl.DataFrame, pl.Series]]:
    """Divide `(X, y)` por la hora del target y purga ambas fronteras.

    `X` trae la hora del origen en `hora`; `y` tiene las mismas filas y orden.
    Los cortes se aplican a `hora + horizonte`. Además, train solo conserva
    targets anteriores a `inicio_validation - horizonte`, y validation,
    targets desde `inicio_validation` hasta antes de `inicio_test - horizonte`.
    Test conserva targets desde `inicio_test`. Las filas purgadas se excluyen,
    sin trasladarlas a otra parte. Así el ajuste y la climatología no usan
    targets posteriores al primer origen nominal del bloque siguiente.

    Rechaza largos distintos, ausencia de `hora` de tipo fecha, entradas
    vacías, horizonte menor que 1, cortes invertidos y particiones vacías.
    """
    if X.height != y.len():
        raise ValueError("X e y deben tener el mismo largo")
    if X.height == 0:
        raise ValueError("X e y no pueden estar vacíos")
    _validar_hora(X)
    if horizonte < 1:
        raise ValueError("El horizonte debe ser al menos 1")

    inicio_validation = _como_fecha(inicio_validation, X["hora"])
    inicio_test = _como_fecha(inicio_test, X["hora"])
    if inicio_test <= inicio_validation:
        raise ValueError("inicio_test debe ser posterior a inicio_validation")

    adelanto = timedelta(hours=horizonte)
    hora_target = X["hora"] + adelanto
    primer_origen_validation = inicio_validation - adelanto
    primer_origen_test = inicio_test - adelanto

    mascaras = {
        "train": hora_target < primer_origen_validation,
        "validation": (hora_target >= inicio_validation)
        & (hora_target < primer_origen_test),
        "test": hora_target >= inicio_test,
    }

    partes = {}
    for nombre, mascara in mascaras.items():
        X_parte = X.filter(mascara)
        if X_parte.height == 0:
            raise ValueError(f"La partición {nombre} quedó vacía")
        partes[nombre] = (X_parte, y.filter(mascara))
    return partes


def rmse(y_real: pl.Series, y_pred: pl.Series) -> float:
    """Raíz del error cuadrático medio entre dos series alineadas."""
    real = np.asarray(y_real, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if real.shape != pred.shape:
        raise ValueError("y_real e y_pred deben tener el mismo largo")
    if real.size == 0:
        raise ValueError("No hay valores para calcular el RMSE")
    return float(np.sqrt(np.mean((real - pred) ** 2)))


def mae(y_real: pl.Series, y_pred: pl.Series) -> float:
    """Error absoluto medio entre dos series alineadas, en °C."""
    real = np.asarray(y_real, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if real.shape != pred.shape:
        raise ValueError("y_real e y_pred deben tener el mismo largo")
    if real.size == 0:
        raise ValueError("No hay valores para calcular el MAE")
    return float(np.mean(np.abs(real - pred)))


def reporte_reglas(
    pronosticos: pl.DataFrame,
    reales: pl.Series,
    descartados: pl.Series,
) -> pl.DataFrame:
    """Reporta MAE y RMSE por regla sobre las mismas filas disponibles.

    `pronosticos` tiene una columna por regla y las mismas filas, en el mismo
    orden, que `reales`. `descartados` marca las filas sin pronóstico de
    persistencia diaria: se excluyen de todas las reglas y de `reales`.
    Devuelve una tabla con `regla`, `n`, `MAE` y `RMSE`.
    """
    if pronosticos.height != reales.len():
        raise ValueError("pronosticos y reales deben tener el mismo largo")
    if "diaria" not in pronosticos.columns:
        raise ValueError("pronosticos debe incluir la columna diaria")
    if pronosticos.height == 0:
        raise ValueError("pronosticos no tiene filas")
    if reales.null_count() > 0:
        raise ValueError("reales no puede tener valores faltantes")

    if descartados.len() != pronosticos.height:
        raise ValueError("descartados debe tener el mismo largo")
    datos = pronosticos.filter(~descartados)
    reales = reales.filter(~descartados)

    if sum(datos.null_count().row(0)) > 0:
        raise ValueError("Quedan pronósticos faltantes después de descartar")
    if reales.len() == 0:
        raise ValueError("No quedan filas para evaluar")

    filas = []
    for regla in datos.columns:
        filas.append(
            {
                "regla": regla,
                "n": reales.len(),
                "MAE": mae(reales, datos[regla]),
                "RMSE": rmse(reales, datos[regla]),
            }
        )
    return pl.DataFrame(filas)


def persistence_forecast(X: pl.DataFrame) -> pl.Series:
    """Baseline de persistencia: repite la temperatura del origen."""
    if "temp_lag_0" not in X.columns:
        raise ValueError("X debe incluir temp_lag_0")
    return X["temp_lag_0"].alias("pronostico")


def daily_persistence_forecast(
    X: pl.DataFrame,
    temperaturas: pl.DataFrame,
    horizonte: int = 12,
) -> pl.Series:
    """Baseline de persistencia diaria: «igual que ayer» a la hora del target.

    El target de cada fila es la temperatura de `hora + horizonte`. La regla
    la pronostica con la temperatura observada 24 horas antes de ese target,
    en `hora + horizonte - 24`, que se busca por hora en `temperaturas`
    (columnas `hora` y `temp`, por ejemplo, la estación limpia). Con
    `horizonte` entre 1 y 24, esa hora no es posterior al origen, así que la
    regla no usa información futura. Devuelve una serie `pronostico` con una
    fila por cada fila de `X`, en el mismo orden.

    No rellena los huecos. Cuando la hora de ayer no está observada, la regla
    no tiene pronóstico y devuelve `null`. Al comparar las reglas en el
    notebook, esas filas se excluyen de todas ellas; las features con
    faltantes se conservan para LightGBM.
    """
    if not 1 <= horizonte <= 24:
        raise ValueError("El horizonte debe estar entre 1 y 24 horas")
    _validar_hora(X)
    if not {"hora", "temp"} <= set(temperaturas.columns):
        raise ValueError("temperaturas debe incluir hora y temp")

    hora_target = X["hora"] + timedelta(hours=horizonte)
    ayer = pl.DataFrame({"hora": hora_target - timedelta(hours=24)})
    return ayer.join(
        temperaturas.select("hora", "temp"),
        on="hora",
        how="left",
        maintain_order="left",
    )["temp"].alias("pronostico")


def climatology_forecast(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X: pl.DataFrame,
    horizonte: int = 12,
) -> pl.Series:
    """Baseline de climatología: media de train por mes y hora del objetivo.

    `X_train` e `y_train` son las filas de entrenamiento: `X_train` aporta la
    hora del origen (columna `hora`) e `y_train`, la temperatura que se
    observó `horizonte` horas después. La media se calcula con la hora a la
    que se pronostica, no con la del origen. Si una combinación de mes y hora
    no aparece en train, usa la media general de train. Devuelve una serie
    `pronostico` con una fila por cada fila de `X`.
    """
    if X_train.height != y_train.len():
        raise ValueError("X_train e y_train deben tener el mismo largo")
    if X_train.height == 0:
        raise ValueError("X_train no puede estar vacío")
    if horizonte < 1:
        raise ValueError("El horizonte debe ser al menos 1")
    _validar_hora(X_train)
    _validar_hora(X)

    adelanto = timedelta(hours=horizonte)
    hora_target_train = X_train["hora"] + adelanto
    promedios = (
        pl.DataFrame(
            {
                "mes": hora_target_train.dt.month(),
                "hora_del_dia": hora_target_train.dt.hour(),
                "temp": y_train,
            }
        )
        .group_by("mes", "hora_del_dia")
        .agg(pl.col("temp").mean().alias("climatologia"))
    )

    hora_target = X["hora"] + adelanto
    return (
        pl.DataFrame(
            {
                "mes": hora_target.dt.month(),
                "hora_del_dia": hora_target.dt.hour(),
            }
        )
        .join(
            promedios,
            on=["mes", "hora_del_dia"],
            how="left",
            maintain_order="left",
        )["climatologia"]
        .fill_null(y_train.mean())
        .alias("pronostico")
    )


def make_expanding_folds(
    indice: pl.Series,
    anios: tuple[int, ...],
    purga_horas: int = 24,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye folds anuales con ventana expansiva y purga.

    Para cada año `Y`, el train son todas las filas anteriores al 1 de enero
    de `Y` menos `purga_horas`, y la validation son las filas de `Y`. La purga
    deja un hueco de al menos el horizonte máximo: sin ella, el objetivo de la
    última fila de train caería dentro del año que se evalúa.

    `indice` es la columna `hora` de `X`. Devuelve posiciones enteras de esa
    columna, no fechas: son las que usa `walk_forward` para seleccionar filas.
    """
    if indice.len() == 0:
        raise ValueError("indice no puede estar vacío")
    if not isinstance(indice.dtype, pl.Datetime):
        raise ValueError("indice debe ser una columna de tipo fecha")
    if len(anios) == 0:
        raise ValueError("Debe indicar al menos un año")
    if purga_horas < 0:
        raise ValueError("purga_horas no puede ser negativa")

    zona = indice.dtype.time_zone
    tz = ZoneInfo(zona) if zona is not None else None
    purga = timedelta(hours=purga_horas)

    resultado = []
    for anio in anios:
        inicio = datetime(anio, 1, 1, tzinfo=tz)
        fin = datetime(anio + 1, 1, 1, tzinfo=tz)
        en_train = indice < inicio - purga
        en_validation = (indice >= inicio) & (indice < fin)

        indices_train = np.flatnonzero(en_train.to_numpy())
        indices_validation = np.flatnonzero(en_validation.to_numpy())
        if indices_train.size == 0:
            raise ValueError(f"El train del año {anio} quedó vacío")
        if indices_validation.size == 0:
            raise ValueError(f"La validation del año {anio} quedó vacía")
        resultado.append((indices_train, indices_validation))
    return resultado


def walk_forward(
    X: pl.DataFrame,
    y: pl.Series,
    modelo,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> pl.DataFrame:
    """Evalúa `modelo` sobre folds de ventana expansiva o móvil.

    `X` trae la hora del origen en la columna `hora`; el modelo no la recibe:
    se ajusta y predice con las demás columnas. Para cada fold ajusta una
    copia del modelo con las filas de train y predice las de validation.
    Devuelve una tabla con las columnas `hora` (el origen), `fold`, `real` y
    `pronostico`.
    """
    if X.height != y.len():
        raise ValueError("X e y deben tener el mismo largo")
    if "hora" not in X.columns:
        raise ValueError("X debe incluir la columna hora")
    if len(folds) == 0:
        raise ValueError("Debe entregar al menos un fold")

    columnas = [columna for columna in X.columns if columna != "hora"]
    tablas = []
    for numero, (indices_train, indices_validation) in enumerate(
        folds, start=1
    ):
        indices_train = np.asarray(indices_train, dtype=np.int64)
        indices_validation = np.asarray(indices_validation, dtype=np.int64)
        if indices_train.size == 0 or indices_validation.size == 0:
            raise ValueError(f"El fold {numero} tiene una partición vacía")
        if indices_train.max() >= indices_validation.min():
            raise ValueError(
                f"El train del fold {numero} no puede incluir validation "
                "ni filas posteriores a su inicio"
            )

        X_train = X[indices_train]
        X_validation = X[indices_validation]
        modelo_fold = clone(modelo)
        modelo_fold.fit(X_train.select(columnas), y.gather(indices_train))
        pronostico = np.asarray(
            modelo_fold.predict(X_validation.select(columnas)), dtype=float
        )
        tablas.append(
            pl.DataFrame(
                {
                    "hora": X_validation["hora"],
                    "fold": pl.Series(
                        [numero] * X_validation.height, dtype=pl.Int64
                    ),
                    "real": y.gather(indices_validation).cast(pl.Float64),
                    "pronostico": pronostico,
                }
            )
        )
    return pl.concat(tablas)


def recall_evento(real: pl.Series, alerta: pl.Series) -> float:
    """Proporción de eventos reales que la alerta avisó."""
    real = np.asarray(real, dtype=bool)
    alerta = np.asarray(alerta, dtype=bool)
    eventos = real.sum()
    if eventos == 0:
        raise ValueError("No hay eventos reales para evaluar el recall")
    return float((real & alerta).sum() / eventos)


def tasa_falsas_alarmas(real: pl.Series, alerta: pl.Series) -> float:
    """Proporción de noches sin evento en que la alerta se encendió."""
    real = np.asarray(real, dtype=bool)
    alerta = np.asarray(alerta, dtype=bool)
    calmas = (~real).sum()
    if calmas == 0:
        raise ValueError("No hay noches sin evento para evaluar la tasa")
    return float((~real & alerta).sum() / calmas)


def costo_alerta(
    real: pl.Series,
    alerta: pl.Series,
    costo_por_alerta: float = 400_000.0,
    costo_helada: float = 20_000_000.0,
) -> float:
    """Costo total de una regla de alerta.

    Toda noche con alerta cuesta `costo_por_alerta`, la parafina, sea una
    falsa alarma o una helada real. Cada helada sin aviso cuesta
    `costo_helada`, porque el evento no se evitó.
    """
    real = np.asarray(real, dtype=bool)
    alerta = np.asarray(alerta, dtype=bool)
    if real.shape != alerta.shape:
        raise ValueError("real y alerta deben tener el mismo largo")

    no_avisadas = (real & ~alerta).sum()
    return float(alerta.sum() * costo_por_alerta + no_avisadas * costo_helada)


def mejor_umbral(
    real: pl.Series,
    pronostico: pl.Series,
    costo_por_alerta: float = 400_000.0,
    costo_helada: float = 20_000_000.0,
) -> tuple[float, float]:
    """Busca el umbral de temperatura que minimiza el costo de la alerta.

    Recorre los valores pronosticados y se queda con el primero que alcanza el
    costo mínimo. Devuelve `(umbral, costo)`: se enciende la alerta cuando el
    pronóstico es menor que el umbral. El umbral se elige en validation y se
    aplica una sola vez en test.
    """
    real = np.asarray(real, dtype=bool)
    pronostico = np.asarray(pronostico, dtype=float)
    if real.shape != pronostico.shape:
        raise ValueError("real y pronostico deben tener el mismo largo")
    if real.size == 0:
        raise ValueError("No hay noches para elegir el umbral")
    if np.isnan(pronostico).any():
        raise ValueError("pronostico no puede tener valores faltantes")

    mejor = None
    # Candidatos de menor a mayor: ante un empate queda el umbral más bajo,
    # es decir, el primero que alcanza el costo mínimo.
    for candidato in np.unique(pronostico):
        costo = costo_alerta(
            real,
            pronostico < candidato,
            costo_por_alerta=costo_por_alerta,
            costo_helada=costo_helada,
        )
        if mejor is None or costo < mejor[1]:
            mejor = (float(candidato), costo)
    return mejor
