"""Andamiaje del panel de la Parte 2. Renómbrenlo si quieren.

MATERIAL PROVISTO. Lo que ya está escrito acá —los imports, la configuración
de la página y el cargador de datos— es infraestructura y no se evalúa: son
las mismas líneas para cualquier panel sobre este dataset. Lo que sí se evalúa
son las cuatro secciones de más abajo.

Las secciones están en un orden que funciona, pero no es obligatorio:
reordenarlas o renombrarlas no descuenta. Lo que se corrige es que las cuatro
cosas estén y que cada gráfico explique por qué está ahí.

Para trabajar:

    uv run streamlit run mi_panel.py

Streamlit reejecuta el archivo completo cada vez que alguien mueve un control,
así que el navegador se actualiza solo al guardar.
"""

from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

RUTA_DATOS = Path(__file__).parent / "data" / "raw" / "penguins.csv"

st.set_page_config(
    page_title="Pingüinos de Palmer", page_icon="🐧", layout="wide"
)
st.title("🐧 Pingüinos del archipiélago de Palmer")


@st.cache_data
def cargar_datos() -> pl.DataFrame:
    """Lee el CSV sin modificarlo.

    `null_values=["NA"]` es necesario: el archivo viene de R, donde `NA` marca
    los faltantes. Sin ese argumento, polars lee las columnas numéricas como
    texto. Con él, los nulos quedan adentro — que es lo que queremos, porque
    este laboratorio no limpia nada.
    """
    return pl.read_csv(RUTA_DATOS, null_values=["NA"])


df = cargar_datos()

# colores fijos para los graficos
COLORES_ESPECIE = {
    "Adelie": "sandybrown",
    "Chinstrap": "mediumpurple",
    "Gentoo": "steelblue",
}
SEXOS_VALIDOS = ["MALE", "FEMALE"]

st.markdown(
    "Panel para revisar los datos de los pingüinos de Palmer: Contiene una "
    "tabla interactiva, un informe de calidad y cuatro gráficos."
)

# --- 1) La tabla interactiva -----------------------------------------------
#
# Una tabla con el dataset que el lector pueda ordenar por cualquier columna y
# filtrar con al menos dos controles: uno categórico y uno de rango numérico.
# Esos mismos filtros deben afectar también a los cuatro gráficos. Los
# registros sin valor en la columna del filtro numérico quedan fuera de la
# selección filtrada. Si ningún registro cumple los filtros, muestren un aviso.
#
# Ordenar y buscar los trae `st.dataframe` de fábrica, sin programar nada.
# Filtrar no: los controles devuelven la selección y ustedes filtran el
# DataFrame antes de pasárselo a la tabla. Denle formato a las columnas, que
# `flipper_length_mm` no es un encabezado para mostrarle a un cliente.
#
#   https://docs.streamlit.io/develop/api-reference/data/st.dataframe
#   https://docs.streamlit.io/develop/api-reference/data/st.column_config
#   https://docs.streamlit.io/develop/api-reference/widgets/st.multiselect
#   https://docs.streamlit.io/develop/api-reference/widgets/st.slider

especies = sorted(df["species"].unique().to_list())
islas = sorted(df["island"].unique().to_list())
masa_min = int(df["body_mass_g"].min())
masa_max = int(df["body_mass_g"].max())

with st.sidebar:
    st.header("Filtros")
    especies_sel = st.multiselect("Especie", especies, default=especies)
    islas_sel = st.multiselect("Isla", islas, default=islas)
    rango_masa = st.slider(
        "Masa corporal (g)",
        min_value=masa_min,
        max_value=masa_max,
        value=(masa_min, masa_max),
        step=50,
        help="Los registros sin masa corporal quedan fuera de la selección.",
    )
    st.caption(
        "Los filtros se aplican a la tabla y a los cuatro gráficos. "
        "Mientras que al informe de calidad usa todos los datos y no se aplican los filtros."
    )


# registros sin masa corporal quedan fuera
df_filtrado = df.filter(
    pl.col("species").is_in(especies_sel),
    pl.col("island").is_in(islas_sel),
    pl.col("body_mass_g").is_between(*rango_masa),
)

st.header("Registros")
st.caption(
    f"Mostrando {df_filtrado.height} de {df.height} registros. Hagan clic en "
    "una columna para ordenar; para filtrar utilizar la barra lateral."
)

if df_filtrado.is_empty():
    st.warning(
        "Ningún registro cumple los filtros seleccionados. Ampliar el rango "
        "de masa corporal, o agregar otras especies o islas."
    )
else:
    st.dataframe(
        df_filtrado,
        hide_index=True,
        column_config={
            "species": st.column_config.TextColumn("Especie"),
            "island": st.column_config.TextColumn("Isla"),
            "culmen_length_mm": st.column_config.NumberColumn(
                "Largo del pico (mm)", format="%.1f"
            ),
            "culmen_depth_mm": st.column_config.NumberColumn(
                "Alto del pico (mm)", format="%.1f"
            ),
            "flipper_length_mm": st.column_config.NumberColumn(
                "Largo de la aleta (mm)", format="%d"
            ),
            "body_mass_g": st.column_config.NumberColumn(
                "Masa corporal (g)", format="%d"
            ),
            "sex": st.column_config.TextColumn("Sexo"),
        },
    )


# --- 2) La calidad de los datos --------------------------------------------
#
# Un informe visible en la página, calculado sobre el CSV completo aunque se
# apliquen filtros: qué columnas tienen nulos y cuántos, cuál es el valor
# inesperado de `sex` y cómo pueden afectar esos problemas los recuentos,
# filtros o gráficos del panel.
#
# Las cifras se calculan desde `df`, no se escriben a mano: si el
# archivo cambiara, un número escrito a mano queda mintiendo.
#
#   https://docs.streamlit.io/develop/api-reference/status/st.warning

st.header("Calidad de los datos")
st.caption(
    "Calculado sobre el CSV completo: los filtros no cambian estas cifras."
)

nulos = (
    df.null_count()
    .unpivot(variable_name="Columna", value_name="Nulos")
    .filter(pl.col("Nulos") > 0)
)
filas_con_nulos = df.filter(pl.any_horizontal(pl.all().is_null())).height
sexos_inesperados = df.filter(
    pl.col("sex").is_not_null() & ~pl.col("sex").is_in(SEXOS_VALIDOS)
)
nulos_sexo = df["sex"].null_count()
nulos_masa = df["body_mass_g"].null_count()

col_nulos, col_sexo = st.columns(2)
with col_nulos:
    st.subheader("Valores faltantes")
    st.metric("Filas con al menos un nulo", f"{filas_con_nulos} de {df.height}")
    st.dataframe(nulos, hide_index=True)
with col_sexo:
    st.subheader("Valores inesperados en `sex`")
    valores = ", ".join(
        f"`{v}`" for v in sexos_inesperados["sex"].unique().to_list()
    )
    st.metric(
        "Registros con un valor distinto de MALE/FEMALE",
        sexos_inesperados.height,
    )
    st.markdown(f"Valor(es) encontrado(s): {valores}")
    st.dataframe(sexos_inesperados, hide_index=True)

st.warning(
    f"**Cómo afecta esto a los recuentos y al panel.** "
    f"Para los {nulos_masa} registros sin masa corporal, estos nunca entran en la "
    f"selección filtrada, porque el filtro de masa no puede ubicarlos en "
    f"ningún rango: el total de la tabla ya parte en "
    f"{df.height - nulos_masa} y no en {df.height}. "
    f"En los gráficos que usan `sex` hay {nulos_sexo} registros sin sexo y "
    f"{sexos_inesperados.height} con un valor inválido; si no se excluyen, "
    f"aparecen como categorías extra y los recuentos por sexo no suman lo "
    f"que se espera. En los gráficos que miden variables numéricas (pico, aleta) donde faltan datos, "
    f"los registros correspondientes no se muestran, así que sus totales son "
    f"menores que los de la tabla."
)


# --- 3) Los cuatro gráficos ------------------------------------------------
#
# Cuatro gráficos a elección, de al menos dos tipos distintos. Pueden reusar
# los de la Parte 1 o construir otros. Van a necesitar `plotly.express`:
# impórtenlo arriba, con el resto.
#
# Cada gráfico lleva, JUNTO A ÉL Y VISIBLE EN LA PÁGINA, por qué esa
# información es útil y por qué eligieron esa visualización. Un comentario en
# el código no cuenta: quien abre el panel no lee el código.
#
#   https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
#   https://docs.streamlit.io/develop/api-reference/text/st.caption
#   https://docs.streamlit.io/develop/api-reference/layout/st.columns

st.header("Gráficos")

if df_filtrado.is_empty():
    st.warning(
        "No hay registros que graficar con los filtros actuales. Ajustar los "
        "filtros de la barra lateral."
    )
else:
    fila1_izq, fila1_der = st.columns(2)
    # Grafico 1
    with fila1_izq:
        conteo = df_filtrado.group_by("island", "species").len("registros")
        fig_conteo = px.bar(
            conteo.sort("island", "species"),
            x="island",
            y="registros",
            color="species",
            color_discrete_map=COLORES_ESPECIE,
            labels={
                "island": "Isla",
                "registros": "Registros",
                "species": "Especie",
            },
            title="Registros por isla y especie",
        )
        st.plotly_chart(fig_conteo)
        st.caption(
            "**Por qué es útil:** Porque permite ver cuántos registros hay detrás de "
            "cada grupo y muestra que no todas las especies viven en todas "
            "las islas (Gentoo solo en Biscoe, Chinstrap solo en Dream), algo "
            "clave antes de comparar islas.  \n"
            "**Por qué barras apiladas:** "
            "Porque permite comparar recuentos entre categorías, se lee mejor por largo de "
            "barra, y el apilado muestra a la vez el total por isla y su "
            "composición."
        )
    # Grafico 2
    with fila1_der:
        fig_dispersion = px.scatter(
            df_filtrado,
            x="flipper_length_mm",
            y="body_mass_g",
            color="species",
            color_discrete_map=COLORES_ESPECIE,
            labels={
                "flipper_length_mm": "Largo de la aleta (mm)",
                "body_mass_g": "Masa corporal (g)",
                "species": "Especie",
            },
            title="Aletas más largas, pingüinos más pesados",
        )
        st.plotly_chart(fig_dispersion)
        st.caption(
            "**Por qué es útil:** Porque muestra que el largo de la aleta y la masa "
            "crecen juntos, distinguiéndose que la especie Gentoo forma un grupo aparte, más grande.  \n"
            "**Por qué dispersión:** Es la forma directa de ver la relación "
            "entre dos variables numéricas registro por registro y al usar color "
            "por especie revela la relación entre las variables y la especie."
        )

    fila2_izq, fila2_der = st.columns(2)
    # Grafico 3
    with fila2_izq:
        df_con_sexo = df_filtrado.filter(pl.col("sex").is_in(SEXOS_VALIDOS))
        excluidos = df_filtrado.height - df_con_sexo.height
        fig_cajas = px.box(
            df_con_sexo,
            x="species",
            y="body_mass_g",
            color="sex",
            color_discrete_map={"MALE": "steelblue", "FEMALE": "sandybrown"},
            labels={
                "species": "Especie",
                "body_mass_g": "Masa corporal (g)",
                "sex": "Sexo",
            },
            title="Masa corporal por especie y sexo",
        )
        st.plotly_chart(fig_cajas)
        st.caption(
            "**Por qué es útil:** compara la masa entre especies y, dentro de "
            "cada una, entre machos y hembras: los machos pesan más en las "
            "tres.  \n"
            "**Por qué cajas:** Porque resume métricas como mediana, dispersión y valores "
            "atípicos de varios grupos lado a lado, sin amontonar puntos.  \n"
            f"**Importante:** Para este gráfico se excluyeron {excluidos} "
            "registros de la selección actual sin sexo o con un valor inválido, que "
            "de otro modo aparecerían como otro grupo."
        )
    # Grafico 4
    with fila2_der:
        fig_histograma = px.histogram(
            df_filtrado,
            x="culmen_length_mm",
            color="species",
            color_discrete_map=COLORES_ESPECIE,
            # Un panel por especie, con el eje X compartido: las barras no se
            # superponen y se sigue viendo dónde cae cada distribución.
            facet_row="species",
            category_orders={"species": list(COLORES_ESPECIE)},
            labels={
                "culmen_length_mm": "Largo del pico (mm)",
                "species": "Especie",
            },
            title="Distribución del largo del pico",
            height=500,
        )
        # Cada panel ya dice su especie: se quita el "Especie=" de la
        # etiqueta y la leyenda, que repetiría lo mismo.
        fig_histograma.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1])
        )
        fig_histograma.update_layout(showlegend=False)
        fig_histograma.update_yaxes(title_text="Registros")
        st.plotly_chart(fig_histograma)
        st.caption(
            "**Por qué es útil:** Del gráfico anterior se ve que Adelie y Chinstrap pesan casi lo mismo, "
            "pero en este gráfico se puede visualizar que el largo del pico los separa, ya que Adelie tiene el pico más "
            "corto.  \n"
            " **Por qué histograma por especie:**  Porque muestra la forma de "
            "la distribución de una variable numérica, al tener un panel por especie "
            "con el mismo eje X evita que las barras se mezclen y se fusionen los colores, además "
            "deja comparar directamente dónde cae cada una."
        )

# --- 4) El tema --------------------------------------------------------------
#
# Este no se programa acá: vive en `.streamlit/config.toml`, al lado de este
# archivo. Ya existe, con las claves comentadas — descoméntenlas y decidan sus
# colores.
#
# Para comprobar que el suyo está haciendo algo: renombren el archivo,
# reinicien el panel y vean si cambia. Si no cambia, no lo configuraron.
#
#   https://docs.streamlit.io/develop/concepts/configuration/theming
