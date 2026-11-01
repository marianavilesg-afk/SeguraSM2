import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="SeguraSM", page_icon="🚦")

st.title("🚦 SeguraSM")
st.subheader("Predicción de la gravedad de siniestros viales en Santa Marta")

RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_seguraSM.joblib")

@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA)

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error("No se pudo cargar el modelo. Archivos que ve la app:")
    st.write(os.listdir(os.path.dirname(RUTA)))
    st.exception(e)
    st.stop()

enc = modelo.named_steps["prep"].named_transformers_["cat"]
tipos = list(enc.categories_[0])
marcas = list(enc.categories_[1])

st.write("Ingresa los datos del vehículo y el modelo estimará si el siniestro sería **con heridos** o **con muertos**.")

col1, col2 = st.columns(2)
with col1:
    tipo = st.selectbox("Tipo de vehículo", tipos)
    marca = st.selectbox("Marca", marcas)
with col2:
    edad = st.slider("Edad del vehículo (años)", 0, 40, 5)
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
             "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    mes_nombre = st.selectbox("Mes del siniestro", meses)
    mes = meses.index(mes_nombre) + 1

if st.button("PREDECIR", type="primary"):
    entrada = pd.DataFrame([{"TIPO_VEHICULO": tipo, "MARCA_AGRUP": marca,
                             "EDAD_VEHICULO": edad, "MES": mes}])
    prob = modelo.predict_proba(entrada)[0, 1]
    pred = modelo.predict(entrada)[0]

    if pred == 1:
        st.error(f"Predicción: **CON MUERTOS**  (probabilidad: {prob:.0%})")
    else:
        st.success(f"Predicción: **CON HERIDOS**  (probabilidad de muertos: {prob:.0%})")
    st.progress(float(prob))

st.warning("⚠️ Esto es una predicción, no una verdad absoluta. El modelo usa pocas variables "
           "(tipo, marca, edad del vehículo y mes) y puede equivocarse. Debe usarse solo como "
           "apoyo, nunca para decidir por sí solo.")
st.caption("Proyecto de grado — Machine Learning | Datos: Datos Abiertos Colombia (Ley 2251-2022)")
