import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
# Metrica de precisión
from sklearn.metrics import r2_score

# configuración y nombre de la pestañita
st.set_page_config(page_title="Proyecto_SDashboard", layout="wide", initial_sidebar_state="expanded")

# Estilo de la pagina, negro chido,
st.markdown("""
    <style>
    .stMetric {
        background-color: #1e293b; 
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True) # quitando el bloqueo de html

# Titulo
st.title("Dashboard Multimodelo por Ricardo Hernández")
st.markdown("Carga tus datos, selecciona tus variables y obtén predicciones usando Regresión Lineal, un Árbol de Decisión y una Red Neuronal.")

# Barra lateral con carga de datos y boton de seleccion
st.sidebar.header("1. Conjunto de Datos 🗁")
tipo_ingreso = st.sidebar.radio("Método de ingreso:", ["Tabla Manual 𓂃✍︎", "Subir archivo CSV"])

# Datos precargados de ejemplo
df_default = pd.DataFrame({
    'Distancia': [2, 4, 6, 8, 10],
    'Tiempo_min': [12, 19, 29, 37, 45]
})
# Condicion dependiendo el tipo de ingreso 
if tipo_ingreso == "Tabla Manual 𓂃✍︎":
    st.sidebar.write("Agrega o edita filas aquí:")
    
    # st.data_editor permite editar la tabla precargada en la misma web
    df = st.sidebar.data_editor(df_default, num_rows="dynamic")
else:
    # uploaded_file permite subir archivos 
    uploaded_file = st.sidebar.file_uploader("Arrastra tu archivo .csv 🗎", type=["csv"])
    # Comprobacion si si subio archivos
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) # pandas lee el archivo  
    else:
        st.sidebar.info("Carga un CSV para empezar.")
        df = df_default

# Configuracion de las varaibles de entreno
st.sidebar.header("2. Variables a Entrenar ⚡︎")

columnas = df.columns.tolist()
# Eleccion de variables dependientes e independientes
col_x = st.sidebar.selectbox("Variable Independiente (Eje X):", columnas, index=0)
col_y = st.sidebar.selectbox("Variable Dependiente (Objetivo Y):", columnas, index=1 if len(columnas)>1 else 0)

# Prediccion
st.sidebar.header("3. Predicción 🔮")
# Valor a predecir
x_nuevo = st.sidebar.number_input(f"Valor de [{col_x}] a predecir:", value=0.0, step=0.5)

st.sidebar.header("Configuración de Modelos ⚙️")
# Arbol: del 1 al 10, empezando en 3
profundidad_arbol = st.sidebar.slider("Profundidad del Árbol:", min_value=1, max_value=15, value=3)
# Neurona: del 1 al 100, empezando en 10
cantidad_neuronas = st.sidebar.slider("Neuronas en capa oculta:", min_value=1, max_value=150, value=10)

# Preparamos los datos
# scikit-learn requiere que X sea 2D (DataFrame) y y sea 1D (Serie)
X = df[[col_x]] # Matriz 2D o tabla
y = df[col_y] # Lista

# ENTRENAMIENTO
# Regresión Lineal
modelo_rl = LinearRegression() # tipo de modelo
modelo_rl.fit(X, y) # entrena con los datos obtenidos
pred_rl = modelo_rl.predict([[x_nuevo]])[0] # Prediccion con el nuevo dato que buscamos
r2_rl = r2_score(y, modelo_rl.predict(X)) # Calificacion de la prediccion

# 2. Árbol de Decisión (Ahora usa tu slider)
modelo_arbol = DecisionTreeRegressor(max_depth=profundidad_arbol, random_state=42)
modelo_arbol.fit(X, y)
pred_arbol = modelo_arbol.predict([[x_nuevo]])[0] # Abre esa lista y dame el primer elemento que encuentres 
r2_arbol = r2_score(y, modelo_arbol.predict(X))

# 3. Red Neuronal (Ahora usa tu slider)
modelo_neurona = MLPRegressor(hidden_layer_sizes=(cantidad_neuronas,), max_iter=2000, random_state=42, solver='lbfgs')
# la documentación oficial de scikit-learn dice que ese es el mejor optimizador cuando tienes bases de datos chiquititas
modelo_neurona.fit(X, y)
pred_neurona = modelo_neurona.predict([[x_nuevo]])[0]
r2_neurona = r2_score(y, modelo_neurona.predict(X))

# Pagina principal y resultados
st.subheader("Resultados de los Modelos ᯓ★")

# Menú horizontal interactivo
tab1, tab2, tab3 = st.tabs(["1. Regresión Lineal 📈", "2. Árbol de Decisión 🌳", "3. Red Neuronal🧠"])

with tab1:
    # coef_ es B1 y intercept_ es B0
    st.markdown(f"**Ecuación de la recta:** $Y = {modelo_rl.coef_[0]:.4f}X + ({modelo_rl.intercept_:.4f})$") # Escribela ecuacion de la recta
    col1, col2 = st.columns(2) #Divicion de las columanas en 2 del mismo tamaño
    col1.metric(label=f"Predicción para X={x_nuevo}", value=f"{pred_rl:.4f}")
    col2.metric(label="Puntaje de Precisión (R²)", value=f"{r2_rl:.4f}")

with tab2:
    st.markdown("**Configuración:** Profundidad máxima de 3 niveles.")
    col1, col2 = st.columns(2)
    col1.metric(label=f"Predicción para X={x_nuevo}", value=f"{pred_arbol:.4f}")
    col2.metric(label="Puntaje de Precisión (R²)", value=f"{r2_arbol:.4f}")

with tab3:
    st.markdown("**Estructura:** Perceptrón Multicapa (1 capa oculta de 100 neuronas).")
    col1, col2 = st.columns(2)
    col1.metric(label=f"Predicción para X={x_nuevo}", value=f"{pred_neurona:.4f}")
    col2.metric(label="Puntaje de Precisión (R²)", value=f"{r2_neurona:.4f}")

# --- VISUALIZACIÓN DE DATOS ---
st.divider()
st.subheader(f"Gráfica de los Datos Históricos: {col_x} vs {col_y}")
# Gráfico interactivo rápido nativo de Streamlit
st.line_chart(df.set_index(col_x)[[col_y]])

# Ejecucion: streamlit run Proyecto_web_python.py
# Detener: Ctrol + C
