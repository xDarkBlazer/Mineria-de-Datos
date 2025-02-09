import streamlit as st
import pickle
import gzip
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model():
    """Carga el modelo desde un archivo comprimido y verifica su integridad."""
    try:
        with gzip.open('model_trained_regressor.pkl.gz', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def load_scaler():
    """Carga el escalador utilizado en el entrenamiento, si existe."""
    try:
        with gzip.open('scaler.pkl.gz', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception:
        return None

def model_page(model_loader, title):
    st.title(title)
    st.write("Introduce las características de la casa para predecir su precio.")

    # Definir nombres y valores por defecto de las características
    feature_names = [
        ("Tasa de criminalidad (CRIM)", 0.1),
        ("Proporción de terrenos residenciales (ZN)", 25.0),
        ("Proporción de acres de negocios (INDUS)", 5.0),
        ("Variable ficticia Charles River (CHAS)", 0),  # Debe ser entero
        ("Concentración de óxidos de nitrógeno (NOX)", 0.5),
        ("Número promedio de habitaciones (RM)", 6.0),
        ("Proporción de unidades antiguas (AGE)", 60.0),
        ("Distancia a centros de empleo (DIS)", 3.0),
        ("Índice de accesibilidad a autopistas (RAD)", 1),
        ("Tasa de impuesto a la propiedad (TAX)", 300.0),
        ("Proporción alumno-maestro (PTRATIO)", 15.0),
        ("Índice de población afroamericana (B)", 400.0),
        ("Porcentaje de población de estatus bajo (LSTAT)", 10.0)
    ]
    
    # Crear entradas con valores por defecto corregidos
    inputs = []
    for feature, default in feature_names:
        if feature == "Variable ficticia Charles River (CHAS)":
            value = st.radio(feature, [0, 1], index=int(default))  # Asegurar que sea int
        else:
            value = st.number_input(feature, min_value=0.0, value=float(default), format="%.4f")
        inputs.append(value)
    
    if st.button("Predecir Precio"):
        model = model_loader()
        scaler = load_scaler()

        if model is not None:
            try:
                # Convertir a numpy array y asegurarse de que CHAS y RAD sean enteros
                features_array = np.array(inputs).reshape(1, -1)
                features_array[:, [3, 8]] = features_array[:, [3, 8]].astype(int)  # CHAS y RAD deben ser enteros
                
                # Aplicar escalado si es necesario
                if scaler:
                    features_array = scaler.transform(features_array)

                # Realizar la predicción
                prediction = model.predict(features_array)
                
                # Mostrar el resultado
                st.success(f"El precio predicho de la casa es: ${prediction[0]:,.2f}")

            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")

def main():
    st.sidebar.title("Navegación")
    page = st.sidebar.selectbox("Elige una página", ["Modelo A", "Modelo B"])

    if page == "Modelo A":
        model_page(load_model, "Predicción de Precios de Viviendas - Modelo A")
    else:
        model_page(load_model, "Predicción de Precios de Viviendas - Modelo B")

    st.sidebar.write("El mejor modelo fue un KernelRidge, este se comparó contra un modelo de ElasticNET y resultó siendo el mejor usando el método de GridSearch.")
    st.sidebar.write("Este modelo fue estandarizado con StandardScaler, con el fin de normalizar los datos restando la media y dividiendo por la desviación estándar de cada característica. Este procedimiento mejora considerablemente el accuracy de modelos sensibles a la escala de las características, tales como el Kernel")
