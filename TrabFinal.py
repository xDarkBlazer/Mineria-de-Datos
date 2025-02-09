import streamlit as st
import pickle
import gzip
import numpy as np
import pandas as pd

def load_model():
    """Carga el modelo desde un archivo comprimido y verifica su integridad."""
    try:
        with gzip.open('model_trained_regressor.pkl.gz', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def make_predictions(model, data):
    """Realiza predicciones utilizando el modelo."""
    predictions = model.predict(data)
    return predictions

def model_page(model_loader, title):
    st.title(title)
    st.write("Carga un Excel para predecir los Retiros.")

    uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.write("Datos cargados:")
            st.write(data)

            model = model_loader()
            if model is not None:
                if st.button("Predecir datos"):
                    predictions = make_predictions(model, data)
                    st.write("Predicciones:")
                    st.write(predictions)
            else:
                st.error("No se pudo cargar el modelo.")
        except Exception as e:
            st.error(f"Error al procesar el archivo Excel: {e}")
    else:
        st.warning("Por favor, carga un archivo Excel para continuar.")

def main():
    st.sidebar.title("Navegación")
    page = st.sidebar.selectbox("Elige un modelo", ["Dense", "RNN", "LSTM", "GRU", "Modelo A", "Modelo B"])

    if page == "Dense":
        model_page(load_model, "Predicción de Retiros - Dense")
    elif page == "RNN":
        model_page(load_model, "Predicción de Retiros - RNN")
    elif page == "LSTM":
        model_page(load_model, "Predicción de Retiros - LSTM")
    elif page == "Modelo A":
        model_page(load_model, "Predicción de Precios de Viviendas - Modelo A")
    elif page == "Modelo B":
        model_page(load_model, "Predicción de Precios de Viviendas - Modelo B")
    else:
        model_page(load_model, "Predicción de Retiros - GRU")

    st.sidebar.write("El mejor modelo fue un KernelRidge, este se comparó contra un modelo de ElasticNET y resultó siendo el mejor usando el método de GridSearch.")
    st.sidebar.write("Este modelo fue estandarizado con StandardScaler, con el fin de normalizar los datos restando la media y dividiendo por la desviación estándar de cada característica. Este procedimiento mejora considerablemente el accuracy de modelos sensibles a la escala de las características, tales como el Kernel")

if __name__ == "__main__":
    main()


