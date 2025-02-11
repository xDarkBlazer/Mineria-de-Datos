import streamlit as st
import pickle
import gzip
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_rnn_model():
    """Carga el modelo RNN desde un archivo comprimido y verifica su integridad."""
    try:
        with gzip.open('RNN.pkl.gz', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo RNN: {e}")
        return None

def load_lstm_model():
    """Carga el modelo LSTM desde un archivo comprimido y verifica su integridad."""
    try:
        with gzip.open('LSTM.pkl.gz', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo LSTM: {e}")
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
                    
                    # Generate Predictions
                    y_pred = predictions  # Assuming predictions is the output of the model
                    y_test = data.iloc[:, -1].values  # Assuming last column is the target variable

                    # Plot actual vs predicted
                    plt.figure(figsize=(12, 5))
                    time = np.arange(len(y_test))
                    plt.plot(time, y_test, label='Actual Retiro', linestyle='dashed', alpha=0.8)
                    plt.plot(time, y_pred, label='Model Prediction', alpha=0.8)

                    plt.legend()
                    plt.title(f"Time Series Predictions - {title}")
                    plt.xlabel("Time Step")
                    plt.ylabel("Retiro (Escalado)")
                    plt.xlim(0, 80)
                    st.pyplot(plt)
                    
                    # Calculate and display metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)

                    st.write(f"**Mean Absolute Error (MAE):** {mae}")
                    st.write(f"**Mean Squared Error (MSE):** {mse}")
            else:
                st.error("No se pudo cargar el modelo.")
        except Exception as e:
            st.error(f"Error al procesar el archivo Excel: {e}")
    else:
        st.warning("Por favor, carga un archivo Excel para continuar.")

def display_image_from_url(url, caption):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    st.image(image, caption=caption)

def main():
    st.sidebar.title("Navegación")
    page = st.sidebar.selectbox("Elige un modelo", ["Descripción del Problema", "Descriptiva de los Datos", "Dense", "RNN", "LSTM", "GRU"])

    if page == "Descripción del Problema":
        model_page(load_rnn_model, "Descripción del Problema")
    elif page == "Descriptiva de los Datos":
        model_page(load_rnn_model, "Descriptiva de los Datos")
    elif page == "Dense":
        model_page(load_rnn_model, "Predicción de Retiros - Dense")
    elif page == "RNN":
        model_page(load_rnn_model, "Predicción de Retiros - RNN")
    elif page == "LSTM":
        model_page(load_lstm_model, "Predicción de Retiros - LSTM")
        
    st.sidebar.write("El mejor modelo fue un KernelRidge, este se comparó contra un modelo de ElasticNET y resultó siendo el mejor usando el método de GridSearch.")
    st.sidebar.write("Este modelo fue estandarizado con StandardScaler, con el fin de normalizar los datos restando la media y dividiendo por la desviación estándar de cada característica. Este procedimiento mejora considerablemente el accuracy de modelos sensibles a la escala de las características, tales como el Kernel")

if __name__ == "__main__":
    main()



