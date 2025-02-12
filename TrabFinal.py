import streamlit as st
import pickle
import gzip
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
st.set_page_config(page_title="Predicción de Retiros Bancarios", page_icon="🏦", layout="wide")
# Custom CSS to change the color of the sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: rgb(148, 180, 59);
    }
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-18e3th9, [data-testid="stSidebar"] .css-1dp5vir {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True)

# Encabezado principal
# Añadir imagen como encabezado en todas las páginas
imagen_path = "UNAL-Manizales.png"  # Reemplazar con la ruta correcta si es necesario
st.image(imagen_path, width=150)
st.title("📊 Predicción de Retiros Bancarios")
st.markdown("### Un análisis detallado con modelos de Deep Learning")

# Funciones para cargar los modelos
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

def load_gru_model():
    """Carga el modelo GRU desde un archivo comprimido y verifica su integridad."""
    try:
        with gzip.open('GRU.pkl.gz', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo GRU: {e}")
        return None

def load_dense_model():
    """Carga el modelo Dense desde un archivo comprimido y verifica su integridad."""
    try:
        with gzip.open('Dense.pkl.gz', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo Dense: {e}")
        return None

def make_predictions(model, data):
    """Realiza predicciones utilizando el modelo."""
    predictions = model.predict(data)
    return predictions

def describe_data(data):
    st.write("#### Resumen Estadístico de los Datos")
    st.write(data.describe())

    st.write("#### Histograma de los Datos")
    for column in data.columns:
        plt.figure()
        data[column].hist(bins=20)
        plt.title(f"Histograma de {column}")
        st.pyplot(plt)

    st.write("#### Boxplot de los Datos")
    plt.figure()
    data.boxplot()
    plt.title("Boxplot de los Datos")
    st.pyplot(plt)
    
    st.write("#### Gráfico de Violín de los Datos")
    plt.figure()
    sns.violinplot(data=data)
    plt.title("Gráfico de Violín de los Datos")
    st.pyplot(plt)

def model_page(model_loader, title, reshape_data=False):
    st.subheader(f"Modelo: {title}")
    st.write("Cargue un archivo Excel para predecir los Retiros.")

    uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.write("### Datos Cargados:")
            st.dataframe(data)

            if reshape_data:
                # Generar ventanas de 30 días solapadas
                X = []
                for i in range(len(data) - 30 + 1):
                    X.append(data.values[i:i + 30])
                X = np.array(X).reshape(-1, 30)
            else:
                X = data.values

            model = model_loader()
            if model is not None:
                if st.button("Predecir datos"):
                    # Predicciones sobre los datos de entrenamiento
                    predictions_train = make_predictions(model, X)
                    st.write("#### Predicciones (Datos de Entrenamiento)")
                    st.dataframe(predictions_train)

                    # Plot actual vs predicted (Datos de Entrenamiento)
                    y_test = data.iloc[:len(predictions_train), -1].values
                    plt.figure(figsize=(12, 5))
                    time = np.arange(len(y_test))
                    plt.plot(time, y_test, label='Actual Retiro', linestyle='dashed', alpha=0.8)
                    plt.plot(time, predictions_train, label='Model Prediction', alpha=0.8)
                    plt.legend()
                    plt.title(f"Time Series Predictions - {title} (Datos de Entrenamiento)")
                    plt.xlabel("Time Step")
                    plt.ylabel("Retiro")
                    plt.xlim(0, len(y_test))
                    st.pyplot(plt)

                    # Predicciones futuras (t+1 en adelante)
                    future_predictions = []
                    last_window = X[-1].reshape(1, -1)  # Usar la última ventana de datos para predecir
                    for _ in range(10):  # Generar 10 predicciones futuras (ajustar según se requiera)
                        next_pred = make_predictions(model, last_window)
                        future_predictions.append(next_pred[0])
                        last_window = np.roll(last_window, -1)
                        last_window[0, -1] = next_pred

                    st.write("#### Predicciones Futuras (t+1 en adelante)")
                    st.dataframe(future_predictions)

                    # Plot future predictions
                    plt.figure(figsize=(12, 5))
                    future_time = np.arange(len(y_test), len(y_test) + len(future_predictions))
                    plt.plot(time, y_test, label='Actual Retiro', linestyle='dashed', alpha=0.8)
                    plt.plot(future_time, future_predictions, label='Future Prediction', alpha=0.8)
                    plt.legend()
                    plt.title(f"Time Series Predictions - {title} (Futuro)")
                    plt.xlabel("Time Step")
                    plt.ylabel("Retiro (Escalado)")
                    st.pyplot(plt)

                    # Calculate and display metrics
                    mae = mean_absolute_error(y_test, predictions_train)
                    mse = mean_squared_error(y_test, predictions_train)

                    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
                    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
            else:
                st.error("No se pudo cargar el modelo.")
        except Exception as e:
            st.error(f"Error al procesar el archivo Excel: {e}")
    else:
        st.warning("Por favor, carga un archivo Excel para continuar.")

def descriptive_page():
    st.subheader("Descriptiva de los Datos")
    st.write("Cargue un archivo Excel para ver la descriptiva de los datos.")

    uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.write("### Datos Cargados:")
            st.dataframe(data)

            describe_data(data)
        except Exception as e:
            st.error(f"Error al procesar el archivo Excel: {e}")
    else:
        st.warning("Por favor, carga un archivo Excel para continuar.")

def display_image_from_url(url, caption):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    st.image(image, caption=caption, use_column_width=True)

def main():
    st.sidebar.title("Navegación")
    page = st.sidebar.selectbox("Elige un modelo", ["Descripción del Problema", "Descriptiva de los Datos", "Dense", "RNN", "LSTM", "GRU", "Conclusiones"])

    if page == "Descripción del Problema":
        st.sidebar.write("Por favor elija un modelo para predecir los retiros bancarios.")
        imagen_path2 = "Captura (1).PNG"  # Reemplazar con la ruta correcta si es necesario
        st.image(imagen_path2, width=550)
    
    elif page == "Descriptiva de los Datos":
        descriptive_page()
    
    elif page == "Dense":
        st.sidebar.write("Mejores hiperparámetros encontrados: {'depth': 5, 'epochs': 6, 'num_units': 160, 'optimizer': 'rmsprop', 'activation': 'relu', 'batch_size': 96, 'learning_rate': 0.00014794817129905585}")
        model_page(load_dense_model, "Predicción de Retiros - Dense", reshape_data=True)
    
    elif page == "RNN":
        st.sidebar.write("Mejores hiperparámetros encontrados: {'depth': 3, 'epochs': 5, 'num_units': 80, 'optimizer': 'adam', 'activation': 'relu', 'batch_size': 80, 'learning_rate': 0.0001645624539394657}")
        model_page(load_rnn_model, "Predicción de Retiros - RNN")
    
    elif page == "LSTM":
        st.sidebar.write("Mejores hiperparámetros encontrados: {'depth': 2, 'epochs': 5, 'num_units': 144, 'optimizer': 'adam', 'activation': 'relu', 'batch_size': 56, 'learning_rate': 0.00022336904639532176}")
        model_page(load_lstm_model, "Predicción de Retiros - LSTM")
    
    elif page == "GRU":
        st.sidebar.write("Mejores hiperparámetros encontrados: {'depth': 2, 'epochs': 6, 'num_units': 160, 'optimizer': 'adam', 'activation': 'relu', 'batch_size': 64, 'learning_rate': 0.012659683585100143}")
        model_page(load_gru_model , "Predicción de Retiros - GRU")

    elif page == "Conclusiones":
        imagen_path3 = "DENSED.PNG"  # Reemplazar con la ruta correcta si es necesario
        st.image(imagen_path3, width=450)
        imagen_path4 = "RNND.PNG"  # Reemplazar con la ruta correcta si es necesario
        st.image(imagen_path4, width=450)
        imagen_path5 = "LSTMD.PNG"  # Reemplazar con la ruta correcta si es necesario
        st.image(imagen_path5, width=450)
        imagen_path6 = "GRUD.PNG"  # Reemplazar con la ruta correcta si es necesario
        st.image(imagen_path6, width=450)
        
st.sidebar.write("### Minería de Datos:") 
st.sidebar.write("Edwin Armando Torres Sánchez.")

if __name__ == "__main__":
    main()



