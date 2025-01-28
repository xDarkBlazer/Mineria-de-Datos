import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image):
  image = image.convert('L') #escala grises
  image = image.resize((28,28))
  image_array = img_to_array(image)/255.0
  image_array = np.expand_dims(image_array, axis=0)
  return image_array
  

def main():
  st.title("Clasificación de la base de datos MNIST")
  st.markdown("Sube una imagen para clasificar")

  uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG:)", type = ["jpg", "png", "jpeg"])

  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    preprocessed_image = preprocess_image(image)
    st.image(preprocessed_image, caption = "imagen subida")


if __name__ == "__main__":
  main()
