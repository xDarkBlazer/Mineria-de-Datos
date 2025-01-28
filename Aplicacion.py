import streamlit as st

def main ():
  st.title("Clasificación de la base de datos mnist.")
  st.markdown("Sube una imagen para clasificar")
  uploaded_file = st.file_uploaded("Seleccione una imagen (PNG, JPG, JPEG:)", type = ["png","jpg","jpeg"])

if __name__=='__main__':
  main()
