import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["STREAMLIT_SERVER_PORT"] = "7860"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image
import pandas as pd

import streamlit.web.cli as stcli
import sys

MODEL_PATH = "biomass_model_multi_input.keras"
SCALER_PATH = "metadata_scaler.joblib"
SAMPLE_IMAGE_PATH = "sample_image.jpg"
IMG_SIZE = 224

TARGET_NAMES = [
    "Trevo Seco ( T )",
    "Matéria Morta (Palha)",
    "Relva Verde ( R )",
    "Peso Total Seco",
    "Total de Matéria Verde (R + T)"
]

@st.cache_resource
def load_keras_model(path):
    print(f"Carregando modelo de: {path}")
    try:
        model = tf.keras.models.load_model(path)
        print("Modelo carregado com sucesso.")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

@st.cache_resource
def load_joblib_scaler(path):
    print(f"Carregando scaler de: {path}")
    try:
        scaler = joblib.load(path)
        print("Scaler carregado com sucesso.")
        return scaler
    except Exception as e:
        st.error(f"Erro ao carregar o scaler: {e}")
        return None

model = None
scaler = None

def process_inputs(image, ndvi, height, scaler_obj):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    img_batch = np.expand_dims(img_array, axis=0)
    img_batch = tf.cast(img_batch, tf.float32)

    meta_array = np.array([[ndvi, height]])
    meta_scaled = scaler_obj.transform(meta_array)
    meta_scaled = tf.cast(meta_scaled, tf.float32)

    return {"image_input": img_batch, "meta_input": meta_scaled}


st.set_page_config(page_title="Previsor de Biomassa CSIRO", layout="centered")
st.title("Previsor de Biomassa de Pastagem")
st.write("""
Esta app demonstra um modelo de teste **Multi-Input** (Imagem + Metadados) 
para prever 5 componentes de biomassa de pastagem.
""")
st.write("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Inputs do Utilizador")
    
    uploaded_file = st.file_uploader(
        "Carregue uma imagem de pastagem...", 
        type=["jpg", "jpeg", "png"]
    )
    
    st.caption("Não tem uma imagem? Baixe este exemplo:")
    if os.path.exists(SAMPLE_IMAGE_PATH):
        try:
            with open(SAMPLE_IMAGE_PATH, "rb") as f:
                image_bytes = f.read()
            
            st.image(
                image_bytes, 
                caption="Imagem de Exemplo da Competição", 
                use_container_width=True
            )
            
            st.download_button(
                label="Baixe esta Imagem de Exemplo",
                data=image_bytes,
                file_name=SAMPLE_IMAGE_PATH,
                mime="image/jpeg"
            )
        except Exception as e:
            st.error(f"Erro ao carregar a imagem de exemplo: {e}")
    else:
        st.warning()
    
    st.write("---")
    ndvi_input = st.number_input(
        "Insira o valor de NDVI (Índice de Vegetação)",
        min_value=-1.0, max_value=1.0, value=0.65, step=0.01
    )
    height_input = st.number_input(
        "Insira a Altura Média (cm)",
        min_value=0.0, max_value=50.0, value=15.0, step=0.5
    )
    
    predict_button = st.button("Executar Previsão")


with col2:
    st.subheader("2. Resultados da Previsão")
    
    if predict_button:
        if model is None or scaler is None:
            with st.spinner("Carregando modelo e scaler..."):
                model = load_keras_model(MODEL_PATH)
                scaler = load_joblib_scaler(SCALER_PATH)
        if uploaded_file is None:
            st.error("Por favor, carregue uma imagem primeiro.")
        elif model is None or scaler is None:
            st.error("Modelo ou Scaler não carregados. Verifique os arquivos.")
        else:
            with st.spinner("Executando o modelo..."):
                image = Image.open(uploaded_file)
                processed_inputs = process_inputs(
                    image, ndvi_input, height_input, scaler
                )
                prediction = model.predict(processed_inputs)
                preds = prediction[0]
                preds[preds < 0] = 0 
                
                results_df = pd.DataFrame({
                    "Componente de Biomassa": TARGET_NAMES, 
                    "Previsão (gramas)": preds.round(2)
                })
                
                st.success("Previsão Concluída!")
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )
                st.image(
                    image, 
                    caption="Imagem Carregada", 
                    use_container_width=True
                )
                