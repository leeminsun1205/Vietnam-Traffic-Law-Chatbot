import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from kaggle_secrets import UserSecretsClient 
import streamlit as st

# --- Model Loading Functions ---
# @st.cache_resource
def load_embedding_model(model_name):
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        return model
    except Exception as e:
        st.error(f"Lỗi tải Embedding Model ({model_name}): {e}")
        return None

@st.cache_resource
def load_reranker_model(model_name):
    if not model_name or model_name == 'Không sử dụng':
        return None
    try:
        model = CrossEncoder(model_name)
        return model
    except Exception as e:
        st.error(f"Lỗi tải Reranker Model ({model_name}): {e}")
        return None

# @st.cache_resource
def load_gemini_model(model_name):
    user_secrets = UserSecretsClient()
    google_api_key = user_secrets.get_secret("GOOGLE_API_KEY")

    if google_api_key:
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel(model_name)
        return model
    else:
        st.error("Không tìm thấy GOOGLE_API_KEY.")
        return None