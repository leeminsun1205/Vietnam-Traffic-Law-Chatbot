# model_loader.py
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from kaggle_secrets import UserSecretsClient
import streamlit as st
import config # Import config để lấy danh sách model

# --- Model Loading Functions ---

@st.cache_resource
def load_single_embedding_model(model_name: str):
    """
    Tải một embedding model cụ thể.
    Hàm này được cache riêng để có thể được gọi từ nhiều nơi nếu cần tải lẻ.
    """
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        # st.info(f"Đã tải Embedding Model: {model_name}") # Bỏ comment nếu cần debug
        return model
    except Exception as e:
        st.error(f"Lỗi tải Embedding Model ({model_name}): {e}")
        return None

@st.cache_resource
def load_all_embedding_models() -> dict:
    """
    Tải tất cả các embedding model từ config.AVAILABLE_EMBEDDING_MODELS.
    Trả về một dictionary: {model_name: model_object}.
    """
    loaded_models = {}
    st.write(f"Bắt đầu tải {len(config.AVAILABLE_EMBEDDING_MODELS)} embedding model(s)...") #
    for model_name in config.AVAILABLE_EMBEDDING_MODELS: #
        with st.spinner(f"Đang tải Embedding model: {model_name}..."):
            model = load_single_embedding_model(model_name) # Gọi hàm đã cache
            if model:
                loaded_models[model_name] = model
                st.success(f"Tải thành công Embedding model: {model_name.split('/')[-1]}")
            else:
                st.warning(f"Không tải được Embedding model: {model_name}")
    if not loaded_models:
        st.error("Không có Embedding model nào được tải thành công!")
    return loaded_models

@st.cache_resource
def load_single_reranker_model(model_name: str):
    """
    Tải một reranker model cụ thể.
    Hàm này được cache riêng.
    """
    if not model_name or model_name == 'Không sử dụng':
        return None
    try:
        model = CrossEncoder(model_name)
        # st.info(f"Đã tải Reranker Model: {model_name}") # Bỏ comment nếu cần debug
        return model
    except Exception as e:
        st.error(f"Lỗi tải Reranker Model ({model_name}): {e}")
        return None

@st.cache_resource
def load_all_reranker_models() -> dict:
    """
    Tải tất cả các reranker model từ config.AVAILABLE_RERANKER_MODELS.
    Trả về một dictionary: {model_name: model_object_or_None}.
    """
    loaded_models = {}
    st.write(f"Bắt đầu tải {len(config.AVAILABLE_RERANKER_MODELS)} reranker model(s)...") #
    for model_name in config.AVAILABLE_RERANKER_MODELS: #
        if model_name == 'Không sử dụng':
            loaded_models[model_name] = None
            st.success("Đã ghi nhận tùy chọn 'Không sử dụng' cho Reranker.")
            continue
        with st.spinner(f"Đang tải Reranker model: {model_name}..."):
            model = load_single_reranker_model(model_name) # Gọi hàm đã cache
            # model có thể là None nếu có lỗi, hoặc nếu model_name là 'Không sử dụng'
            loaded_models[model_name] = model # Lưu cả None nếu không tải được hoặc là 'Không sử dụng'
            if model:
                st.success(f"Tải thành công Reranker model: {model_name.split('/')[-1]}")
            else:
                st.warning(f"Không tải được Reranker model: {model_name} (hoặc đã chọn 'Không sử dụng').")
    return loaded_models

@st.cache_resource
def load_gemini_model(model_name: str):
    """
    Tải một Gemini model cụ thể. Được cache.
    """
    # Kiểm tra xem UserSecretsClient có sẵn không (ví dụ, khi chạy trên Kaggle)
    try:
        user_secrets = UserSecretsClient()
        google_api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    except Exception: # Bắt lỗi chung nếu không có UserSecretsClient
        google_api_key = os.environ.get("GOOGLE_API_KEY") # Thử lấy từ biến môi trường
        if not google_api_key:
            st.error("Không tìm thấy GOOGLE_API_KEY trong Kaggle Secrets hoặc biến môi trường.")
            return None

    if google_api_key:
        genai.configure(api_key=google_api_key)
        try:
            model = genai.GenerativeModel(model_name)
            # st.info(f"Đã tải Gemini Model: {model_name}") # Bỏ comment nếu cần debug
            return model
        except Exception as e:
            st.error(f"Lỗi khi khởi tạo Gemini Model ({model_name}): {e}")
            return None
    else:
        # Thông báo lỗi đã được đưa ra ở trên nếu không tìm thấy key
        return None