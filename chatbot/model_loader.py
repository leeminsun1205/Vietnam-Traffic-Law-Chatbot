# model_loader.py
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from kaggle_secrets import UserSecretsClient
import streamlit as st
import config # Import config để lấy danh sách model
from data_loader import load_or_create_rag_components
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
    
def initialize_app_resources():
    """
    Tải tất cả embedding models, reranker models,
    và khởi tạo RAG components cho mỗi embedding model.
    """
    initialization_successful = True

    # 1. Tải tất cả embedding models (nếu chưa có trong session)
    if not st.session_state.app_loaded_embedding_models:
        with st.status("Đang tải các Embedding Models...", expanded=True) as emb_status:
            st.session_state.app_loaded_embedding_models = load_all_embedding_models() #
            if st.session_state.app_loaded_embedding_models:
                emb_status.update(label=f"Đã tải {len(st.session_state.app_loaded_embedding_models)} Embedding Model(s).", state="complete")
            else:
                emb_status.update(label="Lỗi: Không tải được Embedding Model nào!", state="error")
                initialization_successful = False

    # 2. Tải tất cả reranker models (nếu chưa có trong session)
    if not st.session_state.app_loaded_reranker_models:
        with st.status("Đang tải các Reranker Models...", expanded=True) as rer_status:
            st.session_state.app_loaded_reranker_models = load_all_reranker_models() #
            if st.session_state.app_loaded_reranker_models: # Luôn có key 'Không sử dụng'
                rer_count = len([m for m_name, m in st.session_state.app_loaded_reranker_models.items() if m is not None and m_name != 'Không sử dụng'])
                rer_status.update(label=f"Đã tải {rer_count} Reranker Model(s) (và tùy chọn 'Không sử dụng').", state="complete")
            else: # Trường hợp này gần như không xảy ra nếu config.AVAILABLE_RERANKER_MODELS không rỗng
                rer_status.update(label="Cảnh báo: Không có Reranker model nào được tải.", state="warning")


    # 3. Chuẩn bị RAG components cho từng embedding model đã tải thành công
    if initialization_successful and st.session_state.app_loaded_embedding_models:
        for model_name, emb_model_obj in st.session_state.app_loaded_embedding_models.items():
            if model_name not in st.session_state.app_rag_components_per_embedding_model:
                with st.status(f"Đang chuẩn bị RAG cho: {model_name.split('/')[-1]}...", expanded=True) as rag_status:
                    current_rag_data_prefix = config.get_rag_data_prefix(model_name) #
                    try:
                        # emb_model_obj đã được tải và cache ở trên
                        vector_db, retriever = load_or_create_rag_components(emb_model_obj, current_rag_data_prefix) #
                        if vector_db and retriever:
                            st.session_state.app_rag_components_per_embedding_model[model_name] = (vector_db, retriever)
                            rag_status.update(label=f"RAG cho '{model_name.split('/')[-1]}' đã sẵn sàng.", state="complete")
                        else:
                            rag_status.update(label=f"Lỗi tạo RAG cho '{model_name.split('/')[-1]}'.", state="error")
                            initialization_successful = False # Nếu một RAG không được, coi như lỗi
                            break # Dừng nếu có lỗi
                    except Exception as e:
                        rag_status.update(label=f"Exception khi tạo RAG cho '{model_name.split('/')[-1]}': {e}", state="error")
                        initialization_successful = False
                        break # Dừng nếu có lỗi
    elif not st.session_state.app_loaded_embedding_models: # Nếu không có embedding model nào được tải
        initialization_successful = False

    return initialization_successful
