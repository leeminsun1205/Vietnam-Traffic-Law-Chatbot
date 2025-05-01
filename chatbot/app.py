# app.py
import streamlit as st
import os
import logging
import time # Có thể dùng để thêm delay nếu xử lý quá nhanh

# Import các thành phần đã tách file
import config
import utils
import data_loader
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient # Chỉ dùng nếu chạy trên Kaggle

# Cấu hình logging (ghi ra console khi chạy streamlit)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Cấu hình Trang Streamlit ---
st.set_page_config(page_title="Chatbot Luật GTĐB", layout="wide", initial_sidebar_state="collapsed")

# --- Hàm Cache để tải Model ---
@st.cache_resource
def load_embedding_model(model_name):
    logging.info(f"CACHE MISS: Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        logging.info("Embedding model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Lỗi tải Embedding Model ({model_name}): {e}")
        return None

@st.cache_resource
def load_reranker_model(model_name):
    logging.info(f"CACHE MISS: Loading reranker model: {model_name}")
    try:
        model = CrossEncoder(model_name)
        logging.info("Reranker model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Lỗi tải Reranker Model ({model_name}): {e}")
        return None

@st.cache_resource
def load_gemini_model(model_name):
    logging.info(f"CACHE MISS: Loading/Configuring Gemini model: {model_name}")
    
    try:
        user_secrets = UserSecretsClient()
        google_api_key = user_secrets.get_secret("GOOGLE_API_KEY")
        source = "Kaggle secrets"
    except Exception: # Ngoại lệ chung nếu không ở trong Kaggle
        google_api_key = None
        source = "Không tìm thấy"

    if google_api_key:
        logging.info(f"Tìm thấy Google API Key từ: {source}")
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel(model_name)
        logging.info("Gemini model configured successfully.")
        return model
    else:
        st.error("Không tìm thấy GOOGLE_API_KEY trong Streamlit secrets hoặc Kaggle secrets.")
        logging.error("GOOGLE_API_KEY not found.")
        return None

# --- Hàm Cache để Khởi tạo DB và Retriever ---
@st.cache_resource
def cached_load_or_create_components(_embedding_model): # Thêm _ để streamlit biết nó phụ thuộc vào embedding model
    """Wrapper cho data_loader để dùng với cache của Streamlit."""
    if _embedding_model is None:
         st.error("Không thể khởi tạo DB/Retriever vì Embedding Model lỗi.")
         return None, None
    # Gọi hàm xử lý chính từ data_loader.py
    print(_embedding_model is not None)
    vector_db, hybrid_retriever = data_loader.load_or_create_rag_components(_embedding_model)
    print('aaaa')
    return vector_db, hybrid_retriever

# --- Giao diện chính của Ứng dụng ---
st.title("⚖️ Chatbot Hỏi Đáp Luật Giao Thông Đường Bộ VN")
st.caption(f"Dựa trên QC41, TT36 (2024) và các VB liên quan (hiệu lực 2025). Model: {os.path.basename(config.embedding_model_name)}, {os.path.basename(config.reranking_model_name)}")

# --- Khởi tạo hệ thống ---
init_ok = False
with st.status("Đang khởi tạo hệ thống...", expanded=True) as status:
    st.write(f"Tải embedding model: {config.embedding_model_name}...")
    g_embedding_model = load_embedding_model(config.embedding_model_name)

    st.write(f"Tải reranker model: {config.reranking_model_name}...")
    g_reranking_model = load_reranker_model(config.reranking_model_name)

    st.write(f"Tải/Cấu hình Gemini model: {config.gemini_model_name}...")
    g_gemini_model = load_gemini_model(config.gemini_model_name)

    models_loaded = all([g_embedding_model, g_reranking_model, g_gemini_model])

    st.write("Chuẩn bị cơ sở dữ liệu và retriever...")
    g_vector_db, g_hybrid_retriever = None, None
    print(models_loaded)
    if models_loaded: 
        g_vector_db, g_hybrid_retriever = cached_load_or_create_components(g_embedding_model)
    retriever_ready = g_hybrid_retriever is not None
   
    if models_loaded and retriever_ready:
        status.update(label="✅ Hệ thống đã sẵn sàng!", state="complete", expanded=False)
        init_ok = True
    else:
        status.update(label=" Lỗi Khởi Tạo!", state="error", expanded=True)
        if not models_loaded: st.error("Một hoặc nhiều mô hình AI không thể tải.")
        if not retriever_ready: st.error("Không thể chuẩn bị cơ sở dữ liệu/retriever.")

# --- Phần tương tác ---
if init_ok:
    # Sử dụng form để nhóm input và button
    with st.form("query_form"):
        user_query = st.text_area("Nhập câu hỏi của bạn:", height=100, placeholder="Ví dụ: Mức phạt khi không đội mũ bảo hiểm?")
        submitted = st.form_submit_button("Tra cứu 🚀")

    if submitted and user_query:
        st.markdown("---")
        with st.spinner("⏳ Đang xử lý..."):
            start_time = time.time()

            # 1. Query Augmentation
            st.write(f"*{time.time() - start_time:.2f}s: Mở rộng câu hỏi...*")
            all_queries, summarizing_q = utils.generate_query_variations(
                user_query, g_gemini_model, num_variations=config.NUM_QUERY_VARIATIONS
            )

            # 2. Hybrid Search
            st.write(f"*{time.time() - start_time:.2f}s: Tìm kiếm tài liệu liên quan...*")
            collected_docs_data = {}
            for q_idx, query_variant in enumerate(all_queries):
                variant_results = g_hybrid_retriever.hybrid_search(
                    query_variant, g_embedding_model,
                    vector_search_k=config.VECTOR_K_PER_QUERY,
                    final_k=config.HYBRID_K_PER_QUERY
                )
                for item in variant_results:
                    doc_index = item['index']
                    if doc_index not in collected_docs_data:
                        collected_docs_data[doc_index] = {'doc': item['doc']}
            num_unique_docs = len(collected_docs_data)
            st.write(f"*{time.time() - start_time:.2f}s: Tìm thấy {num_unique_docs} tài liệu ứng viên.*")

            # Chuẩn bị cho Re-rank
            unique_docs_for_reranking_input = []
            if num_unique_docs > 0:
                unique_docs_for_reranking_input = [{'doc': data['doc'], 'index': idx}
                                                  for idx, data in collected_docs_data.items()]
                if len(unique_docs_for_reranking_input) > config.MAX_DOCS_FOR_RERANK:
                    unique_docs_for_reranking_input = unique_docs_for_reranking_input[:config.MAX_DOCS_FOR_RERANK]

            # 3. Re-ranking
            final_relevant_documents = []
            if unique_docs_for_reranking_input:
                st.write(f"*{time.time() - start_time:.2f}s: Đánh giá và xếp hạng lại {len(unique_docs_for_reranking_input)} tài liệu...*")
                reranked_results = utils.rerank_documents(
                    summarizing_q,
                    unique_docs_for_reranking_input,
                    g_reranking_model
                )
                final_relevant_documents = reranked_results[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                st.write(f"*{time.time() - start_time:.2f}s: Chọn lọc top {len(final_relevant_documents)} tài liệu.*")
            # else: st.write("   * Không có tài liệu để re-rank.") # Không cần hiển thị

            # 4. Generate Answer
            final_answer = "..."
            if final_relevant_documents:
                st.write(f"*{time.time() - start_time:.2f}s: Tổng hợp câu trả lời...*")
                final_answer = utils.generate_answer_with_gemini(
                    user_query,
                    final_relevant_documents,
                    g_gemini_model
                )
            else:
                st.write(f"*{time.time() - start_time:.2f}s: Không đủ ngữ cảnh, đang tạo câu trả lời chung...*")
                # Vẫn gọi generate nhưng không có context
                final_answer = utils.generate_answer_with_gemini(user_query, [], g_gemini_model)

            end_time = time.time()
            st.write(f"*{end_time - start_time:.2f}s: Hoàn tất!*")

        # --- Hiển thị Kết quả ---
        st.markdown("---")
        st.header("📖 Câu trả lời:")
        st.markdown(final_answer) # Hiển thị câu trả lời từ LLM

        # --- Hiển thị Ảnh (nếu có) ---
        if final_relevant_documents:
            st.markdown("---")
            # Sử dụng expander để không chiếm nhiều diện tích nếu không cần
            with st.expander("Xem Hình Ảnh Biển Báo Liên Quan (Nếu có)"):
                displayed_images = set()
                image_found_in_context = False
                cols = st.columns(5) # Hiển thị tối đa 5 ảnh/hàng
                col_idx = 0
                for item in final_relevant_documents:
                    doc = item.get('doc')
                    if doc:
                        metadata = doc.get('metadata', {})
                        image_path = metadata.get('sign_image_path') # Lấy đường dẫn ảnh
                        sign_code = metadata.get('sign_code') # Lấy mã hiệu biển báo

                        if image_path and image_path not in displayed_images:
                            # Quan trọng: Điều chỉnh đường dẫn ảnh này cho đúng với môi trường deploy
                            # Ví dụ: Nếu ảnh nằm trong thư mục 'images' cùng cấp app.py
                            full_image_path = image_path # Giả sử đường dẫn đã đúng
                            # Hoặc full_image_path = os.path.join("images", os.path.basename(image_path))

                            if os.path.exists(full_image_path):
                                with cols[col_idx % 5]:
                                    st.image(full_image_path, caption=f"{sign_code}" if sign_code else None, use_column_width=True)
                                displayed_images.add(image_path)
                                image_found_in_context = True
                                col_idx += 1
                            # else: print(f"Ảnh không tồn tại: {full_image_path}") # Debug

                if not image_found_in_context:
                    st.write("_Không tìm thấy hình ảnh biển báo trong các tài liệu tham khảo._")

    elif submitted and not user_query:
        st.warning("🤔 Vui lòng nhập câu hỏi!")

elif not init_ok:
    st.error("⚠️ Hệ thống chưa thể khởi động do lỗi. Vui lòng kiểm tra lại cấu hình và đảm bảo có kết nối mạng để tải model lần đầu.")