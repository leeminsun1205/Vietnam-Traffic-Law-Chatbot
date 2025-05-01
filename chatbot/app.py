# app.py
import streamlit as st
import os
import time 
import config
import utils
import data_loader

MAX_HISTORY_TURNS = 5

# --- Hàm Cache để Khởi tạo DB và Retriever ---
@st.cache_resource
def cached_load_or_create_components(_embedding_model): 
    vector_db, hybrid_retriever = data_loader.load_or_create_rag_components(_embedding_model)
    return vector_db, hybrid_retriever

# --- Cấu hình Trang Streamlit ---
st.set_page_config(page_title="Chatbot Luật GTĐB", layout="wide", initial_sidebar_state="collapsed")

# --- Khởi tạo Session State cho Lịch sử Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Giao diện chính của Ứng dụng ---
st.title("⚖️ Chatbot Hỏi Đáp Luật Giao Thông Đường Bộ VN")
st.caption(f"Dựa trên các văn bản Luật, Nghị Định, Thông tư về Luật giao thông đường bộ Việt Nam.")

# --- Hiển thị Lịch sử Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Khởi tạo hệ thống ---
init_ok = False
with st.status("Đang khởi tạo hệ thống...", expanded=True) as status:
    g_embedding_model = utils.load_embedding_model(config.embedding_model_name)
    g_reranking_model = utils.load_reranker_model(config.reranking_model_name)
    g_gemini_model = utils.load_gemini_model(config.gemini_model_name)
    models_loaded = all([g_embedding_model, g_reranking_model, g_gemini_model])
    g_vector_db, g_hybrid_retriever = cached_load_or_create_components(g_embedding_model)
    retriever_ready = g_hybrid_retriever is not None
    if retriever_ready:
        status.update(label="✅ Hệ thống đã sẵn sàng!", state="complete", expanded=False)
        init_ok = True
    else:
        status.update(label=" Lỗi Khởi Tạo!", state="error", expanded=True)
        if not retriever_ready: st.error("Không thể chuẩn bị cơ sở dữ liệu/retriever.")

# --- Input và Xử lý ---
if init_ok:
    # Sử dụng st.chat_input thay cho form để có giao diện chat quen thuộc hơn
    if user_query := st.chat_input("Nhập câu hỏi của bạn về Luật GTĐB..."):
        # 1. Thêm và hiển thị tin nhắn của người dùng
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Xử lý và tạo phản hồi từ bot
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Tạo placeholder để cập nhật nội dung bot
            full_response = ""
            with st.spinner("⏳ Đang xử lý..."):
                start_time = time.time()

                # --- Query Augmentation, Relevance Check & Direct Answer ---
                # Không truyền history vào đây
                relevance_status, direct_answer, _, summarizing_q = utils.generate_query_variations(
                    user_query, g_gemini_model, num_variations=config.NUM_QUERY_VARIATIONS
                )

                # --- Kiểm tra mức độ liên quan ---
                if relevance_status == 'invalid':
                    if direct_answer and direct_answer.strip():
                        full_response = direct_answer
                    else:
                        full_response = "⚠️ Câu hỏi của bạn có vẻ không liên quan đến Luật Giao thông Đường bộ Việt Nam."
                    message_placeholder.markdown(full_response + f"\n\n*({time.time() - start_time:.2f}s)*")

                # --- Nếu câu hỏi hợp lệ, tiếp tục xử lý RAG ---
                else:
                    status_updates = [f"*{time.time() - start_time:.2f}s: Phân tích câu hỏi...*"]
                    message_placeholder.markdown(" ".join(status_updates) + "...")

                    # --- Lấy lịch sử gần đây cho LLM thứ 2 ---
                    # Lấy tối đa MAX_HISTORY_TURNS * 2 tin nhắn cuối cùng (user + assistant)
                    recent_chat_history = st.session_state.messages[-(MAX_HISTORY_TURNS * 2):-1] # Bỏ qua tin nhắn cuối cùng của user (đã có trong query_text)

                    # 2a. Hybrid Search (Dùng summarizing_q)
                    status_updates.append(f"\n*{time.time() - start_time:.2f}s: Tìm kiếm tài liệu...*")
                    message_placeholder.markdown(" ".join(status_updates) + "...")
                    # ... (code hybrid_search dùng summarizing_q) ...
                    variant_results = g_hybrid_retriever.hybrid_search(
                            summarizing_q, g_embedding_model, # Tìm kiếm bằng câu hỏi tóm tắt
                            vector_search_k=config.VECTOR_K_PER_QUERY,
                            final_k=config.HYBRID_K_PER_QUERY
                    )
                    collected_docs_data = {}
                    for item in variant_results: # Thu thập kết quả
                        doc_index = item['index']
                        if doc_index not in collected_docs_data:
                           collected_docs_data[doc_index] = {'doc': item['doc']}
                    num_unique_docs = len(collected_docs_data)
                    status_updates.append(f"\n*{time.time() - start_time:.2f}s: Tìm thấy {num_unique_docs} tài liệu.*")
                    message_placeholder.markdown(" ".join(status_updates) + "...")

                    unique_docs_for_reranking_input = []
                    if num_unique_docs > 0:
                         # ... (code chuẩn bị doc cho rerank) ...
                        unique_docs_for_reranking_input = [{'doc': data['doc'], 'index': idx}
                                                  for idx, data in collected_docs_data.items()]
                        if len(unique_docs_for_reranking_input) > config.MAX_DOCS_FOR_RERANK:
                            unique_docs_for_reranking_input = unique_docs_for_reranking_input[:config.MAX_DOCS_FOR_RERANK]


                    # 2b. Re-ranking (Dùng summarizing_q)
                    final_relevant_documents = []
                    if unique_docs_for_reranking_input:
                        status_updates.append(f"\n*{time.time() - start_time:.2f}s: Xếp hạng lại tài liệu...*")
                        message_placeholder.markdown(" ".join(status_updates) + "...")
                        reranked_results = utils.rerank_documents(
                            summarizing_q, # Rerank bằng câu hỏi tóm tắt
                            unique_docs_for_reranking_input,
                            g_reranking_model
                        )
                        final_relevant_documents = reranked_results[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                        status_updates.append(f"\n*{time.time() - start_time:.2f}s: Chọn top {len(final_relevant_documents)}.*")
                        message_placeholder.markdown(" ".join(status_updates) + "...")

                    # 2c. Generate Answer (Truyền history vào đây)
                    answer_mode = 'Đầy đủ' # Hoặc lấy từ st.radio nếu bạn muốn giữ lại lựa chọn đó
                    status_updates.append(f"\n*{time.time() - start_time:.2f}s: Tổng hợp câu trả lời ({answer_mode})...*")
                    message_placeholder.markdown(" ".join(status_updates))

                    if final_relevant_documents:
                        full_response = utils.generate_answer_with_gemini(
                            user_query, # Dùng câu hỏi gốc của user cho LLM cuối
                            final_relevant_documents,
                            g_gemini_model,
                            mode=answer_mode,
                            chat_history=recent_chat_history # << Truyền lịch sử chat gần đây
                        )
                    else:
                        # Vẫn có thể truyền history để LLM biết ngữ cảnh chung
                        full_response = utils.generate_answer_with_gemini(
                            user_query,
                            [],
                            g_gemini_model,
                            mode=answer_mode,
                            chat_history=recent_chat_history
                        )

                    # Cập nhật placeholder với câu trả lời cuối cùng
                    message_placeholder.markdown(full_response + f"\n\n*({time.time() - start_time:.2f}s)*")

                    # --- Hiển thị Ảnh (nếu có) ---
                    if final_relevant_documents:
                        st.markdown("---")
                        
                        with st.expander("Xem Hình Ảnh Biển Báo Liên Quan (Nếu có)"):
                            displayed_images = set()
                            image_found_in_context = False
                            cols = st.columns(5) # Hiển thị tối đa 5 ảnh/hàng
                            col_idx = 0
                            for item in final_relevant_documents:
                                doc = item.get('doc')
                                if doc:
                                    metadata = doc.get('metadata', {})
                                    image_path = metadata.get('sign_image_path') 
                                    sign_code = metadata.get('sign_code')

                                    if image_path and image_path not in displayed_images:
                                        
                                        full_image_path = image_path 
                                        # Hoặc full_image_path = os.path.join("images", os.path.basename(image_path))

                                        if os.path.exists(full_image_path):
                                            with cols[col_idx % 5]:
                                                st.image(full_image_path, caption=f"{sign_code}" if sign_code else None, use_column_width=True)
                                            displayed_images.add(image_path)
                                            image_found_in_context = True
                                            col_idx += 1

                            if not image_found_in_context:
                                st.write("_Không tìm thấy hình ảnh biển báo trong các tài liệu tham khảo._")

            st.session_state.messages.append({"role": "assistant", "content": full_response})
elif not init_ok:
    st.error("⚠️ Hệ thống chưa thể khởi động do lỗi. Vui lòng kiểm tra lại cấu hình và đảm bảo có kết nối mạng để tải model lần đầu.")