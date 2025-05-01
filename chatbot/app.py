# app.py
import streamlit as st
import os
import time 
import config
import utils
import data_loader

# --- Hàm Cache để Khởi tạo DB và Retriever ---
@st.cache_resource
def cached_load_or_create_components(_embedding_model): 
    vector_db, hybrid_retriever = data_loader.load_or_create_rag_components(_embedding_model)
    return vector_db, hybrid_retriever

# --- Cấu hình Trang Streamlit ---
st.set_page_config(page_title="Chatbot Luật GTĐB", layout="wide", initial_sidebar_state="collapsed")

# --- Giao diện chính của Ứng dụng ---
st.title("⚖️ Chatbot Hỏi Đáp Luật Giao Thông Đường Bộ VN")
st.caption(f"Dựa trên các văn bản Luật, Nghị Định, Thông tư về Luật giao thông đường bộ Việt Nam.")

# --- Khởi tạo hệ thống ---
init_ok = False
with st.status("Đang khởi tạo hệ thống...", expanded=True) as status:
    st.write(f"Tải embedding model: {config.embedding_model_name}...")
    g_embedding_model = utils.load_embedding_model(config.embedding_model_name)

    st.write(f"Tải reranker model: {config.reranking_model_name}...")
    g_reranking_model = utils.load_reranker_model(config.reranking_model_name)

    st.write(f"Cấu hình Gemini model: {config.gemini_model_name}...")
    g_gemini_model = utils.load_gemini_model(config.gemini_model_name)

    models_loaded = all([g_embedding_model, g_reranking_model, g_gemini_model])

    st.write("Chuẩn bị cơ sở dữ liệu và retriever...")
    g_vector_db, g_hybrid_retriever = cached_load_or_create_components(g_embedding_model)
    retriever_ready = g_hybrid_retriever is not None
    if retriever_ready:
        status.update(label="✅ Hệ thống đã sẵn sàng!", state="complete", expanded=False)
        init_ok = True
    else:
        status.update(label=" Lỗi Khởi Tạo!", state="error", expanded=True)
        if not retriever_ready: st.error("Không thể chuẩn bị cơ sở dữ liệu/retriever.")

# --- Phần tương tác ---
if init_ok:
    with st.form("query_form"):
        user_query = st.text_area("Nhập câu hỏi của bạn:", height=100, placeholder="Ví dụ: Mức phạt khi không đội mũ bảo hiểm?")
        submitted = st.form_submit_button("Tra cứu 🚀")

    if submitted and user_query:
        st.markdown("---")
        with st.spinner("⏳ Đang xử lý..."):
            start_time = time.time()

            # --- 1. Query Augmentation, Relevance Check & Direct Answer ---
            st.write(f"*{time.time() - start_time:.2f}s: Phân tích câu hỏi...*")
            relevance_status, direct_answer, all_queries, summarizing_q = utils.generate_query_variations(
                user_query, g_gemini_model, num_variations=config.NUM_QUERY_VARIATIONS
            )

            # --- Kiểm tra mức độ liên quan ---
            if relevance_status == 'invalid':
                st.markdown("---")
                st.header("📖 Câu trả lời:")
                if direct_answer and direct_answer.strip():
                    st.markdown(direct_answer) # Hiển thị câu trả lời trực tiếp từ LLM
                else:
                    # Fallback nếu LLM không tạo câu trả lời trực tiếp
                    st.warning("⚠️ Câu hỏi của bạn có vẻ không liên quan đến Luật Giao thông Đường bộ Việt Nam.")
                end_time_invalid = time.time()
                st.write(f"*{end_time_invalid - start_time:.2f}s: Hoàn tất!*")
                st.stop() # Dừng xử lý tại đây

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
                final_answer = utils.generate_answer_with_gemini(user_query, [], g_gemini_model)

            end_time = time.time()
            st.write(f"*{end_time - start_time:.2f}s: Hoàn tất!*")

        # --- Hiển thị Kết quả ---
        st.markdown("---")
        st.header("📖 Câu trả lời:")
        st.markdown(final_answer) 

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

    elif submitted and not user_query:
        st.warning("🤔 Vui lòng nhập câu hỏi!")

elif not init_ok:
    st.error("⚠️ Hệ thống chưa thể khởi động do lỗi. Vui lòng kiểm tra lại cấu hình và đảm bảo có kết nối mạng để tải model lần đầu.")