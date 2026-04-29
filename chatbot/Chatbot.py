# app.py (Chatbot.py)
import traceback
import log

log.traceback = traceback
import streamlit as st
import time
import config 
import utils 
import traceback
from model_loader import load_gemini_model, initialize_app_resources
from reranker import rerank_documents 
from generation import generate_answer_with_gemini 

# --- Trang Streamlit cho Chatbot ---
st.set_page_config(page_title="Chatbot Luật GTĐB", layout="wide", initial_sidebar_state="auto")

# --- Khởi tạo session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn, tôi là chatbot Luật Giao thông Đường bộ. Bạn cần hỗ trợ gì?"}]

# Khởi tạo session state cho mô hình 
if "selected_emb_name" not in st.session_state:
    st.session_state.selected_emb_name = config.DEFAULT_EMBEDDING_MODEL 
if "selected_secondary_emb_name" not in st.session_state:
    secondary_default = None
    for model_name in config.AVAILABLE_EMBEDDING_MODELS:
        if model_name != config.DEFAULT_EMBEDDING_MODEL:
            secondary_default = model_name
            break
    st.session_state.selected_secondary_emb_name = secondary_default if secondary_default else (config.AVAILABLE_EMBEDDING_MODELS[1] if len(config.AVAILABLE_EMBEDDING_MODELS) > 1 else config.DEFAULT_EMBEDDING_MODEL)
if "hybrid_component_mode" not in st.session_state:
    st.session_state.hybrid_component_mode = "2 Dense + 1 Sparse"
if "selected_gem_name" not in st.session_state:
    st.session_state.selected_gem_name = config.DEFAULT_GEMINI_MODEL 
if "selected_reranker_name" not in st.session_state:
    st.session_state.selected_reranker_name = config.DEFAULT_RERANKER_MODEL 

# Khởi tạo session state cho chế độ
if "answer_mode" not in st.session_state: 
    st.session_state.answer_mode = 'Ngắn gọn'
if "retrieval_query_mode" not in st.session_state: 
    st.session_state.retrieval_query_mode = 'Mở rộng'
if "retrieval_method" not in st.session_state: 
    st.session_state.retrieval_method = 'Kết hợp'

# Tải trước TOÀN BỘ models và RAG components 
if "app_loaded_embedding_models" not in st.session_state:
    st.session_state.app_loaded_embedding_models = {}
if "app_loaded_reranker_models" not in st.session_state:
    st.session_state.app_loaded_reranker_models = {}
if "app_rag_components_per_embedding_model" not in st.session_state:
    st.session_state.app_rag_components_per_embedding_model = {}

# --- Sidebar cho trang Chatbot---
with st.sidebar:
    st.title("Tùy chọn Cấu hình")

    current_emb_name_sb = st.session_state.selected_emb_name
    current_secondary_emb_name_sb = st.session_state.selected_secondary_emb_name
    current_hybrid_component_mode = st.session_state.hybrid_component_mode
    current_gem_name_sb = st.session_state.selected_gem_name
    current_reranker_name_sb = st.session_state.selected_reranker_name
    current_answer_mode = st.session_state.answer_mode
    current_retrieval_query_mode = st.session_state.retrieval_query_mode
    current_retrieval_method = st.session_state.retrieval_method

    # Mode radio
    answer_mode_choice = st.radio(
        "Chọn chế độ trả lời:", 
        options=['Ngắn gọn', 'Đầy đủ'],
        key="answer_mode", 
        index=['Ngắn gọn', 'Đầy đủ'].index(current_answer_mode),
        horizontal=True, 
        help="Mức độ chi tiết của câu trả lời."
    )

    st.header("Cấu hình truy vấn")

    retrieval_query_mode_choice = st.radio(
        "Nguồn câu hỏi cho truy vấn:", 
        options=['Đơn giản', 'Mở rộng', 'Đa dạng'],
        key="retrieval_query_mode",
        index = ['Đơn giản', 'Mở rộng', 'Đa dạng'].index(current_retrieval_query_mode), 
        horizontal=True, 
        help=(
            "**Đơn giản:** Chỉ dùng câu hỏi gốc.\n"
            "**Mở rộng:** Chỉ dùng câu hỏi mở rộng từ câu hỏi gốc (do AI tạo).\n"
            "**Đa dạng:** Dùng cả câu hỏi gốc và các biến thể từ câu hỏi gốc(do AI tạo)."
        )
    )

    retrieval_method_choice = st.radio(
        "Phương thức truy vấn:", 
        options=['Ngữ nghĩa', 'Từ khóa', 'Kết hợp'],
        key="retrieval_method",
        index=['Ngữ nghĩa', 'Từ khóa', 'Kết hợp'].index(current_retrieval_method), 
        horizontal=True, 
        help=(
            "**Dense:** Tìm kiếm dựa trên vector ngữ nghĩa (nhanh, hiểu ngữ cảnh).\n"
            "**Sparse:** Tìm kiếm dựa trên từ khóa (BM25) (nhanh, chính xác từ khóa).\n"
            "**Hybrid:** Kết hợp cả Dense và Sparse (cân bằng, có thể tốt nhất)."
        )
    )

    if current_retrieval_method == 'Kết hợp':
        hybrid_component_mode_choice = st.radio(
            "Cấu hình thành phần Hybrid:",
            options=["1 Dense + 1 Sparse", "2 Dense + 1 Sparse"],
            key="hybrid_component_mode",
            index=["1 Dense + 1 Sparse", "2 Dense + 1 Sparse"].index(current_hybrid_component_mode),
            horizontal=True,
            help="Chọn số lượng Dense encoders sử dụng trong phương thức Kết hợp."
        )

    st.header("Mô hình")

    # Model selectbox
    avail_emb_names = list(st.session_state.get("app_loaded_embedding_models", {}).keys())
    if not avail_emb_names:
        avail_emb_names = config.AVAILABLE_EMBEDDING_MODELS 
    # Selectbox cho Embedding Model
    selected_emb_name_ui = st.selectbox(
        "Chọn mô hình Embedding:",
        options=avail_emb_names,
        key = "selected_emb_name",
        index=avail_emb_names.index(current_emb_name_sb)
            if current_emb_name_sb in avail_emb_names else 0,
        help="Chọn mô hình để vector hóa tài liệu và câu hỏi."
    )
    
    if current_retrieval_method == 'Kết hợp' and st.session_state.hybrid_component_mode == "2 Dense + 1 Sparse": 
        options_for_secondary = [
            name for name in avail_emb_names 
            if name != st.session_state.selected_emb_name
        ]

        current_secondary_val = st.session_state.selected_secondary_emb_name
        
        if not options_for_secondary: 
            st.warning("Cần ít nhất 2 embedding models khác nhau để sử dụng chế độ Hybrid 2-Dense.")
            st.session_state.selected_secondary_emb_name = None
        elif current_secondary_val == st.session_state.selected_emb_name or current_secondary_val not in options_for_secondary:
            st.session_state.selected_secondary_emb_name = options_for_secondary[0]
            current_secondary_val = options_for_secondary[0]

        idx_secondary = 0
        if current_secondary_val and options_for_secondary:
            try:
                idx_secondary = options_for_secondary.index(current_secondary_val)
            except ValueError: 
                st.session_state.selected_secondary_emb_name = options_for_secondary[0]
                idx_secondary = 0
        elif not options_for_secondary:
             st.session_state.selected_secondary_emb_name = None 

        if options_for_secondary: 
            selected_secondary_emb_name_ui = st.selectbox(
                "Chọn mô hình Embedding Phụ (cho Hybrid 2-Dense):",
                options=options_for_secondary,
                key="selected_secondary_emb_name",
                index=idx_secondary,
                help="Chọn mô hình embedding thứ hai. Danh sách này đã loại trừ mô hình Embedding Chính."
            )

    # Selectbox cho Gemini Model
    selected_gem_name_ui = st.selectbox(
        "Chọn mô hình Gemini:",
        options=config.AVAILABLE_GEMINI_MODELS, 
        key = "selected_gem_name",
        index=config.AVAILABLE_GEMINI_MODELS.index(current_gem_name_sb) 
            if current_gem_name_sb in config.AVAILABLE_GEMINI_MODELS else 0, 
        help="Chọn mô hình ngôn ngữ lớn để xử lý yêu cầu."
    )

    avail_reranker_names = list(st.session_state.get("app_loaded_reranker_models", {}).keys())
    if not avail_reranker_names: 
        avail_reranker_names = config.AVAILABLE_RERANKER_MODELS 
    # Selectbox cho Reranker Model
    selected_reranker_name_ui = st.selectbox(
        "Chọn mô hình Reranker:",
        options=avail_reranker_names,
        key = "selected_reranker_name",
        index=avail_reranker_names.index(current_reranker_name_sb)
            if current_reranker_name_sb in avail_reranker_names else 0,
        help="Chọn mô hình để xếp hạng lại kết quả tìm kiếm. 'Không sử dụng' để tắt."
    )

    st.markdown("---")
    st.header("Quản lý Hội thoại")
    if st.button("⚠️ Xóa Lịch Sử Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Chào bạn, tôi là chatbot Luật Giao thông Đường bộ. Bạn cần hỗ trợ gì?"}]
        st.success("Đã xóa lịch sử chat!")
        time.sleep(1)
        st.rerun()
    st.markdown("---")

# --- Giao diện chính của Ứng dụng ---
st.title("⚖️ Chatbot Hỏi Đáp Luật Giao Thông Đường Bộ VN")
st.caption(f"Dựa trên các văn bản Luật, Nghị Định, Thông tư về Luật giao thông đường bộ Việt Nam.")

# --- Khởi tạo tài nguyên cho trang Chatbot ---
app_init_status_placeholder = st.empty() 
if "app_resources_initialized" not in st.session_state:
    st.session_state.app_resources_initialized = False

if not st.session_state.app_resources_initialized:
    with st.spinner("Đang khởi tạo toàn bộ hệ thống và các mô hình... Quá trình này có thể mất vài phút."):
        system_fully_ready = initialize_app_resources()
        st.session_state.app_resources_initialized = system_fully_ready 

# Kiểm tra sau khi đã khởi tạo
if st.session_state.app_resources_initialized:
    app_init_status_placeholder.success("✅ Hệ thống và tất cả mô hình đã sẵn sàng!")

    current_selected_emb_name = st.session_state.selected_emb_name
    current_selected_reranker_name = st.session_state.selected_reranker_name
    current_selected_gem_name = st.session_state.selected_gem_name

    active_emb_obj = st.session_state.app_loaded_embedding_models.get(current_selected_emb_name)
    active_rag_comps = st.session_state.app_rag_components_per_embedding_model.get(current_selected_emb_name)
    active_retriever = active_rag_comps[1] if active_rag_comps else None
    active_reranker_obj = st.session_state.app_loaded_reranker_models.get(current_selected_reranker_name)
    active_gem_obj = load_gemini_model(current_selected_gem_name)

    # --- Kiểm tra lại các active components ---
    proceed_with_chat = True
    if not active_emb_obj:
        st.error(f"Lỗi nghiêm trọng: Không tìm thấy Embedding model '{current_selected_emb_name.split('/')[-1]}' đã tải.")
        proceed_with_chat = False
    if not active_retriever:
        st.error(f"Lỗi nghiêm trọng: Không tìm thấy Retriever cho '{current_selected_emb_name.split('/')[-1]}'.")
        proceed_with_chat = False
    if not active_gem_obj:
        st.error(f"Lỗi nghiêm trọng: Không tải được Gemini model '{current_selected_gem_name}'.")
        proceed_with_chat = False

    # --- Input và Xử lý ---
    if proceed_with_chat:
        # --- Cập nhật Caption hiển thị cấu hình ---

        caption_text = (
            f"Embedding Chính: `{current_selected_emb_name.split('/')[-1]}` | "
            f"Mô hình: `{current_selected_gem_name}` | "
            f"Trả lời: `{current_answer_mode}` | "
            f"Nguồn câu hỏi: `{current_retrieval_query_mode}` | "
            f"Loại truy vấn: `{current_retrieval_method}` | "
            f"Reranker: `{current_selected_reranker_name.split('/')[-1] if current_selected_reranker_name != 'Không sử dụng' else 'Tắt'}`"
        )
        if st.session_state.retrieval_method == 'Kết hợp':
            caption_text += f" | Cấu hình Hybrid: `{st.session_state.hybrid_component_mode}`"
            if st.session_state.hybrid_component_mode == "2 Dense + 1 Sparse" and st.session_state.selected_secondary_emb_name:
                caption_text += f" | Embedding Phụ: `{st.session_state.selected_secondary_emb_name.split('/')[-1]}`"
        st.caption(caption_text)

        # --- Hiển thị Lịch sử Chat ---
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                content_to_display = message["content"]
                if message["role"] == "assistant":
                    docs_for_this_message = message.get("relevant_docs_for_display", [])
                    content_to_display = utils.render_html_for_assistant_message(content_to_display, docs_for_this_message) #
                st.markdown(content_to_display, unsafe_allow_html=True)

        if user_query := st.chat_input("Nhập câu hỏi của bạn về Luật GTĐB..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                raw_llm_output = ""
                processing_log = []
                final_relevant_documents_for_display_main = []
                relevance_status = 'valid'
                try:
                    start_time = time.time()
                    processing_log.append(f"[{time.time() - start_time:.2f}s] Bắt đầu xử lý...")
                    message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    use_reranker_flag_main = active_reranker_obj is not None and current_selected_reranker_name != 'Không sử dụng'

                    if use_reranker_flag_main:
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Reranker '{current_selected_reranker_name.split('/')[-1]}' đang hoạt động.")
                    else:
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Reranker không được sử dụng.")
                    message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    history_for_llm1_main = st.session_state.messages[-(config.MAX_HISTORY_TURNS * 2):-1] 
                    processing_log.append(f"[{time.time() - start_time:.2f}s] Phân tích câu hỏi \"{user_query}\"...")
                    message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    use_two_dense_hybrid_from_session = (st.session_state.hybrid_component_mode == "2 Dense + 1 Sparse")
                    secondary_embedding_model_object_main = None
                    secondary_vector_db_main = None

                    if st.session_state.retrieval_method == 'Kết hợp':
                        selected_secondary_emb_name = st.session_state.get("selected_secondary_emb_name")
                        
                        if use_two_dense_hybrid_from_session: 
                            if selected_secondary_emb_name: 
                                secondary_embedding_model_object_main = st.session_state.app_loaded_embedding_models.get(selected_secondary_emb_name)
                                secondary_rag_components = st.session_state.app_rag_components_per_embedding_model.get(selected_secondary_emb_name)
                                if secondary_rag_components:
                                    secondary_vector_db_main = secondary_rag_components[0]

                                if secondary_embedding_model_object_main and secondary_vector_db_main:
                                    if secondary_embedding_model_object_main != active_emb_obj or \
                                    (secondary_embedding_model_object_main == active_emb_obj and selected_secondary_emb_name != st.session_state.selected_emb_name): # Đảm bảo khác biệt thực sự
                                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Hybrid mode (2-Dense) với Embedding Phụ: {selected_secondary_emb_name.split('/')[-1]}.")
                                    else:
                                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Embedding Phụ giống Embedding Chính. Chuyển sang Hybrid mode (1-Dense).")
                                        use_two_dense_hybrid_from_session = False # Ghi đè nếu không hợp lệ
                                else:
                                    processing_log.append(f"[{time.time() - start_time:.2f}s]: Không tìm thấy component cho Embedding Phụ. Chuyển sang Hybrid mode (1-Dense).")
                                    use_two_dense_hybrid_from_session = False # Ghi đè
                            else:
                                processing_log.append(f"[{time.time() - start_time:.2f}s]: Chưa chọn Embedding Phụ cho chế độ 2-Dense. Chuyển sang Hybrid mode (1-Dense).")
                                use_two_dense_hybrid_from_session = False # Ghi đè
                        else: # Trường hợp "1 Dense + 1 Sparse" được chọn
                            processing_log.append(f"[{time.time() - start_time:.2f}s]: Hybrid mode (1-Dense) đã được chọn.")

                    relevance_status, direct_answer, all_queries, summarizing_q = utils.generate_query_variations(
                        original_query=user_query,
                        gemini_model=active_gem_obj,
                        chat_history=history_for_llm1_main,
                        num_variations=config.NUM_QUERY_VARIATIONS 
                    )
                    
                    if relevance_status == 'invalid':
                        full_response = direct_answer if direct_answer and direct_answer.strip() else "⚠️ Câu hỏi của bạn có vẻ không liên quan đến Luật Giao thông Đường bộ Việt Nam."
                        processing_log.append(f"[{time.time() - start_time:.2f}s] Hoàn tất (Câu hỏi không liên quan).")
                    else:
                        recent_chat_history_main = st.session_state.messages[-(config.MAX_HISTORY_TURNS * 2):-1] 
                        queries_to_search_main = []
                        query_source_log_main = ""
                        retrieval_query_mode_main = st.session_state.retrieval_query_mode
                        if retrieval_query_mode_main == 'Đơn giản':
                            queries_to_search_main = [user_query]
                            query_source_log_main = "câu hỏi gốc"
                        elif retrieval_query_mode_main == 'Mở rộng':
                            queries_to_search_main = [summarizing_q] if summarizing_q else [user_query]
                            query_source_log_main = "câu hỏi mở rộng từ câu gốc"
                        elif retrieval_query_mode_main == 'Đa dạng':
                            queries_to_search_main = all_queries if all_queries else [user_query]
                            query_source_log_main = f"câu hỏi gốc và {max(0, len(all_queries)-1)} biến thể"

                        retrieval_method_main = st.session_state.retrieval_method
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Bắt đầu Truy vấn (Nguồn: {query_source_log_main}, Phương thức: {retrieval_method_main})...")
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")

                        collected_docs_data_main = {}
                        retrieval_start_time = time.time()
                        
                        for q_variant in queries_to_search_main:
                            if not q_variant: continue
                            
                            search_results = active_retriever.search(
                                q_variant,
                                active_emb_obj,
                                method=st.session_state.retrieval_method,
                                k=config.VECTOR_K_PER_QUERY if st.session_state.retrieval_method != 'Kết hợp' else config.HYBRID_K_PER_QUERY,
                                secondary_embedding_model=secondary_embedding_model_object_main if use_two_dense_hybrid_from_session else None,
                                secondary_vector_db=secondary_vector_db_main if use_two_dense_hybrid_from_session else None,
                                use_two_dense_if_hybrid=use_two_dense_hybrid_from_session
                            )

                            for item_res in search_results:
                                doc_idx = item_res['index']
                                if doc_idx not in collected_docs_data_main:
                                    collected_docs_data_main[doc_idx] = item_res
                        retrieval_time = time.time() - retrieval_start_time
                        num_unique_docs_main = len(collected_docs_data_main)
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Truy vấn ({retrieval_time:.2f}s) tìm thấy {num_unique_docs_main} tài liệu ứng viên.")
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")

                        retrieved_docs_list_main = list(collected_docs_data_main.values())
                        sort_reverse_main = (retrieval_method_main != 'Ngữ nghĩa')
                        retrieved_docs_list_main.sort(key=lambda x: x.get('score', 0 if sort_reverse_main else float('inf')), reverse=sort_reverse_main)

                        reranked_documents_for_llm_main = []
                        rerank_time = 0.0

                        if use_reranker_flag_main and num_unique_docs_main > 0:
                            rerank_start_time = time.time()
                            query_for_reranking_main = summarizing_q if summarizing_q else user_query
                            processing_log.append(f"[{time.time() - start_time:.2f}s]: Xếp hạng lại {min(num_unique_docs_main, config.MAX_DOCS_FOR_RERANK)} tài liệu bằng '{current_selected_reranker_name.split('/')[-1]}'...") #
                            message_placeholder.markdown(" ".join(processing_log) + "⏳")

                            docs_to_rerank_input_main = retrieved_docs_list_main[:config.MAX_DOCS_FOR_RERANK] #
                            rerank_input_formatted_main = [{'doc': item['doc'], 'index': item['index']} for item in docs_to_rerank_input_main]

                            reranked_results_list_main = rerank_documents( #
                                query_for_reranking_main,
                                rerank_input_formatted_main,
                                active_reranker_obj
                            )
                            reranked_documents_for_llm_main = reranked_results_list_main[:config.FINAL_NUM_RESULTS_AFTER_RERANK] #
                            rerank_time = time.time() - rerank_start_time
                            processing_log.append(f"[{time.time() - start_time:.2f}s]: Xếp hạng ({rerank_time:.2f}s) hoàn tất, chọn top {len(reranked_documents_for_llm_main)}.")
                            message_placeholder.markdown(" ".join(processing_log) + "⏳")
                        elif num_unique_docs_main > 0:
                            processing_log.append(f"[{time.time() - start_time:.2f}s]: Bỏ qua Xếp hạng, lấy top {config.FINAL_NUM_RESULTS_AFTER_RERANK} kết quả Retrieval.") 
                            temp_docs_main = retrieved_docs_list_main[:config.FINAL_NUM_RESULTS_AFTER_RERANK] 
                            reranked_documents_for_llm_main = [
                                {'doc': item['doc'], 'score': item.get('score'), 'original_index': item['index']}
                                for item in temp_docs_main
                            ]
                            message_placeholder.markdown(" ".join(processing_log) + "⏳")
                        else:
                            processing_log.append(f"[{time.time() - start_time:.2f}s]: Không tìm thấy tài liệu liên quan.")
                            message_placeholder.markdown(" ".join(processing_log) + "⏳")

                        final_relevant_documents_for_display_main = reranked_documents_for_llm_main

                        answer_mode_main = st.session_state.answer_mode
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Tổng hợp câu trả lời (chế độ: {answer_mode_main})...")
                        message_placeholder.markdown(" ".join(processing_log))

                        raw_llm_output = generate_answer_with_gemini( 
                            query_text=user_query,
                            relevant_documents=reranked_documents_for_llm_main,
                            gemini_model=active_gem_obj,
                            mode=answer_mode_main,
                            chat_history=recent_chat_history_main
                        )
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Hoàn tất!")

                    with st.expander("Xem chi tiết quá trình xử lý", expanded=False):
                        log_content_main = "\n".join(processing_log)
                        st.markdown(f"```text\n{log_content_main}\n```")

                    if raw_llm_output:
                        content_for_display_main = utils.render_html_for_assistant_message(raw_llm_output, final_relevant_documents_for_display_main) #
                        message_placeholder.markdown(content_for_display_main, unsafe_allow_html=True)
                        full_response = raw_llm_output
                    else:
                        message_placeholder.markdown(full_response, unsafe_allow_html=True)

                except Exception as e_main:
                    st.error(f"🐞 Đã xảy ra lỗi: {e_main}")
                    st.expander("Xem Traceback Lỗi").code(traceback.format_exc())
                    full_response = f"🐞 Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại hoặc thay đổi cấu hình."
                    message_placeholder.markdown(full_response, unsafe_allow_html=True)
                finally:
                    if user_query and full_response:
                        utils.log_qa_to_json(user_query, full_response) #

                    if full_response:
                        assistant_message_main = {"role": "assistant", "content": full_response}
                        if relevance_status != 'invalid' and final_relevant_documents_for_display_main:
                            assistant_message_main["relevant_docs_for_display"] = final_relevant_documents_for_display_main
                        else:
                            assistant_message_main["relevant_docs_for_display"] = []
                        st.session_state.messages.append(assistant_message_main)
    else:
        st.error("⚠️ Chatbot không thể hoạt động do thiếu các thành phần cần thiết. Vui lòng kiểm tra thông báo lỗi ở trên.")

elif not st.session_state.app_resources_initialized:
    app_init_status_placeholder.error("⚠️ Hệ thống CHƯA SẴN SÀNG. Lỗi trong quá trình tải model hoặc tạo RAG. Vui lòng kiểm tra log chi tiết trong các khối 'status' ở trên (nếu có) hoặc làm mới trang để thử lại.")