# app.py (Chatbot.py)
import streamlit as st
import time
import config 
import utils 
from model_loader import load_gemini_model, initialize_app_resources
from reranker import rerank_documents 
from generation import generate_answer_with_gemini 

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(page_title="Chatbot Luật GTĐB", layout="wide", initial_sidebar_state="auto")

# --- Khởi tạo Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn, tôi là chatbot Luật Giao thông Đường bộ. Bạn cần hỗ trợ gì?"}]

# Model selection
if "selected_embedding_model_name" not in st.session_state:
    st.session_state.selected_embedding_model_name = config.DEFAULT_EMBEDDING_MODEL 
if "selected_gemini_model_name" not in st.session_state:
    st.session_state.selected_gemini_model_name = config.DEFAULT_GEMINI_MODEL 
if "selected_reranker_model_name" not in st.session_state:
    st.session_state.selected_reranker_model_name = config.DEFAULT_RERANKER_MODEL 
# Mode selection
if "answer_mode" not in st.session_state: 
    st.session_state.answer_mode = 'Ngắn gọn'
if "retrieval_query_mode" not in st.session_state: 
    st.session_state.retrieval_query_mode = 'Mở rộng'
if "retrieval_method" not in st.session_state: 
    st.session_state.retrieval_method = 'Hybrid'

# --- Tải trước TOÀN BỘ models và RAG components ---
if "app_loaded_embedding_models" not in st.session_state:
    st.session_state.app_loaded_embedding_models = {}
if "app_loaded_reranker_models" not in st.session_state:
    st.session_state.app_loaded_reranker_models = {}
if "app_rag_components_per_embedding_model" not in st.session_state:
    st.session_state.app_rag_components_per_embedding_model = {}

# --- Sidebar ---
with st.sidebar:
    st.title("Tùy chọn")
    st.header("Mô hình")

    current_embedding_name_sb = st.session_state.selected_embedding_model_name
    current_gemini_name_sb = st.session_state.selected_gemini_model_name
    current_reranker_name_sb = st.session_state.selected_reranker_model_name
    current_answer_mode = st.session_state.answer_mode
    current_retrieval_query_mode = st.session_state.retrieval_query_mode
    current_retrieval_method = st.session_state.retrieval_method

    # Model selectbox
    # Selectbox cho Embedding Model
    available_loaded_embedding_names = list(st.session_state.get("app_loaded_embedding_models", {}).keys())
    if not available_loaded_embedding_names:
        available_loaded_embedding_names = config.AVAILABLE_EMBEDDING_MODELS 
    selected_embedding_model_name_ui = st.selectbox(
        "Chọn mô hình Embedding:",
        options=available_loaded_embedding_names,
        index=available_loaded_embedding_names.index(current_embedding_name_sb)
            if current_embedding_name_sb in available_loaded_embedding_names else 0,
        help="Chọn mô hình để vector hóa tài liệu và câu hỏi."
    )
    # Cập nhật session state nếu có thay đổi từ UI
    if selected_embedding_model_name_ui != st.session_state.selected_embedding_model_name:
        st.session_state.selected_embedding_model_name = selected_embedding_model_name_ui
        st.rerun() 

    # Selectbox cho Gemini Model
    selected_gemini_model_name_ui = st.selectbox(
        "Chọn mô hình Gemini:",
        options=config.AVAILABLE_GEMINI_MODELS, 
        index=config.AVAILABLE_GEMINI_MODELS.index(current_gemini_name_sb) 
            if current_gemini_name_sb in config.AVAILABLE_GEMINI_MODELS else 0, 
        help="Chọn mô hình ngôn ngữ lớn để xử lý yêu cầu."
    )
    if selected_gemini_model_name_ui != st.session_state.selected_gemini_model_name:
        st.session_state.selected_gemini_model_name = selected_gemini_model_name_ui
        st.rerun()

    # Selectbox cho Reranker Model
    available_loaded_reranker_names = list(st.session_state.get("app_loaded_reranker_models", {}).keys())
    if not available_loaded_reranker_names: # Fallback
        available_loaded_reranker_names = config.AVAILABLE_RERANKER_MODELS 
    selected_reranker_model_name_ui = st.selectbox(
        "Chọn mô hình Reranker:",
        options=available_loaded_reranker_names,
        index=available_loaded_reranker_names.index(current_reranker_name_sb)
            if current_reranker_name_sb in available_loaded_reranker_names else 0,
        help="Chọn mô hình để xếp hạng lại kết quả tìm kiếm. 'Không sử dụng' để tắt. Các mô hình đã được tải trước."
    )
    if selected_reranker_model_name_ui != st.session_state.selected_reranker_model_name:
        st.session_state.selected_reranker_model_name = selected_reranker_model_name_ui
        st.rerun() 

    # Mode radio
    answer_mode_choice = st.radio(
        "Chọn chế độ trả lời:", 
        options=['Ngắn gọn', 'Đầy đủ'],
        index=['Ngắn gọn', 'Đầy đủ'].index(current_answer_mode),
        # key="answer_mode", 
        horizontal=True, 
        help="Mức độ chi tiết của câu trả lời."
    )
    if answer_mode_choice != st.session_state.answer_mode:
        st.session_state.answer_mode = answer_mode_choice
        st.rerun()

    st.header("Cấu hình truy vấn")
    retrieval_query_mode_choice = st.radio(
        "Nguồn câu hỏi cho Retrieval:", 
        options=['Đơn giản', 'Mở rộng', 'Đa dạng'],
        index = ['Đơn giản', 'Mở rộng', 'Đa dạng'].index(current_retrieval_query_mode),
        # key="retrieval_query_mode", 
        horizontal=True, 
        help=(
            "**Đơn giản:** Chỉ dùng câu hỏi gốc.\n"
            "**Mở rộng:** Chỉ dùng câu hỏi mở rộng từ câu hỏi gốc (do AI tạo).\n"
            "**Đa dạng:** Dùng cả câu hỏi gốc và các biến thể từ câu hỏi gốc(do AI tạo)."
        )
    )
    if retrieval_query_mode_choice != st.session_state.retrieval_query_mode:
        st.session_state.retrieval_query_mode = retrieval_query_mode_choice
        st.rerun()

    retrieval_method_choice = st.radio(
        "Phương thức Retrieval:", 
        options=['Dense', 'Sparse', 'Hybrid'],
        index=['Dense', 'Sparse', 'Hybrid'].index(current_retrieval_method),
        # key="retrieval_method", 
        horizontal=True, 
        help=(
            "**Dense:** Tìm kiếm dựa trên vector ngữ nghĩa (nhanh, hiểu ngữ cảnh).\n"
            "**Sparse:** Tìm kiếm dựa trên từ khóa (BM25) (nhanh, chính xác từ khóa).\n"
            "**Hybrid:** Kết hợp cả Dense và Sparse (cân bằng, có thể tốt nhất)."
        )
    )
    if retrieval_method_choice != st.session_state.retrieval_method:
            st.session_state.retrieval_method = retrieval_method_choice
            st.rerun()

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

# --- Cập nhật Caption hiển thị cấu hình ---
reranker_status_display_main = st.session_state.selected_reranker_model_name
if reranker_status_display_main == 'Không sử dụng':
    reranker_status_display_main = "Tắt"
else:
    reranker_status_display_main = reranker_status_display_main.split('/')[-1]

st.caption(
    f"Embedding: `{st.session_state.selected_embedding_model_name.split('/')[-1]}` | "
    f"Mô hình: `{st.session_state.selected_gemini_model_name}` | Trả lời: `{st.session_state.answer_mode}` | "
    f"Nguồn Query: `{st.session_state.retrieval_query_mode}` | Retrieval: `{st.session_state.retrieval_method}` | "
    f"Reranker: `{reranker_status_display_main}`"
)

# --- Hiển thị Lịch sử Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content_to_display = message["content"]
        if message["role"] == "assistant":
            docs_for_this_message = message.get("relevant_docs_for_display", [])
            content_to_display = utils.render_html_for_assistant_message(content_to_display, docs_for_this_message) #
        st.markdown(content_to_display, unsafe_allow_html=True)


# --- Khởi tạo hệ thống một lần ---
app_init_status_placeholder = st.empty() # Placeholder để hiển thị trạng thái cuối cùng
if "app_resources_initialized" not in st.session_state:
    st.session_state.app_resources_initialized = False

if not st.session_state.app_resources_initialized:
    with st.spinner("Đang khởi tạo toàn bộ hệ thống và các mô hình... Quá trình này có thể mất vài phút."):
        # Khởi tạo toàn bộ models và RAG components
        # Hàm này sẽ cập nhật st.status bên trong nó.
        system_fully_ready = initialize_app_resources()
        st.session_state.app_resources_initialized = system_fully_ready # Lưu trạng thái khởi tạo

# Kiểm tra sau khi đã cố gắng khởi tạo
if st.session_state.app_resources_initialized:
    app_init_status_placeholder.success("✅ Hệ thống và tất cả mô hình đã sẵn sàng!")

    # Lấy các đối tượng model và RAG components CẦN THIẾT cho lần chạy hiện tại
    # dựa trên lựa chọn của người dùng trong session_state
    current_selected_embedding_name_main = st.session_state.selected_embedding_model_name
    current_selected_reranker_name_main = st.session_state.selected_reranker_model_name
    current_selected_gemini_name_main = st.session_state.selected_gemini_model_name

    # Lấy embedding model object đã tải từ session_state
    active_embedding_model_object_main = st.session_state.app_loaded_embedding_models.get(current_selected_embedding_name_main)

    # Lấy RAG components (vector_db, retriever) cho embedding model hiện tại
    active_rag_components_main = st.session_state.app_rag_components_per_embedding_model.get(current_selected_embedding_name_main)
    active_retriever_main = active_rag_components_main[1] if active_rag_components_main else None

    # Lấy reranker model object đã tải (có thể là None)
    active_reranker_model_object_main = st.session_state.app_loaded_reranker_models.get(current_selected_reranker_name_main)

    # Tải (hoặc lấy từ cache của Streamlit) Gemini model được chọn
    active_gemini_llm_main = load_gemini_model(current_selected_gemini_name_main) #

    # --- Kiểm tra lại các active components ---
    proceed_with_chat = True
    if not active_embedding_model_object_main:
        st.error(f"Lỗi nghiêm trọng: Không tìm thấy Embedding model '{current_selected_embedding_name_main.split('/')[-1]}' đã tải.")
        proceed_with_chat = False
    if not active_retriever_main:
        st.error(f"Lỗi nghiêm trọng: Không tìm thấy Retriever cho '{current_selected_embedding_name_main.split('/')[-1]}'.")
        proceed_with_chat = False
    if not active_gemini_llm_main:
        st.error(f"Lỗi nghiêm trọng: Không tải được Gemini model '{current_selected_gemini_name_main}'.")
        proceed_with_chat = False
    # Reranker có thể là None, không cần kiểm tra lỗi ở đây.

    # --- Input và Xử lý ---
    if proceed_with_chat:
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
                    processing_log.append(f"[{time.time() - start_time:.2f}s] Bắt đầu xử lý với: "
                                          f"Emb='{current_selected_embedding_name_main.split('/')[-1]}', "
                                          f"Gem='{current_selected_gemini_name_main}', "
                                          f"Rer='{current_selected_reranker_name_main.split('/')[-1]}'")
                    message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    use_reranker_flag_main = active_reranker_model_object_main is not None and current_selected_reranker_name_main != 'Không sử dụng'

                    if use_reranker_flag_main:
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Reranker '{current_selected_reranker_name_main.split('/')[-1]}' đang hoạt động.")
                    else:
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Reranker không được sử dụng.")
                    message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    history_for_llm1_main = st.session_state.messages[-(config.MAX_HISTORY_TURNS * 2):-1] #
                    log_hist_llm1_main = "(có dùng lịch sử)" if history_for_llm1_main else "(không dùng lịch sử)"
                    processing_log.append(f"[{time.time() - start_time:.2f}s] Phân tích câu hỏi {log_hist_llm1_main}...")
                    message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    relevance_status, direct_answer, all_queries, summarizing_q = utils.generate_query_variations( #
                        original_query=user_query,
                        gemini_model=active_gemini_llm_main,
                        chat_history=history_for_llm1_main,
                        num_variations=config.NUM_QUERY_VARIATIONS #
                    )

                    if relevance_status == 'invalid':
                        full_response = direct_answer if direct_answer and direct_answer.strip() else "⚠️ Câu hỏi của bạn có vẻ không liên quan đến Luật Giao thông Đường bộ Việt Nam."
                        processing_log.append(f"[{time.time() - start_time:.2f}s] Hoàn tất (Câu hỏi không liên quan).")
                    else:
                        recent_chat_history_main = st.session_state.messages[-(config.MAX_HISTORY_TURNS * 2):-1] #
                        queries_to_search_main = []
                        query_source_log_main = ""
                        retrieval_query_mode_main = st.session_state.retrieval_query_mode
                        if retrieval_query_mode_main == 'Đơn giản':
                            queries_to_search_main = [user_query]
                            query_source_log_main = "câu hỏi gốc"
                        elif retrieval_query_mode_main == 'Mở rộng':
                            queries_to_search_main = [summarizing_q] if summarizing_q else [user_query]
                            query_source_log_main = "câu hỏi tóm tắt/gốc"
                        elif retrieval_query_mode_main == 'Đa dạng':
                            queries_to_search_main = all_queries if all_queries else [user_query]
                            query_source_log_main = f"câu hỏi gốc và {max(0, len(all_queries)-1)} biến thể"

                        retrieval_method_main = st.session_state.retrieval_method
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Bắt đầu Retrieval (Nguồn: {query_source_log_main}, Phương thức: {retrieval_method_main})...")
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")

                        collected_docs_data_main = {}
                        retrieval_start_time = time.time()
                        for q_variant in queries_to_search_main:
                            if not q_variant: continue
                            search_results = active_retriever_main.search( #
                                q_variant,
                                active_embedding_model_object_main,
                                method=retrieval_method_main,
                                k=config.VECTOR_K_PER_QUERY #
                            )
                            for item_res in search_results:
                                doc_idx = item_res['index']
                                if doc_idx not in collected_docs_data_main:
                                    collected_docs_data_main[doc_idx] = item_res
                        retrieval_time = time.time() - retrieval_start_time
                        num_unique_docs_main = len(collected_docs_data_main)
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Retrieval ({retrieval_time:.2f}s) tìm thấy {num_unique_docs_main} tài liệu ứng viên.")
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")

                        retrieved_docs_list_main = list(collected_docs_data_main.values())
                        sort_reverse_main = (retrieval_method_main != 'Dense')
                        retrieved_docs_list_main.sort(key=lambda x: x.get('score', 0 if sort_reverse_main else float('inf')), reverse=sort_reverse_main)

                        reranked_documents_for_llm_main = []
                        rerank_time = 0.0

                        if use_reranker_flag_main and num_unique_docs_main > 0:
                            rerank_start_time = time.time()
                            query_for_reranking_main = summarizing_q if summarizing_q else user_query
                            processing_log.append(f"[{time.time() - start_time:.2f}s]: Xếp hạng lại {min(num_unique_docs_main, config.MAX_DOCS_FOR_RERANK)} tài liệu bằng '{current_selected_reranker_name_main.split('/')[-1]}'...") #
                            message_placeholder.markdown(" ".join(processing_log) + "⏳")

                            docs_to_rerank_input_main = retrieved_docs_list_main[:config.MAX_DOCS_FOR_RERANK] #
                            rerank_input_formatted_main = [{'doc': item['doc'], 'index': item['index']} for item in docs_to_rerank_input_main]

                            reranked_results_list_main = rerank_documents( #
                                query_for_reranking_main,
                                rerank_input_formatted_main,
                                active_reranker_model_object_main
                            )
                            reranked_documents_for_llm_main = reranked_results_list_main[:config.FINAL_NUM_RESULTS_AFTER_RERANK] #
                            rerank_time = time.time() - rerank_start_time
                            processing_log.append(f"[{time.time() - start_time:.2f}s]: Rerank ({rerank_time:.2f}s) hoàn tất, chọn top {len(reranked_documents_for_llm_main)}.")
                            message_placeholder.markdown(" ".join(processing_log) + "⏳")
                        elif num_unique_docs_main > 0:
                            processing_log.append(f"[{time.time() - start_time:.2f}s]: Bỏ qua Rerank, lấy top {config.FINAL_NUM_RESULTS_AFTER_RERANK} kết quả Retrieval.") #
                            temp_docs_main = retrieved_docs_list_main[:config.FINAL_NUM_RESULTS_AFTER_RERANK] #
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

                        raw_llm_output = generate_answer_with_gemini( #
                            query_text=user_query,
                            relevant_documents=reranked_documents_for_llm_main,
                            gemini_model=active_gemini_llm_main,
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
                    import traceback
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
    else: # proceed_with_chat is False
        st.error("⚠️ Chatbot không thể hoạt động do thiếu các thành phần model cần thiết. Vui lòng kiểm tra thông báo lỗi ở trên.")

elif not st.session_state.app_resources_initialized:
    app_init_status_placeholder.error("⚠️ Hệ thống CHƯA SẴN SÀNG. Lỗi trong quá trình tải model hoặc tạo RAG. Vui lòng kiểm tra log chi tiết trong các khối 'status' ở trên (nếu có) hoặc làm mới trang để thử lại.")