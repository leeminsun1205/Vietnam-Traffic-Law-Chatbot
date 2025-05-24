# app.py
import streamlit as st
import time
import config
import utils 
from model_loader import load_embedding_model, load_gemini_model, load_reranker_model
from data_loader import load_or_create_rag_components #Đảm bảo import đúng
from reranker import rerank_documents
from generation import generate_answer_with_gemini

# @st.cache_resource # Sử dụng cache của Streamlit cho hàm này rất quan trọng
def cached_load_or_create_components(embedding_model_name: str, _embedding_model_object):
    # Hàm này sẽ tạo hoặc tải VectorDB và Retriever cho một embedding model cụ thể.
    # _embedding_model_object được truyền vào để đảm bảo cache hoạt động đúng khi object thay đổi.
    current_rag_data_prefix = config.get_rag_data_prefix(embedding_model_name)
    vector_db, retriever = load_or_create_rag_components(_embedding_model_object, current_rag_data_prefix)
    return vector_db, retriever

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(page_title="Chatbot Luật GTĐB", layout="wide", initial_sidebar_state="auto")

# --- Khởi tạo Session State cho Lịch sử Chat và Cấu hình ---

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn, tôi là chatbot Luật Giao thông Đường bộ. Bạn cần hỗ trợ gì?"}]

# Model chính
if "selected_embedding_model" not in st.session_state:
    st.session_state.selected_embedding_model = config.DEFAULT_EMBEDDING_MODEL
# Model phụ cho hybrid
if "additional_hybrid_models" not in st.session_state:
    st.session_state.additional_hybrid_models = [] # Khởi tạo là list rỗng

if "selected_gemini_model" not in st.session_state:
    st.session_state.selected_gemini_model = config.DEFAULT_GEMINI_MODEL
if "answer_mode" not in st.session_state:
    st.session_state.answer_mode = 'Ngắn gọn'
if "retrieval_query_mode" not in st.session_state:
    st.session_state.retrieval_query_mode = 'Mở rộng' 
if "retrieval_method" not in st.session_state:
    st.session_state.retrieval_method = 'hybrid'
if "selected_reranker_model" not in st.session_state:
    st.session_state.selected_reranker_model = config.DEFAULT_RERANKER_MODEL

# --- Sidebar ---
with st.sidebar:
    st.title("Tùy chọn")
    st.header("Mô hình")

    # Chọn mô hình Embedding chính
    selected_embedding_model_name = st.selectbox(
        "Chọn mô hình Embedding chính:",
        options=config.AVAILABLE_EMBEDDING_MODELS,
        index=config.AVAILABLE_EMBEDDING_MODELS.index(st.session_state.selected_embedding_model),
        key="selected_embedding_model", # Giữ key này cho session state
        help="Chọn mô hình embedding chính. Mô hình này sẽ được dùng cho tìm kiếm 'dense' thuần túy, làm cơ sở cho BM25 (sparse) và là một phần của tìm kiếm 'hybrid'."
    )

    # Chọn mô hình Embedding phụ (chỉ hiển thị khi retrieval_method là 'hybrid')
    if st.session_state.retrieval_method == 'hybrid':
        available_for_additional = [
            m for m in config.AVAILABLE_EMBEDDING_MODELS 
            if m != st.session_state.selected_embedding_model # Loại trừ model chính đã chọn
        ]
        
        # Lấy lựa chọn hiện tại từ session_state, đảm bảo nó là list
        current_additional_selection = st.session_state.get("additional_hybrid_models", [])
        if not isinstance(current_additional_selection, list):
            current_additional_selection = []
        
        # Lọc ra những lựa chọn cũ không còn hợp lệ (ví dụ model chính thay đổi)
        valid_current_additional_selection = [m for m in current_additional_selection if m in available_for_additional]

        if available_for_additional:
            selected_additional_models = st.multiselect(
                "Chọn thêm mô hình Embedding phụ cho Hybrid (tối đa 2):",
                options=available_for_additional,
                default=valid_current_additional_selection, # default là list đã lọc
                # key="additional_hybrid_models_multiselect", # Sử dụng key mới nếu cần phân biệt, hoặc cập nhật session_state trực tiếp
                help="Kết hợp thêm tối đa 2 mô hình embedding khác. Tổng số mô hình dense trong hybrid sẽ là 1 (chính) + số lượng chọn ở đây."
            )
            if len(selected_additional_models) > 2:
                st.warning("Bạn chỉ có thể chọn tối đa 2 mô hình embedding phụ. Sẽ chỉ lấy 2 mô hình đầu tiên được chọn.")
                st.session_state.additional_hybrid_models = selected_additional_models[:2]
            else:
                st.session_state.additional_hybrid_models = selected_additional_models
        else:
            st.markdown("<p style='font-size:0.9em; font-style:italic;'>Không có mô hình embedding phụ nào khác để chọn.</p>", unsafe_allow_html=True)
            st.session_state.additional_hybrid_models = [] # Reset nếu không có lựa chọn
    else:
        # Nếu không phải hybrid, xóa lựa chọn model phụ
        st.session_state.additional_hybrid_models = []


    selected_model_llm = st.selectbox(
        "Chọn mô hình Gemini:",
        options=config.AVAILABLE_GEMINI_MODELS,
        index=config.AVAILABLE_GEMINI_MODELS.index(st.session_state.selected_gemini_model),
        key="selected_gemini_model",
        help="Chọn mô hình ngôn ngữ lớn để xử lý yêu cầu."
    )
    selected_reranker = st.selectbox(
        "Chọn mô hình Reranker:",
        options=config.AVAILABLE_RERANKER_MODELS,
        index=config.AVAILABLE_RERANKER_MODELS.index(st.session_state.selected_reranker_model),
        key="selected_reranker_model",
        help="Chọn mô hình để xếp hạng lại kết quả tìm kiếm. Chọn 'Không sử dụng' để tắt."
    )
    answer_mode_choice = st.radio(
        "Chọn chế độ trả lời:", options=['Ngắn gọn', 'Đầy đủ'], key="answer_mode", horizontal=True,
        help="Mức độ chi tiết của câu trả lời."
    )

    st.header("Cấu hình truy vấn")
    retrieval_query_mode_choice = st.radio(
        "Nguồn câu hỏi cho Retrieval:", options=['Đơn giản', 'Mở rộng', 'Đa dạng'], key="retrieval_query_mode", horizontal=True,
        help=(
            "**Đơn giản:** Chỉ dùng câu hỏi gốc.\n"
            "**Mở rộng:** Chỉ dùng câu hỏi mở rộng từ câu hỏi gốc (do AI tạo).\n"
            "**Đa dạng:** Dùng cả câu hỏi gốc và các biến thể từ câu hỏi gốc(do AI tạo)."
        )
    )
    # Cập nhật index cho radio retrieval_method
    current_retrieval_method_index = ['dense', 'sparse', 'hybrid'].index(st.session_state.retrieval_method)
    retrieval_method_choice = st.radio(
        "Phương thức Retrieval:", options=['dense', 'sparse', 'hybrid'], index=current_retrieval_method_index, key="retrieval_method", horizontal=True,
        help=(
            "**dense:** Tìm kiếm dựa trên vector ngữ nghĩa (nhanh, hiểu ngữ cảnh).\n"
            "**sparse:** Tìm kiếm dựa trên từ khóa (BM25) (nhanh, chính xác từ khóa).\n"
            "**hybrid:** Kết hợp cả dense và sparse, có thể bao gồm nhiều nguồn dense (cân bằng, có thể tốt nhất)."
        )
    )

    st.markdown("---") 
    st.header("Quản lý Hội thoại")
    if st.button("⚠️ Xóa Lịch Sử Chat"):
        st.session_state.messages = []
        st.success("Đã xóa lịch sử chat!")
        time.sleep(1); st.rerun()
    st.markdown("---")

# --- Giao diện chính của Ứng dụng ---
st.title("⚖️ Chatbot Hỏi Đáp Luật Giao Thông Đường Bộ VN")
st.caption(f"Dựa trên các văn bản Luật, Nghị Định, Thông tư về Luật giao thông đường bộ Việt Nam.")

# --- Cập nhật Caption hiển thị cấu hình ---
reranker_status_display = st.session_state.selected_reranker_model
if reranker_status_display == 'Không sử dụng': reranker_status_display = "Tắt"
else: reranker_status_display = reranker_status_display.split('/')[-1]

# Hiển thị các model phụ nếu có
additional_models_display_caption = "Không"
if st.session_state.retrieval_method == 'hybrid' and st.session_state.additional_hybrid_models:
    additional_names = [name.split('/')[-1] for name in st.session_state.additional_hybrid_models]
    additional_models_display_caption = ", ".join(additional_names) if additional_names else "Không"

st.caption(
    f"Embedding chính: `{st.session_state.selected_embedding_model.split('/')[-1]}` | "
    f"Embeddings phụ (Hybrid): `{additional_models_display_caption}` | "
    f"Mô hình LLM: `{st.session_state.selected_gemini_model}` | Trả lời: `{st.session_state.answer_mode}` | "
    f"Nguồn Query: `{st.session_state.retrieval_query_mode}` | Retrieval: `{st.session_state.retrieval_method}` | "
    f"Reranker: `{reranker_status_display}`"
)

# --- Hiển thị Lịch sử Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content_to_display = message["content"]
        if message["role"] == "assistant":
            docs_for_this_message = message.get("relevant_docs_for_display", [])
            content_to_display = utils.render_html_for_assistant_message(content_to_display, docs_for_this_message)
        st.markdown(content_to_display, unsafe_allow_html=True)

# --- Khởi tạo hệ thống ---
init_ok = False
reranking_model_loaded = None
embedding_model_object = None # Model chính
vector_db = None # VectorDB chính
retriever = None # Retriever chính, sẽ sử dụng model chính và các model phụ

with st.status("Đang khởi tạo hệ thống...", expanded=True) as status:
    # 1. Tải embedding model chính
    embedding_model_object = load_embedding_model(st.session_state.selected_embedding_model)
    
    # 2. Tải VectorDB và Retriever cho model chính
    if embedding_model_object:
        vector_db, retriever = cached_load_or_create_components(
            st.session_state.selected_embedding_model, 
            embedding_model_object
        )
    
    models_loaded_ok = all([embedding_model_object])
    retriever_ready_ok = all([vector_db, retriever])
    
    status_label = "✅ Hệ thống đã sẵn sàng!"
    status_state = "complete"
    init_ok = True

    if not models_loaded_ok: 
        status_label = f"⚠️ Lỗi tải Embedding model chính ({st.session_state.selected_embedding_model})!"
        status_state = "error"; init_ok = False
    elif not retriever_ready_ok:
        status_label = f"⚠️ Lỗi khởi tạo VectorDB/Retriever cho model chính!"
        status_state = "error"; init_ok = False

    # 3. Tải các embedding model phụ và VDB của chúng nếu cần (chỉ thông báo, không chặn init chính)
    # Các object này sẽ được load lại lúc xử lý query để đảm bảo tính nhất quán của cache
    if init_ok and st.session_state.retrieval_method == 'hybrid' and st.session_state.additional_hybrid_models:
        status.write("Đang kiểm tra các embedding model phụ...")
        for model_name_add_init in st.session_state.additional_hybrid_models:
            emb_obj_add_init = load_embedding_model(model_name_add_init)
            if emb_obj_add_init:
                vdb_add_init, _ = cached_load_or_create_components(model_name_add_init, emb_obj_add_init)
                if not vdb_add_init:
                    status.write(f"Lưu ý: Không thể tải VectorDB cho model phụ '{model_name_add_init.split('/')[-1]}' lúc khởi tạo.")
            else:
                status.write(f"Lưu ý: Không thể tải Embedding Model phụ '{model_name_add_init.split('/')[-1]}' lúc khởi tạo.")
        status.write("Kiểm tra model phụ hoàn tất.")


    if init_ok:
        status.update(label=status_label, state=status_state, expanded=False)
    else:
        status.update(label=status_label, state=status_state, expanded=True)


# --- Input và Xử lý ---
if init_ok and retriever: # Đảm bảo retriever chính đã được khởi tạo
    if user_query := st.chat_input("Nhập câu hỏi của bạn về Luật GTĐB..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            raw_llm_output = ""
            processing_log = []
            final_relevant_documents = [] 
            relevance_status = 'valid'

            try:
                start_time = time.time()
                processing_log.append(f"[{time.time() - start_time:.2f}s] Bắt đầu xử lý...")
                message_placeholder.markdown(" ".join(processing_log) + "⏳")

                # Tải reranker model
                if st.session_state.selected_reranker_model != 'Không sử dụng':
                    reranking_model_loaded = load_reranker_model(st.session_state.selected_reranker_model)
                else: reranking_model_loaded = None
                
                # Tải LLM (Gemini)
                selected_llm_name = st.session_state.selected_gemini_model
                selected_gemini_llm = load_gemini_model(selected_llm_name)
                if not selected_gemini_llm:
                     raise ValueError(f"Không thể tải model Gemini: {selected_llm_name}")
                processing_log.append(f"[{time.time() - start_time:.2f}s]: Model LLM '{selected_llm_name}' đã sẵn sàng.")
                message_placeholder.markdown(" ".join(processing_log) + "⏳")

                current_reranker_model_obj = reranking_model_loaded
                use_reranker_flag = current_reranker_model_obj is not None and st.session_state.selected_reranker_model != 'Không sử dụng'
                if use_reranker_flag:
                     processing_log.append(f"[{time.time() - start_time:.2f}s]: Reranker '{st.session_state.selected_reranker_model.split('/')[-1]}' đã sẵn sàng.")
                else:
                     processing_log.append(f"[{time.time() - start_time:.2f}s]: Reranker không được sử dụng.")
                message_placeholder.markdown(" ".join(processing_log) + "⏳")
                
                # --- Bước A: Phân loại relevancy và tạo biến thể/tóm tắt ---
                history_for_llm1 = st.session_state.messages[-(config.MAX_HISTORY_TURNS * 2):-1]
                log_hist_llm1 = "(có dùng lịch sử)" if history_for_llm1 else "(không dùng lịch sử)"
                processing_log.append(f"[{time.time() - start_time:.2f}s] Phân tích câu hỏi {log_hist_llm1}...")
                message_placeholder.markdown(" ".join(processing_log) + "⏳")
                
                relevance_status, direct_answer, all_queries, summarizing_q = utils.generate_query_variations(
                    original_query=user_query, gemini_model=selected_gemini_llm,
                    chat_history=history_for_llm1, num_variations=config.NUM_QUERY_VARIATIONS
                )
                
                if relevance_status == 'invalid':
                    full_response = direct_answer if direct_answer and direct_answer.strip() else "⚠️ Câu hỏi của bạn có vẻ không liên quan đến Luật Giao thông Đường bộ Việt Nam."
                    processing_log.append(f"[{time.time() - start_time:.2f}s] Hoàn tất (Câu hỏi không liên quan).")
                else: # Câu hỏi hợp lệ, tiếp tục RAG
                    recent_chat_history = st.session_state.messages[-(config.MAX_HISTORY_TURNS * 2):-1]
                    queries_to_search = []
                    query_source_log = ""
                    retrieval_query_mode = st.session_state.retrieval_query_mode
                    if retrieval_query_mode == 'Đơn giản': queries_to_search = [user_query]; query_source_log = "câu hỏi gốc"
                    elif retrieval_query_mode == 'Mở rộng': queries_to_search = [summarizing_q]; query_source_log = "câu hỏi tóm tắt/mở rộng"
                    elif retrieval_query_mode == 'Đa dạng': queries_to_search = all_queries; query_source_log = f"câu hỏi gốc và {len(all_queries)-1} biến thể"
                    
                    retrieval_method = st.session_state.retrieval_method
                    
                    # --- Chuẩn bị các nguồn dense phụ cho Retriever ---
                    additional_dense_sources_for_retriever_search = []
                    if retrieval_method == 'hybrid' and st.session_state.additional_hybrid_models:
                        processing_log.append(f"[{time.time() - start_time:.2f}s] Đang tải các nguồn embedding phụ cho hybrid search...")
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")
                        for model_name_add in st.session_state.additional_hybrid_models:
                            add_emb_model_obj = load_embedding_model(model_name_add) # Hàm này đã có @st.cache_resource
                            if add_emb_model_obj:
                                # Hàm này cũng đã có @st.cache_resource
                                add_vector_db, _ = cached_load_or_create_components(model_name_add, add_emb_model_obj)
                                if add_vector_db:
                                    additional_dense_sources_for_retriever_search.append((add_emb_model_obj, add_vector_db))
                                    processing_log.append(f"[{time.time() - start_time:.2f}s] Nguồn phụ '{model_name_add.split('/')[-1]}' đã được nạp.")
                                else:
                                    processing_log.append(f"[{time.time() - start_time:.2f}s] LƯU Ý: Không tải được VectorDB cho nguồn phụ '{model_name_add.split('/')[-1]}'. Bỏ qua.")
                            else:
                                processing_log.append(f"[{time.time() - start_time:.2f}s] LƯU Ý: Không tải được Embedding Model cho nguồn phụ '{model_name_add.split('/')[-1]}'. Bỏ qua.")
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")
                    
                    num_additional_sources = len(additional_dense_sources_for_retriever_search)
                    hybrid_info_log = ""
                    if retrieval_method == 'hybrid':
                        hybrid_info_log = f" (Chính: {st.session_state.selected_embedding_model.split('/')[-1]}"
                        if num_additional_sources > 0:
                            add_names = [name.split('/')[-1] for name in st.session_state.additional_hybrid_models[:num_additional_sources]] # Chỉ log các model đã load thành công
                            hybrid_info_log += f", Phụ: {', '.join(add_names)}"
                        hybrid_info_log += ")"
                    
                    processing_log.append(f"[{time.time() - start_time:.2f}s]: Bắt đầu Retrieval (Nguồn Query: {query_source_log}, Phương thức: {retrieval_method}{hybrid_info_log})...")
                    message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    collected_docs_data = {} 
                    retrieval_start_time = time.time()
                    
                    for q_idx, current_query_variant in enumerate(queries_to_search):
                        if not current_query_variant: continue
                        # Gọi phương thức search của retriever chính
                        # embedding_model_object là model chính đã được tải ở phần init
                        search_results_variant = retriever.search(
                            current_query_variant,
                            embedding_model_object, # Model object cho VDB chính của retriever
                            method=retrieval_method,
                            k=config.VECTOR_K_PER_QUERY, 
                            additional_dense_sources=additional_dense_sources_for_retriever_search
                        )
                        for item_res in search_results_variant:
                            doc_index = item_res['index']
                            if doc_index not in collected_docs_data:
                                collected_docs_data[doc_index] = item_res
                            else: # Nếu trùng, cập nhật score nếu score mới tốt hơn
                                # Score RRF/BM25: cao hơn tốt hơn. Score L2 distance (dense thuần túy): thấp hơn tốt hơn.
                                if (retrieval_method == 'hybrid' or retrieval_method == 'sparse') and item_res['score'] > collected_docs_data[doc_index]['score']:
                                    collected_docs_data[doc_index] = item_res
                                elif retrieval_method == 'dense' and item_res['score'] < collected_docs_data[doc_index]['score']:
                                     collected_docs_data[doc_index] = item_res


                    retrieval_time = time.time() - retrieval_start_time
                    num_unique_docs = len(collected_docs_data)
                    processing_log.append(f"[{time.time() - start_time:.2f}s]: Retrieval ({retrieval_time:.2f}s) tìm thấy {num_unique_docs} tài liệu ứng viên.")
                    message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    retrieved_docs_list = list(collected_docs_data.values())
                    # Sắp xếp dựa trên phương thức retrieval cuối cùng đã dùng
                    # Hybrid và Sparse: score cao tốt hơn. Dense: score thấp tốt hơn (distance).
                    sort_reverse_final = (retrieval_method == 'hybrid' or retrieval_method == 'sparse')
                    retrieved_docs_list.sort(key=lambda x: x.get('score', 0 if sort_reverse_final else float('inf')), reverse=sort_reverse_final)

                    final_relevant_documents = []
                    rerank_time_val = 0.0
                    
                    if use_reranker_flag and num_unique_docs > 0:
                        rerank_start_time_val = time.time()
                        query_for_reranking_step = summarizing_q if summarizing_q else user_query
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Xếp hạng lại {min(num_unique_docs, config.MAX_DOCS_FOR_RERANK)} tài liệu bằng '{st.session_state.selected_reranker_model.split('/')[-1]}'...")
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")
                        
                        docs_to_rerank_input = retrieved_docs_list[:config.MAX_DOCS_FOR_RERANK]
                        # rerank_documents_with_indices yêu cầu input dạng [{'doc': ..., 'index': ...}]
                        rerank_input_formatted = [{'doc': item_rr['doc'], 'index': item_rr['index']} for item_rr in docs_to_rerank_input]

                        reranked_results = rerank_documents(
                            query_for_reranking_step,
                            rerank_input_formatted, 
                            current_reranker_model_obj 
                        )
                        final_relevant_documents = reranked_results[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                        rerank_time_val = time.time() - rerank_start_time_val
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Rerank ({rerank_time_val:.2f}s) hoàn tất, chọn top {len(final_relevant_documents)}.")
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")
                    elif num_unique_docs > 0:
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Bỏ qua Rerank, lấy trực tiếp top {config.FINAL_NUM_RESULTS_AFTER_RERANK} kết quả Retrieval.")
                        final_relevant_documents = retrieved_docs_list[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")
                    else:
                         processing_log.append(f"[{time.time() - start_time:.2f}s]: Không tìm thấy tài liệu liên quan từ Retrieval.")
                         message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    answer_mode = st.session_state.answer_mode
                    processing_log.append(f"[{time.time() - start_time:.2f}s]: Tổng hợp câu trả lời (chế độ: {answer_mode})...")
                    message_placeholder.markdown(" ".join(processing_log)) # Không có icon ở đây
                    
                    raw_llm_output = generate_answer_with_gemini(
                        query_text=user_query, relevant_documents=final_relevant_documents, 
                        gemini_model=selected_gemini_llm, mode=answer_mode,
                        chat_history=recent_chat_history
                    )
                    processing_log.append(f"[{time.time() - start_time:.2f}s]: Hoàn tất!")

                with st.expander("Xem chi tiết quá trình xử lý", expanded=False):
                    st.markdown(f"```text\n{'\n'.join(processing_log)}\n```")
                
                if raw_llm_output:
                    content_for_immediate_display = utils.render_html_for_assistant_message(raw_llm_output, final_relevant_documents)
                    message_placeholder.markdown(content_for_immediate_display, unsafe_allow_html=True)
                    full_response = raw_llm_output
                else: # Trường hợp full_response đã được gán (ví dụ câu hỏi không liên quan)
                    message_placeholder.markdown(full_response, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"🐞 Đã xảy ra lỗi: {e}") 
                full_response = f"🐞 Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại hoặc thay đổi cấu hình."
                if message_placeholder: message_placeholder.markdown(full_response, unsafe_allow_html=True)
                else: st.markdown(full_response, unsafe_allow_html=True) # Fallback
            finally:
                if user_query and full_response:
                    utils.log_qa_to_json(user_query, full_response)
                
                if full_response: 
                    assistant_message = {"role": "assistant", "content": full_response}
                    # Chỉ thêm relevant_docs nếu câu hỏi hợp lệ và có tài liệu
                    if relevance_status != 'invalid' and 'final_relevant_documents' in locals() and final_relevant_documents:
                        assistant_message["relevant_docs_for_display"] = final_relevant_documents
                    else: # Bao gồm cả trường hợp invalid hoặc không có docs
                        assistant_message["relevant_docs_for_display"] = []
                    st.session_state.messages.append(assistant_message)

elif not init_ok:
    st.error("⚠️ Hệ thống chưa thể khởi động do lỗi tải mô hình chính hoặc dữ liệu VectorDB/Retriever chính. Vui lòng kiểm tra lại.")
else: # init_ok nhưng retriever chưa sẵn sàng (trường hợp hiếm)
    st.error("⚠️ Retriever chính chưa sẵn sàng. Vui lòng kiểm tra cấu hình và thử làm mới trang.")