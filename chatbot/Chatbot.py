# app.py
import streamlit as st
import time
import config
import utils 
from model_loader import load_embedding_model, load_gemini_model, load_reranker_model
from data_loader import load_or_create_rag_components
from reranker import rerank_documents
from generation import generate_answer_with_gemini

# @st.cache_resource
def cached_load_or_create_components(_embedding_model):
    vector_db, hybrid_retriever = load_or_create_rag_components(_embedding_model)
    return vector_db, hybrid_retriever

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(page_title="Chatbot Luật GTĐB", layout="wide", initial_sidebar_state="auto")

# --- Khởi tạo Session State cho Lịch sử Chat và Cấu hình ---

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn, tôi là chatbot Luật Giao thông Đường bộ. Bạn cần hỗ trợ gì?"}]

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

    st.header("Mô hình & Trả lời")

    selected_model = st.selectbox(
        "Chọn mô hình Gemini:",
        options=config.AVAILABLE_GEMINI_MODELS,
        index=config.AVAILABLE_GEMINI_MODELS.index(st.session_state.selected_gemini_model),
        key="selected_gemini_model",
        help="Chọn mô hình ngôn ngữ lớn để xử lý yêu cầu."
    )

    answer_mode_choice = st.radio(
        "Chọn chế độ trả lời:",
        options=['Ngắn gọn', 'Đầy đủ'],
        key="answer_mode",
        horizontal=True,
        help="Mức độ chi tiết của câu trả lời."
    )

    st.header("Cấu hình truy vấn")

    retrieval_query_mode_choice = st.radio(
        "Nguồn câu hỏi cho Retrieval:",
        options=['Đơn giản', 'Mở rộng', 'Đa dạng'],
        key="retrieval_query_mode", 
        horizontal=True,
        help=(
            "**Đơn giản:** Chỉ dùng câu hỏi gốc.\n"
            "**Mở rộng:** Chỉ dùng câu hỏi mở rộng từ câu hỏi gốc (do AI tạo).\n"
            "**Đa dạng:** Dùng cả câu hỏi gốc và các biến thể từ câu hỏi gốc(do AI tạo)."
        )
    )

    retrieval_method_choice = st.radio(
        "Phương thức Retrieval:",
        options=['dense', 'sparse', 'hybrid'],
        index=['dense', 'sparse', 'hybrid'].index(st.session_state.retrieval_method), 
        key="retrieval_method",
        horizontal=True,
        help=(
            "**dense:** Tìm kiếm dựa trên vector ngữ nghĩa (nhanh, hiểu ngữ cảnh).\n"
            "**sparse:** Tìm kiếm dựa trên từ khóa (BM25) (nhanh, chính xác từ khóa).\n"
            "**hybrid:** Kết hợp cả dense và sparse (cân bằng, có thể tốt nhất)."
        )
    )

    selected_reranker = st.selectbox(
        "Chọn mô hình Reranker:",
        options=config.AVAILABLE_RERANKER_MODELS,
        index=config.AVAILABLE_RERANKER_MODELS.index(st.session_state.selected_reranker_model),
        key="selected_reranker_model",
        help="Chọn mô hình để xếp hạng lại kết quả tìm kiếm. Chọn 'Không sử dụng' để tắt."
    )

    st.markdown("---") 

    st.header("Quản lý Hội thoại")
    if st.button("⚠️ Xóa Lịch Sử Chat"):
        st.session_state.messages = []
        st.success("Đã xóa lịch sử chat!")
        time.sleep(1)
        st.rerun()
    st.markdown("---")

# --- Giao diện chính của Ứng dụng ---
st.title("⚖️ Chatbot Hỏi Đáp Luật Giao Thông Đường Bộ VN")
st.caption(f"Dựa trên các văn bản Luật, Nghị Định, Thông tư về Luật giao thông đường bộ Việt Nam.")

# --- Cập nhật Caption hiển thị cấu hình ---
reranker_status_display = st.session_state.selected_reranker_model
if reranker_status_display == 'Không sử dụng':
    reranker_status_display = "Tắt"
else:
    # Lấy tên model ngắn gọn hơn để hiển thị nếu cần
    reranker_status_display = reranker_status_display.split('/')[-1]
st.caption(f"Mô hình: `{st.session_state.selected_gemini_model}` | Trả lời: `{st.session_state.answer_mode}` | Nguồn Query: `{st.session_state.retrieval_query_mode}` | Retrieval: `{st.session_state.retrieval_method}` | Reranker: `{reranker_status_display}`")

# --- Hiển thị Lịch sử Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content_to_display = message["content"]
        if message["role"] == "assistant":
            # Lấy relevant_docs từ message, nếu không có thì dùng list rỗng
            docs_for_this_message = message.get("relevant_docs_for_display", [])
            content_to_display = utils.render_html_for_assistant_message(content_to_display, docs_for_this_message)
        st.markdown(content_to_display, unsafe_allow_html=True)

# --- Khởi tạo hệ thống ---
init_ok = False
reranking_model_loaded = None

with st.status("Đang khởi tạo hệ thống...", expanded=True) as status:
    embedding_model = load_embedding_model(config.embedding_model_name)
    if st.session_state.selected_reranker_model != 'Không sử dụng':
        reranking_model_loaded = load_reranker_model(st.session_state.selected_reranker_model)
    else:
        reranking_model_loaded = None
    models_loaded = all([embedding_model])
    vector_db, hybrid_retriever = cached_load_or_create_components(embedding_model)
    retriever_ready = hybrid_retriever is not None

    status_label = "✅ Hệ thống đã sẵn sàng!"
    status_state = "complete"
    init_ok = True

    if not models_loaded: 
        status_label = "⚠️ Lỗi tải Embedding model!"
        status_state = "error"
        init_ok = False
    elif not retriever_ready:
        status_label = "⚠️ Lỗi khởi tạo VectorDB hoặc Retriever!"
        status_state = "error"
        init_ok = False
    elif st.session_state.selected_reranker_model != 'Không sử dụng' and not reranking_model_loaded:
        status_label = f"⚠️ Lỗi tải Reranker model ({st.session_state.selected_reranker_model}). Reranking sẽ bị tắt."
        status_state = "warning" # Có thể coi là warning, hệ thống vẫn chạy được nhưng không có rerank
        # init_ok vẫn có thể là True, nhưng reranking_model_loaded sẽ là None
        init_ok = False

    if init_ok:
        status.update(label=status_label, state=status_state, expanded=False)
    else:
        status.update(label=status_label, state=status_state, expanded=True)

# --- Input và Xử lý ---
if init_ok:
    if user_query := st.chat_input("Nhập câu hỏi của bạn về Luật GTĐB..."):
        # 1. Thêm và hiển thị tin nhắn của người dùng
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Xử lý và tạo phản hồi từ bot
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

                # --- Tải model Gemini đã chọn ---
                selected_model_name = st.session_state.selected_gemini_model
                selected_gemini_llm = load_gemini_model(selected_model_name)
                if not selected_gemini_llm:
                     raise ValueError(f"Không thể tải model Gemini: {selected_model_name}")
                processing_log.append(f"[{time.time() - start_time:.2f}s]: Model '{selected_model_name}' đã sẵn sàng.")
                message_placeholder.markdown(" ".join(processing_log) + "⏳")

                # Lấy reranker model đã tải (có thể là None)
                current_reranker_model = reranking_model_loaded
                use_reranker_flag = current_reranker_model is not None and st.session_state.selected_reranker_model != 'Không sử dụng'

                if use_reranker_flag:
                     processing_log.append(f"[{time.time() - start_time:.2f}s]: Reranker model '{st.session_state.selected_reranker_model}' đã sẵn sàng.")
                else:
                     processing_log.append(f"[{time.time() - start_time:.2f}s]: Reranker không được sử dụng hoặc không tải được.")
                message_placeholder.markdown(" ".join(processing_log) + "⏳")

                history_for_llm1 = st.session_state.messages[-(config.MAX_HISTORY_TURNS * 2):-1]
                log_hist_llm1 = "(có dùng lịch sử)"
                processing_log.append(f"[{time.time() - start_time:.2f}s] Phân tích câu hỏi {log_hist_llm1}...")
                message_placeholder.markdown(" ".join(processing_log) + "⏳")

                # --- Bước A: Phân loại relevancy và tạo biến thể/tóm tắt ---
                relevance_status, direct_answer, all_queries, summarizing_q = utils.generate_query_variations(
                    original_query=user_query,
                    gemini_model=selected_gemini_llm,
                    chat_history=history_for_llm1,
                    num_variations=config.NUM_QUERY_VARIATIONS
                )
                st.write(summarizing_q)
                # --- Kiểm tra mức độ liên quan ---
                if relevance_status == 'invalid':
                    if direct_answer and direct_answer.strip():
                        full_response = direct_answer
                    else:
                        full_response = "⚠️ Câu hỏi của bạn có vẻ không liên quan đến Luật Giao thông Đường bộ Việt Nam."
                    processing_log.append(f"[{time.time() - start_time:.2f}s] Hoàn tất (Câu hỏi không liên quan).")

                # --- Nếu câu hỏi hợp lệ, tiếp tục xử lý RAG ---
                else:
                    # --- Lấy lịch sử gần đây cho LLM thứ 2 (tạo câu trả lời) ---
                    recent_chat_history = st.session_state.messages[-(config.MAX_HISTORY_TURNS * 2):-1]

                    # --- Xác định query(s) để tìm kiếm dựa trên retrieval_query_mode ---
                    queries_to_search = []
                    query_source_log = ""
                    retrieval_query_mode = st.session_state.retrieval_query_mode
                    if retrieval_query_mode == 'Đơn giản':
                        queries_to_search = [user_query]
                        query_source_log = "câu hỏi gốc"
                    elif retrieval_query_mode == 'Mở rộng':
                        queries_to_search = [summarizing_q]
                        query_source_log = "câu hỏi tóm tắt"
                    elif retrieval_query_mode == 'Đa dạng':
                        queries_to_search = all_queries # all_queries đã bao gồm user_query
                        query_source_log = f"câu hỏi gốc và {len(all_queries)-1} biến thể"

                    # --- Lấy cấu hình retrieval và rerank ---
                    retrieval_method = st.session_state.retrieval_method

                    processing_log.append(f"[{time.time() - start_time:.2f}s]: Bắt đầu Retrieval (Nguồn: {query_source_log}, Phương thức: {retrieval_method})...")
                    message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    # --- Thực hiện Retrieval ---
                    collected_docs_data = {} # Dict lưu kết quả {index: {'doc': ..., 'score': ...}}
                    retrieval_start_time = time.time()
        
                    for q_idx, current_query in enumerate(queries_to_search):
                        # Gọi phương thức search mới của retriever
                        search_results = hybrid_retriever.search(
                            current_query,
                            embedding_model,
                            method=retrieval_method,
                            k=config.VECTOR_K_PER_QUERY # Lấy nhiều hơn để có đủ cho rerank/fusion
                        )
                        # Tổng hợp kết quả, tránh trùng lặp index
                        for item in search_results:
                            doc_index = item['index']
                            if doc_index not in collected_docs_data:
                                collected_docs_data[doc_index] = item # Lưu cả score từ retrieval
                            # Optional: Nếu muốn cập nhật score (ví dụ: lấy score cao nhất nếu trùng) - phức tạp hơn
                    retrieval_time = time.time() - retrieval_start_time
                    num_unique_docs = len(collected_docs_data)
                    processing_log.append(f"[{time.time() - start_time:.2f}s]: Retrieval ({retrieval_time:.2f}s) tìm thấy {num_unique_docs} tài liệu ứng viên.")
                    message_placeholder.markdown(" ".join(processing_log) + "⏳")


                    # --- Chuẩn bị dữ liệu cho bước tiếp theo (Rerank hoặc lấy trực tiếp) ---
                    # Chuyển dict thành list và sắp xếp theo score (cao xuống thấp cho sparse/hybrid, thấp lên cao cho dense)
                    retrieved_docs_list = list(collected_docs_data.values())
                    sort_reverse = (retrieval_method != 'dense') # Dense sắp xếp ngược lại
                    retrieved_docs_list.sort(key=lambda x: x.get('score', 0 if sort_reverse else float('inf')), reverse=sort_reverse)

                    # --- Bước Rerank (Nếu được bật) ---
                    final_relevant_documents = [] # List các dict {'doc': ..., 'score': ..., 'original_index': ...}
                    rerank_time = 0.0
                    rerank_start_time = time.time()

                    if use_reranker_flag and num_unique_docs > 0:
                        # Lấy query phù hợp để rerank (thường là câu tóm tắt hoặc câu gốc)
                        query_for_reranking = summarizing_q if summarizing_q else user_query
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Xếp hạng lại {min(num_unique_docs, config.MAX_DOCS_FOR_RERANK)} tài liệu bằng '{st.session_state.selected_reranker_model}'...")
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")

                        # Chọn top N docs để rerank
                        docs_to_rerank = retrieved_docs_list[:config.MAX_DOCS_FOR_RERANK]
                        rerank_input = [{'doc': item['doc'], 'index': item['index']} for item in docs_to_rerank]
                        reranked_results = rerank_documents(
                            query_for_reranking,
                            rerank_input, # Đảm bảo đúng định dạng đầu vào
                            current_reranker_model
                        )
                        # Lấy top K kết quả cuối cùng sau rerank
                        final_relevant_documents = reranked_results[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                        rerank_time = time.time() - rerank_start_time
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Rerank ({rerank_time:.2f}s) hoàn tất, chọn top {len(final_relevant_documents)}.")
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    elif num_unique_docs > 0: # Không dùng reranker nhưng có kết quả retrieval
                        processing_log.append(f"[{time.time() - start_time:.2f}s]: Bỏ qua Rerank, lấy trực tiếp top {config.FINAL_NUM_RESULTS_AFTER_RERANK} kết quả Retrieval.")
                        final_relevant_documents = retrieved_docs_list[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                        message_placeholder.markdown(" ".join(processing_log) + "⏳")
                    else: # Không có kết quả retrieval
                         processing_log.append(f"[{time.time() - start_time:.2f}s]: Không tìm thấy tài liệu liên quan.")
                         message_placeholder.markdown(" ".join(processing_log) + "⏳")

                    # --- Bước Generate Answer ---
                    answer_mode = st.session_state.answer_mode
                    processing_log.append(f"[{time.time() - start_time:.2f}s]: Tổng hợp câu trả lời (chế độ: {answer_mode})...")
                    message_placeholder.markdown(" ".join(processing_log))
                    raw_llm_output = generate_answer_with_gemini(
                        query_text=user_query,
                        relevant_documents=final_relevant_documents, 
                        gemini_model=selected_gemini_llm,
                        mode=answer_mode,
                        chat_history=recent_chat_history
                    )
                    processing_log.append(f"[{time.time() - start_time:.2f}s]: Hoàn tất!")

                # Hiển thị log xử lý
                with st.expander("Xem chi tiết quá trình xử lý", expanded=False):
                    log_content = "\n".join(processing_log)
                    st.markdown(f"```text\n{log_content}\n```")
                # Hiển thị câu trả lời cuối cùng
                if raw_llm_output:
                    content_for_immediate_display = utils.render_html_for_assistant_message(raw_llm_output, final_relevant_documents)
                    message_placeholder.markdown(content_for_immediate_display, unsafe_allow_html=True)
                    full_response = raw_llm_output
                else:
                    message_placeholder.markdown(full_response, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"🐞 Đã xảy ra lỗi: {e}") 
                full_response = f"🐞 Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại hoặc thay đổi cấu hình."
                if message_placeholder:
                    message_placeholder.markdown(full_response, unsafe_allow_html=True)
                else:
                    st.markdown(full_response) 
            finally:
                if user_query and full_response:
                    utils.log_qa_to_json(user_query, full_response)
                # Đảm bảo tin nhắn của assistant luôn được thêm vào history
                if full_response: 
                    assistant_message = {"role": "assistant", "content": full_response}
                    if relevance_status != 'invalid' and 'final_relevant_documents' in locals() and final_relevant_documents:
                        assistant_message["relevant_docs_for_display"] = final_relevant_documents
                    elif relevance_status == 'invalid': 
                        assistant_message["relevant_docs_for_display"] = [] 
                    else: 
                        assistant_message["relevant_docs_for_display"] = []

                    st.session_state.messages.append(assistant_message)

elif not init_ok:
    st.error("⚠️ Hệ thống chưa thể khởi động do lỗi tải mô hình hoặc dữ liệu. Vui lòng kiểm tra lại.")