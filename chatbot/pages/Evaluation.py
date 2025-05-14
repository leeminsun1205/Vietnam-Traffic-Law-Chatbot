# pages/2_Evaluation.py
import time
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import sys
import numpy as np # Import numpy for isinstance checks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import utils
import data_loader
from retriever import HybridRetriever
from utils import precision_at_k, recall_at_k, f1_at_k, mrr_at_k, ndcg_at_k, calculate_average_metrics

def run_retrieval_evaluation(
    eval_data: list,
    hybrid_retriever: HybridRetriever,
    embedding_model,
    reranking_model,
    gemini_model,
    eval_config: dict,
    progress_bar_placeholder, # Truyền placeholder vào hàm
    status_text_placeholder # Truyền placeholder vào hàm
    ):

    results_list = []
    k_values = [3, 5, 10]

    # --- Lấy cấu hình từ eval_config ---
    retrieval_query_mode = eval_config.get('retrieval_query_mode', 'Tổng quát')
    retrieval_method = eval_config.get('retrieval_method', 'hybrid')
    use_reranker = eval_config.get('use_reranker', True)

    # Tạo thanh tiến trình và text status bên trong placeholder được truyền vào
    progress_bar = progress_bar_placeholder.progress(0)
    status_text = status_text_placeholder.empty()


    total_items = len(eval_data)
    queries_per_batch = 15 # Giới hạn số lượng query trước khi tạm dừng
    wait_time_seconds = 60 # Thời gian tạm dừng


    # Đặt cờ hủy bỏ về False khi bắt đầu chạy (đảm bảo reset sau khi rerun)
    st.session_state.cancel_eval_requested = False


    for i, item in enumerate(eval_data):
        # --- KIỂM TRA YÊU CẦU HỦY BỎ ---
        if st.session_state.cancel_eval_requested:
            status_text.warning(f"Đã hủy bỏ quá trình đánh giá tại query {i}/{total_items}.")
            break # Thoát khỏi vòng lặp chính

        # Tạm dừng sau mỗi batch
        if i > 0 and i % queries_per_batch == 0:
            pause_msg = f"Đã xử lý {i}/{total_items} queries. Tạm dừng {wait_time_seconds} giây..."
            status_text.text(pause_msg)
            time.sleep(wait_time_seconds)
            # Kiểm tra lại yêu cầu hủy bỏ sau khi tạm dừng
            if st.session_state.cancel_eval_requested:
                status_text.warning(f"Đã hủy bỏ quá trình đánh giá tại query {i}/{total_items}.")
                break # Thoát khỏi vòng lặp chính

            status_text.text(f"Tiếp tục xử lý query {i+1}/{total_items}...")


        query_id = item.get("query_id"); original_query = item.get("query")
        relevant_chunk_ids = set(item.get("relevant_chunk_ids", []))

        status_text.text(f"Đang xử lý query {i+1}/{total_items}: {query_id} (QueryMode: {retrieval_query_mode}, Method: {retrieval_method}, Rerank: {'Bật' if use_reranker else 'Tắt'})")


        start_time = time.time()
        # --- Khởi tạo query_metrics với các trường cấu hình ---
        query_metrics = {
            "query_id": query_id, "query": original_query,
            "retrieval_query_mode": retrieval_query_mode,
            "retrieval_method": retrieval_method,
            "use_reranker": use_reranker,
            "status": "error", "retrieved_ids": [], "relevant_ids": list(relevant_chunk_ids),
            "processing_time": 0.0, 'summarizing_query': '',
            'variation_time': 0.0, 'search_time': 0.0, 'rerank_time': 0.0,
            'num_variations_generated': 0, 'num_unique_docs_found': 0, 'num_docs_reranked': 0,
            'num_retrieved_before_rerank': 0, 'num_retrieved_after_rerank': 0
        }
        # Vòng lặp khởi tạo metrics, tự động dùng k_values mới
        for k in k_values:
            query_metrics[f'precision@{k}'] = 0.0; query_metrics[f'recall@{k}'] = 0.0
            query_metrics[f'f1@{k}'] = 0.0; query_metrics[f'mrr@{k}'] = 0.0; query_metrics[f'ndcg@{k}'] = 0.0

        try:
            # Bước 1: Tạo variations/summarizing query (luôn chạy)
            variation_start = time.time()
            relevance_status, _, all_queries, summarizing_query = utils.generate_query_variations(
                original_query=original_query, gemini_model=gemini_model,
                chat_history=None, # Không dùng lịch sử chat cho đánh giá retrieval
                num_variations=config.NUM_QUERY_VARIATIONS
            )
            query_metrics["variation_time"] = time.time() - variation_start
            query_metrics["summarizing_query"] = summarizing_query
            query_metrics["num_variations_generated"] = len(all_queries) - 1 # Trừ query gốc

            if relevance_status == 'invalid':
                query_metrics["status"] = "skipped_irrelevant"
                query_metrics["processing_time"] = time.time() - start_time
                results_list.append(query_metrics)
                progress_bar.progress((i + 1) / total_items)
                continue

            # --- Bước 2: Xác định query(s) để tìm kiếm ---
            queries_to_search = []
            if retrieval_query_mode == 'Đơn giản': queries_to_search = [original_query]
            elif retrieval_query_mode == 'Tổng quát': queries_to_search = [summarizing_query]
            elif retrieval_query_mode == 'Sâu': queries_to_search = all_queries

            # --- Bước 3: Thực hiện Retrieval ---
            collected_docs_data = {}
            search_start = time.time()
            if hybrid_retriever: # Chỉ search nếu retriever khả dụng
                 for q_variant in queries_to_search:
                    if not q_variant: continue
                    search_results = hybrid_retriever.search(
                        q_variant, embedding_model,
                        method=retrieval_method,
                        k=config.VECTOR_K_PER_QUERY
                    )
                    for item in search_results:
                        doc_index = item.get('index')
                        # Đảm bảo doc_index là số nguyên hợp lệ trước khi thêm vào dict
                        if isinstance(doc_index, int) and doc_index >= 0 and doc_index not in collected_docs_data:
                            collected_docs_data[doc_index] = item
            query_metrics["search_time"] = time.time() - search_start
            query_metrics["num_unique_docs_found"] = len(collected_docs_data)

            # --- Chuẩn bị danh sách kết quả retrieval ---
            retrieved_docs_list = list(collected_docs_data.values())
            sort_reverse = (retrieval_method != 'dense') # Dense sort theo distance (nhỏ tốt hơn), Sparse/Hybrid sort theo score (lớn tốt hơn)
            retrieved_docs_list.sort(key=lambda x: x.get('score', 0 if sort_reverse else float('inf')), reverse=sort_reverse)
            query_metrics["num_retrieved_before_rerank"] = len(retrieved_docs_list)


            # --- Bước 4: Re-ranking (Nếu bật và model khả dụng) ---
            final_docs_for_metrics = []
            rerank_time = 0.0 # Khởi tạo rerank_time
            rerank_start = time.time()

            # Chỉ thực hiện reranking nếu use_reranker BẬT VÀ reranking_model TỒN TẠI VÀ có kết quả retrieval
            if use_reranker and reranking_model and retrieved_docs_list:
                query_for_reranking = summarizing_query if summarizing_query else original_query # Sử dụng câu hỏi tóm tắt hoặc gốc cho rerank
                docs_to_rerank = retrieved_docs_list[:config.MAX_DOCS_FOR_RERANK] # Chỉ rerank top N

                query_metrics["num_docs_reranked"] = len(docs_to_rerank)

                # Chuyển đổi định dạng đầu vào cho rerank_documents
                rerank_input = [{'doc': item.get('doc', {}), 'index': item.get('index')} for item in docs_to_rerank if 'doc' in item]

                if rerank_input: # Chỉ gọi rerank nếu có input
                    reranked_results = utils.rerank_documents(
                        query_for_reranking, rerank_input, reranking_model
                    )
                    # Lấy top K kết quả cuối cùng sau rerank
                    final_docs_for_metrics = reranked_results[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                    rerank_time = time.time() - rerank_start
                else:
                     # Không có docs để rerank
                     rerank_time = 0.0
                     query_metrics["num_docs_reranked"] = 0

            elif retrieved_docs_list: # Không dùng reranker HOẶC reranker model không tải được, nhưng có kết quả retrieval
                final_docs_for_metrics = retrieved_docs_list[:config.FINAL_NUM_RESULTS_AFTER_RERANK] # Lấy top K trực tiếp từ retrieval
                rerank_time = 0.0
                query_metrics["num_docs_reranked"] = 0
            else: # Không có kết quả retrieval nào
                 rerank_time = 0.0
                 query_metrics["num_docs_reranked"] = 0

            query_metrics["rerank_time"] = rerank_time # Lưu thời gian rerank đã tính
            query_metrics["num_retrieved_after_rerank"] = len(final_docs_for_metrics)


            # --- Bước 5: Lấy IDs và Tính Metrics ---
            retrieved_ids = []
            for res in final_docs_for_metrics:
                doc = res.get('doc', {}); original_index = res.get('original_index', res.get('index')) # Lấy original_index từ kết quả rerank nếu có, fallback về index ban đầu
                chunk_id = None

                # Cố gắng lấy chunk_id từ doc data
                if isinstance(doc, dict):
                     # Ưu tiên lấy từ metadata trước
                    metadata = doc.get('metadata', {})
                    if isinstance(metadata, dict):
                        chunk_id = metadata.get('chunk_id') or metadata.get('id')
                     # Nếu không có trong metadata, thử lấy trực tiếp từ doc
                    if not chunk_id:
                        chunk_id = doc.get('id')

                # Fallback lấy từ original_index nếu chunk_id không tìm thấy
                # Điều này yêu cầu truy cập lại dữ liệu gốc từ retriever instance
                if not chunk_id and isinstance(original_index, (int, np.integer)):
                    if hybrid_retriever and hasattr(hybrid_retriever, 'documents') and isinstance(hybrid_retriever.documents, list) and 0 <= original_index < len(hybrid_retriever.documents):
                        doc_from_retriever = hybrid_retriever.documents[original_index]
                        if isinstance(doc_from_retriever, dict):
                            metadata_from_retriever = doc_from_retriever.get('metadata', {})
                            if isinstance(metadata_from_retriever, dict):
                                chunk_id = metadata_from_retriever.get('chunk_id') or metadata_from_retriever.get('id')
                            if not chunk_id:
                                chunk_id = doc_from_retriever.get('id')


                if chunk_id is not None: # Kiểm tra None thay vì chỉ True/False
                    retrieved_ids.append(str(chunk_id)) # Đảm bảo là string


            query_metrics["retrieved_ids"] = retrieved_ids

            query_metrics["status"] = "evaluated"
            # Vòng lặp tính metrics, tự động dùng k_values mới
            for k in k_values:
                query_metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'f1@{k}'] = f1_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'mrr@{k}'] = mrr_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved_ids, relevant_chunk_ids, k)


        except Exception as e:
            query_metrics["status"] = "error_runtime"
            query_metrics["error_message"] = str(e)
        finally:
            query_metrics["processing_time"] = time.time() - start_time
            results_list.append(query_metrics)
            progress_bar.progress((i + 1) / total_items)

    # Cập nhật thanh tiến trình và trạng thái sau khi hoàn thành hoặc hủy bỏ
    if st.session_state.cancel_eval_requested:
         progress_bar.progress((i + 1) / total_items) # Hiển thị tiến độ tại thời điểm hủy
         # Trạng thái hủy bỏ đã được hiển thị bên trong vòng lặp
    else:
        status_text.text(f"Hoàn thành đánh giá {total_items} queries!")


    return pd.DataFrame(results_list)

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Đánh giá Retrieval", layout="wide")
st.title("📊 Đánh giá Hệ thống Retrieval")

st.markdown("""
Trang này cho phép bạn chạy đánh giá hiệu suất của hệ thống retrieval và reranking
dựa trên một tập dữ liệu câu hỏi và các chunk tài liệu liên quan (ground truth).
Sử dụng cấu hình **hiện tại được chọn trên sidebar của trang này**.
""")

# --- sidebar ---
with st.sidebar:
    st.title("Tùy chọn Đánh giá")

    # --- Khởi tạo hoặc kiểm tra tất cả các biến Session State cần thiết ---
    # Đảm bảo tất cả các key được sử dụng đều có giá trị mặc định nếu chưa tồn tại
    if "selected_gemini_model" not in st.session_state:
        st.session_state.selected_gemini_model = config.DEFAULT_GEMINI_MODEL
    if "retrieval_query_mode" not in st.session_state:
        st.session_state.retrieval_query_mode = 'Tổng quát'
    if "retrieval_method" not in st.session_state:
        st.session_state.retrieval_method = 'hybrid'
    if "use_reranker" not in st.session_state:
        st.session_state.use_reranker = True

    # State cho dữ liệu và kết quả đánh giá
    if 'eval_data' not in st.session_state:
        st.session_state.eval_data = None
    if 'eval_results_df' not in st.session_state:
        st.session_state.eval_results_df = None
    if 'eval_run_completed' not in st.session_state:
        st.session_state.eval_run_completed = False
    if 'eval_uploaded_filename' not in st.session_state:
        st.session_state.eval_uploaded_filename = "" # Sử dụng chuỗi rỗng cho trạng thái ban đầu
    if 'last_eval_config' not in st.session_state:
        st.session_state.last_eval_config = {}

    # State cho tiến trình và hủy bỏ
    if 'cancel_eval_requested' not in st.session_state:
        st.session_state.cancel_eval_requested = False
    if 'status_message' not in st.session_state:
         st.session_state.status_message = "Chưa sẵn sàng." # Trạng thái hiển thị chính

    # State cho các instance hệ thống (models, retriever)
    if 'g_embedding_model' not in st.session_state:
        st.session_state.g_embedding_model = None
    if 'g_reranking_model_loaded' not in st.session_state:
        st.session_state.g_reranking_model_loaded = None
    if 'retriever_instance' not in st.session_state:
        st.session_state.retriever_instance = None
    if 'vector_db_instance' not in st.session_state: # Cũng lưu vector_db nếu cần
         st.session_state.vector_db_instance = None


    st.header("Mô hình")
    st.selectbox(
        "Chọn mô hình Gemini (để tạo query variations):",
        options=config.AVAILABLE_GEMINI_MODELS,
        index=config.AVAILABLE_GEMINI_MODELS.index(st.session_state.selected_gemini_model), # Đọc từ state
        key="selected_gemini_model", # Ghi vào state khi thay đổi
        help="Chọn mô hình ngôn ngữ lớn để phân tích và tạo biến thể câu hỏi cho Retrieval."
    )

    st.header("Cấu hình Retrieval")

    st.radio(
        "Nguồn câu hỏi cho Retrieval:",
        options=['Đơn giản', 'Tổng quát', 'Sâu'],
        index=['Đơn giản', 'Tổng quát', 'Sâu'].index(st.session_state.retrieval_query_mode), # Đọc từ state
        key="retrieval_query_mode", # Ghi vào state khi thay đổi
        horizontal=True,
        help=(
            "**Đơn giản:** Chỉ dùng câu hỏi gốc.\n"
            "**Tổng quát:** Chỉ dùng câu hỏi tóm tắt (do AI tạo).\n"
            "**Sâu:** Dùng cả câu hỏi gốc và các biến thể (do AI tạo)."
        )
    )

    st.radio(
        "Phương thức Retrieval:",
        options=['dense', 'sparse', 'hybrid'],
        index=['dense', 'sparse', 'hybrid'].index(st.session_state.retrieval_method), # Đọc từ state
        key="retrieval_method", # Ghi vào state khi thay đổi
        horizontal=True,
        help=(
            "**dense:** Tìm kiếm dựa trên vector ngữ nghĩa.\n"
            "**sparse:** Tìm kiếm dựa trên từ khóa (BM25).\n"
            "**hybrid:** Kết hợp cả dense và sparse."
        )
    )

    # Widget đọc và ghi vào st.session_state['use_reranker']
    st.toggle(
        "Sử dụng Reranker",
        value=st.session_state.use_reranker, # Đọc từ state
        key="use_reranker", # Ghi vào state khi thay đổi
        help="Bật để sử dụng mô hình CrossEncoder xếp hạng lại kết quả tìm kiếm."
    )


st.subheader("Trạng thái Hệ thống Cơ bản")

# Sử dụng st.status để hiển thị trạng thái khởi tạo, chỉ chạy logic khởi tạo nếu chưa thành công
with st.status(st.session_state.status_message, expanded=True) as status:
    # Chỉ chạy logic tải model và retriever nếu chưa được tải thành công
    if st.session_state.g_embedding_model is None or st.session_state.retriever_instance is None:
         try:
            status.update(label="Đang tải Embedding Model...", state="running", expanded=True)
            st.session_state.g_embedding_model = utils.load_embedding_model(config.embedding_model_name)

            # Chỉ tải reranker model nếu cấu hình sidebar bật VÀ embedding model đã tải thành công
            use_reranker_current = st.session_state.get('use_reranker', True)
            if use_reranker_current and st.session_state.g_embedding_model:
                 status.update(label="Đang tải Reranker Model...", state="running", expanded=True)
                 st.session_state.g_reranking_model_loaded = utils.load_reranker_model(config.reranking_model_name)
            else:
                 st.session_state.g_reranking_model_loaded = None # Đảm bảo None nếu tắt hoặc embedding lỗi


            # Tải hoặc tạo RAG components
            if st.session_state.g_embedding_model: # Chỉ tạo/tải retriever nếu embedding model đã tải
                status.update(label="Đang tải hoặc tạo Vector Database và Retriever...", state="running", expanded=True)
                st.session_state.vector_db_instance, st.session_state.retriever_instance = data_loader.load_or_create_rag_components(st.session_state.g_embedding_model)
            else:
                 st.session_state.vector_db_instance = None
                 st.session_state.retriever_instance = None


            if st.session_state.retriever_instance and st.session_state.g_embedding_model:
                st.session_state.status_message = "✅ Hệ thống đã sẵn sàng!"
                status.update(label=st.session_state.status_message, state="complete", expanded=False)

                # Thông báo về reranker model nếu không tải được hoặc bị tắt
                if use_reranker_current and not st.session_state.g_reranking_model_loaded: # Kiểm tra lại trạng thái tải và cấu hình
                     st.warning("⚠️ Không tải được Reranker Model. Chức năng rerank sẽ không hoạt động.")
                elif not use_reranker_current: # Dùng biến mới đọc từ state
                     st.info("Chức năng Rerank đang **Tắt** trong cấu hình sidebar.")


            else: # Lỗi khởi tạo Retriever hoặc Embedding Model
                missing = [comp for comp, loaded in [("Retriever/VectorDB", st.session_state.retriever_instance), ("Embedding Model", st.session_state.g_embedding_model)] if not loaded]
                st.session_state.status_message = f"⚠️ Lỗi khởi tạo: {', '.join(missing)}."
                status.update(label=st.session_state.status_message, state="error", expanded=True)


         except Exception as e:
            st.session_state.status_message = f"⚠️ Lỗi nghiêm trọng khi khởi tạo hệ thống: {e}"
            status.update(label=st.session_state.status_message, state="error", expanded=True)
    else:
        # Nếu đã tải thành công trong lần rerun trước
        status.update(label=st.session_state.status_message, state="complete", expanded=False)
        # Thông báo về reranker model nếu không tải được hoặc bị tắt (hiển thị lại sau rerun nếu cần)
        use_reranker_current = st.session_state.get('use_reranker', True)
        if use_reranker_current and not st.session_state.g_reranking_model_loaded:
             st.warning("⚠️ Không tải được Reranker Model. Chức năng rerank sẽ không hoạt động.")
        elif not use_reranker_current:
             st.info("Chức năng Rerank đang **Tắt** trong cấu hình sidebar.")


# Chỉ cho phép các hành động tiếp theo nếu hệ thống cơ bản đã sẵn sàng (embedding và retriever)
system_ready = st.session_state.g_embedding_model is not None and st.session_state.retriever_instance is not None

if system_ready:
    # --- Hiển thị Cấu hình Đánh giá sẽ sử dụng (đọc từ session state, giờ do sidebar quản lý) ---
    st.caption(f"Mô hình: `{st.session_state.selected_gemini_model}` | Nguồn Query: `{st.session_state.retrieval_query_mode}` | Retrieval: `{st.session_state.retrieval_method}` | Reranker: `{'Bật' if st.session_state.use_reranker else 'Tắt'}`")

    # Tạo dict cấu hình cho hàm đánh giá - Đọc trực tiếp từ st.session_state
    # Các giá trị này giờ được đảm bảo tồn tại do sidebar hoặc khởi tạo sớm
    eval_config_dict = {
        'retrieval_query_mode': st.session_state.retrieval_query_mode,
        'retrieval_method': st.session_state.retrieval_method,
        'use_reranker': st.session_state.use_reranker,
        'gemini_model_name': st.session_state.selected_gemini_model,
        'embedding_model_name': config.embedding_model_name, # Lấy từ config file
        # Cập nhật tên reranker model dựa trên trạng thái tải và cấu hình
        'reranker_model_name': config.reranking_model_name if st.session_state.use_reranker and st.session_state.g_reranking_model_loaded else ("DISABLED_BY_CONFIG" if st.session_state.use_reranker else "DISABLED_BY_CONFIG"),
    }
    # Kiểm tra cuối cùng cho reranker model để truyền vào hàm run_retrieval_evaluation
    reranker_model_for_run = st.session_state.g_reranking_model_loaded if st.session_state.use_reranker and st.session_state.g_reranking_model_loaded else None


    st.subheader("Tải Lên File Đánh giá")
    uploaded_file = st.file_uploader(
        "Chọn file JSON dữ liệu đánh giá...", type=["json"], key="eval_file_uploader" # Thêm key để dễ reset
    )

    # Logic xử lý file tải lên
    # Kiểm tra nếu file mới được tải lên HOẶC nếu trạng thái dữ liệu không khớp với tên file (do xóa trạng thái)
    # uploaded_file_name kiểm tra None để reset khi clear state
    if uploaded_file is not None:
        if st.session_state.eval_uploaded_filename is None or uploaded_file.name != st.session_state.eval_uploaded_filename:
             try:
                # Reset trạng thái liên quan đến kết quả cũ
                st.session_state.eval_data = None # Clear data first
                st.session_state.eval_uploaded_filename = uploaded_file.name # Cập nhật tên file ngay
                st.session_state.eval_run_completed = False
                st.session_state.eval_results_df = None
                st.session_state.last_eval_config = {}
                st.session_state.cancel_eval_requested = False # Reset cờ hủy
                st.session_state.status_message = f"Sẵn sàng đánh giá với dữ liệu từ: {uploaded_file.name}" # Cập nhật status

                eval_data_list = json.loads(uploaded_file.getvalue().decode('utf-8'))
                st.session_state.eval_data = eval_data_list

                st.success(st.session_state.status_message)
                # st.rerun() # Rerun sau khi tải file thành công để cập nhật giao diện
             except Exception as e:
                st.error(f"Lỗi xử lý file JSON: {e}")
                # Reset trạng thái nếu lỗi
                st.session_state.eval_data = None
                st.session_state.eval_uploaded_filename = None # Đặt lại None để có thể tải lại file cùng tên
                st.session_state.eval_run_completed = False
                st.session_state.eval_results_df = None
                st.session_state.last_eval_config = {}
                st.session_state.cancel_eval_requested = False # Reset cờ hủy
                st.session_state.status_message = "Lỗi tải file đánh giá."


    if st.session_state.eval_data is not None:
        st.info(f"Sẵn sàng đánh giá với dữ liệu từ: **{st.session_state.eval_uploaded_filename}**.")

        if st.checkbox("Hiển thị dữ liệu mẫu (5 dòng)", key="show_eval_data_preview"):
            st.dataframe(pd.DataFrame(st.session_state.eval_data).head())

        # Placeholders cho thanh tiến trình và text status trong quá trình chạy
        progress_bar_placeholder = st.empty()
        status_text_placeholder = st.empty()


        # Xác định trạng thái đang chạy để disable nút
        is_running = st.session_state.eval_data is not None and not st.session_state.eval_run_completed and not st.session_state.cancel_eval_requested

        # Nút bắt đầu và hủy đánh giá
        col_eval_btn, col_cancel_btn = st.columns(2)

        with col_eval_btn:
            # Disable nút Bắt đầu nếu đang chạy hoặc chưa có dữ liệu
            if st.button("🚀 Bắt đầu Đánh giá", key="start_eval_button", disabled=is_running or st.session_state.eval_data is None):
                 # Lưu cấu hình hiện tại từ st.session_state vào last_eval_config trước khi chạy
                 current_config_for_save = {
                    'retrieval_query_mode': st.session_state.retrieval_query_mode,
                    'retrieval_method': st.session_state.retrieval_method,
                    'use_reranker': st.session_state.use_reranker,
                    'gemini_model_name': st.session_state.selected_gemini_model,
                    'embedding_model_name': config.embedding_model_name,
                    'reranker_model_name': config.reranking_model_name if st.session_state.use_reranker and st.session_state.g_reranking_model_loaded else ("DISABLED_BY_CONFIG" if st.session_state.use_reranker else "DISABLED_BY_CONFIG"),
                 }
                 st.session_state.last_eval_config = current_config_for_save.copy() # Lưu bản sao
                 st.session_state.eval_run_completed = False # Đặt lại cờ hoàn thành
                 st.session_state.eval_results_df = None # Xóa kết quả cũ khi chạy mới

                 # Reset cờ hủy khi bắt đầu chạy mới
                 st.session_state.cancel_eval_requested = False
                 st.session_state.status_message = "Đang chạy đánh giá..." # Cập nhật status


                 with st.spinner(f"Đang tải model Gemini: {st.session_state.selected_gemini_model}..."):
                     # Tải Gemini model dựa trên lựa chọn mới nhất từ sidebar (đã có trong session state)
                     # Nên tải lại mỗi lần chạy mới để đảm bảo đúng model được chọn
                     g_gemini_model_eval = utils.load_gemini_model(st.session_state.selected_gemini_model)


                 if g_gemini_model_eval:
                    st.info(f"Model Gemini '{st.session_state.selected_gemini_model}' đã sẵn sàng.")
                    # Sử dụng st.spinner để hiển thị trạng thái chạy
                    # (Spinner này bao quanh hàm run_retrieval_evaluation)
                    with st.spinner(""): # Spinner rỗng, text status được quản lý bởi placeholder bên dưới
                        start_eval_time = time.time()
                        results_df = run_retrieval_evaluation(
                            eval_data=st.session_state.eval_data,
                            hybrid_retriever=st.session_state.retriever_instance, # Lấy instance từ state
                            embedding_model=st.session_state.g_embedding_model, # Lấy model từ state
                            reranking_model=reranker_model_for_run, # Truyền model (hoặc None)
                            gemini_model=g_gemini_model_eval, # Truyền Gemini model đã tải
                            eval_config=st.session_state.last_eval_config, # Truyền dict config đã lưu (đảm bảo nhất)
                            progress_bar_placeholder=progress_bar_placeholder, # Truyền placeholder
                            status_text_placeholder=status_text_placeholder # Truyền placeholder
                        )
                        total_eval_time = time.time() - start_eval_time

                        # Cập nhật status sau khi hàm chạy xong
                        if st.session_state.cancel_eval_requested:
                             st.session_state.status_message = f"Đánh giá bị hủy bỏ sau {total_eval_time:.2f} giây."
                             st.warning(st.session_state.status_message)
                        else:
                             st.session_state.status_message = f"Hoàn thành đánh giá sau {total_eval_time:.2f} giây."
                             st.success(st.session_state.status_message)


                        st.session_state.eval_results_df = results_df
                        # Chỉ set complete nếu không bị hủy bỏ
                        if not st.session_state.cancel_eval_requested:
                            st.session_state.eval_run_completed = True

                        st.session_state.cancel_eval_requested = False # Reset cờ hủy sau khi kết thúc chạy
                        # st.rerun() # Streamlit sẽ tự động rerun sau callback của nút

                 else: # Nếu không tải được Gemini model
                      st.session_state.status_message = f"⚠️ Lỗi tải mô hình Gemini: {st.session_state.selected_gemini_model}"
                      st.error(st.session_state.status_message)
                      st.session_state.cancel_eval_requested = False # Đảm bảo cờ hủy được reset
                      st.session_state.eval_run_completed = False # Đảm bảo trạng thái không phải completed


        with col_cancel_btn:
            # Chỉ hiển thị nút Hủy nếu quá trình đánh giá đang chạy (is_running là True)
            if is_running:
                 if st.button("❌ Hủy Đánh giá", key="cancel_eval_button"):
                    st.session_state.cancel_eval_requested = True # Đặt cờ yêu cầu hủy
                    st.info("Đang yêu cầu hủy bỏ quá trình đánh giá...")
                    st.session_state.status_message = "Đang yêu cầu hủy bỏ..." # Cập nhật status hiển thị
                    # st.rerun() # Streamlit sẽ tự động rerun sau callback của nút


    # --- Hiển thị Kết quả ---
    # Chỉ hiển thị kết quả nếu đã hoàn thành VÀ không có yêu cầu hủy bỏ đang chờ xử lý
    if st.session_state.eval_run_completed and st.session_state.eval_results_df is not None and not st.session_state.cancel_eval_requested:
        st.subheader("Kết quả Đánh giá")
        detailed_results_df = st.session_state.eval_results_df
        last_config = st.session_state.last_eval_config # Đọc config đã chạy

        # --- Hiển thị lại cấu hình đã chạy ---
        st.markdown("**Cấu hình đã sử dụng cho lần chạy cuối:**")
        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
        cfg_col1.metric("Nguồn Query", last_config.get('retrieval_query_mode', 'N/A'))
        cfg_col2.metric("Ret. Method", last_config.get('retrieval_method', 'N/A'))
        cfg_col3.metric("Reranker", "Bật" if last_config.get('use_reranker', False) else "Tắt")
        st.caption(f"Gemini: `{last_config.get('gemini_model_name', 'N/A')}`, Embedding: `{last_config.get('embedding_model_name', 'N/A')}`, Reranker: `{last_config.get('reranker_model_name', 'N/A')}`")


        avg_metrics, num_eval, num_skipped_error = calculate_average_metrics(detailed_results_df)

        st.metric("Tổng số Queries", len(detailed_results_df))
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Queries Đánh giá Hợp lệ", num_eval)
        col_res2.metric("Queries Bỏ qua / Lỗi", num_skipped_error)

        if avg_metrics:
            st.markdown("#### Metrics Trung bình @K (trên các queries hợp lệ)")
            # Đã bỏ K=1
            k_values_display = [3, 5, 10]
            cols_k = st.columns(len(k_values_display))
            for idx, k in enumerate(k_values_display):
                with cols_k[idx]:
                    st.markdown(f"**K = {k}**")
                    st.text(f"Precision: {avg_metrics.get(f'avg_precision@{k}', 0.0):.4f}")
                    st.text(f"Recall:    {avg_metrics.get(f'avg_recall@{k}', 0.0):.4f}")
                    st.text(f"F1:        {avg_metrics.get(f'avg_f1@{k}', 0.0):.4f}")
                    st.text(f"MRR:       {avg_metrics.get(f'avg_mrr@{k}', 0.0):.4f}")
                    st.text(f"NDCG:      {avg_metrics.get(f'avg_ndcg@{k}', 0.0):.4f}")

            st.markdown("#### Thông tin Hiệu năng & Số lượng Trung bình")
            col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
            col_perf1.metric("Avg Total Time (s)", f"{avg_metrics.get('avg_processing_time', 0.0):.3f}")
            col_perf2.metric("Avg Variation Time (s)", f"{avg_metrics.get('avg_variation_time', 0.0):.3f}")
            col_perf3.metric("Avg Search Time (s)", f"{avg_metrics.get('avg_search_time', 0.0):.3f}")
            col_perf4.metric("Avg Rerank Time (s)", f"{avg_metrics.get('avg_rerank_time', 0.0):.3f}")

            col_count1, col_count2, col_count3, col_count4 = st.columns(4)
            col_count1.metric("Avg Variations Gen", f"{avg_metrics.get('avg_num_variations_generated', 0.0):.1f}")
            col_count2.metric("Avg Docs Found", f"{avg_metrics.get('avg_num_unique_docs_found', 0.0):.1f}")
            col_count3.metric("Avg Docs Reranked", f"{avg_metrics.get('avg_num_docs_reranked', 0.0):.1f}")
            col_count4.metric("Avg Final Docs", f"{avg_metrics.get('avg_num_retrieved_after_rerank', 0.0):.1f}")


        else:
            st.warning("Không thể tính metrics trung bình (không có query hợp lệ).")


        with st.expander("Xem Kết quả Chi tiết cho từng Query"):
            display_columns = [
                'query_id', 'query', 'status',
                'retrieval_query_mode','retrieval_method', 'use_reranker',
                'precision@3', 'recall@3', 'f1@3', 'mrr@3', 'ndcg@3', # Chỉ giữ K=3, 5, 10
                'precision@5', 'recall@5', 'f1@5', 'mrr@5', 'ndcg@5',
                'precision@10', 'recall@10', 'f1@10', 'mrr@10', 'ndcg@10',
                'processing_time', 'variation_time', 'search_time', 'rerank_time',
                'num_variations_generated','num_unique_docs_found', 'num_retrieved_before_rerank','num_docs_reranked', 'num_retrieved_after_rerank',
                'retrieved_ids', 'relevant_ids', 'summarizing_query', 'error_message'
            ]
            # Lọc lại các cột hiển thị để chỉ giữ lại các cột thực sự có trong DataFrame
            # Điều này quan trọng vì các metrics @1 không còn được tính
            existing_display_columns = [col for col in display_columns if col in detailed_results_df.columns]
            st.dataframe(detailed_results_df[existing_display_columns])


        st.subheader("Lưu Kết quả Chi tiết")
        try:
            results_json = detailed_results_df.to_json(orient='records', indent=2, force_ascii=False)
            results_csv = detailed_results_df.to_csv(index=False).encode('utf-8')

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sử dụng config đã chạy để tạo tên file
            last_config = st.session_state.get('last_eval_config', {}) # Đảm bảo lấy từ state
            qmode_suffix = last_config.get('retrieval_query_mode', 'na').lower()[:3]
            method_suffix = last_config.get('retrieval_method', 'na').lower()
            rerank_suffix = "rr" if last_config.get('use_reranker', False) else "norr"
            model_suffix = last_config.get('gemini_model_name', 'gemini').split('/')[-1].replace('.','-')[:15]

            base_filename = f"eval_{qmode_suffix}_{method_suffix}_{rerank_suffix}_{model_suffix}_{timestamp}"
            fname_json = f"{base_filename}.json"
            fname_csv = f"{base_filename}.csv"

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("💾 Tải về JSON", results_json, fname_json, "application/json", key="dl_json")
            with col_dl2:
                st.download_button("💾 Tải về CSV", results_csv, fname_csv, "text/csv", key="dl_csv")
        except Exception as e:
            st.error(f"Lỗi khi chuẩn bị file kết quả: {e}")

    # --- Quản lý Trạng thái Đánh giá ---
    st.markdown("---")
    st.subheader("Quản lý Trạng thái Đánh giá")
    # Nút xóa: reset toàn bộ trạng thái liên quan đến đánh giá và uploader
    if st.button("Xóa File Đã Tải và Kết Quả", key="clear_eval_state"):
        st.session_state.eval_data = None
        st.session_state.eval_uploaded_filename = None # Đặt lại None để có thể tải lại file cùng tên
        st.session_state.eval_run_completed = False
        st.session_state.eval_results_df = None
        st.session_state.last_eval_config = {}
        st.session_state.cancel_eval_requested = False # Reset cờ hủy
        st.session_state.status_message = "Đã xóa dữ liệu đánh giá. Sẵn sàng tải file mới." # Cập nhật trạng thái hiển thị

        # KHÔNG Cần gán None vào st.session_state["eval_file_uploader"]
        # Chỉ cần xóa các biến trạng thái dữ liệu và rerun
        st.success(st.session_state.status_message)
        st.rerun() # Kích hoạt rerun để giao diện cập nhật

else:
    # Nếu hệ thống chưa sẵn sàng (embedding hoặc retriever lỗi/chưa tải xong)
    st.warning("⚠️ Hệ thống cơ bản chưa sẵn sàng. Vui lòng kiểm tra lỗi ở mục 'Trạng thái Hệ thống Cơ bản'.")
    # Trạng thái chi tiết đã được hiển thị bởi st.status block