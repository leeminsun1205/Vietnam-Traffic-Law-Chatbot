# pages/2_Evaluation.py
import time
import streamlit as st

# --- Debug: Kiểm tra trạng thái ngay khi script tải (giữ lại hoặc xóa tùy ý) ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Evaluation Page Script Started (Removing K=1 metrics).")
logging.info(f"State on load - Gemini Model: {st.session_state.get('selected_gemini_model', 'NOT_FOUND')}")
logging.info(f"State on load - Query Mode: {st.session_state.get('retrieval_query_mode', 'NOT_FOUND')}")
logging.info(f"State on load - Retrieval Method: {st.session_state.get('retrieval_method', 'NOT_FOUND')}")
logging.info(f"State on load - Use Reranker: {st.session_state.get('use_reranker', 'NOT_FOUND')}")
# Đã bỏ log History LLM1 ở đây
logging.info("--------------------------------------")
# --- Kết thúc Debug ---


# ... Tiếp tục các lệnh import khác
import pandas as pd
import json
import math
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import config, utils, data_loader, retriever sau khi điều chỉnh path
import config
import utils
import data_loader
from retriever import HybridRetriever

# --- Các hàm tính toán metrics (giữ nguyên) ---
def precision_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0: return 0.0
    retrieved_at_k = retrieved_ids[:k]; relevant_set = set(relevant_ids)
    if not relevant_set: return 0.0
    intersect = set(retrieved_at_k) & relevant_set
    return len(intersect) / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    if not relevant_set: return 1.0
    retrieved_at_k = retrieved_ids[:k]
    intersect = set(retrieved_at_k) & relevant_set
    return len(intersect) / len(relevant_set)

def f1_at_k(retrieved_ids, relevant_ids, k):
    prec = precision_at_k(retrieved_ids, relevant_ids, k); rec = recall_at_k(retrieved_ids, relevant_ids, k)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def mrr_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids);
    if not relevant_set: return 0.0
    retrieved_at_k = retrieved_ids[:k]
    for rank, doc_id in enumerate(retrieved_at_k, 1):
        if doc_id in relevant_set: return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids);
    if not relevant_set: return 1.0
    retrieved_at_k = retrieved_ids[:k]; dcg = 0.0; idcg = 0.0
    for i, doc_id in enumerate(retrieved_at_k):
        relevance = 1.0 if doc_id in relevant_set else 0.0
        dcg += relevance / math.log2(i + 2)
    num_relevant_in_total = len(relevant_set)
    for i in range(min(k, num_relevant_in_total)):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def run_retrieval_evaluation(
    eval_data: list,
    hybrid_retriever: HybridRetriever,
    embedding_model,
    reranking_model, # Có thể là None nếu không dùng rerank
    gemini_model,
    eval_config: dict # Chứa retrieval_query_mode, retrieval_method, use_reranker
    ):

    results_list = []
    # Đã bỏ K=1
    k_values = [3, 5, 10] # Các giá trị K để tính metrics

    # --- Lấy cấu hình từ eval_config ---
    retrieval_query_mode = eval_config.get('retrieval_query_mode', 'Tổng quát')
    retrieval_method = eval_config.get('retrieval_method', 'hybrid')
    use_reranker = eval_config.get('use_reranker', True)
    dummy_history = None # Luôn là None vì không dùng history trong evaluation

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_items = len(eval_data)
    queries_per_batch = 15 # Giới hạn số lượng query trước khi tạm dừng
    wait_time_seconds = 60 # Thời gian tạm dừng

    for i, item in enumerate(eval_data):
        # Tạm dừng sau mỗi batch
        if i > 0 and i % queries_per_batch == 0:
            pause_msg = f"Đã xử lý {i}/{total_items} queries. Tạm dừng {wait_time_seconds} giây..."
            logging.info(pause_msg)
            status_text.text(pause_msg)
            time.sleep(wait_time_seconds)
            status_text.text(f"Tiếp tục xử lý query {i+1}/{total_items}...")

        query_id = item.get("query_id"); original_query = item.get("query")
        relevant_chunk_ids = set(item.get("relevant_chunk_ids", []))
        if not query_id or not original_query:
            logging.warning(f"Bỏ qua mục {i} do thiếu query_id hoặc query.")
            continue

        status_text.text(f"Đang xử lý query {i+1}/{total_items}: {query_id} (QueryMode: {retrieval_query_mode}, Method: {retrieval_method}, Rerank: {use_reranker})")
        logging.info(f"Eval - Processing QID: {query_id} (QueryMode: {retrieval_query_mode}, Method: {retrieval_method}, Rerank: {use_reranker})")

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
            # dummy_history giờ luôn là None
            variation_start = time.time()
            relevance_status, _, all_queries, summarizing_query = utils.generate_query_variations(
                original_query=original_query, gemini_model=gemini_model,
                chat_history=dummy_history, # Sử dụng biến này (luôn None)
                num_variations=config.NUM_QUERY_VARIATIONS
            )
            query_metrics["variation_time"] = time.time() - variation_start
            query_metrics["summarizing_query"] = summarizing_query
            query_metrics["num_variations_generated"] = len(all_queries) - 1

            if relevance_status == 'invalid':
                query_metrics["status"] = "skipped_irrelevant"
                query_metrics["processing_time"] = time.time() - start_time
                results_list.append(query_metrics)
                progress_bar.progress((i + 1) / total_items)
                logging.info(f"QID {query_id} skipped as irrelevant.")
                continue

            # --- Bước 2: Xác định query(s) để tìm kiếm ---
            queries_to_search = []
            if retrieval_query_mode == 'Đơn giản': queries_to_search = [original_query]
            elif retrieval_query_mode == 'Tổng quát': queries_to_search = [summarizing_query]
            elif retrieval_query_mode == 'Sâu': queries_to_search = all_queries

            # --- Bước 3: Thực hiện Retrieval ---
            collected_docs_data = {}
            search_start = time.time()
            for q_variant in queries_to_search:
                if not q_variant: continue
                search_results = hybrid_retriever.search(
                    q_variant, embedding_model,
                    method=retrieval_method,
                    k=config.VECTOR_K_PER_QUERY
                )
                for item in search_results:
                    doc_index = item.get('index')
                    if isinstance(doc_index, int) and doc_index >= 0 and doc_index not in collected_docs_data:
                        collected_docs_data[doc_index] = item
            query_metrics["search_time"] = time.time() - search_start
            query_metrics["num_unique_docs_found"] = len(collected_docs_data)
            logging.debug(f"QID {query_id}: Retrieval found {len(collected_docs_data)} unique docs.")

            # --- Chuẩn bị danh sách kết quả retrieval ---
            retrieved_docs_list = list(collected_docs_data.values())
            sort_reverse = (retrieval_method != 'dense')
            retrieved_docs_list.sort(key=lambda x: x.get('score', 0 if sort_reverse else float('inf')), reverse=sort_reverse)
            query_metrics["num_retrieved_before_rerank"] = len(retrieved_docs_list)


            # --- Bước 4: Re-ranking (Nếu bật) ---
            final_docs_for_metrics = []
            rerank_start = time.time()

            if use_reranker and retrieved_docs_list:
                query_for_reranking = summarizing_query if summarizing_query else original_query
                docs_to_rerank = retrieved_docs_list[:config.MAX_DOCS_FOR_RERANK]
                query_metrics["num_docs_reranked"] = len(docs_to_rerank)
                logging.debug(f"QID {query_id}: Reranking {len(docs_to_rerank)} docs with query: '{query_for_reranking[:50]}...'")

                rerank_input = [{'doc': item['doc'], 'index': item['index']} for item in docs_to_rerank]

                reranked_results = utils.rerank_documents(
                    query_for_reranking, rerank_input, reranking_model
                )
                final_docs_for_metrics = reranked_results[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                query_metrics["rerank_time"] = time.time() - rerank_start
                logging.debug(f"QID {query_id}: Reranking finished, selected {len(final_docs_for_metrics)} docs.")

            elif retrieved_docs_list:
                final_docs_for_metrics = retrieved_docs_list[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                query_metrics["rerank_time"] = 0.0
                query_metrics["num_docs_reranked"] = 0
                logging.debug(f"QID {query_id}: Skipped reranking, taking top {len(final_docs_for_metrics)} retrieval results.")
            else:
                 query_metrics["rerank_time"] = 0.0
                 query_metrics["num_docs_reranked"] = 0
                 logging.debug(f"QID {query_id}: No docs to rerank or select.")

            query_metrics["num_retrieved_after_rerank"] = len(final_docs_for_metrics)

            # --- Bước 5: Lấy IDs và Tính Metrics ---
            retrieved_ids = []
            for res in final_docs_for_metrics:
                doc_data = res.get('doc', {})
                chunk_id = None
                if isinstance(doc_data, dict):
                    chunk_id = doc_data.get('id')
                    if not chunk_id:
                        metadata = doc_data.get('metadata', {})
                        if isinstance(metadata, dict):
                            chunk_id = metadata.get('id') or metadata.get('chunk_id')
                if chunk_id:
                    retrieved_ids.append(str(chunk_id))

            query_metrics["retrieved_ids"] = retrieved_ids
            logging.debug(f"QID {query_id}: Final retrieved IDs for metrics (top {len(retrieved_ids)}): {retrieved_ids}")

            query_metrics["status"] = "evaluated"
            # Vòng lặp tính metrics, tự động dùng k_values mới
            for k in k_values:
                query_metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'f1@{k}'] = f1_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'mrr@{k}'] = mrr_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved_ids, relevant_chunk_ids, k)


        except Exception as e:
            logging.exception(f"Error evaluating QID {query_id}: {e}")
            query_metrics["status"] = "error_runtime"
            query_metrics["error_message"] = str(e)
        finally:
            query_metrics["processing_time"] = time.time() - start_time
            results_list.append(query_metrics)
            progress_bar.progress((i + 1) / total_items)

    status_text.text(f"Hoàn thành đánh giá {total_items} queries!")
    logging.info(f"Finished evaluation for {total_items} queries.")
    return pd.DataFrame(results_list)


def calculate_average_metrics(df_results: pd.DataFrame):
    evaluated_df = df_results[df_results['status'] == 'evaluated'].copy()
    num_evaluated = len(evaluated_df)
    num_skipped_error = len(df_results) - num_evaluated

    if num_evaluated == 0:
        logging.warning("No queries were successfully evaluated. Cannot calculate average metrics.")
        return None, num_evaluated, num_skipped_error

    avg_metrics = {}
    # Đã bỏ K=1
    k_values = [3, 5, 10]
    metric_keys_k = [f'{m}@{k}' for k in k_values for m in ['precision', 'recall', 'f1', 'mrr', 'ndcg']]
    timing_keys = ['processing_time', 'variation_time', 'search_time', 'rerank_time']
    count_keys = ['num_variations_generated', 'num_unique_docs_found', 'num_docs_reranked', 'num_retrieved_before_rerank', 'num_retrieved_after_rerank']

    all_keys_to_average = metric_keys_k + timing_keys + count_keys

    for key in all_keys_to_average:
        if key in evaluated_df.columns:
            evaluated_df[key] = pd.to_numeric(evaluated_df[key], errors='coerce')
            total = evaluated_df[key].sum(skipna=True)
            valid_count = evaluated_df[key].notna().sum()
            avg_metrics[f'avg_{key}'] = total / valid_count if valid_count > 0 else 0.0
        else:
            logging.warning(f"Metric key '{key}' not found in results DataFrame for averaging.")
            avg_metrics[f'avg_{key}'] = 0.0

    logging.info(f"Calculated average metrics over {num_evaluated} evaluated queries.")
    return avg_metrics, num_evaluated, num_skipped_error


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

    # --- Initialize session state keys for sidebar widgets if they don't exist ---
    # Attempt to get initial state from Chatbot state if it exists, otherwise use defaults
    DEFAULT_EVAL_CONFIG_STATE = {
        "selected_gemini_model": st.session_state.get("selected_gemini_model", config.DEFAULT_GEMINI_MODEL),
        "retrieval_query_mode": st.session_state.get("retrieval_query_mode", 'Tổng quát'),
        "retrieval_method": st.session_state.get("retrieval_method", 'hybrid'),
        "use_reranker": st.session_state.get("use_reranker", True),
    }

    for key, default_value in DEFAULT_EVAL_CONFIG_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            # logging.info(f"Initialized missing key '{key}' in Evaluation sidebar state with default: {default_value}") # Optional logging

    # Đảm bảo key use_history_llm1 tồn tại với giá trị False mặc định cho Evaluation nếu nó bị thiếu
    # Không có widget điều khiển cho nó ở đây.
    if 'use_history_for_llm1' not in st.session_state:
        st.session_state.use_history_for_llm1 = False


    st.header("Mô hình")
    # Widget đọc và ghi vào st.session_state['selected_gemini_model']
    st.selectbox(
        "Chọn mô hình Gemini (để tạo query variations):",
        options=config.AVAILABLE_GEMINI_MODELS,
        index=config.AVAILABLE_GEMINI_MODELS.index(st.session_state.get('selected_gemini_model', config.DEFAULT_GEMINI_MODEL)), # Đọc từ state
        key="selected_gemini_model", # Ghi vào state khi thay đổi
        help="Chọn mô hình ngôn ngữ lớn để phân tích và tạo biến thể câu hỏi cho Retrieval."
    )


    st.header("Cấu hình Retrieval")

    # Widget đọc và ghi vào st.session_state['retrieval_query_mode']
    st.radio(
        "Nguồn câu hỏi cho Retrieval:",
        options=['Đơn giản', 'Tổng quát', 'Sâu'],
        index=['Đơn giản', 'Tổng quát', 'Sâu'].index(st.session_state.get('retrieval_query_mode', 'Tổng quát')), # Đọc từ state
        key="retrieval_query_mode", # Ghi vào state khi thay đổi
        horizontal=True,
        help=(
            "**Đơn giản:** Chỉ dùng câu hỏi gốc.\n"
            "**Tổng quát:** Chỉ dùng câu hỏi tóm tắt (do AI tạo).\n"
            "**Sâu:** Dùng cả câu hỏi gốc và các biến thể (do AI tạo)."
        )
    )

    # Widget đọc và ghi vào st.session_state['retrieval_method']
    st.radio(
        "Phương thức Retrieval:",
        options=['dense', 'sparse', 'hybrid'],
        index=['dense', 'sparse', 'hybrid'].index(st.session_state.get('retrieval_method', 'hybrid')), # Đọc từ state
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
        value=st.session_state.get('use_reranker', True), # Đọc từ state
        key="use_reranker", # Ghi vào state khi thay đổi
        help="Bật để sử dụng mô hình CrossEncoder xếp hạng lại kết quả tìm kiếm."
    )

    # Đã bỏ cài đặt cho History LLM1 ở sidebar

# --- Khởi tạo hoặc kiểm tra Session State (Tiếp tục) ---
# Phần khởi tạo state riêng của Evaluation (giữ nguyên)
if 'eval_data' not in st.session_state: st.session_state.eval_data = None
if 'eval_results_df' not in st.session_state: st.session_state.eval_results_df = None
if 'eval_run_completed' not in st.session_state: st.session_state.eval_run_completed = False
if 'eval_uploaded_filename' not in st.session_state: st.session_state.eval_uploaded_filename = ""
# last_eval_config không cần khởi tạo ở đây vì nó chỉ được set khi bắt đầu đánh giá


st.subheader("Trạng thái Hệ thống Cơ bản")
init_ok = False
retriever_instance = None
g_embedding_model = None
g_reranking_model_loaded = None # Đổi tên biến để tránh nhầm lẫn

with st.spinner("Kiểm tra và khởi tạo tài nguyên cốt lõi..."):
    try:
        g_embedding_model = utils.load_embedding_model(config.embedding_model_name)
        # Tải reranker model nhưng chỉ dùng nếu use_reranker_eval là True (đọc từ state)
        g_reranking_model_loaded = utils.load_reranker_model(config.reranking_model_name)

        _, retriever_instance = data_loader.load_or_create_rag_components(g_embedding_model)

        # Đọc giá trị use_reranker từ session state (được quản lý bởi sidebar)
        use_reranker_current = st.session_state.get('use_reranker', True)

        if retriever_instance and g_embedding_model:
            init_ok = True
            st.success("✅ VectorDB, Retriever, Embedding Model đã sẵn sàng.")
            logging.info("Core components initialized successfully for evaluation.")
            # Thông báo về reranker model nếu không tải được hoặc bị tắt
            if not g_reranking_model_loaded:
                 st.warning("⚠️ Không tải được Reranker Model. Chức năng rerank sẽ không hoạt động.")
                 logging.warning("Reranker model failed to load, reranking will be disabled if attempted.")
            elif not use_reranker_current: # Dùng biến mới đọc từ state
                 st.info("Reranker Model đã tải, nhưng chức năng Rerank đang **Tắt** trong cấu hình sidebar.")

        else:
            missing = [comp for comp, loaded in [("Retriever/VectorDB", retriever_instance), ("Embedding Model", g_embedding_model)] if not loaded]
            st.error(f"⚠️ Lỗi khởi tạo: {', '.join(missing)}.")
            logging.error(f"Failed to initialize components: {', '.join(missing)}.")

    except Exception as e:
        st.error(f"⚠️ Lỗi nghiêm trọng khi khởi tạo hệ thống: {e}")
        logging.exception("Critical error during system initialization for evaluation.")

if init_ok:
    # --- Hiển thị Cấu hình Đánh giá sẽ sử dụng (đọc từ session state, giờ do sidebar quản lý) ---
    st.caption(f"Mô hình: `{st.session_state.get('selected_gemini_model', 'N/A')}` | Nguồn Query: `{st.session_state.get('retrieval_query_mode', 'N/A')}` | Retrieval: `{st.session_state.get('retrieval_method', 'N/A')}` | Reranker: `{'Bật' if st.session_state.get('use_reranker', False) else 'Tắt'}`")

    # Tạo dict cấu hình cho hàm đánh giá - Đọc trực tiếp từ st.session_state
    # Các giá trị này giờ được đảm bảo tồn tại do sidebar hoặc khởi tạo sớm
    eval_config_dict = {
        'retrieval_query_mode': st.session_state.get('retrieval_query_mode', 'Tổng quát'),
        'retrieval_method': st.session_state.get('retrieval_method', 'hybrid'),
        'use_reranker': st.session_state.get('use_reranker', True),
        'use_history_llm1': False, # Giá trị này luôn là False cho evaluation
        'gemini_model_name': st.session_state.get('selected_gemini_model', config.DEFAULT_GEMINI_MODEL),
        'embedding_model_name': config.embedding_model_name,
        # Cập nhật tên reranker model dựa trên trạng thái tải và cấu hình
        'reranker_model_name': config.reranking_model_name if st.session_state.get('use_reranker', True) and g_reranking_model_loaded else ("DISABLED_BY_CONFIG" if st.session_state.get('use_reranker', True) else "DISABLED_BY_CONFIG"),
    }
    # Kiểm tra cuối cùng cho reranker model để truyền vào hàm run_retrieval_evaluation
    reranker_model_for_run = g_reranking_model_loaded if st.session_state.get('use_reranker', True) and g_reranking_model_loaded else None


    st.subheader("Tải Lên File Đánh giá")
    uploaded_file = st.file_uploader(
        "Chọn file JSON dữ liệu đánh giá...", type=["json"], key="eval_file_uploader"
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.eval_uploaded_filename:
            try:
                eval_data_list = json.loads(uploaded_file.getvalue().decode('utf-8'))
                st.session_state.eval_data = eval_data_list
                st.session_state.eval_uploaded_filename = uploaded_file.name
                st.session_state.eval_run_completed = False
                # Reset last_eval_config khi tải file mới để tránh hiển thị kết quả cũ với cấu hình sai
                st.session_state.last_eval_config = {}
                st.success(f"Đã tải file '{uploaded_file.name}' ({len(eval_data_list)} câu hỏi).")
                logging.info(f"Loaded evaluation file: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Lỗi xử lý file JSON: {e}")
                logging.exception("Error processing uploaded JSON file.")
                st.session_state.eval_data = None; st.session_state.eval_uploaded_filename = ""
                st.session_state.eval_run_completed = False

    if st.session_state.eval_data is not None:
        st.info(f"Sẵn sàng đánh giá với dữ liệu từ: **{st.session_state.eval_uploaded_filename}**.")

        if st.checkbox("Hiển thị dữ liệu mẫu (5 dòng)", key="show_eval_data_preview"):
            st.dataframe(pd.DataFrame(st.session_state.eval_data).head())

        # Nút bắt đầu đánh giá
        if st.button("🚀 Bắt đầu Đánh giá", key="start_eval_button"):
             # Lưu cấu hình hiện tại từ st.session_state vào last_eval_config trước khi chạy
             # Đây là cấu hình mà người dùng đã chọn trên sidebar của trang Evaluation
             current_config_for_save = {
                'retrieval_query_mode': st.session_state.get('retrieval_query_mode', 'Tổng quát'),
                'retrieval_method': st.session_state.get('retrieval_method', 'hybrid'),
                'use_reranker': st.session_state.get('use_reranker', True),
                'use_history_llm1': False, # Giá trị này luôn là False cho evaluation
                'gemini_model_name': st.session_state.get('selected_gemini_model', config.DEFAULT_GEMINI_MODEL),
                'embedding_model_name': config.embedding_model_name,
                'reranker_model_name': config.reranking_model_name if st.session_state.get('use_reranker', True) and g_reranking_model_loaded else ("DISABLED_BY_CONFIG" if st.session_state.get('use_reranker', True) else "DISABLED_BY_CONFIG"),
             }
             st.session_state.last_eval_config = current_config_for_save.copy() # Lưu bản sao

             with st.spinner(f"Đang tải model Gemini: {st.session_state.get('selected_gemini_model', config.DEFAULT_GEMINI_MODEL)}..."):
                 # Tải Gemini model dựa trên lựa chọn mới nhất từ sidebar (đã có trong session state)
                 g_gemini_model_eval = utils.load_gemini_model(st.session_state.get('selected_gemini_model', config.DEFAULT_GEMINI_MODEL))


             if g_gemini_model_eval:
                st.info(f"Model Gemini '{st.session_state.get('selected_gemini_model', config.DEFAULT_GEMINI_MODEL)}' đã sẵn sàng.")
                with st.spinner("⏳ Đang chạy đánh giá..."):
                    start_eval_time = time.time()
                    results_df = run_retrieval_evaluation(
                        eval_data=st.session_state.eval_data,
                        hybrid_retriever=retriever_instance,
                        embedding_model=g_embedding_model,
                        reranking_model=reranker_model_for_run, # Truyền model (hoặc None)
                        gemini_model=g_gemini_model_eval, # Truyền Gemini model đã tải
                        eval_config=st.session_state.last_eval_config # Truyền dict config đã lưu (đảm bảo nhất)
                    )
                    total_eval_time = time.time() - start_eval_time
                    st.success(f"Hoàn thành đánh giá sau {total_eval_time:.2f} giây.")
                    logging.info(f"Evaluation completed in {total_eval_time:.2f} seconds.")

                    st.session_state.eval_results_df = results_df
                    st.session_state.eval_run_completed = True
                    st.rerun() # Rerun để hiển thị kết quả


    # --- Hiển thị Kết quả ---
    if st.session_state.eval_run_completed and st.session_state.eval_results_df is not None:
        st.subheader("Kết quả Đánh giá")
        detailed_results_df = st.session_state.eval_results_df
        last_config = st.session_state.last_eval_config # Đọc config đã chạy

        # --- Hiển thị lại cấu hình đã chạy ---
        st.markdown("**Cấu hình đã sử dụng cho lần chạy cuối:**")
        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
        cfg_col1.metric("Nguồn Query", last_config.get('retrieval_query_mode', 'N/A'))
        cfg_col2.metric("Ret. Method", last_config.get('retrieval_method', 'N/A'))
        cfg_col3.metric("Reranker", "Bật" if last_config.get('use_reranker', False) else "Tắt")
        cfg_col4.metric("History LLM1", "Tắt") # Luôn hiển thị Tắt vì không dùng history
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
                'retrieval_query_mode','retrieval_method', 'use_reranker', # Đã bỏ use_history_llm1 khỏi cột hiển thị
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
            qmode_suffix = last_config.get('retrieval_query_mode', 'na').lower()[:3]
            method_suffix = last_config.get('retrieval_method', 'na').lower()
            rerank_suffix = "rr" if last_config.get('use_reranker', False) else "norr"
            # Đã bỏ hist_suffix khỏi tên file
            model_suffix = last_config.get('gemini_model_name', 'gemini').split('/')[-1].replace('.','-')[:15]

            # Tên file không còn chứa thông tin history
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
            logging.exception("Error preparing evaluation results for download.")

    # --- Quản lý Trạng thái Đánh giá ---
    st.markdown("---")
    st.subheader("Quản lý Trạng thái Đánh giá")
    if st.button("Xóa File Đã Tải và Kết Quả", key="clear_eval_state"):
        st.session_state.eval_data = None
        st.session_state.eval_uploaded_filename = ""
        st.session_state.eval_run_completed = False
        st.session_state.eval_results_df = None
        st.session_state.last_eval_config = {}
        # Reset các cài đặt sidebar về mặc định khi xóa trạng thái
        st.session_state.selected_gemini_model = config.DEFAULT_GEMINI_MODEL
        st.session_state.retrieval_query_mode = 'Tổng quát'
        st.session_state.retrieval_method = 'hybrid'
        st.session_state.use_reranker = True
        st.session_state.use_history_llm1 = False # Luôn reset use_history_llm1 về False cho Evaluation

        st.success("Đã xóa trạng thái đánh giá.")
        logging.info("Evaluation state cleared.")
        time.sleep(1); st.rerun()

else:
    st.warning("⚠️ Hệ thống cơ bản chưa sẵn sàng. Vui lòng kiểm tra lỗi và khởi động lại.")
    logging.warning("Evaluation page cannot proceed as core components are not ready.")