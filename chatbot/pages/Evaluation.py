# pages/2_Evaluation.py
import time
import streamlit as st
import pandas as pd
import json
import time
import math
import os
import logging
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import utils
import data_loader
# Bỏ import SimpleVectorDatabase vì không dùng trực tiếp
from retriever import HybridRetriever # Chỉ cần import HybridRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Các hàm tính toán metrics (giữ nguyên) ---
def precision_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0: return 0.0
    retrieved_at_k = retrieved_ids[:k]; relevant_set = set(relevant_ids)
    if not relevant_set: return 0.0
    intersect = set(retrieved_at_k) & relevant_set
    return len(intersect) / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    if not relevant_set: return 1.0 # Nếu không có relevant thì recall là 1? Hoặc 0? Xem lại logic nếu cần
    retrieved_at_k = retrieved_ids[:k]
    intersect = set(retrieved_at_k) & relevant_set
    return len(intersect) / len(relevant_set)

def f1_at_k(retrieved_ids, relevant_ids, k):
    prec = precision_at_k(retrieved_ids, relevant_ids, k); rec = recall_at_k(retrieved_ids, relevant_ids, k)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def mrr_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids);
    if not relevant_set: return 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant_set: return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids);
    if not relevant_set: return 1.0 # Nếu không có relevant thì NDCG là 1? Hoặc 0? Xem lại logic
    retrieved_at_k = retrieved_ids[:k]; dcg = 0.0; idcg = 0.0
    for i, doc_id in enumerate(retrieved_at_k):
        if doc_id in relevant_set: dcg += 1.0 / math.log2(i + 2)
    # IDCG tính dựa trên số lượng relevant thực tế, tối đa là k
    for i in range(min(k, len(relevant_set))): idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def run_retrieval_evaluation(
    eval_data: list,
    hybrid_retriever: HybridRetriever,
    embedding_model,
    reranking_model,
    gemini_model,
    eval_config: dict # Chứa retrieval_mode và use_history_for_llm1
    ):

    results_list = []
    k_values = [1, 3, 5, 10] # Các giá trị K để tính metrics

    # Lấy cấu hình từ eval_config
    retrieval_mode = eval_config.get('retrieval_mode', 'Tổng quát') # Mặc định là Tổng quát
    use_history = eval_config.get('use_history_for_llm1', False)
    # Tạo lịch sử giả nếu cần cho bước tạo variation
    dummy_history = [{"role": "user", "content": "Câu hỏi trước đó (nếu có)"}] if use_history else None

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_items = len(eval_data)
    queries_per_batch = 15 # Giới hạn số lượng query trước khi tạm dừng (tránh lỗi API)
    wait_time_seconds = 60 # Thời gian tạm dừng (giây)

    for i, item in enumerate(eval_data):
        # Tạm dừng sau mỗi batch để tránh rate limit của API (nếu cần)
        if i > 0 and i % queries_per_batch == 0:
            pause_msg = f"Đã xử lý {i}/{total_items} queries. Tạm dừng {wait_time_seconds} giây để tránh lỗi API..."
            logging.info(pause_msg)
            status_text.text(pause_msg)
            time.sleep(wait_time_seconds)
            status_text.text(f"Tiếp tục xử lý query {i+1}/{total_items}...")

        query_id = item.get("query_id"); original_query = item.get("query")
        relevant_chunk_ids = set(item.get("relevant_chunk_ids", [])) # Dùng set để kiểm tra nhanh hơn
        if not query_id or not original_query:
            logging.warning(f"Bỏ qua mục {i} do thiếu query_id hoặc query.")
            continue # Bỏ qua nếu thiếu thông tin cơ bản

        status_text.text(f"Đang xử lý query {i+1}/{total_items}: {query_id} (Mode: {retrieval_mode})")
        logging.info(f"Eval - Processing QID: {query_id} with Mode: {retrieval_mode}")

        start_time = time.time()
        # Khởi tạo dictionary chứa kết quả cho query này
        query_metrics = {
            "query_id": query_id, "query": original_query,
            "retrieval_mode": retrieval_mode, # Lưu lại chế độ đang dùng
            "use_history_llm1": use_history, # Lưu lại trạng thái dùng history
            "status": "error", # Trạng thái mặc định là lỗi
            "retrieved_ids": [], "relevant_ids": list(relevant_chunk_ids), # Lưu cả relevant ids để đối chiếu
            "processing_time": 0.0
        }
        # Khởi tạo tất cả các metrics về 0.0
        for k in k_values:
            query_metrics[f'precision@{k}'] = 0.0
            query_metrics[f'recall@{k}'] = 0.0
            query_metrics[f'f1@{k}'] = 0.0
            query_metrics[f'mrr@{k}'] = 0.0
            query_metrics[f'ndcg@{k}'] = 0.0
        # Khởi tạo các thông số thời gian và số lượng
        timing_keys = ['variation_time', 'search_time', 'rerank_time']
        count_keys = ['num_variations_generated', 'num_unique_docs_found', 'num_docs_reranked']
        for k in timing_keys + count_keys: query_metrics[k] = 0.0
        query_metrics['summarizing_query'] = '' # Khởi tạo summarizing_query

        try:
            # Bước 1: Tạo variations và summarizing query (luôn chạy bước này)
            variation_start = time.time()
            relevance_status, _, all_queries, summarizing_query = utils.generate_query_variations(
                original_query=original_query,
                gemini_model=gemini_model,
                chat_history=dummy_history, # Truyền dummy_history
                num_variations=config.NUM_QUERY_VARIATIONS # Đảm bảo dùng config
            )
            query_metrics["variation_time"] = time.time() - variation_start
            query_metrics["summarizing_query"] = summarizing_query
            # all_queries đã bao gồm original_query trong hàm generate_query_variations
            query_metrics["num_variations_generated"] = len(all_queries) -1 # Số biến thể = tổng - 1 (câu gốc)


            # Kiểm tra relevancy (nếu cần bỏ qua các query không liên quan)
            if relevance_status == 'invalid':
                query_metrics["status"] = "skipped_irrelevant"
                query_metrics["processing_time"] = time.time() - start_time
                results_list.append(query_metrics)
                progress_bar.progress((i + 1) / total_items)
                logging.info(f"QID {query_id} skipped as irrelevant.")
                continue # Chuyển sang query tiếp theo

            # Bước 2: Retrieval dựa trên retrieval_mode
            collected_docs_data = {} # Dict để lưu các docs tìm được {index: {'doc': ..., 'hybrid_score': ...}}
            search_start = time.time()
            query_for_reranking = original_query # Query mặc định để rerank

            if retrieval_mode == 'Đơn giản':
                logging.debug(f"QID {query_id}: Running Simple Search with: '{original_query}'")
                variant_results = hybrid_retriever.hybrid_search(
                    original_query, embedding_model, # <<< Dùng câu gốc
                    vector_search_k=config.VECTOR_K_PER_QUERY,
                    final_k=config.HYBRID_K_PER_QUERY
                )
                for res_item in variant_results:
                    idx = res_item.get('index')
                    # Chỉ thêm nếu là số nguyên và hợp lệ
                    if isinstance(idx, int) and idx >= 0:
                        collected_docs_data[idx] = res_item # Ghi đè nếu đã có (không cần thiết vì chỉ search 1 lần)
                query_for_reranking = original_query

            elif retrieval_mode == 'Tổng quát':
                logging.debug(f"QID {query_id}: Running General Search with: '{summarizing_query}'")
                variant_results = hybrid_retriever.hybrid_search(
                    summarizing_query, embedding_model, # <<< Dùng câu tóm tắt
                    vector_search_k=config.VECTOR_K_PER_QUERY,
                    final_k=config.HYBRID_K_PER_QUERY
                )
                for res_item in variant_results:
                    idx = res_item.get('index')
                    if isinstance(idx, int) and idx >= 0:
                        collected_docs_data[idx] = res_item
                query_for_reranking = summarizing_query

            elif retrieval_mode == 'Sâu':
                logging.debug(f"QID {query_id}: Running Deep Search with {len(all_queries)} queries...")
                # all_queries đã chứa original_query
                for q_variant in all_queries:
                    logging.debug(f"  - Deep searching with variant: '{q_variant[:100]}...'")
                    variant_results = hybrid_retriever.hybrid_search(
                        q_variant, embedding_model, # <<< Dùng từng query trong all_queries
                        vector_search_k=config.VECTOR_K_PER_QUERY,
                        final_k=config.HYBRID_K_PER_QUERY # Lấy K kết quả cho mỗi variant
                    )
                    for res_item in variant_results:
                        idx = res_item.get('index')
                        # Chỉ thêm nếu chưa có để tránh trùng lặp từ các variant khác nhau
                        if isinstance(idx, int) and idx >= 0 and idx not in collected_docs_data:
                             collected_docs_data[idx] = res_item
                # Cho chế độ Sâu, rerank bằng câu tóm tắt (hoặc có thể đổi thành câu gốc nếu muốn)
                query_for_reranking = summarizing_query

            query_metrics["search_time"] = time.time() - search_start
            query_metrics["num_unique_docs_found"] = len(collected_docs_data)
            logging.debug(f"QID {query_id}: Found {len(collected_docs_data)} unique docs.")

            # Chuẩn bị danh sách docs để rerank
            unique_docs_list = list(collected_docs_data.values())
            # Sắp xếp theo hybrid_score giảm dần (nếu có) trước khi cắt
            unique_docs_list.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
            docs_for_reranking_input = unique_docs_list[:config.MAX_DOCS_FOR_RERANK]
            query_metrics["num_docs_reranked"] = len(docs_for_reranking_input)
            logging.debug(f"QID {query_id}: Prepared {len(docs_for_reranking_input)} docs for reranking.")


            # Bước 3: Re-ranking
            rerank_start = time.time()
            reranked_results = [] # Khởi tạo list rỗng
            if docs_for_reranking_input: # Chỉ rerank nếu có tài liệu
                logging.debug(f"QID {query_id}: Reranking with query: '{query_for_reranking[:100]}...'")
                reranked_results = utils.rerank_documents(
                    query_for_reranking, # Query dùng để rerank đã xác định ở bước trước
                    docs_for_reranking_input, # Danh sách các dict {'doc': ..., 'index': ...}
                    reranking_model
                )
            else:
                 logging.debug(f"QID {query_id}: No documents to rerank.")
            query_metrics["rerank_time"] = time.time() - rerank_start

            # Bước 4: Lấy IDs và Tính Metrics
            # Lấy top K kết quả cuối cùng sau rerank
            final_retrieved_docs = reranked_results[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
            retrieved_ids = []
            for res in final_retrieved_docs:
                # Cố gắng lấy 'id' hoặc 'chunk_id' từ metadata hoặc từ key 'id' trực tiếp
                doc_data = res.get('doc', {})
                chunk_id = None
                if isinstance(doc_data, dict):
                    # Ưu tiên lấy từ key 'id' nếu có, sau đó mới đến metadata
                    chunk_id = doc_data.get('id')
                    if not chunk_id:
                        metadata = doc_data.get('metadata', {})
                        if isinstance(metadata, dict):
                             chunk_id = metadata.get('id') or metadata.get('chunk_id')

                if chunk_id:
                    retrieved_ids.append(str(chunk_id)) # Đảm bảo ID là string

            query_metrics["retrieved_ids"] = retrieved_ids
            logging.debug(f"QID {query_id}: Final retrieved IDs (top {len(retrieved_ids)}): {retrieved_ids}")


            query_metrics["status"] = "evaluated" # Đánh dấu là đã đánh giá thành công
            # Tính toán tất cả các metrics với các giá trị K
            for k in k_values:
                query_metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'f1@{k}'] = f1_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'mrr@{k}'] = mrr_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved_ids, relevant_chunk_ids, k)
                logging.debug(f"  Metrics @{k}: P={query_metrics[f'precision@{k}']:.4f}, R={query_metrics[f'recall@{k}']:.4f}, F1={query_metrics[f'f1@{k}']:.4f}, MRR={query_metrics[f'mrr@{k}']:.4f}, NDCG={query_metrics[f'ndcg@{k}']:.4f}")


        except Exception as e:
            logging.exception(f"Error evaluating QID {query_id}: {e}") # Log cả traceback
            query_metrics["status"] = "error_runtime"
            query_metrics["error_message"] = str(e) # Ghi lại thông báo lỗi
        finally:
            query_metrics["processing_time"] = time.time() - start_time
            results_list.append(query_metrics) # Thêm kết quả (kể cả lỗi) vào danh sách
            progress_bar.progress((i + 1) / total_items) # Cập nhật progress bar

    status_text.text(f"Hoàn thành đánh giá {total_items} queries!")
    logging.info(f"Finished evaluation for {total_items} queries.")
    return pd.DataFrame(results_list)


def calculate_average_metrics(df_results: pd.DataFrame):
    # Chỉ tính trung bình trên các query được đánh giá thành công
    evaluated_df = df_results[df_results['status'] == 'evaluated'].copy() # Tạo bản sao để tránh SettingWithCopyWarning
    num_evaluated = len(evaluated_df)
    num_skipped_error = len(df_results) - num_evaluated

    if num_evaluated == 0:
        logging.warning("No queries were successfully evaluated. Cannot calculate average metrics.")
        return None, num_evaluated, num_skipped_error

    avg_metrics = {}
    k_values = [1, 3, 5, 10] # Các giá trị K đã tính
    # Danh sách các cột metrics cần tính trung bình
    metric_keys_k = [f'{m}@{k}' for k in k_values for m in ['precision', 'recall', 'f1', 'mrr', 'ndcg']]
    timing_keys = ['processing_time', 'variation_time', 'search_time', 'rerank_time']
    count_keys = ['num_variations_generated', 'num_unique_docs_found', 'num_docs_reranked']

    all_keys_to_average = metric_keys_k + timing_keys + count_keys

    for key in all_keys_to_average:
        if key in evaluated_df.columns: # Kiểm tra xem cột có tồn tại không
             # Chuyển đổi cột sang dạng số, lỗi sẽ thành NaN
             evaluated_df[key] = pd.to_numeric(evaluated_df[key], errors='coerce')
             # Tính tổng bỏ qua NaN
             total = evaluated_df[key].sum(skipna=True)
             # Đếm số lượng giá trị không phải NaN để chia trung bình
             valid_count = evaluated_df[key].notna().sum()
             # Tính trung bình, tránh chia cho 0
             avg_metrics[f'avg_{key}'] = total / valid_count if valid_count > 0 else 0.0
        else:
             logging.warning(f"Metric key '{key}' not found in results DataFrame for averaging.")
             avg_metrics[f'avg_{key}'] = 0.0 # Đặt giá trị mặc định nếu cột không tồn tại

    logging.info(f"Calculated average metrics over {num_evaluated} evaluated queries.")
    return avg_metrics, num_evaluated, num_skipped_error


# --- Giao diện Streamlit ---
st.set_page_config(page_title="Đánh giá Retrieval", layout="wide")
st.title("📊 Đánh giá Hệ thống Retrieval")

st.markdown("""
Trang này cho phép bạn chạy đánh giá hiệu suất của hệ thống retrieval (tìm kiếm + xếp hạng lại)
dựa trên một tập dữ liệu có chứa các câu hỏi và các chunk tài liệu liên quan (ground truth).
Kết quả đánh giá cho lần chạy gần nhất sẽ được lưu trong phiên làm việc này.
""")

# --- Khởi tạo hoặc kiểm tra Session State ---
if 'eval_data' not in st.session_state:
    st.session_state.eval_data = None # Dữ liệu đánh giá đã tải
if 'eval_results_df' not in st.session_state:
    st.session_state.eval_results_df = None # DataFrame kết quả
if 'eval_run_completed' not in st.session_state:
    st.session_state.eval_run_completed = False # Đánh dấu đã chạy xong
if 'eval_uploaded_filename' not in st.session_state:
    st.session_state.eval_uploaded_filename = "" # Tên file đã tải
if 'last_eval_config' not in st.session_state:
    st.session_state.last_eval_config = {} # Lưu cấu hình của lần chạy cuối

st.subheader("Trạng thái Hệ thống Cơ bản")
init_ok = False
retriever_instance = None
g_embedding_model = None
g_reranking_model = None

# Sử dụng spinner để hiển thị quá trình khởi tạo
with st.spinner("Kiểm tra và khởi tạo tài nguyên cốt lõi (models, vectorDB, retriever)..."):
    try:
        # Tải các model cần thiết (cache resource sẽ hoạt động nếu đã tải trước đó)
        g_embedding_model = utils.load_embedding_model(config.embedding_model_name)
        g_reranking_model = utils.load_reranker_model(config.reranking_model_name)

        # Tải hoặc tạo VectorDB và Retriever (cache resource sẽ hoạt động)
        # Giả sử cached_load_or_create_components trả về (vector_db, hybrid_retriever)
        _, retriever_instance = data_loader.load_or_create_rag_components(g_embedding_model)

        # Kiểm tra tất cả thành phần đã sẵn sàng chưa
        if retriever_instance and g_embedding_model and g_reranking_model:
            init_ok = True
            st.success("✅ VectorDB, Retriever, Embedding Model, Reranker Model đã sẵn sàng.")
            logging.info("Core components initialized successfully.")
        else:
            missing_components = []
            if not retriever_instance: missing_components.append("Retriever")
            if not g_embedding_model: missing_components.append("Embedding Model")
            if not g_reranking_model: missing_components.append("Reranker Model")
            st.error(f"⚠️ Lỗi: Không thể khởi tạo các thành phần: {', '.join(missing_components)}.")
            logging.error(f"Failed to initialize components: {', '.join(missing_components)}.")

    except Exception as e:
        st.error(f"⚠️ Lỗi nghiêm trọng trong quá trình khởi tạo hệ thống: {e}")
        logging.exception("Critical error during system initialization.")

# Chỉ hiển thị phần còn lại nếu hệ thống cơ bản OK
if init_ok:
    st.subheader("Cấu hình Đánh giá")
    st.markdown("Đánh giá sẽ được thực hiện với các cấu hình **hiện tại** được chọn trong **Sidebar của trang Chatbot chính**.")

    # Lấy cấu hình hiện tại từ session state của trang Chatbot
    # Cung cấp giá trị mặc định nếu key không tồn tại
    current_gemini_model = st.session_state.get('selected_gemini_model', config.DEFAULT_GEMINI_MODEL)
    current_retrieval_mode = st.session_state.get('retrieval_mode', 'Tổng quát') # Mặc định 'Tổng quát'
    current_use_history = st.session_state.get('use_history_for_llm1', False) # Mặc định True

    # Hiển thị cấu hình sẽ sử dụng
    col1, col2, col3 = st.columns(3)
    with col1: st.info(f"**Chế độ Retrieval:** `{current_retrieval_mode}`")
    with col2: st.info(f"**Model Gemini (Variations):** `{current_gemini_model}`")
    with col3: st.info(f"**Sử dụng Lịch sử (LLM1):** `{current_use_history}`")

    # Tạo dict cấu hình sẽ truyền vào hàm đánh giá
    eval_config_dict = {
        'retrieval_mode': current_retrieval_mode,
        'use_history_for_llm1': False,
        'gemini_model_name': current_gemini_model, 
    }

    st.subheader("Tải Lên File Đánh giá")
    uploaded_file = st.file_uploader(
        "Chọn file JSON chứa dữ liệu đánh giá (định dạng: [{'query_id': ..., 'query': ..., 'relevant_chunk_ids': [...]}, ...])",
        type=["json"],
        key="eval_file_uploader", # Key để giữ trạng thái của uploader
        accept_multiple_files=False # Chỉ cho phép tải 1 file
    )

    # Xử lý file được tải lên
    if uploaded_file is not None:
        # Chỉ xử lý lại nếu tên file thay đổi (tránh load lại khi rerun không cần thiết)
        if uploaded_file.name != st.session_state.eval_uploaded_filename:
            try:
                # Đọc nội dung file
                file_content_bytes = uploaded_file.getvalue()
                # Decode UTF-8 và parse JSON
                eval_data_list = json.loads(file_content_bytes.decode('utf-8'))

                # Kiểm tra định dạng cơ bản của dữ liệu
                if not isinstance(eval_data_list, list) or not all(isinstance(item, dict) for item in eval_data_list):
                     raise ValueError("Dữ liệu không phải là một danh sách các dictionary.")
                if not all('query_id' in item and 'query' in item and 'relevant_chunk_ids' in item for item in eval_data_list):
                     raise ValueError("Mỗi mục phải chứa 'query_id', 'query', và 'relevant_chunk_ids'.")

                # Lưu dữ liệu và trạng thái vào session state
                st.session_state.eval_data = eval_data_list
                st.session_state.eval_uploaded_filename = uploaded_file.name
                # Reset trạng thái đánh giá cũ khi có file mới
                st.session_state.eval_run_completed = False
                st.session_state.eval_results_df = None
                st.session_state.last_eval_config = {} # Xóa config cũ
                st.success(f"Đã tải và xác thực thành công file '{uploaded_file.name}' chứa {len(eval_data_list)} câu hỏi.")
                logging.info(f"Successfully loaded and validated evaluation file: {uploaded_file.name}")

            except json.JSONDecodeError as e:
                st.error(f"Lỗi: File tải lên không phải là định dạng JSON hợp lệ. Chi tiết: {e}")
                logging.error(f"JSONDecodeError while processing uploaded file: {e}")
                # Reset trạng thái nếu file lỗi
                st.session_state.eval_data = None
                st.session_state.eval_uploaded_filename = ""
                st.session_state.eval_run_completed = False
                st.session_state.eval_results_df = None
            except ValueError as e:
                 st.error(f"Lỗi: Dữ liệu trong file không đúng định dạng yêu cầu. Chi tiết: {e}")
                 logging.error(f"ValueError (Invalid data format) while processing uploaded file: {e}")
                 # Reset trạng thái nếu file lỗi
                 st.session_state.eval_data = None
                 st.session_state.eval_uploaded_filename = ""
                 st.session_state.eval_run_completed = False
                 st.session_state.eval_results_df = None
            except Exception as e:
                st.error(f"Lỗi không xác định khi xử lý file: {e}")
                logging.exception("Unhandled error during evaluation file processing.")
                # Reset trạng thái nếu có lỗi khác
                st.session_state.eval_data = None
                st.session_state.eval_uploaded_filename = ""
                st.session_state.eval_run_completed = False
                st.session_state.eval_results_df = None
        # else:
        #     # Nếu tên file không đổi, không cần làm gì cả, dữ liệu đã có trong session state
        #     pass


    # Nếu đã có dữ liệu đánh giá trong session state
    if st.session_state.eval_data is not None:
        st.info(f"Đang sử dụng dữ liệu từ file: **{st.session_state.eval_uploaded_filename}** ({len(st.session_state.eval_data)} câu hỏi).")

        # Cho phép xem trước 5 dòng đầu
        if st.checkbox("Hiển thị dữ liệu mẫu (5 dòng đầu)", key="show_eval_data_preview"):
            st.dataframe(pd.DataFrame(st.session_state.eval_data).head())

        # Nút bắt đầu đánh giá
        if st.button("🚀 Bắt đầu Đánh giá", key="start_eval_button", help="Chạy đánh giá với cấu hình hiện tại và dữ liệu đã tải lên."):
            # Tải model Gemini đã chọn (không cache vì có thể thay đổi)
            with st.spinner(f"Đang tải model Gemini: {current_gemini_model}..."):
                # Cần đảm bảo hàm load_gemini_model xử lý lỗi API key hoặc tải model
                g_gemini_model = utils.load_gemini_model(current_gemini_model)

            if g_gemini_model:
                st.info(f"Model Gemini '{current_gemini_model}' đã sẵn sàng.")
                logging.info(f"Gemini model '{current_gemini_model}' loaded for evaluation.")
                with st.spinner("⏳ Đang chạy đánh giá... Quá trình này có thể mất vài phút hoặc lâu hơn tùy thuộc vào số lượng câu hỏi và cấu hình."):
                    start_eval_time = time.time()
                    # Gọi hàm đánh giá
                    results_df = run_retrieval_evaluation(
                        eval_data=st.session_state.eval_data,
                        hybrid_retriever=retriever_instance, # Đã khởi tạo ở trên
                        embedding_model=g_embedding_model, # Đã khởi tạo ở trên
                        reranking_model=g_reranking_model, # Đã khởi tạo ở trên
                        gemini_model=g_gemini_model, # Model Gemini vừa tải
                        eval_config=eval_config_dict # Dict cấu hình đã tạo
                    )
                    total_eval_time = time.time() - start_eval_time
                    st.success(f"Hoàn thành đánh giá sau {total_eval_time:.2f} giây.")
                    logging.info(f"Evaluation completed in {total_eval_time:.2f} seconds.")

                    # Lưu kết quả và trạng thái vào session state
                    st.session_state.eval_results_df = results_df
                    st.session_state.eval_run_completed = True
                    st.session_state.last_eval_config = eval_config_dict # Lưu lại cấu hình đã chạy
                    st.rerun() # Rerun để hiển thị kết quả
            else:
                st.error(f"Không thể tải model Gemini: {current_gemini_model}. Không thể chạy đánh giá.")
                logging.error(f"Failed to load Gemini model '{current_gemini_model}'. Evaluation cannot proceed.")

    # --- Hiển thị Kết quả (Nếu đã chạy xong) ---
    if st.session_state.eval_run_completed and st.session_state.eval_results_df is not None:
        st.subheader("Kết quả Đánh giá")
        # Lấy DataFrame kết quả từ session state
        detailed_results_df = st.session_state.eval_results_df
        # Lấy cấu hình của lần chạy này từ session state
        last_config = st.session_state.last_eval_config

        # Hiển thị lại cấu hình đã sử dụng cho lần đánh giá này
        st.markdown("**Cấu hình đã sử dụng cho lần đánh giá này:**")
        config_cols = st.columns(3)
        config_cols[0].metric("Chế độ Retrieval", last_config.get('retrieval_mode', 'N/A'))
        config_cols[1].metric("Model Gemini", last_config.get('gemini_model_name', 'N/A'))
        config_cols[2].metric("Sử dụng History LLM1", str(last_config.get('use_history_for_llm1', 'N/A')))


        # Tính toán metrics trung bình
        avg_metrics, num_eval, num_skipped_error = calculate_average_metrics(detailed_results_df)

        st.metric("Tổng số Queries trong File", len(detailed_results_df))
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Số Queries Đánh giá Hợp lệ", num_eval)
        col_res2.metric("Số Queries Bỏ qua / Lỗi", num_skipped_error)

        if avg_metrics:
            st.markdown("#### Metrics Trung bình (trên các queries hợp lệ)")
            k_values_display = [1, 3, 5, 10] # Các giá trị K để hiển thị
            cols_k = st.columns(len(k_values_display)) # Tạo cột cho mỗi K
            for idx, k in enumerate(k_values_display):
                with cols_k[idx]:
                    st.markdown(f"**K = {k}**")
                    # Sử dụng .get để tránh lỗi nếu key không tồn tại, mặc định là 0.0
                    st.text(f"Precision: {avg_metrics.get(f'avg_precision@{k}', 0.0):.4f}")
                    st.text(f"Recall:    {avg_metrics.get(f'avg_recall@{k}', 0.0):.4f}")
                    st.text(f"F1:        {avg_metrics.get(f'avg_f1@{k}', 0.0):.4f}")
                    st.text(f"MRR:       {avg_metrics.get(f'avg_mrr@{k}', 0.0):.4f}")
                    st.text(f"NDCG:      {avg_metrics.get(f'avg_ndcg@{k}', 0.0):.4f}")

            st.markdown("#### Thông tin Hiệu năng Trung bình (trên các queries hợp lệ)")
            col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
            col_perf1.metric("Avg Total Time/Query (s)", f"{avg_metrics.get('avg_processing_time', 0.0):.3f}")
            col_perf2.metric("Avg Variation Time (s)", f"{avg_metrics.get('avg_variation_time', 0.0):.3f}")
            col_perf3.metric("Avg Search Time (s)", f"{avg_metrics.get('avg_search_time', 0.0):.3f}")
            col_perf4.metric("Avg Rerank Time (s)", f"{avg_metrics.get('avg_rerank_time', 0.0):.3f}")

        else:
            st.warning("Không có query nào được đánh giá thành công, không thể tính metrics trung bình.")


        with st.expander("Xem Kết quả Chi tiết cho từng Query"):
            # Chọn các cột muốn hiển thị trong bảng kết quả chi tiết
            display_columns = [
                'query_id', 'query', 'status', 'retrieval_mode', 'use_history_llm1',
                'precision@1', 'recall@1', 'f1@1','mrr@1', 'ndcg@1', # Metrics @1
                'precision@3', 'recall@3', 'f1@3', 'mrr@3', 'ndcg@3', # Metrics @3
                'precision@5', 'recall@5', 'f1@5', 'mrr@5', 'ndcg@5', # Metrics @5
                'precision@10', 'recall@10', 'f1@10', 'mrr@10', 'ndcg@10', # Metrics @10
                'processing_time', 'variation_time', 'search_time', 'rerank_time',
                'num_variations_generated','num_unique_docs_found', 'num_docs_reranked',
                'retrieved_ids', 'relevant_ids', 'error_message' # Thêm error_message
            ]
            # Lọc ra những cột thực sự tồn tại trong DataFrame để tránh lỗi
            existing_display_columns = [col for col in display_columns if col in detailed_results_df.columns]
            # Hiển thị DataFrame với các cột đã chọn
            st.dataframe(detailed_results_df[existing_display_columns])

        st.subheader("Lưu Kết quả Chi tiết")
        try:
            # Chuẩn bị dữ liệu để tải về
            results_json = detailed_results_df.to_json(orient='records', indent=2, force_ascii=False)
            results_csv = detailed_results_df.to_csv(index=False).encode('utf-8') # CSV cần encode

            # Tạo tên file động dựa trên thời gian và cấu hình
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_suffix = last_config.get('retrieval_mode', 'unknown').lower().replace(' ', '_')
            hist_suffix = "hist" if last_config.get('use_history_for_llm1', False) else "nohist"
            model_suffix = last_config.get('gemini_model_name', 'gemini').split('/')[-1] # Lấy phần cuối tên model
            base_filename = f"eval_{mode_suffix}_{hist_suffix}_{model_suffix}_{timestamp}"
            fname_json = f"{base_filename}.json"
            fname_csv = f"{base_filename}.csv"

            # Tạo nút tải về
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="💾 Tải về JSON", data=results_json, file_name=fname_json, mime="application/json",
                    key="download_json_eval"
                )
            with col_dl2:
                st.download_button(
                    label="💾 Tải về CSV", data=results_csv, file_name=fname_csv, mime="text/csv",
                    key="download_csv_eval"
                )
        except Exception as e:
            st.error(f"Lỗi khi chuẩn bị file kết quả để tải về: {e}")
            logging.exception("Error preparing evaluation results for download.")

    # --- Quản lý Trạng thái Đánh giá ---
    st.markdown("---")
    st.subheader("Quản lý Trạng thái Đánh giá")
    if st.button("Xóa File Đã Tải và Kết Quả Đánh Giá", key="clear_eval_state", help="Xóa dữ liệu và kết quả đánh giá hiện tại khỏi bộ nhớ phiên."):
        # Reset tất cả các session state liên quan đến đánh giá
        st.session_state.eval_data = None
        st.session_state.eval_uploaded_filename = ""
        st.session_state.eval_run_completed = False
        st.session_state.eval_results_df = None
        st.session_state.last_eval_config = {}
        st.success("Đã xóa trạng thái đánh giá. Vui lòng tải lại file nếu muốn chạy lại.")
        logging.info("Evaluation state cleared.")
        time.sleep(1) # Chờ một chút để người dùng đọc thông báo
        st.rerun() # Rerun để cập nhật giao diện

# Trường hợp hệ thống cơ bản không khởi tạo được
else:
    st.warning("⚠️ Hệ thống cơ bản chưa sẵn sàng (Models, VectorDB, Retriever). Vui lòng kiểm tra lại trang Chatbot chính hoặc khởi động lại ứng dụng nếu cần.")
    logging.warning("Evaluation page cannot proceed as core components are not ready.")