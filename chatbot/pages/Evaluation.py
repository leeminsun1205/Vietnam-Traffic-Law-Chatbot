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
from retriever import HybridRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Các hàm tính toán metrics (giữ nguyên) ---
def precision_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0: return 0.0
    retrieved_at_k = retrieved_ids[:k]; relevant_set = set(relevant_ids)
    if not relevant_set: return 0.0 # Nếu không có relevant thì precision là 0
    intersect = set(retrieved_at_k) & relevant_set
    return len(intersect) / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    if not relevant_set: return 1.0 # Nếu không có relevant thì coi như đã tìm thấy tất cả (recall = 1)
    retrieved_at_k = retrieved_ids[:k]
    intersect = set(retrieved_at_k) & relevant_set
    return len(intersect) / len(relevant_set)

def f1_at_k(retrieved_ids, relevant_ids, k):
    prec = precision_at_k(retrieved_ids, relevant_ids, k); rec = recall_at_k(retrieved_ids, relevant_ids, k)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def mrr_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids);
    if not relevant_set: return 0.0 # Nếu không có relevant thì MRR là 0
    retrieved_at_k = retrieved_ids[:k] # Chỉ xét top K
    for rank, doc_id in enumerate(retrieved_at_k, 1):
        if doc_id in relevant_set: return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids);
    if not relevant_set: return 1.0 # Nếu không có relevant thì coi như list trả về là hoàn hảo (NDCG=1)
    retrieved_at_k = retrieved_ids[:k]; dcg = 0.0; idcg = 0.0
    # Calculate DCG@k
    for i, doc_id in enumerate(retrieved_at_k):
        # Giả sử relevancy là 1 nếu doc_id nằm trong relevant_set, ngược lại là 0
        relevance = 1.0 if doc_id in relevant_set else 0.0
        dcg += relevance / math.log2(i + 2) # i+2 vì rank bắt đầu từ 1 (log2(1+1))
    # Calculate IDCG@k
    num_relevant_in_total = len(relevant_set)
    # IDCG được tính bằng cách giả sử các tài liệu relevant nhất nằm ở đầu danh sách
    for i in range(min(k, num_relevant_in_total)):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def run_retrieval_evaluation(
    eval_data: list,
    hybrid_retriever: HybridRetriever,
    embedding_model,
    reranking_model, # Có thể là None nếu không dùng rerank
    gemini_model,
    eval_config: dict # Chứa retrieval_query_mode, retrieval_method, use_reranker, use_history_for_llm1
    ):

    results_list = []
    k_values = [1, 3, 5, 10] # Các giá trị K để tính metrics

    # --- Lấy cấu hình từ eval_config ---
    retrieval_query_mode = eval_config.get('retrieval_query_mode', 'Tổng quát')
    retrieval_method = eval_config.get('retrieval_method', 'hybrid')
    use_reranker = eval_config.get('use_reranker', True)
    use_history_llm1 = eval_config.get('use_history_for_llm1', True) # Sửa key
    dummy_history = [{"role": "user", "content": "..."}] if use_history_llm1 else None

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
            "use_history_llm1": use_history_llm1,
            "status": "error", "retrieved_ids": [], "relevant_ids": list(relevant_chunk_ids),
            "processing_time": 0.0, 'summarizing_query': '',
            'variation_time': 0.0, 'search_time': 0.0, 'rerank_time': 0.0,
            'num_variations_generated': 0, 'num_unique_docs_found': 0, 'num_docs_reranked': 0,
            'num_retrieved_before_rerank': 0, 'num_retrieved_after_rerank': 0
        }
        for k in k_values:
            query_metrics[f'precision@{k}'] = 0.0; query_metrics[f'recall@{k}'] = 0.0
            query_metrics[f'f1@{k}'] = 0.0; query_metrics[f'mrr@{k}'] = 0.0; query_metrics[f'ndcg@{k}'] = 0.0

        try:
            # Bước 1: Tạo variations/summarizing query (luôn chạy)
            variation_start = time.time()
            relevance_status, _, all_queries, summarizing_query = utils.generate_query_variations(
                original_query=original_query, gemini_model=gemini_model,
                chat_history=dummy_history, num_variations=config.NUM_QUERY_VARIATIONS
            )
            query_metrics["variation_time"] = time.time() - variation_start
            query_metrics["summarizing_query"] = summarizing_query
            query_metrics["num_variations_generated"] = len(all_queries) - 1

            if relevance_status == 'invalid':
                query_metrics["status"] = "skipped_irrelevant"
                # Các metrics khác giữ nguyên giá trị 0
                query_metrics["processing_time"] = time.time() - start_time
                results_list.append(query_metrics)
                progress_bar.progress((i + 1) / total_items)
                logging.info(f"QID {query_id} skipped as irrelevant.")
                continue

            # --- Bước 2: Xác định query(s) để tìm kiếm ---
            queries_to_search = []
            st.write(queries_to_search)
            if retrieval_query_mode == 'Đơn giản': queries_to_search = [original_query]
            elif retrieval_query_mode == 'Tổng quát': queries_to_search = [summarizing_query]
            elif retrieval_query_mode == 'Sâu': queries_to_search = all_queries

            # --- Bước 3: Thực hiện Retrieval ---
            collected_docs_data = {}
            search_start = time.time()
            st.write(queries_to_search)
            for q_variant in queries_to_search:
                if not q_variant: continue # Bỏ qua nếu query rỗng
                # Gọi hàm search mới của retriever
                search_results = hybrid_retriever.search(
                    q_variant, embedding_model,
                    method=retrieval_method,
                    k=config.VECTOR_K_PER_QUERY # Lấy K đủ lớn cho bước sau
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
            sort_reverse = (retrieval_method != 'dense') # Dense sắp xếp score (distance) tăng dần
            retrieved_docs_list.sort(key=lambda x: x.get('score', 0 if sort_reverse else float('inf')), reverse=sort_reverse)
            query_metrics["num_retrieved_before_rerank"] = len(retrieved_docs_list)


            # --- Bước 4: Re-ranking (Nếu bật) ---
            final_docs_for_metrics = [] # Danh sách kết quả cuối cùng để tính metrics
            rerank_start = time.time()

            if use_reranker and retrieved_docs_list:
                query_for_reranking = summarizing_query if summarizing_query else original_query
                docs_to_rerank = retrieved_docs_list[:config.MAX_DOCS_FOR_RERANK]
                query_metrics["num_docs_reranked"] = len(docs_to_rerank)
                logging.debug(f"QID {query_id}: Reranking {len(docs_to_rerank)} docs with query: '{query_for_reranking[:50]}...'")

                # Đảm bảo input cho rerank đúng định dạng list of dicts {'doc': ..., 'index': ...}
                rerank_input = [{'doc': item['doc'], 'index': item['index']} for item in docs_to_rerank]

                reranked_results = utils.rerank_documents(
                    query_for_reranking, rerank_input, reranking_model
                )
                # Lấy top K kết quả sau rerank
                final_docs_for_metrics = reranked_results[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                query_metrics["rerank_time"] = time.time() - rerank_start
                logging.debug(f"QID {query_id}: Reranking finished, selected {len(final_docs_for_metrics)} docs.")

            elif retrieved_docs_list: # Không rerank, lấy trực tiếp từ retrieval
                final_docs_for_metrics = retrieved_docs_list[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                query_metrics["rerank_time"] = 0.0 # Không tốn thời gian rerank
                query_metrics["num_docs_reranked"] = 0 # Không có docs nào được rerank
                logging.debug(f"QID {query_id}: Skipped reranking, taking top {len(final_docs_for_metrics)} retrieval results.")
            else: # Không có kết quả retrieval
                 query_metrics["rerank_time"] = 0.0
                 query_metrics["num_docs_reranked"] = 0
                 logging.debug(f"QID {query_id}: No docs to rerank or select.")

            query_metrics["num_retrieved_after_rerank"] = len(final_docs_for_metrics)

            # --- Bước 5: Lấy IDs và Tính Metrics ---
            retrieved_ids = []
            # Cần lấy 'id' hoặc 'chunk_id' từ final_docs_for_metrics
            for res in final_docs_for_metrics:
                doc_data = res.get('doc', {})
                chunk_id = None
                if isinstance(doc_data, dict):
                    chunk_id = doc_data.get('id') # Ưu tiên key 'id'
                    if not chunk_id:
                        metadata = doc_data.get('metadata', {})
                        if isinstance(metadata, dict):
                            chunk_id = metadata.get('id') or metadata.get('chunk_id')
                if chunk_id:
                    retrieved_ids.append(str(chunk_id)) # Đảm bảo là string

            query_metrics["retrieved_ids"] = retrieved_ids
            logging.debug(f"QID {query_id}: Final retrieved IDs for metrics (top {len(retrieved_ids)}): {retrieved_ids}")

            query_metrics["status"] = "evaluated"
            # Tính toán metrics
            for k in k_values:
                query_metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'f1@{k}'] = f1_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'mrr@{k}'] = mrr_at_k(retrieved_ids, relevant_chunk_ids, k)
                query_metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved_ids, relevant_chunk_ids, k)
                # logging.debug(f"  Metrics @{k}: P={query_metrics[f'precision@{k}']:.4f}, R={query_metrics[f'recall@{k}']:.4f}, F1={query_metrics[f'f1@{k}']:.4f}, MRR={query_metrics[f'mrr@{k}']:.4f}, NDCG={query_metrics[f'ndcg@{k}']:.4f}")


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
    k_values = [1, 3, 5, 10]
    metric_keys_k = [f'{m}@{k}' for k in k_values for m in ['precision', 'recall', 'f1', 'mrr', 'ndcg']]
    timing_keys = ['processing_time', 'variation_time', 'search_time', 'rerank_time']
    # Thêm các keys số lượng mới
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
Sử dụng cấu hình hiện tại từ trang Chatbot chính.
""")

# --- Khởi tạo hoặc kiểm tra Session State ---
if 'eval_data' not in st.session_state: st.session_state.eval_data = None
if 'eval_results_df' not in st.session_state: st.session_state.eval_results_df = None
if 'eval_run_completed' not in st.session_state: st.session_state.eval_run_completed = False
if 'eval_uploaded_filename' not in st.session_state: st.session_state.eval_uploaded_filename = ""
if 'last_eval_config' not in st.session_state: st.session_state.last_eval_config = {}

st.subheader("Trạng thái Hệ thống Cơ bản")
init_ok = False
retriever_instance = None
g_embedding_model = None
g_reranking_model = None

with st.spinner("Kiểm tra và khởi tạo tài nguyên cốt lõi..."):
    try:
        g_embedding_model = utils.load_embedding_model(config.embedding_model_name)
        g_reranking_model = utils.load_reranker_model(config.reranking_model_name)
        _, retriever_instance = data_loader.load_or_create_rag_components(g_embedding_model)

        if retriever_instance and g_embedding_model: # Reranker có thể không cần nếu use_reranker=False
            init_ok = True
            st.success("✅ VectorDB, Retriever, Embedding Model, Reranker Model đã sẵn sàng.")
            logging.info("Core components initialized successfully for evaluation.")
            # Ghi chú về reranker model nếu không tải được nhưng init vẫn ok
            if not g_reranking_model:
                 st.warning("⚠️ Không tải được Reranker Model. Chức năng rerank sẽ không hoạt động.")
                 logging.warning("Reranker model failed to load, reranking will be disabled if attempted.")
        else:
            missing = [comp for comp, loaded in [("Retriever/VectorDB", retriever_instance), ("Embedding Model", g_embedding_model)] if not loaded]
            st.error(f"⚠️ Lỗi khởi tạo: {', '.join(missing)}.")
            logging.error(f"Failed to initialize components: {', '.join(missing)}.")

    except Exception as e:
        st.error(f"⚠️ Lỗi nghiêm trọng khi khởi tạo hệ thống: {e}")
        logging.exception("Critical error during system initialization for evaluation.")

if init_ok:
    st.subheader("Cấu hình Đánh giá")
    st.markdown("Đánh giá sẽ sử dụng cấu hình **hiện tại** từ **Sidebar của trang Chatbot**.")

    # --- Lấy cấu hình hiện tại từ session state của trang Chatbot ---
    current_gemini_model = st.session_state.get('selected_gemini_model', config.DEFAULT_GEMINI_MODEL)
    current_retrieval_query_mode = st.session_state.get('retrieval_query_mode', 'Tổng quát')
    current_use_history_llm1 = st.session_state.get('use_history_for_llm1', True)
    current_retrieval_method = st.session_state.get('retrieval_method', 'hybrid')
    current_use_reranker = st.session_state.get('use_reranker', True)

    # --- Hiển thị cấu hình sẽ sử dụng ---
    st.markdown("**Cấu hình sẽ sử dụng:**")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        st.info(f"**Nguồn Query:** `{current_retrieval_query_mode}`")
        st.info(f"**Ret. Method:** `{current_retrieval_method}`")
    with cfg_col2:
        st.info(f"**Reranker:** `{'Bật' if current_use_reranker else 'Tắt'}`")
        st.info(f"**History LLM1:** `{'Bật' if current_use_history_llm1 else 'Tắt'}`")
    with cfg_col3:
        st.info(f"**Gemini Model:** `{current_gemini_model}`")


    # Tạo dict cấu hình cho hàm đánh giá
    eval_config_dict = {
        'retrieval_query_mode': current_retrieval_query_mode,
        'retrieval_method': current_retrieval_method,
        'use_reranker': current_use_reranker,
        'use_history_for_llm1': current_use_history_llm1,
        'gemini_model_name': current_gemini_model,
        # Thêm tên model khác nếu cần lưu vào kết quả
        'embedding_model_name': config.embedding_model_name,
        'reranker_model_name': config.reranking_model_name if current_use_reranker else None,
    }
    # Kiểm tra nếu reranker bị tắt nhưng model không tải được
    reranker_model_to_pass = g_reranking_model if current_use_reranker else None
    if current_use_reranker and not g_reranking_model:
         st.warning("Reranker đang được bật trong cấu hình nhưng model reranker không tải được. Reranking sẽ bị bỏ qua.")
         eval_config_dict['use_reranker'] = False # Ghi đè config nếu model không có
         eval_config_dict['reranker_model_name'] = "FAILED_TO_LOAD"
         reranker_model_to_pass = None


    st.subheader("Tải Lên File Đánh giá")
    uploaded_file = st.file_uploader(
        "Chọn file JSON dữ liệu đánh giá...", type=["json"], key="eval_file_uploader"
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.eval_uploaded_filename:
            try:
                eval_data_list = json.loads(uploaded_file.getvalue().decode('utf-8'))
                # Thêm kiểm tra định dạng cơ bản ở đây nếu cần
                st.session_state.eval_data = eval_data_list
                st.session_state.eval_uploaded_filename = uploaded_file.name
                st.session_state.eval_run_completed = False
                st.session_state.eval_results_df = None
                st.session_state.last_eval_config = {}
                st.success(f"Đã tải file '{uploaded_file.name}' ({len(eval_data_list)} câu hỏi).")
                logging.info(f"Loaded evaluation file: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Lỗi xử lý file JSON: {e}")
                logging.exception("Error processing uploaded JSON file.")
                st.session_state.eval_data = None; st.session_state.eval_uploaded_filename = ""
                st.session_state.eval_run_completed = False; st.session_state.eval_results_df = None

    if st.session_state.eval_data is not None:
        st.info(f"Sẵn sàng đánh giá với dữ liệu từ: **{st.session_state.eval_uploaded_filename}**.")

        if st.checkbox("Hiển thị dữ liệu mẫu (5 dòng)", key="show_eval_data_preview"):
            st.dataframe(pd.DataFrame(st.session_state.eval_data).head())

        if st.button("🚀 Bắt đầu Đánh giá", key="start_eval_button"):
            with st.spinner(f"Đang tải model Gemini: {current_gemini_model}..."):
                g_gemini_model = utils.load_gemini_model(current_gemini_model)

            if g_gemini_model:
                st.info(f"Model Gemini '{current_gemini_model}' đã sẵn sàng.")
                with st.spinner("⏳ Đang chạy đánh giá..."):
                    start_eval_time = time.time()
                    results_df = run_retrieval_evaluation(
                        eval_data=st.session_state.eval_data,
                        hybrid_retriever=retriever_instance,
                        embedding_model=g_embedding_model,
                        reranking_model=reranker_model_to_pass, # Truyền model reranker (hoặc None)
                        gemini_model=g_gemini_model,
                        eval_config=eval_config_dict # Truyền dict config
                    )
                    st.write('HHAHAH')
                    total_eval_time = time.time() - start_eval_time
                    st.success(f"Hoàn thành đánh giá sau {total_eval_time:.2f} giây.")
                    logging.info(f"Evaluation completed in {total_eval_time:.2f} seconds.")

                    st.session_state.eval_results_df = results_df
                    st.session_state.eval_run_completed = True
                    st.session_state.last_eval_config = eval_config_dict # Lưu config đã chạy
                    st.rerun()
            else:
                st.error(f"Không thể tải model Gemini: {current_gemini_model}.")
                logging.error(f"Failed to load Gemini model '{current_gemini_model}'.")

    # --- Hiển thị Kết quả ---
    if st.session_state.eval_run_completed and st.session_state.eval_results_df is not None:
        st.subheader("Kết quả Đánh giá")
        detailed_results_df = st.session_state.eval_results_df
        last_config = st.session_state.last_eval_config

        # --- Hiển thị lại cấu hình đã chạy ---
        st.markdown("**Cấu hình đã sử dụng:**")
        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
        cfg_col1.metric("Nguồn Query", last_config.get('retrieval_query_mode', 'N/A'))
        cfg_col2.metric("Ret. Method", last_config.get('retrieval_method', 'N/A'))
        cfg_col3.metric("Reranker", "Bật" if last_config.get('use_reranker', False) else "Tắt")
        cfg_col4.metric("History LLM1", "Bật" if last_config.get('use_history_for_llm1', False) else "Tắt")
        # Thêm thông tin model nếu có trong config
        st.caption(f"Gemini: `{last_config.get('gemini_model_name', 'N/A')}`, Embedding: `{last_config.get('embedding_model_name', 'N/A')}`, Reranker: `{last_config.get('reranker_model_name', 'N/A')}`")


        avg_metrics, num_eval, num_skipped_error = calculate_average_metrics(detailed_results_df)

        st.metric("Tổng số Queries", len(detailed_results_df))
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Queries Đánh giá Hợp lệ", num_eval)
        col_res2.metric("Queries Bỏ qua / Lỗi", num_skipped_error)

        if avg_metrics:
            st.markdown("#### Metrics Trung bình @K (trên các queries hợp lệ)")
            k_values_display = [1, 3, 5, 10]
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
            # --- Cập nhật các cột hiển thị ---
            display_columns = [
                'query_id', 'query', 'status',
                'retrieval_query_mode','retrieval_method', 'use_reranker', 'use_history_llm1', # Cấu hình
                'precision@1', 'recall@1', 'f1@1','mrr@1', 'ndcg@1',
                'precision@3', 'recall@3', 'f1@3', 'mrr@3', 'ndcg@3',
                'precision@5', 'recall@5', 'f1@5', 'mrr@5', 'ndcg@5',
                'precision@10', 'recall@10', 'f1@10', 'mrr@10', 'ndcg@10', # Thêm @10
                'processing_time', 'variation_time', 'search_time', 'rerank_time', # Thời gian
                'num_variations_generated','num_unique_docs_found', 'num_retrieved_before_rerank','num_docs_reranked', 'num_retrieved_after_rerank', # Số lượng
                'retrieved_ids', 'relevant_ids', 'summarizing_query', 'error_message' # Thông tin khác
            ]
            existing_display_columns = [col for col in display_columns if col in detailed_results_df.columns]
            st.dataframe(detailed_results_df[existing_display_columns])

        st.subheader("Lưu Kết quả Chi tiết")
        try:
            results_json = detailed_results_df.to_json(orient='records', indent=2, force_ascii=False)
            results_csv = detailed_results_df.to_csv(index=False).encode('utf-8')

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # --- Cập nhật tên file để bao gồm cấu hình mới ---
            qmode_suffix = last_config.get('retrieval_query_mode', 'na').lower()[:3] # Lấy 3 chữ cái đầu
            method_suffix = last_config.get('retrieval_method', 'na').lower()
            rerank_suffix = "rr" if last_config.get('use_reranker', False) else "norr"
            hist_suffix = "hist" if last_config.get('use_history_for_llm1', False) else "nohist"
            model_suffix = last_config.get('gemini_model_name', 'gemini').split('/')[-1].replace('.','-')[:15] # Giới hạn độ dài tên model

            base_filename = f"eval_{qmode_suffix}_{method_suffix}_{rerank_suffix}_{hist_suffix}_{model_suffix}_{timestamp}"
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
        st.success("Đã xóa trạng thái đánh giá.")
        logging.info("Evaluation state cleared.")
        time.sleep(1); st.rerun()

else:
    st.warning("⚠️ Hệ thống cơ bản chưa sẵn sàng. Vui lòng kiểm tra lỗi và khởi động lại.")
    logging.warning("Evaluation page cannot proceed as core components are not ready.")