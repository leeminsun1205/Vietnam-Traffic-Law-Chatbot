# pages/Evaluation.py
import time
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from model_loader import load_gemini_model, initialize_evaluation_page_resources
from reranker import rerank_documents
from utils import (
    generate_query_variations,
    precision_at_k, recall_at_k, f1_at_k, mrr_at_k, ndcg_at_k,
    calculate_average_metrics
)

# --- Hàm chỉ để sinh và thu thập biến thể ---
def generate_and_collect_variations_only(eval_data: list, gemini_model_object, num_variations: int) -> dict:
    """
    Sinh biến thể cho tất cả các query trong eval_data và trả về một dict
    với query_id làm key.
    """
    collected_variations_output = {}
    progress_bar_var_only = st.progress(0)
    status_text_var_only = st.empty()
    total_items_var_only = len(eval_data)
    queries_per_batch_var_only = 15
    wait_time_seconds_var_only = 61

    if not gemini_model_object:
        st.error("Lỗi: Gemini model không được cung cấp cho việc sinh biến thể.")
        return {}

    for i, item_data_var in enumerate(eval_data):
        if i > 0 and i % queries_per_batch_var_only == 0:
            pause_msg_var = f"Đang sinh biến thể {i}/{total_items_var_only}. Tạm dừng {wait_time_seconds_var_only} giây..."
            status_text_var_only.text(pause_msg_var)
            time.sleep(wait_time_seconds_var_only)
            status_text_var_only.text(f"Tiếp tục sinh biến thể cho query {i+1}/{total_items_var_only}...")

        query_id_item_var = item_data_var.get("query_id", f"item_{i+1}")
        original_query_item_var = item_data_var.get("query")
        status_text_var_only.text(f"Đang sinh biến thể cho query {i+1}/{total_items_var_only}: {query_id_item_var}...")

        try:
            relevance_status_gen, direct_ans_gen, all_q_list_gen, summarizing_q_str_gen = generate_query_variations(
                original_query=original_query_item_var,
                gemini_model=gemini_model_object,
                chat_history=None,
                num_variations=num_variations
            )
            collected_variations_output[query_id_item_var] = {
                "original_query": original_query_item_var,
                "relevance_status": relevance_status_gen,
                "direct_answer_if_invalid": direct_ans_gen,
                "all_queries": all_q_list_gen,
                "summarizing_query": summarizing_q_str_gen,
                "llm_model_used_for_generation": gemini_model_object.model_name
            }
        except Exception as e_gen_var_item:
            st.warning(f"Lỗi khi sinh biến thể cho query_id '{query_id_item_var}': {e_gen_var_item}. Mục này sẽ có thông tin lỗi.")
            collected_variations_output[query_id_item_var] = {
                "original_query": original_query_item_var,
                "relevance_status": "error_generating_variations",
                "direct_answer_if_invalid": "",
                "all_queries": [original_query_item_var],
                "summarizing_query": original_query_item_var,
                "error_message": str(e_gen_var_item),
                "llm_model_used_for_generation": gemini_model_object.model_name
            }
        finally:
            progress_bar_var_only.progress((i + 1) / total_items_var_only)

    status_text_var_only.success(f"Hoàn thành sinh biến thể cho {total_items_var_only} câu hỏi!")
    return collected_variations_output

# --- Hàm chạy đánh giá chính ---
def run_retrieval_evaluation(
    eval_data: list,
    retriever_instance_for_eval,
    embedding_model_object_for_eval,
    reranking_model_object_for_eval,
    gemini_model_object_for_eval,
    eval_config_params: dict,
    preloaded_query_variations: dict = None
    ):
    results_list = []
    k_values_metrics = [3, 5, 10]

    retrieval_query_mode_eval = eval_config_params.get('retrieval_query_mode', 'Mở rộng')
    retrieval_method_eval = eval_config_params.get('retrieval_method', 'Kết hợp')
    hybrid_component_mode_eval = eval_config_params.get('eval_hybrid_component_mode', '2 Dense + 1 Sparse')
    selected_reranker_name_eval_run = eval_config_params.get('selected_reranker_model_name', 'Không sử dụng')
    use_reranker_eval_run = reranking_model_object_for_eval is not None and selected_reranker_name_eval_run != 'Không sử dụng'
    variation_mode_run = eval_config_params.get('variation_mode_used', "Luôn sinh mới (qua LLM)")

    progress_bar_eval = st.progress(0)
    status_text_area_eval = st.empty()
    total_items_eval = len(eval_data)
    queries_per_batch_eval = 15
    if variation_mode_run == "Sử dụng file biến thể đã tải lên":
        wait_time_seconds_eval = 0
    else:
        wait_time_seconds_eval = 60

    for i, item_eval in enumerate(eval_data):
        if i > 0 and i % queries_per_batch_eval == 0:
            pause_msg_eval = f"Đã xử lý {i}/{total_items_eval} queries. Tạm dừng {wait_time_seconds_eval} giây..."
            status_text_area_eval.text(pause_msg_eval)
            time.sleep(wait_time_seconds_eval)

        query_id_eval = item_eval.get("query_id", f"item_{i+1}")
        original_query_eval = item_eval.get("query")
        relevant_chunk_ids_eval = set(str(cid) for cid in item_eval.get("relevant_chunk_ids", []))

        emb_name_disp_eval = eval_config_params.get('embedding_model_name', 'N/A').split('/')[-1]
        rer_name_disp_eval = selected_reranker_name_eval_run.split('/')[-1] if selected_reranker_name_eval_run != 'Không sử dụng' else "Tắt"
        var_mode_disp = variation_mode_run.split('(')[0].strip() if '(' in variation_mode_run else variation_mode_run

        status_text_area_eval.text(
            f"Đang xử lý query {i+1}/{total_items_eval}: {query_id_eval}\n"
            f"(Emb: {emb_name_disp_eval}, QueryMode: {retrieval_query_mode_eval}, "
            f"Method: {retrieval_method_eval}, Reranker: {rer_name_disp_eval}, VarMode: {var_mode_disp})"
        )

        start_time_eval_q = time.time()
        query_metrics_dict = {
            "query_id": query_id_eval, "query": original_query_eval,
            "embedding_model_name": eval_config_params.get('embedding_model_name', 'N/A'),
            "retrieval_query_mode": retrieval_query_mode_eval,
            "retrieval_method": retrieval_method_eval,
            "selected_reranker_model": selected_reranker_name_eval_run,
            "variation_mode_run": variation_mode_run,
            "variation_source": "N/A",
            "llm_model_for_variation": "N/A",
            "status": "error_default", "retrieved_ids": [], "relevant_ids": list(relevant_chunk_ids_eval),
            "processing_time": 0.0, 'summarizing_query': '',
            'variation_time': 0.0, 'search_time': 0.0, 'rerank_time': 0.0,
            'num_variations_generated': 0, 'num_unique_docs_found': 0, 'num_docs_reranked': 0,
            'num_retrieved_before_rerank': 0, 'num_retrieved_after_rerank': 0, 'error_message': ''
        }
        for k_val_m in k_values_metrics:
            query_metrics_dict[f'precision@{k_val_m}'] = 0.0
            query_metrics_dict[f'recall@{k_val_m}'] = 0.0
            query_metrics_dict[f'f1@{k_val_m}'] = 0.0
            query_metrics_dict[f'mrr@{k_val_m}'] = 0.0
            query_metrics_dict[f'ndcg@{k_val_m}'] = 0.0

        try:
            var_start_processing_time = time.time()
            relevance_status_q_eval = "valid"
            all_queries_q_eval = [original_query_eval]
            summarizing_q_q_eval = original_query_eval

            if preloaded_query_variations and query_id_eval in preloaded_query_variations:
                query_metrics_dict["variation_source"] = "File"
                var_data = preloaded_query_variations[query_id_eval]
                relevance_status_q_eval = var_data.get("relevance_status", "valid")
                all_queries_q_eval = var_data.get("all_queries", [original_query_eval])
                summarizing_q_q_eval = var_data.get("summarizing_query", original_query_eval)
                query_metrics_dict["llm_model_for_variation"] = var_data.get("llm_model_used_for_generation", "From File")
                if relevance_status_q_eval == "error_generating_variations":
                    st.sidebar.warning(f"Query ID {query_id_eval}: Lỗi khi tạo biến thể từ file trước đó. Sử dụng fallback.")
            elif gemini_model_object_for_eval:
                query_metrics_dict["variation_source"] = "LLM"
                query_metrics_dict["llm_model_for_variation"] = gemini_model_object_for_eval.model_name
                relevance_status_q_eval, _, all_queries_q_eval, summarizing_q_q_eval = generate_query_variations(
                    original_query=original_query_eval,
                    gemini_model=gemini_model_object_for_eval,
                    chat_history=None,
                    num_variations=config.NUM_QUERY_VARIATIONS
                )
            else:
                query_metrics_dict["variation_source"] = "Fallback Original"
                query_metrics_dict["llm_model_for_variation"] = "N/A"

            query_metrics_dict["variation_time"] = time.time() - var_start_processing_time
            query_metrics_dict["summarizing_query"] = summarizing_q_q_eval
            query_metrics_dict["num_variations_generated"] = len(all_queries_q_eval) - 1 if isinstance(all_queries_q_eval, list) and len(all_queries_q_eval) > 0 else 0

            if relevance_status_q_eval == 'invalid' or relevance_status_q_eval == 'error_generating_variations':
                query_metrics_dict["status"] = "skipped_irrelevant" if relevance_status_q_eval == 'invalid' else "skipped_variation_error"
            else:
                queries_to_search_eval_run = []
                if retrieval_query_mode_eval == 'Đơn giản': 
                    queries_to_search_eval_run = [original_query_eval]
                elif retrieval_query_mode_eval == 'Mở rộng': 
                    queries_to_search_eval_run = [summarizing_q_q_eval] if summarizing_q_q_eval else [original_query_eval]
                elif retrieval_query_mode_eval == 'Đa dạng': 
                    queries_to_search_eval_run = all_queries_q_eval if all_queries_q_eval else [original_query_eval]
                collected_docs_data_eval_run = {}
                search_start_eval_q_run = time.time()
            
                for q_var_eval_run in queries_to_search_eval_run:
                    if not q_var_eval_run: continue
                    use_two_dense_eval_hybrid_runtime = (st.session_state.eval_hybrid_component_mode == "2 Dense + 1 Sparse") 
                    eval_secondary_emb_obj_runtime = None
                    eval_secondary_vector_db_runtime = None
                    if retrieval_method_eval == 'Kết hợp': # Điều kiện mới
                        eval_selected_secondary_emb_name_runtime = eval_config_params.get("secondary_embedding_model_name")
                        if use_two_dense_eval_hybrid_runtime:
                            if eval_selected_secondary_emb_name_runtime:
                                eval_secondary_emb_obj_runtime = st.session_state.eval_loaded_embedding_models.get(eval_selected_secondary_emb_name_runtime)
                                secondary_rag_comps_eval_runtime = st.session_state.eval_rag_components_per_embedding_model.get(eval_selected_secondary_emb_name_runtime)
                                if secondary_rag_comps_eval_runtime:
                                    eval_secondary_vector_db_runtime = secondary_rag_comps_eval_runtime[0]

                                if not (eval_secondary_emb_obj_runtime and eval_secondary_vector_db_runtime and eval_secondary_emb_obj_runtime != embedding_model_object_for_eval):
                                    use_two_dense_eval_hybrid_runtime = False 
                                else:
                                    use_two_dense_eval_hybrid_runtime = True
                    st.write(eval_secondary_emb_obj_runtime, eval_secondary_emb_obj_runtime)

                    search_results_eval_run = retriever_instance_for_eval.search(
                        q_var_eval_run,
                        embedding_model_object_for_eval, 
                        method=retrieval_method_eval,
                        k=config.VECTOR_K_PER_QUERY if retrieval_method_eval != 'Kết hợp' else config.HYBRID_K_PER_QUERY,
                        secondary_embedding_model=eval_secondary_emb_obj_runtime if use_two_dense_eval_hybrid_runtime else None,
                        secondary_vector_db=eval_secondary_vector_db_runtime if use_two_dense_eval_hybrid_runtime else None,
                        use_two_dense_if_hybrid=use_two_dense_eval_hybrid_runtime 
                    )

                    for res_item_eval_run in search_results_eval_run:
                        doc_idx_eval_run = res_item_eval_run.get('index')
                        if isinstance(doc_idx_eval_run, int) and doc_idx_eval_run >= 0 and doc_idx_eval_run not in collected_docs_data_eval_run:
                            collected_docs_data_eval_run[doc_idx_eval_run] = res_item_eval_run
                query_metrics_dict["search_time"] = time.time() - search_start_eval_q_run
                query_metrics_dict["num_unique_docs_found"] = len(collected_docs_data_eval_run)

                retrieved_docs_list_eval_run = list(collected_docs_data_eval_run.values())
                sort_reverse_eval_run = (retrieval_method_eval != 'Ngữ nghĩa')
                retrieved_docs_list_eval_run.sort(key=lambda x: x.get('score', 0 if sort_reverse_eval_run else float('inf')), reverse=sort_reverse_eval_run)
                query_metrics_dict["num_retrieved_before_rerank"] = len(retrieved_docs_list_eval_run)

                final_docs_for_metrics_eval_run = []
                rerank_start_eval_q_time_run = time.time()

                if use_reranker_eval_run and retrieved_docs_list_eval_run:
                    query_for_reranking_eval_run = summarizing_q_q_eval if summarizing_q_q_eval else original_query_eval
                    docs_to_rerank_input_eval_run = retrieved_docs_list_eval_run[:config.MAX_DOCS_FOR_RERANK]
                    query_metrics_dict["num_docs_reranked"] = len(docs_to_rerank_input_eval_run)
                    rerank_input_fmt_eval_run = [{'doc': item_d_run['doc'], 'index': item_d_run['index']} for item_d_run in docs_to_rerank_input_eval_run]

                    reranked_results_eval_run = rerank_documents(
                        query_for_reranking_eval_run,
                        rerank_input_fmt_eval_run,
                        reranking_model_object_for_eval
                    )
                    final_docs_for_metrics_eval_run = reranked_results_eval_run[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                    query_metrics_dict["rerank_time"] = time.time() - rerank_start_eval_q_time_run
                elif retrieved_docs_list_eval_run:
                    temp_final_docs_eval_run = retrieved_docs_list_eval_run[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                    final_docs_for_metrics_eval_run = [
                        {'doc': item_run['doc'], 'score': item_run.get('score'), 'original_index': item_run['index']}
                        for item_run in temp_final_docs_eval_run
                    ]
                    query_metrics_dict["rerank_time"] = 0.0
                    query_metrics_dict["num_docs_reranked"] = 0
                else:
                    query_metrics_dict["rerank_time"] = 0.0
                    query_metrics_dict["num_docs_reranked"] = 0

                query_metrics_dict["num_retrieved_after_rerank"] = len(final_docs_for_metrics_eval_run)
                retrieved_ids_eval_q_run = []
                for res_final_eval_run in final_docs_for_metrics_eval_run:
                    doc_data_eval_run = res_final_eval_run.get('doc', {})
                    chunk_id_eval_run = None
                    if isinstance(doc_data_eval_run, dict):
                        chunk_id_eval_run = doc_data_eval_run.get('id') or \
                                        doc_data_eval_run.get('chunk_id') or \
                                        doc_data_eval_run.get('metadata', {}).get('id') or \
                                        doc_data_eval_run.get('metadata', {}).get('chunk_id')
                    if chunk_id_eval_run is not None:
                        retrieved_ids_eval_q_run.append(str(chunk_id_eval_run))

                query_metrics_dict["retrieved_ids"] = retrieved_ids_eval_q_run
                query_metrics_dict["status"] = "evaluated"
                for k_val_m_calc_run in k_values_metrics:
                    query_metrics_dict[f'precision@{k_val_m_calc_run}'] = precision_at_k(retrieved_ids_eval_q_run, relevant_chunk_ids_eval, k_val_m_calc_run)
                    query_metrics_dict[f'recall@{k_val_m_calc_run}'] = recall_at_k(retrieved_ids_eval_q_run, relevant_chunk_ids_eval, k_val_m_calc_run)
                    query_metrics_dict[f'f1@{k_val_m_calc_run}'] = f1_at_k(retrieved_ids_eval_q_run, relevant_chunk_ids_eval, k_val_m_calc_run)
                    query_metrics_dict[f'mrr@{k_val_m_calc_run}'] = mrr_at_k(retrieved_ids_eval_q_run, relevant_chunk_ids_eval, k_val_m_calc_run)
                    query_metrics_dict[f'ndcg@{k_val_m_calc_run}'] = ndcg_at_k(retrieved_ids_eval_q_run, relevant_chunk_ids_eval, k_val_m_calc_run)

        except Exception as e_eval_q_run:
            query_metrics_dict["status"] = "error_runtime"
            query_metrics_dict["error_message"] = f"{type(e_eval_q_run).__name__}: {str(e_eval_q_run)}"
            st.sidebar.error(f"Lỗi khi xử lý query {query_id_eval}: {e_eval_q_run}")
            import traceback
            st.sidebar.expander(f"Traceback lỗi query {query_id_eval}").code(traceback.format_exc())
        finally:
            query_metrics_dict["processing_time"] = time.time() - start_time_eval_q
            results_list.append(query_metrics_dict)
            progress_bar_eval.progress((i + 1) / total_items_eval)

    status_text_area_eval.text(f"Hoàn thành đánh giá {total_items_eval} queries!")
    return pd.DataFrame(results_list)

# --- Trang Streamlit cho Đánh giá ---
st.set_page_config(page_title="Đánh giá Truy vấn", layout="wide", initial_sidebar_state="auto")

# Khởi tạo session state cho mô hình
if "eval_selected_emb_name" not in st.session_state:
    st.session_state.eval_selected_emb_name = config.DEFAULT_EMBEDDING_MODEL
if "eval_selected_secondary_emb_name" not in st.session_state:
    secondary_default_eval = None
    for model_name in config.AVAILABLE_EMBEDDING_MODELS:
        if model_name != config.DEFAULT_EMBEDDING_MODEL:
            secondary_default_eval = model_name
            break
    st.session_state.eval_selected_secondary_emb_name = secondary_default_eval if secondary_default_eval else (config.AVAILABLE_EMBEDDING_MODELS[1] if len(config.AVAILABLE_EMBEDDING_MODELS) > 1 else config.DEFAULT_EMBEDDING_MODEL)
if "eval_hybrid_component_mode" not in st.session_state:
    st.session_state.eval_hybrid_component_mode = "2 Dense + 1 Sparse"
if "eval_selected_gemini_model_name" not in st.session_state:
    st.session_state.eval_selected_gemini_model_name = config.DEFAULT_GEMINI_MODEL
if "eval_selected_reranker_mname" not in st.session_state:
    st.session_state.eval_selected_reranker_mname = config.DEFAULT_RERANKER_MODEL

# Khởi tạo session state cho chế độ
if "eval_retrieval_query_mode" not in st.session_state: 
    st.session_state.eval_retrieval_query_mode = 'Mở rộng'
if "eval_retrieval_method" not in st.session_state: 
    st.session_state.eval_retrieval_method = 'Kết hợp'

if 'eval_data' not in st.session_state: 
    st.session_state.eval_data = None
if 'eval_results_df' not in st.session_state: 
    st.session_state.eval_results_df = None
if 'eval_run_completed' not in st.session_state: 
    st.session_state.eval_run_completed = False
if 'eval_uploaded_filename' not in st.session_state: 
    st.session_state.eval_uploaded_filename = None
if "eval_upload_counter" not in st.session_state: 
    st.session_state.eval_upload_counter = 0
if 'eval_last_config_run' not in st.session_state: 
    st.session_state.eval_last_config_run = {}

if "eval_variation_mode" not in st.session_state:
    st.session_state.eval_variation_mode = "Tạo mới từ LLM"
if "eval_save_newly_generated_variations_cb" not in st.session_state:
    st.session_state.eval_save_newly_generated_variations_cb = False
if "eval_uploaded_variations_file_obj_name" not in st.session_state: 
    st.session_state.eval_uploaded_variations_file_obj_name = None
if "eval_variations_data_from_file" not in st.session_state: 
    st.session_state.eval_variations_data_from_file = None
if "eval_generated_variations_for_saving" not in st.session_state: 
    st.session_state.eval_generated_variations_for_saving = None


# --- Tải trước Models và RAG cho Trang Đánh giá ---
if "eval_loaded_embedding_models" not in st.session_state:
    st.session_state.eval_loaded_embedding_models = {}
if "eval_loaded_reranker_models" not in st.session_state:
    st.session_state.eval_loaded_reranker_models = {}
if "eval_rag_components_per_embedding_model" not in st.session_state:
    st.session_state.eval_rag_components_per_embedding_model = {}

# --- Sidebar cho trang Đánh giá ---
with st.sidebar:
    st.title("Tùy chọn Đánh giá")

    current_eval_emb_name_sb = st.session_state.eval_selected_emb_name
    current_eval_secondary_emb_name_sb = st.session_state.eval_selected_secondary_emb_name
    current_eval_hybrid_component_mode = st.session_state.eval_hybrid_component_mode
    current_eval_gem_name_sb = st.session_state.eval_selected_gemini_model_name
    current_eval_reranker_name_sb = st.session_state.eval_selected_reranker_mname
    current_eval_retrieval_query_mode = st.session_state.eval_retrieval_query_mode
    current_eval_retrieval_method = st.session_state.eval_retrieval_method
    
    # Mode radio
    st.header("Cấu hình truy vấn")

    eval_retrieval_query_mode_choice = st.radio(
        "Nguồn câu hỏi cho truy vấn:", 
        options=['Đơn giản', 'Mở rộng', 'Đa dạng'],
        key="eval_retrieval_query_mode",
        index=['Đơn giản', 'Mở rộng', 'Đa dạng'].index(current_eval_retrieval_query_mode), 
        horizontal=True,
        help=(
            "**Đơn giản:** Chỉ dùng câu hỏi gốc.\n"
            "**Mở rộng:** Chỉ dùng câu hỏi mở rộng từ câu hỏi gốc (do AI tạo).\n"
            "**Đa dạng:** Dùng cả câu hỏi gốc và các biến thể từ câu hỏi gốc(do AI tạo)."
        )
    )
    
    eval_retrieval_method_choice = st.radio(
        "Phương thức truy vấn:", 
        options=['Ngữ nghĩa', 'Từ khóa', 'Kết hợp'],
        key="eval_retrieval_method", 
        index=['Ngữ nghĩa', 'Từ khóa', 'Kết hợp'].index(current_eval_retrieval_method),
        horizontal=True,
        help=(
            "**Ngữ nghĩa:** Tìm kiếm dựa trên vector ngữ nghĩa (nhanh, hiểu ngữ cảnh).\n"
            "**Từ khóa:** Tìm kiếm dựa trên từ khóa (BM25) (nhanh, chính xác từ khóa).\n"
            "**Kết hợp:** Kết hợp cả Ngữ nghĩa và Từ khóa (cân bằng, có thể tốt nhất)."
        )
    )
    
    if current_eval_retrieval_method == 'Kết hợp':
        eval_hybrid_component_mode_choice = st.radio(
            "Cấu hình thành phần Hybrid (Đánh giá):",
            options=["1 Dense + 1 Sparse", "2 Dense + 1 Sparse"],
            key="eval_hybrid_component_mode",
            index=["1 Dense + 1 Sparse", "2 Dense + 1 Sparse"].index(current_eval_hybrid_component_mode),
            horizontal=True,
            help="Chọn số lượng Dense encoders sử dụng trong phương thức Kết hợp cho đánh giá."
        )

    st.header("Mô hình")

    # Model selectbox
    eval_avail_emb_names = list(st.session_state.get("eval_loaded_embedding_models", {}).keys())
    if not eval_avail_emb_names: 
        eval_avail_emb_names = config.AVAILABLE_EMBEDDING_MODELS
    # Selectbox cho Embedding Model
    eval_selected_emb_name_ui = st.selectbox(
        "Chọn mô hình Embedding (Đánh giá):", 
        options=eval_avail_emb_names,
        key="eval_selected_emb_name",
        index=eval_avail_emb_names.index(current_eval_emb_name_sb) 
            if current_eval_emb_name_sb in eval_avail_emb_names else 0, 
        help="Chọn mô hình để vector hóa tài liệu và câu hỏi."
    )

    if current_eval_retrieval_method == 'Kết hợp' and st.session_state.eval_hybrid_component_mode == "2 Dense + 1 Sparse": # Cập nhật điều kiện
        options_for_secondary_eval = [
            name for name in eval_avail_emb_names 
            if name != st.session_state.eval_selected_emb_name
        ]
        
        current_eval_secondary_val = st.session_state.eval_selected_secondary_emb_name

        if not options_for_secondary_eval:
            st.warning("Cần ít nhất 2 embedding models khác nhau cho chế độ Hybrid 2-Dense (Đánh giá).")
            st.session_state.eval_selected_secondary_emb_name = None
        elif current_eval_secondary_val == st.session_state.eval_selected_emb_name or current_eval_secondary_val not in options_for_secondary_eval:
            st.session_state.eval_selected_secondary_emb_name = options_for_secondary_eval[0]
            current_eval_secondary_val = options_for_secondary_eval[0]

        idx_secondary_eval = 0
        if current_eval_secondary_val and options_for_secondary_eval:
            try:
                idx_secondary_eval = options_for_secondary_eval.index(current_eval_secondary_val)
            except ValueError:
                st.session_state.eval_selected_secondary_emb_name = options_for_secondary_eval[0]
                idx_secondary_eval = 0
        elif not options_for_secondary_eval:
            st.session_state.eval_selected_secondary_emb_name = None

        if options_for_secondary_eval:
            eval_sel_secondary_emb_name_ui = st.selectbox(
                "Chọn mô hình Embedding Phụ (Đánh giá Hybrid 2-Dense):",
                options=options_for_secondary_eval,
                key="eval_selected_secondary_emb_name",
                index=idx_secondary_eval,
                help="Chọn mô hình embedding thứ hai. Danh sách này đã loại trừ mô hình Embedding Chính (Đánh giá)."
            )

    # Selectbox cho Gemini Model
    eval_sel_gem_name_ui = st.selectbox(
        "Chọn mô hình Gemini (Đánh giá):", 
        options=config.AVAILABLE_GEMINI_MODELS,
        key="eval_selected_gemini_model_name",
        index=config.AVAILABLE_GEMINI_MODELS.index(current_eval_gem_name_sb) 
            if current_eval_gem_name_sb in config.AVAILABLE_GEMINI_MODELS else 0, 
        help="Chọn mô hình ngôn ngữ lớn để xử lý yêu cầu."
    )

    eval_avail_reranker_names = list(st.session_state.get("eval_loaded_reranker_models", {}).keys())
    if not eval_avail_reranker_names: 
        eval_avail_reranker_names = config.AVAILABLE_RERANKER_MODELS
    # Selectbox cho Reranker Model
    eval_selelected_reranker_name_ui = st.selectbox(
        "Chọn mô hình Reranker (Đánh giá):", 
        options=eval_avail_reranker_names,
        key="eval_selected_reranker_mname",
        index=eval_avail_reranker_names.index(current_eval_reranker_name_sb) 
            if current_eval_reranker_name_sb in eval_avail_reranker_names else 0, 
        help="Chọn mô hình để xếp hạng lại kết quả tìm kiếm. 'Không sử dụng' để tắt."
    )

# --- Giao diện chính của Ứng dụng ---
st.title("📊 Đánh giá Hệ thống Retrieval")
st.markdown("Trang này cho phép bạn chạy đánh giá hiệu suất của hệ thống retrieval và reranking với các mô hình đã được tải trước, cùng tùy chọn quản lý biến thể câu hỏi.")

# --- Khởi tạo tài nguyên cho trang Đánh giá ---
eval_init_status_placeholder = st.empty()
if "eval_resources_initialized" not in st.session_state:
    st.session_state.eval_resources_initialized = False

if not st.session_state.eval_resources_initialized:
    with st.spinner("Đang khởi tạo tài nguyên cho trang Đánh giá..."):
        eval_resources_ready = initialize_evaluation_page_resources()
        st.session_state.eval_resources_initialized = eval_resources_ready

# Kiểm tra sau khi đã khởi tạo
if st.session_state.eval_resources_initialized:
    eval_init_status_placeholder.success("✅ Hệ thống và tất cả mô hình đã sẵn sàng!")

    current_eval_selected_emb_name = st.session_state.eval_selected_emb_name
    current_eval_selected_reranker_name = st.session_state.eval_selected_reranker_mname
    current_eval_selected_gem_name = st.session_state.eval_selected_gemini_model_name

    eval_active_emb_obj = st.session_state.eval_loaded_embedding_models.get(current_eval_selected_emb_name)
    eval_active_rag_comps = st.session_state.eval_rag_components_per_embedding_model.get(current_eval_selected_emb_name)
    eval_active_retriever = eval_active_rag_comps[1] if eval_active_rag_comps else None
    eval_active_reranker_obj = st.session_state.eval_loaded_reranker_models.get(current_eval_selected_reranker_name)
    eval_active_gem_obj = load_gemini_model(current_eval_selected_gem_name)

    can_run_evaluation_flow = True
    if st.session_state.eval_variation_mode == "Sử dụng file biến thể đã tải lên":
        if not st.session_state.get("eval_variations_data_from_file"):
            pass
    elif st.session_state.eval_variation_mode == "Tạo mới từ LLM" or st.session_state.eval_variation_mode == "Chỉ sinh và lưu biến thể (không chạy đánh giá)":
        if not eval_active_gem_obj:
            st.error(f"Lỗi: Gemini model '{eval_selected_gem_name}' chưa tải được. Cần thiết để tạo biến thể mới.")
            can_run_evaluation_flow = False

    if st.session_state.eval_variation_mode != "Chỉ sinh và lưu biến thể (không chạy đánh giá)":
        if not eval_active_emb_obj:
            st.error(f"Lỗi: Embedding model '{eval_selected_emb_name.split('/')[-1]}' (Đánh giá) chưa tải.")
            can_run_evaluation_flow = False
        if not eval_active_retriever:
            st.error(f"Lỗi: Retriever cho '{eval_selected_emb_name.split('/')[-1]}' (Đánh giá) chưa sẵn sàng.")
            can_run_evaluation_flow = False

    if can_run_evaluation_flow:
        # --- Cập nhật Caption hiển thị cấu hình ---
        caption_eval_text = (
            f"Embedding Chính: `{eval_selected_emb_name.split('/')[-1]}` | "
            f"Mô hình Gemini: `{eval_selected_gem_name.split('/')[-1]}` | "
            f"Nguồn câu hỏi: `{st.session_state.eval_retrieval_query_mode}` | "
            f"Loại truy vấn: `{st.session_state.eval_retrieval_method}` | "
            f"Reranker: `{eval_selected_reranker_name.split('/')[-1] if eval_selected_reranker_name != 'Không sử dụng' else 'Tắt'}` | "
            f"Chế độ Biến thể: `{st.session_state.eval_variation_mode.split('(')[0].strip()}`"
        )
        if st.session_state.eval_retrieval_method == 'Kết hợp':
            caption_eval_text += f" | Cấu hình Hybrid: `{st.session_state.eval_hybrid_component_mode}`"
            if st.session_state.eval_hybrid_component_mode == "2 Dense + 1 Sparse" and st.session_state.get("eval_selected_secondary_emb_name"):
                eval_active_secondary_emb_name = st.session_state.eval_selected_secondary_emb_name
                caption_eval_text += f" | Embedding Phụ: `{eval_active_secondary_emb_name.split('/')[-1]}`"
        st.caption(caption_eval_text) 

        st.subheader("Cấu hình Biến thể Câu hỏi (Query Variations)")
        variation_mode_options_list = [
            "Tạo mới từ LLM",
            "Chỉ sinh và lưu biến thể (không chạy đánh giá)",
            "Sử dụng file biến thể đã tải lên"
        ]
        current_variation_mode_index = variation_mode_options_list.index(st.session_state.eval_variation_mode) \
            if st.session_state.eval_variation_mode in variation_mode_options_list else 0

        st.session_state.eval_variation_mode = st.radio(
            "Chế độ xử lý biến thể câu hỏi:",
            options=variation_mode_options_list,
            key="eval_variation_mode_radio_selector_main", 
            index=current_variation_mode_index,
            horizontal=False,
            help=(
                "- **Tạo mới từ LLM:** Mỗi lần chạy sẽ gọi LLM để tạo biến thể.\n"
                "- **Chỉ sinh và lưu biến thể:** Chạy LLM để tạo biến thể từ file đánh giá gốc, sau đó cho phép tải xuống file JSON. Không chạy các bước retrieval hay tính metrics.\n"
                "- **Sử dụng file biến thể đã tải lên:** Tải lên file JSON biến thể đã lưu. Hệ thống sẽ dùng các biến thể từ file này thay vì gọi LLM."
            )
        )
        if st.session_state.eval_variation_mode == "Tạo mới từ LLM":
            st.checkbox(
                "Lưu các biến thể câu hỏi được tạo ra file JSON (nếu chạy đánh giá retrieval)?",
                key="eval_save_newly_generated_variations_cb_main",
                help="Nếu chọn và chạy đánh giá retrieval, các biến thể mới sinh sẽ được chuẩn bị để tải về."
            )
        elif st.session_state.eval_variation_mode == "Sử dụng file biến thể đã tải lên":
            eval_uploaded_variations_file_widget_main = st.file_uploader(
                "Tải lên file JSON chứa biến thể câu hỏi đã lưu:",
                type=["json"],
                key="eval_var_file_uploader_widget_main",
                accept_multiple_files=False
            )
            if eval_uploaded_variations_file_widget_main is not None:
                if st.session_state.get("eval_uploaded_variations_file_obj_name") != eval_uploaded_variations_file_widget_main.name:
                    try:
                        variations_data_from_uploaded_file_main = json.loads(eval_uploaded_variations_file_widget_main.getvalue().decode('utf-8'))
                        if not isinstance(variations_data_from_uploaded_file_main, dict) or \
                           not all(isinstance(item_v_main, dict) and "all_queries" in item_v_main for item_v_main in variations_data_from_uploaded_file_main.values()):
                            st.error("File biến thể không đúng định dạng. Cần một JSON object với query_id làm key, và mỗi value chứa 'all_queries'.")
                            st.session_state.eval_variations_data_from_file = None
                        else:
                            st.session_state.eval_variations_data_from_file = variations_data_from_uploaded_file_main
                            st.session_state.eval_uploaded_variations_file_obj_name = eval_uploaded_variations_file_widget_main.name
                            st.success(f"Đã tải và xử lý file biến thể: {eval_uploaded_variations_file_widget_main.name} ({len(variations_data_from_uploaded_file_main)} query_ids).")
                    except Exception as e_var_json_upload_main:
                        st.error(f"Lỗi xử lý file JSON biến thể: {e_var_json_upload_main}")
                        st.session_state.eval_variations_data_from_file = None
            elif st.session_state.get("eval_variations_data_from_file"):
                 st.info(f"Đang sử dụng file biến thể đã tải: {st.session_state.get('eval_uploaded_variations_file_obj_name')}")

        uploader_key_eval_main_uploader = f"eval_main_file_uploader_main_{st.session_state.eval_upload_counter}"
        st.subheader("Tải Lên File Đánh giá Gốc (.json)")
        uploaded_file_eval_main_uploader = st.file_uploader(
            "Chọn file JSON chứa dữ liệu đánh giá gốc (queries, relevant_ids)...",
            type=["json"],
            key=uploader_key_eval_main_uploader
        )
        if uploaded_file_eval_main_uploader is not None:
            if uploaded_file_eval_main_uploader.name != st.session_state.eval_uploaded_filename:
                try:
                    eval_data_list_pg_main_data = json.loads(uploaded_file_eval_main_uploader.getvalue().decode('utf-8'))
                    if not isinstance(eval_data_list_pg_main_data, list) or \
                       (len(eval_data_list_pg_main_data) > 0 and not all(isinstance(item_e_main, dict) and "query" in item_e_main and "query_id" in item_e_main for item_e_main in eval_data_list_pg_main_data)):
                        st.error("File đánh giá gốc không đúng định dạng. Cần một danh sách các object, mỗi object phải có 'query' và 'query_id'.")
                        st.session_state.eval_data = None
                    else:
                        st.session_state.eval_data = eval_data_list_pg_main_data
                        st.session_state.eval_uploaded_filename = uploaded_file_eval_main_uploader.name
                        st.session_state.eval_run_completed = False
                        st.session_state.eval_results_df = None
                        st.session_state.eval_last_config_run = {}
                        st.session_state.eval_generated_variations_for_saving = None
                        st.success(f"Đã tải file đánh giá gốc '{uploaded_file_eval_main_uploader.name}' ({len(eval_data_list_pg_main_data)} câu hỏi).")
                except Exception as e_json_main_data:
                    st.error(f"Lỗi xử lý file JSON đánh giá gốc: {e_json_main_data}")
                    st.session_state.eval_data = None; st.session_state.eval_uploaded_filename = None

        if st.session_state.eval_data is not None:
            st.info(f"Sẵn sàng xử lý với dữ liệu từ: **{st.session_state.eval_uploaded_filename}** ({len(st.session_state.eval_data)} câu hỏi).")
            if st.checkbox("Hiển thị dữ liệu mẫu (5 dòng đầu)", key="eval_show_preview_main_cb"):
                st.dataframe(pd.DataFrame(st.session_state.eval_data).head())

            # --- Nút thực thi ---
            cols_buttons = st.columns(2)
            with cols_buttons[0]:
                if st.session_state.eval_variation_mode == "Chỉ sinh và lưu biến thể (không chạy đánh giá)":
                    if st.button("📝 Bắt đầu: Chỉ Sinh và Lưu Biến thể", key="eval_generate_variations_only_main_btn", use_container_width=True):
                        if not eval_active_gem_obj:
                            st.error("Lỗi: Cần có Gemini model để thực hiện thao tác này.")
                        else:
                            with st.spinner("⏳ Đang sinh biến thể cho tất cả các câu hỏi..."):
                                generated_data_main = generate_and_collect_variations_only(
                                    eval_data=st.session_state.eval_data,
                                    gemini_model_object=eval_active_gem_obj,
                                    num_variations=config.NUM_QUERY_VARIATIONS
                                )
                                if generated_data_main:
                                    st.session_state.eval_generated_variations_for_saving = generated_data_main
                                else:
                                    st.error("Không thể sinh biến thể. Vui lòng kiểm tra log.")
                                # Nút download sẽ xuất hiện ở dưới sau khi rerun hoặc tự động

            with cols_buttons[1]:
                run_eval_button_text_main = "🚀 Bắt đầu Đánh giá Retrieval & Metrics"
                disable_run_eval_button = False
                if st.session_state.eval_variation_mode == "Sử dụng file biến thể đã tải lên":
                    if not st.session_state.get("eval_variations_data_from_file"):
                        run_eval_button_text_main = "⚠️ Vui lòng tải file biến thể"
                        disable_run_eval_button = True
                elif st.session_state.eval_variation_mode == "Chỉ sinh và lưu biến thể (không chạy đánh giá)":
                     disable_run_eval_button = True # Không cho chạy retrieval ở chế độ này

                if st.button(run_eval_button_text_main, key="eval_start_full_eval_main_btn", disabled=disable_run_eval_button, use_container_width=True):
                    proceed_run_main = True
                    variations_to_pass_to_run_main = None
                    if st.session_state.eval_variation_mode == "Sử dụng file biến thể đã tải lên":
                        if not st.session_state.get("eval_variations_data_from_file"):
                            st.error("Bạn đã chọn 'Sử dụng file biến thể đã tải lên' nhưng chưa tải file hoặc file không hợp lệ.")
                            proceed_run_main = False
                        else:
                            variations_to_pass_to_run_main = st.session_state.eval_variations_data_from_file
                    elif st.session_state.eval_variation_mode == "Tạo mới từ LLM":
                        if not eval_active_gem_obj:
                            st.error("Bạn đã chọn 'Tạo mới từ LLM' nhưng Gemini model chưa sẵn sàng.")
                            proceed_run_main = False

                    if proceed_run_main and can_run_evaluation_flow:
                        eval_config_for_this_run_pg_main_run = {
                            'embedding_model_name': eval_selected_emb_name,
                            'retrieval_query_mode': st.session_state.eval_retrieval_query_mode,
                            'retrieval_method': st.session_state.eval_retrieval_method,
                            'selected_reranker_model_name': eval_selected_reranker_name,
                            'gemini_model_name': eval_selected_gem_name,
                            'variation_mode_used': st.session_state.eval_variation_mode,
                        }
                        if st.session_state.eval_retrieval_method == 'Kết hợp':
                            eval_config_for_this_run_pg_main_run['hybrid_component_mode'] = st.session_state.eval_hybrid_component_mode 
                            if st.session_state.eval_hybrid_component_mode == "2 Dense + 1 Sparse":
                                eval_config_for_this_run_pg_main_run['secondary_embedding_model_name'] = st.session_state.get("eval_selected_secondary_emb_name")

                        with st.spinner("⏳ Đang chạy đánh giá Retrieval & Metrics..."):
                            start_eval_time_pg_main_run = time.time()
                            results_df_output_pg_main_run = run_retrieval_evaluation(
                                eval_data=st.session_state.eval_data,
                                retriever_instance_for_eval=eval_active_retriever,
                                embedding_model_object_for_eval=eval_active_emb_obj,
                                reranking_model_object_for_eval=eval_active_reranker_obj,
                                gemini_model_object_for_eval=eval_active_gem_obj,
                                eval_config_params=st.session_state.eval_last_config_run,
                                preloaded_query_variations=variations_to_pass_to_run_main
                            )
                            total_eval_time_pg_main_run = time.time() - start_eval_time_pg_main_run
                            st.success(f"Hoàn thành đánh giá Retrieval & Metrics sau {total_eval_time_pg_main_run:.2f} giây.")
                            st.session_state.eval_results_df = results_df_output_pg_main_run
                            st.session_state.eval_run_completed = True

                            if st.session_state.eval_variation_mode == "Tạo mới từ LLM" and \
                               st.session_state.get("eval_save_newly_generated_variations_cb_main", False) and \
                               results_df_output_pg_main_run is not None and not results_df_output_pg_main_run.empty:
                                # Thu thập lại các biến thể từ kết quả đánh giá nếu cần lưu
                                newly_generated_vars_from_run = {}
                                for _, row in results_df_output_pg_main_run.iterrows():
                                    if row.get("variation_source") == "LLM":
                                        newly_generated_vars_from_run[row["query_id"]] = {
                                            "original_query": row["query"],
                                            "relevance_status": "valid" if row["status"] == "evaluated" or row["status"] == "skipped_irrelevant" else "error_generating_variations", # Cần cách tốt hơn để lấy relevance gốc
                                            "direct_answer_if_invalid": "", 
                                            "all_queries": [row["query"]] + [f"var_{j}" for j in range(int(row.get("num_variations_generated",0)))], # Cần cách lấy all_queries thật sự
                                            "summarizing_query": row.get("summarizing_query", row["query"]),
                                            "llm_model_used_for_generation": row.get("llm_model_for_variation", eval_selected_gem_name)
                                        }

                                if newly_generated_vars_from_run:
                                     st.session_state.eval_generated_variations_for_saving = newly_generated_vars_from_run
                                     st.info("Các biến thể được tạo mới trong quá trình đánh giá đã được chuẩn bị để tải xuống (kết quả có thể khác với nút 'Chỉ Sinh và Lưu' do tối ưu).")
                            st.rerun()

            if st.session_state.get("eval_generated_variations_for_saving"):
                try:
                    variations_json_to_save_main = json.dumps(st.session_state.eval_generated_variations_for_saving, indent=2, ensure_ascii=False)
                    ts_var_save_main = datetime.now().strftime("%Y%m%d_%H%M%S")
                    gem_name_var_save_main = st.session_state.eval_selected_gemini_model_name.split('/')[-1].replace('.','-')[:15]
                    original_eval_file_name_base = os.path.splitext(st.session_state.eval_uploaded_filename)[0] if st.session_state.eval_uploaded_filename else "unknown_eval_data"
                    var_fname_save_main = f"{original_eval_file_name_base}_variations_{gem_name_var_save_main}_{ts_var_save_main}.json"
                    st.download_button(
                        label="📥 Tải về File Biến thể Câu hỏi Đã Tạo (.json)",
                        data=variations_json_to_save_main,
                        file_name=var_fname_save_main,
                        mime="application/json",
                        key="download_generated_variations_main_btn"
                    )
                except Exception as e_dl_gen_var_main:
                    st.error(f"Lỗi khi chuẩn bị file biến thể đã tạo để tải xuống: {e_dl_gen_var_main}")

            if st.session_state.eval_run_completed and st.session_state.eval_results_df is not None:
                st.subheader("Kết quả Đánh giá Chi tiết")
                detailed_results_df_display_pg = st.session_state.eval_results_df
                last_config_run_display_pg = st.session_state.eval_last_config_run

                st.markdown("**Cấu hình đã sử dụng cho lần chạy cuối:**")
                cfg_cols_pg = st.columns(6)
                emb_n_disp_pg = last_config_run_display_pg.get('embedding_model_name', 'N/A').split('/')[-1]
                cfg_cols_pg[0].metric("Embedding", emb_n_disp_pg)
                cfg_cols_pg[1].metric("Query Mode", last_config_run_display_pg.get('retrieval_query_mode', 'N/A'))
                cfg_cols_pg[2].metric("Ret. Method", last_config_run_display_pg.get('retrieval_method', 'N/A'))
                rer_n_disp_pg = last_config_run_display_pg.get('selected_reranker_model_name', 'N/A').split('/')[-1]
                if rer_n_disp_pg == "Không sử dụng".split('/')[-1]: rer_n_disp_pg = "Tắt"
                cfg_cols_pg[3].metric("Reranker", rer_n_disp_pg)
                gem_n_disp_pg = last_config_run_display_pg.get('gemini_model_name', 'N/A').split('/')[-1]
                cfg_cols_pg[4].metric("Gemini (Var)", gem_n_disp_pg)
                var_mode_cfg_disp = last_config_run_display_pg.get('variation_mode_used', 'N/A').split('(')[0].strip()
                cfg_cols_pg[5].metric("Variation Mode", var_mode_cfg_disp)


                avg_metrics_res_pg, num_eval, num_skip_err_pg = calculate_average_metrics(detailed_results_df_display_pg)

                st.metric("Tổng số Queries trong File", len(detailed_results_df_display_pg))
                col_rc1_pg, col_rc2_pg = st.columns(2)
                col_rc1_pg.metric("Queries Đánh giá Hợp lệ", num_eval)
                col_rc2_pg.metric("Queries Bỏ qua / Lỗi Runtime", num_skip_err_pg)

                if avg_metrics_res_pg:
                    st.markdown("#### Metrics Trung bình @K (trên các queries hợp lệ)")
                    k_vals_disp_pg = [3, 5, 10]
                    cols_k_pg = st.columns(len(k_vals_disp_pg))
                    for idx_k_pg, k_v_pg in enumerate(k_vals_disp_pg):
                        with cols_k_pg[idx_k_pg]:
                            st.markdown(f"**K = {k_v_pg}**")
                            st.text(f"Precision: {avg_metrics_res_pg.get(f'avg_precision@{k_v_pg}', 0.0):.4f}")
                            st.text(f"Recall:    {avg_metrics_res_pg.get(f'avg_recall@{k_v_pg}', 0.0):.4f}")
                            st.text(f"F1:        {avg_metrics_res_pg.get(f'avg_f1@{k_v_pg}', 0.0):.4f}")
                            st.text(f"MRR:       {avg_metrics_res_pg.get(f'avg_mrr@{k_v_pg}', 0.0):.4f}")
                            st.text(f"NDCG:      {avg_metrics_res_pg.get(f'avg_ndcg@{k_v_pg}', 0.0):.4f}")

                    st.markdown("#### Thông tin Hiệu năng & Số lượng Trung bình (trên các queries hợp lệ)")
                    perf_col1_pg, perf_col2_pg, perf_col3_pg, perf_col4_pg = st.columns(4)
                    perf_col1_pg.metric("Avg. Query Time (s)", f"{avg_metrics_res_pg.get('avg_processing_time', 0.0):.2f}s")
                    perf_col1_pg.metric("Avg. Variation Time (s)", f"{avg_metrics_res_pg.get('avg_variation_time', 0.0):.3f}s")
                    perf_col2_pg.metric("Avg. Search Time (s)", f"{avg_metrics_res_pg.get('avg_search_time', 0.0):.2f}s")
                    perf_col2_pg.metric("Avg. Rerank Time (s)", f"{avg_metrics_res_pg.get('avg_rerank_time', 0.0):.3f}s")
                    perf_col3_pg.metric("Avg. #Variations Gen", f"{avg_metrics_res_pg.get('avg_num_variations_generated', 0.0):.1f}")
                    perf_col3_pg.metric("Avg. #Docs Reranked", f"{avg_metrics_res_pg.get('avg_num_docs_reranked', 0.0):.1f}")
                    perf_col4_pg.metric("Avg. #Docs After Rerank", f"{avg_metrics_res_pg.get('avg_num_retrieved_after_rerank',0.0):.1f}")
                    perf_col4_pg.metric("Avg. #Unique Docs Found", f"{avg_metrics_res_pg.get('avg_num_unique_docs_found',0.0):.1f}")


                with st.expander("Xem Kết quả Chi tiết từng Query (Raw Data)"):
                    display_cols_eval_results = [
                        'query_id', 'query', 'status', 'error_message',
                        'embedding_model_name', 'retrieval_query_mode','retrieval_method', 'selected_reranker_model',
                        'variation_mode_run', 'variation_source', 'llm_model_for_variation',
                        'precision@3', 'recall@3', 'f1@3', 'mrr@3', 'ndcg@3',
                        'precision@5', 'recall@5', 'f1@5', 'mrr@5', 'ndcg@5',
                        'precision@10', 'recall@10', 'f1@10', 'mrr@10', 'ndcg@10',
                        'processing_time', 'variation_time', 'search_time', 'rerank_time',
                        'num_variations_generated','num_unique_docs_found', 'num_retrieved_before_rerank',
                        'num_docs_reranked', 'num_retrieved_after_rerank',
                        'summarizing_query', 'retrieved_ids', 'relevant_ids'
                    ]
                    existing_display_cols_eval_results = [col for col in display_cols_eval_results if col in detailed_results_df_display_pg.columns]
                    st.dataframe(detailed_results_df_display_pg[existing_display_cols_eval_results])

                st.subheader("Lưu Kết quả Đánh giá Retrieval")
                try:
                    results_json_pg_main_save = detailed_results_df_display_pg.to_json(orient='records', indent=2, force_ascii=False)
                    results_csv_pg_main_save = detailed_results_df_display_pg.to_csv(index=False).encode('utf-8')
                    timestamp_pg_main_save = datetime.now().strftime("%Y%m%d_%H%M%S")

                    emb_sfx_pg_main_save = last_config_run_display_pg.get('embedding_model_name', 'na').split('/')[-1].replace('-', '').replace('_', '')[:10]
                    qmode_sfx_pg_main_save = last_config_run_display_pg.get('retrieval_query_mode', 'na').lower()[:3]
                    method_sfx_pg_main_save = last_config_run_display_pg.get('retrieval_method', 'na').lower()
                    rer_sfx_pg_main_save = "norr"
                    sel_rer_fname_pg_main_save = last_config_run_display_pg.get('selected_reranker_model_name', 'Không sử dụng')
                    if sel_rer_fname_pg_main_save != 'Không sử dụng':
                        rer_sfx_pg_main_save = sel_rer_fname_pg_main_save.split('/')[-1].replace('-', '').replace('_', '')[:10]
                    mod_sfx_pg_main_save = last_config_run_display_pg.get('gemini_model_name', 'gemini').split('/')[-1].replace('.','-')[:15]
                    var_mode_sfx_pg_main_save = last_config_run_display_pg.get('variation_mode_used', 'na').lower().replace(" ","")[:10]

                    base_fname_pg_main_save = f"eval_results_{emb_sfx_pg_main_save}_{qmode_sfx_pg_main_save}_{method_sfx_pg_main_save}_{rer_sfx_pg_main_save}_{mod_sfx_pg_main_save}_{var_mode_sfx_pg_main_save}_{timestamp_pg_main_save}"
                    fname_json_pg_main_save = f"{base_fname_pg_main_save}.json"
                    fname_csv_pg_main_save = f"{base_fname_pg_main_save}.csv"

                    dl_col1_pg_main_save, dl_col2_pg_main_save = st.columns(2)
                    with dl_col1_pg_main_save:
                        st.download_button("💾 Tải về Kết quả Đánh giá (JSON)", results_json_pg_main_save, fname_json_pg_main_save, "application/json", key="dl_json_eval_main_save_btn")
                    with dl_col2_pg_main_save:
                        st.download_button("💾 Tải về Kết quả Đánh giá (CSV)", results_csv_pg_main_save, fname_csv_pg_main_save, "text/csv", key="dl_csv_eval_main_save_btn")
                except Exception as e_file_dl_main_save:
                    st.error(f"Lỗi khi chuẩn bị file kết quả đánh giá: {e_file_dl_main_save}")


            st.markdown("---")
            st.subheader("Quản lý Trạng thái Đánh giá")
            if st.button("Xóa File Đã Tải và Kết Quả Hiện Tại", key="eval_clear_state_button_main_clear"):
                st.session_state.eval_data = None
                st.session_state.eval_upload_counter += 1
                st.session_state.eval_run_completed = False
                st.session_state.eval_results_df = None
                st.session_state.eval_last_config_run = {}
                st.session_state.eval_uploaded_filename = None
                st.session_state.eval_variations_data_from_file = None
                st.session_state.eval_uploaded_variations_file_obj_name = None
                st.session_state.eval_generated_variations_for_saving = None
                st.success("Đã xóa trạng thái đánh giá hiện tại.")
                time.sleep(1)
                st.rerun()
    else:
        st.warning("⚠️ Không thể tiến hành do thiếu các thành phần cần thiết. Vui lòng kiểm tra thông báo lỗi ở trên và cấu hình trong sidebar.")

elif not st.session_state.eval_resources_initialized:
    eval_init_status_placeholder.error("⚠️ Tài nguyên trang Đánh giá CHƯA SẴN SÀNG. Lỗi trong quá trình tải model hoặc tạo RAG. Vui lòng kiểm tra log chi tiết hoặc làm mới trang.")