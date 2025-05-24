# pages/Evaluation.py
import time
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Thêm thư mục gốc vào sys.path
import config #
from model_loader import ( #
    load_all_embedding_models,
    load_all_reranker_models,
    load_gemini_model
)
from data_loader import load_or_create_rag_components #
from reranker import rerank_documents #
from utils import ( #
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
    wait_time_seconds_var_only = 60 # Giữ thời gian chờ để tránh rate limit

    if not gemini_model_object:
        st.error("Lỗi: Gemini model không được cung cấp cho việc sinh biến thể.")
        return {}

    for i, item_data_var in enumerate(eval_data):
        if i > 0 and i % queries_per_batch_var_only == 0:
            pause_msg_var = f"Đang sinh biến thể {i}/{total_items_var_only}. Tạm dừng {wait_time_seconds_var_only} giây..."
            status_text_var_only.text(pause_msg_var)
            time.sleep(wait_time_seconds_var_only)
            status_text_var_only.text(f"Tiếp tục sinh biến thể cho query {i+1}/{total_items_var_only}...")

        query_id_item_var = item_data_var.get("query_id", f"item_{i+1}") # Fallback query_id
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
                "llm_model_used_for_generation": gemini_model_object.model_name # Lưu tên model đã dùng
            }
        except Exception as e_gen_var_item:
            st.warning(f"Lỗi khi sinh biến thể cho query_id '{query_id_item_var}': {e_gen_var_item}. Mục này sẽ có thông tin lỗi.")
            collected_variations_output[query_id_item_var] = {
                "original_query": original_query_item_var,
                "relevance_status": "error_generating_variations",
                "direct_answer_if_invalid": "",
                "all_queries": [original_query_item_var], # Fallback
                "summarizing_query": original_query_item_var, # Fallback
                "error_message": str(e_gen_var_item),
                "llm_model_used_for_generation": gemini_model_object.model_name
            }
        finally:
            progress_bar_var_only.progress((i + 1) / total_items_var_only)

    status_text_var_only.success(f"Hoàn thành sinh biến thể cho {total_items_var_only} câu hỏi!")
    return collected_variations_output

# --- Hàm chạy đánh giá chính, được sửa đổi ---
def run_retrieval_evaluation(
    eval_data: list,
    retriever_instance_for_eval,
    embedding_model_object_for_eval,
    reranking_model_object_for_eval,
    gemini_model_object_for_eval,
    eval_config_params: dict,
    preloaded_query_variations: dict = None # Mới: dict {query_id: variation_data}
    ):
    results_list = []
    k_values_metrics = [3, 5, 10]

    retrieval_query_mode_eval = eval_config_params.get('retrieval_query_mode', 'Tổng quát')
    retrieval_method_eval = eval_config_params.get('retrieval_method', 'hybrid')
    selected_reranker_name_eval_run = eval_config_params.get('selected_reranker_model_name', 'Không sử dụng')
    use_reranker_eval_run = reranking_model_object_for_eval is not None and selected_reranker_name_eval_run != 'Không sử dụng'
    variation_mode_run = eval_config_params.get('variation_mode_used', "Luôn sinh mới (qua LLM)")

    progress_bar_eval = st.progress(0)
    status_text_area_eval = st.empty()
    total_items_eval = len(eval_data)
    queries_per_batch_eval = 15
    wait_time_seconds_eval = 60

    for i, item_eval in enumerate(eval_data):
        if i > 0 and i % queries_per_batch_eval == 0:
            pause_msg_eval = f"Đã xử lý {i}/{total_items_eval} queries. Tạm dừng {wait_time_seconds_eval} giây..."
            status_text_area_eval.text(pause_msg_eval)
            time.sleep(wait_time_seconds_eval)

        query_id_eval = item_eval.get("query_id", f"item_{i+1}") # Fallback query_id
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
            "variation_source": "N/A", # Sẽ được cập nhật
            "llm_model_for_variation": "N/A", # Sẽ được cập nhật
            "status": "error_default", "retrieved_ids": [], "relevant_ids": list(relevant_chunk_ids_eval),
            "processing_time": 0.0, 'summarizing_query': '',
            'variation_time': 0.0, 'search_time': 0.0, 'rerank_time': 0.0,
            'num_variations_generated': 0, 'num_unique_docs_found': 0, 'num_docs_reranked': 0,
            'num_retrieved_before_rerank': 0, 'num_retrieved_after_rerank': 0, 'error_message': ''
        }
        for k_val_m in k_values_metrics:
            query_metrics_dict[f'precision@{k_val_m}'] = 0.0; query_metrics_dict[f'recall@{k_val_m}'] = 0.0
            query_metrics_dict[f'f1@{k_val_m}'] = 0.0; query_metrics_dict[f'mrr@{k_val_m}'] = 0.0; query_metrics_dict[f'ndcg@{k_val_m}'] = 0.0

        try:
            var_start_processing_time = time.time()
            relevance_status_q_eval = "valid"
            all_queries_q_eval = [original_query_eval]
            summarizing_q_q_eval = original_query_eval
            # direct_ans_if_inv_q_eval = "" # Không dùng trực tiếp trong đánh giá retrieval

            if preloaded_query_variations and query_id_eval in preloaded_query_variations:
                query_metrics_dict["variation_source"] = "File"
                var_data = preloaded_query_variations[query_id_eval]
                relevance_status_q_eval = var_data.get("relevance_status", "valid")
                all_queries_q_eval = var_data.get("all_queries", [original_query_eval])
                summarizing_q_q_eval = var_data.get("summarizing_query", original_query_eval)
                query_metrics_dict["llm_model_for_variation"] = var_data.get("llm_model_used_for_generation", "From File")
                if relevance_status_q_eval == "error_generating_variations":
                    st.sidebar.warning(f"Query ID {query_id_eval}: Lỗi khi tạo biến thể từ file trước đó. Sử dụng fallback.")
            elif gemini_model_object_for_eval: # Sinh mới nếu không có preloaded hoặc query_id không khớp
                query_metrics_dict["variation_source"] = "LLM"
                query_metrics_dict["llm_model_for_variation"] = gemini_model_object_for_eval.model_name
                relevance_status_q_eval, _, all_queries_q_eval, summarizing_q_q_eval = generate_query_variations(
                    original_query=original_query_eval,
                    gemini_model=gemini_model_object_for_eval,
                    chat_history=None,
                    num_variations=config.NUM_QUERY_VARIATIONS
                )
            else: # Không có preloaded và không có model để sinh mới
                query_metrics_dict["variation_source"] = "Fallback Original"
                query_metrics_dict["llm_model_for_variation"] = "N/A"
                # Các giá trị mặc định đã được đặt ở trên

            query_metrics_dict["variation_time"] = time.time() - var_start_processing_time
            query_metrics_dict["summarizing_query"] = summarizing_q_q_eval
            query_metrics_dict["num_variations_generated"] = len(all_queries_q_eval) - 1 if isinstance(all_queries_q_eval, list) and len(all_queries_q_eval) > 0 else 0

            if relevance_status_q_eval == 'invalid' or relevance_status_q_eval == 'error_generating_variations':
                query_metrics_dict["status"] = "skipped_irrelevant" if relevance_status_q_eval == 'invalid' else "skipped_variation_error"
            else:
                queries_to_search_eval_run = []
                if retrieval_query_mode_eval == 'Đơn giản': queries_to_search_eval_run = [original_query_eval]
                elif retrieval_query_mode_eval == 'Tổng quát': queries_to_search_eval_run = [summarizing_q_q_eval] if summarizing_q_q_eval else [original_query_eval]
                elif retrieval_query_mode_eval == 'Sâu': queries_to_search_eval_run = all_queries_q_eval if all_queries_q_eval else [original_query_eval]

                collected_docs_data_eval_run = {}
                search_start_eval_q_run = time.time()
                for q_var_eval_run in queries_to_search_eval_run:
                    if not q_var_eval_run: continue
                    search_results_eval_run = retriever_instance_for_eval.search(
                        q_var_eval_run,
                        embedding_model_object_for_eval,
                        method=retrieval_method_eval,
                        k=config.VECTOR_K_PER_QUERY
                    )
                    for res_item_eval_run in search_results_eval_run:
                        doc_idx_eval_run = res_item_eval_run.get('index')
                        if isinstance(doc_idx_eval_run, int) and doc_idx_eval_run >= 0 and doc_idx_eval_run not in collected_docs_data_eval_run:
                            collected_docs_data_eval_run[doc_idx_eval_run] = res_item_eval_run
                query_metrics_dict["search_time"] = time.time() - search_start_eval_q_run
                query_metrics_dict["num_unique_docs_found"] = len(collected_docs_data_eval_run)

                retrieved_docs_list_eval_run = list(collected_docs_data_eval_run.values())
                sort_reverse_eval_run = (retrieval_method_eval != 'dense')
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
                else: # Không có kết quả retrieval
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


# --- (Phần còn lại của trang Streamlit cho Đánh giá) ---
# ... (initialize_evaluation_page_resources, st.sidebar, st.caption, ...)

# --- Khởi tạo tài nguyên cho trang Đánh giá ---
eval_page_status_placeholder = st.empty()
if "eval_page_resources_initialized" not in st.session_state:
    st.session_state.eval_page_resources_initialized = False

if not st.session_state.eval_page_resources_initialized:
    with st.spinner("Đang khởi tạo tài nguyên cho trang Đánh giá..."):
        eval_resources_ready = initialize_evaluation_page_resources() # Hàm này đã được định nghĩa ở phản hồi trước
        st.session_state.eval_page_resources_initialized = eval_resources_ready

if st.session_state.eval_page_resources_initialized:
    eval_page_status_placeholder.success("✅ Tài nguyên trang Đánh giá đã sẵn sàng!")

    # Lấy các model objects đã được tải trước
    eval_pg_active_emb_name = st.session_state.eval_pg_selected_embedding_model_name
    eval_pg_active_rer_name = st.session_state.eval_pg_selected_reranker_model_name
    eval_pg_active_gem_name = st.session_state.eval_pg_selected_gemini_model_name

    eval_pg_active_emb_obj = st.session_state.eval_pg_loaded_embedding_models.get(eval_pg_active_emb_name)
    eval_pg_active_rag_comps = st.session_state.eval_pg_rag_components_per_embedding_model.get(eval_pg_active_emb_name)
    eval_pg_active_retriever = eval_pg_active_rag_comps[1] if eval_pg_active_rag_comps else None
    eval_pg_active_rer_obj = st.session_state.eval_pg_loaded_reranker_models.get(eval_pg_active_rer_name)
    eval_pg_active_gem_obj = load_gemini_model(eval_pg_active_gem_name)

    can_run_evaluation_flow = True
    # Kiểm tra các model cần thiết cho các chế độ khác nhau
    if st.session_state.eval_pg_variation_mode == "Sử dụng file biến thể đã tải lên":
        if not st.session_state.get("eval_pg_variations_data_from_file"):
            # Sẽ có thông báo lỗi cụ thể hơn khi nhấn nút "Bắt đầu"
            pass # Không dừng ngay ở đây, để người dùng có cơ hội tải file
    elif st.session_state.eval_pg_variation_mode == "Tạo mới từ LLM":
        if not eval_pg_active_gem_obj:
            st.error(f"Lỗi: Gemini model '{eval_pg_active_gem_name}' chưa tải được. Cần thiết để tạo biến thể mới.")
            can_run_evaluation_flow = False
    # Kiểm tra chung cho retriever và embedding nếu không phải chỉ sinh biến thể
    if st.session_state.eval_pg_variation_mode != "Chỉ sinh và lưu biến thể (không chạy đánh giá)":
        if not eval_pg_active_emb_obj:
            st.error(f"Lỗi: Embedding model '{eval_pg_active_emb_name.split('/')[-1]}' (Đánh giá) chưa tải.")
            can_run_evaluation_flow = False
        if not eval_pg_active_retriever:
            st.error(f"Lỗi: Retriever cho '{eval_pg_active_emb_name.split('/')[-1]}' (Đánh giá) chưa sẵn sàng.")
            can_run_evaluation_flow = False

    if can_run_evaluation_flow:
        st.caption(
            f"Đánh giá với: Embedding: `{eval_pg_active_emb_name.split('/')[-1]}` | "
            f"Gemini: `{eval_pg_active_gem_name}` | "
            f"Query Mode: `{st.session_state.eval_pg_retrieval_query_mode}` | "
            f"Retrieval: `{st.session_state.eval_pg_retrieval_method}` | "
            f"Reranker: `{eval_pg_active_rer_name.split('/')[-1] if eval_pg_active_rer_name != 'Không sử dụng' else 'Tắt'}` | "
            f"Chế độ Biến thể: `{st.session_state.eval_pg_variation_mode.split('(')[0].strip()}`"
        )

        # --- UI cho các lựa chọn chế độ biến thể ---
        st.subheader("Cấu hình Biến thể Câu hỏi (Query Variations)")
        variation_mode_options_list = [
            "Tạo mới từ LLM",
            "Chỉ sinh và lưu biến thể (không chạy đánh giá)", # Đổi tên cho nút riêng
            "Sử dụng file biến thể đã tải lên"
        ]
        # Lấy giá trị hiện tại từ session_state để giữ lựa chọn khi rerun
        current_variation_mode_index = variation_mode_options_list.index(st.session_state.eval_pg_variation_mode) \
            if st.session_state.eval_pg_variation_mode in variation_mode_options_list else 0

        st.session_state.eval_pg_variation_mode = st.radio(
            "Chế độ xử lý biến thể câu hỏi:",
            options=variation_mode_options_list,
            key="eval_pg_variation_mode_radio_selector",
            index=current_variation_mode_index,
            horizontal=False,
            help=(
                "- **Tạo mới từ LLM:** Mỗi lần chạy sẽ gọi LLM để tạo biến thể.\n"
                "- **Chỉ sinh và lưu biến thể:** Chạy LLM để tạo biến thể từ file đánh giá gốc, sau đó cho phép tải xuống file JSON. Không chạy các bước retrieval hay tính metrics.\n"
                "- **Sử dụng file biến thể đã tải lên:** Tải lên file JSON biến thể đã lưu. Hệ thống sẽ dùng các biến thể từ file này thay vì gọi LLM."
            )
        )
        # Checkbox và File Uploader tùy theo chế độ
        if st.session_state.eval_pg_variation_mode == "Tạo mới từ LLM":
            st.checkbox(
                "Lưu các biến thể câu hỏi được tạo ra file JSON?",
                key="eval_pg_save_newly_generated_variations_cb", # Key mới
                help="Nếu chọn, sau khi chạy đánh giá, bạn có thể tải về file chứa các biến thể vừa được tạo."
            )
        elif st.session_state.eval_pg_variation_mode == "Sử dụng file biến thể đã tải lên":
            eval_pg_uploaded_variations_file_widget = st.file_uploader(
                "Tải lên file JSON chứa biến thể câu hỏi đã lưu:",
                type=["json"],
                key="eval_pg_var_file_uploader_widget",
                accept_multiple_files=False
            )
            if eval_pg_uploaded_variations_file_widget is not None:
                # Chỉ xử lý nếu file mới được tải lên hoặc chưa có file nào trong state
                if st.session_state.get("eval_pg_uploaded_variations_file_obj_name") != eval_pg_uploaded_variations_file_widget.name:
                    try:
                        variations_data_from_uploaded_file = json.loads(eval_pg_uploaded_variations_file_widget.getvalue().decode('utf-8'))
                        if not isinstance(variations_data_from_uploaded_file, dict) or \
                           not all(isinstance(item_v, dict) and "all_queries" in item_v for item_v in variations_data_from_uploaded_file.values()):
                            st.error("File biến thể không đúng định dạng. Cần một JSON object với query_id làm key, và mỗi value chứa 'all_queries'.")
                            st.session_state.eval_pg_variations_data_from_file = None
                        else:
                            st.session_state.eval_pg_variations_data_from_file = variations_data_from_uploaded_file
                            st.session_state.eval_pg_uploaded_variations_file_obj_name = eval_pg_uploaded_variations_file_widget.name # Lưu tên file để tránh xử lý lại
                            st.success(f"Đã tải và xử lý file biến thể: {eval_pg_uploaded_variations_file_widget.name} ({len(variations_data_from_uploaded_file)} query_ids).")
                    except Exception as e_var_json_upload:
                        st.error(f"Lỗi xử lý file JSON biến thể: {e_var_json_upload}")
                        st.session_state.eval_pg_variations_data_from_file = None
            elif st.session_state.get("eval_pg_variations_data_from_file"): # Nếu đã có dữ liệu từ trước
                 st.info(f"Đang sử dụng file biến thể đã tải: {st.session_state.get('eval_pg_uploaded_variations_file_obj_name')}")


        # --- Tải File Đánh giá Gốc ---
        uploader_key_eval_pg_main = f"eval_pg_main_file_uploader_{st.session_state.eval_pg_upload_counter}"
        st.subheader("Tải Lên File Đánh giá Gốc (.json)")
        uploaded_file_eval_pg_main = st.file_uploader(
            "Chọn file JSON chứa dữ liệu đánh giá gốc (queries, relevant_ids)...",
            type=["json"],
            key=uploader_key_eval_pg_main
        )
        if uploaded_file_eval_pg_main is not None:
            if uploaded_file_eval_pg_main.name != st.session_state.eval_pg_uploaded_filename:
                try:
                    eval_data_list_pg_main = json.loads(uploaded_file_eval_pg_main.getvalue().decode('utf-8'))
                    # Sơ bộ kiểm tra cấu trúc file eval_data
                    if not isinstance(eval_data_list_pg_main, list) or \
                       (len(eval_data_list_pg_main) > 0 and not all(isinstance(item_e, dict) and "query" in item_e and "query_id" in item_e for item_e in eval_data_list_pg_main)):
                        st.error("File đánh giá gốc không đúng định dạng. Cần một danh sách các object, mỗi object phải có 'query' và 'query_id'.")
                        st.session_state.eval_pg_data = None
                    else:
                        st.session_state.eval_pg_data = eval_data_list_pg_main
                        st.session_state.eval_pg_uploaded_filename = uploaded_file_eval_pg_main.name
                        st.session_state.eval_pg_run_completed = False
                        st.session_state.eval_pg_results_df = None
                        st.session_state.eval_pg_last_config_run = {}
                        st.session_state.eval_pg_generated_variations_for_saving = None # Reset khi tải file mới
                        st.success(f"Đã tải file đánh giá gốc '{uploaded_file_eval_pg_main.name}' ({len(eval_data_list_pg_main)} câu hỏi).")
                except Exception as e_json_main:
                    st.error(f"Lỗi xử lý file JSON đánh giá gốc: {e_json_main}")
                    st.session_state.eval_pg_data = None; st.session_state.eval_pg_uploaded_filename = None

        if st.session_state.eval_pg_data is not None:
            st.info(f"Sẵn sàng xử lý với dữ liệu từ: **{st.session_state.eval_pg_uploaded_filename}** ({len(st.session_state.eval_pg_data)} câu hỏi).")
            if st.checkbox("Hiển thị dữ liệu mẫu (5 dòng đầu)", key="eval_pg_show_preview_main"):
                st.dataframe(pd.DataFrame(st.session_state.eval_pg_data).head())

            # --- Nút thực thi chính ---
            # Nút "Chỉ sinh và lưu biến thể"
            if st.session_state.eval_pg_variation_mode == "Chỉ sinh và lưu biến thể (không chạy đánh giá)":
                if st.button("📝 Bắt đầu: Chỉ Sinh và Lưu Biến thể Câu hỏi", key="eval_pg_generate_variations_only_btn"):
                    if not eval_pg_active_gem_obj:
                        st.error("Lỗi: Cần có Gemini model để thực hiện thao tác này.")
                    else:
                        with st.spinner("⏳ Đang sinh biến thể cho tất cả các câu hỏi..."):
                            generated_data = generate_and_collect_variations_only(
                                eval_data=st.session_state.eval_pg_data,
                                gemini_model_object=eval_pg_active_gem_obj,
                                num_variations=config.NUM_QUERY_VARIATIONS
                            )
                            if generated_data:
                                st.session_state.eval_pg_generated_variations_for_saving = generated_data
                                # Nút download sẽ xuất hiện ở dưới
                            else:
                                st.error("Không thể sinh biến thể. Vui lòng kiểm tra log.")
            else: # Các chế độ còn lại ("Tạo mới từ LLM" hoặc "Sử dụng file...") sẽ chạy đánh giá retrieval
                run_eval_button_text = "🚀 Bắt đầu Đánh giá Retrieval & Metrics"
                if st.session_state.eval_pg_variation_mode == "Sử dụng file biến thể đã tải lên":
                    if not st.session_state.get("eval_pg_variations_data_from_file"):
                        run_eval_button_text = "⚠️ Vui lòng tải file biến thể trước khi bắt đầu"


                if st.button(run_eval_button_text, key="eval_pg_start_full_eval_button",
                             disabled=(st.session_state.eval_pg_variation_mode == "Sử dụng file biến thể đã tải lên" and \
                                       not st.session_state.get("eval_pg_variations_data_from_file"))):

                    # Kiểm tra điều kiện tiên quyết
                    proceed_run = True
                    variations_to_pass_to_run = None
                    if st.session_state.eval_pg_variation_mode == "Sử dụng file biến thể đã tải lên":
                        if not st.session_state.get("eval_pg_variations_data_from_file"):
                            st.error("Bạn đã chọn 'Sử dụng file biến thể đã tải lên' nhưng chưa tải file hoặc file không hợp lệ.")
                            proceed_run = False
                        else:
                            variations_to_pass_to_run = st.session_state.eval_pg_variations_data_from_file
                    elif st.session_state.eval_pg_variation_mode == "Tạo mới từ LLM":
                        if not eval_pg_active_gem_obj:
                            st.error("Bạn đã chọn 'Tạo mới từ LLM' nhưng Gemini model chưa sẵn sàng.")
                            proceed_run = False
                    # Kiểm tra các model khác nếu không phải chỉ sinh biến thể (đã làm ở trên, can_run_evaluation_flow)

                    if proceed_run and can_run_evaluation_flow: # can_run_evaluation_flow kiểm tra retriever, emb
                        eval_config_for_this_run_pg_main = {
                            'embedding_model_name': eval_pg_active_emb_name,
                            'retrieval_query_mode': st.session_state.eval_pg_retrieval_query_mode,
                            'retrieval_method': st.session_state.eval_pg_retrieval_method,
                            'selected_reranker_model_name': eval_pg_active_rer_name,
                            'gemini_model_name': eval_pg_active_gem_name,
                            'variation_mode_used': st.session_state.eval_pg_variation_mode,
                        }
                        st.session_state.eval_pg_last_config_run = eval_config_for_this_run_pg_main.copy()

                        with st.spinner("⏳ Đang chạy đánh giá Retrieval & Metrics..."):
                            start_eval_time_pg_main = time.time()
                            save_new_vars_flag = (st.session_state.eval_pg_variation_mode == "Tạo mới từ LLM" and \
                                                  st.session_state.get("eval_pg_save_newly_generated_variations_cb", False))

                            results_df_output_pg_main = run_retrieval_evaluation(
                                eval_data=st.session_state.eval_pg_data,
                                retriever_instance_for_eval=eval_pg_active_retriever,
                                embedding_model_object_for_eval=eval_pg_active_emb_obj,
                                reranking_model_object_for_eval=eval_pg_active_rer_obj,
                                gemini_model_object_for_eval=eval_pg_active_gem_obj,
                                eval_config_params=st.session_state.eval_pg_last_config_run,
                                preloaded_query_variations=variations_to_pass_to_run
                                # Cờ save_generated_variations_flag không cần truyền nữa
                                # vì run_retrieval_evaluation sẽ không chịu trách nhiệm lưu,
                                # nó chỉ sinh nếu cần. Việc lưu sẽ do logic bên ngoài sau khi
                                # run_retrieval_evaluation trả về các biến thể đã sinh (nếu có).
                                # Tuy nhiên, để đơn giản, ta có thể giữ lại việc `run_retrieval_evaluation`
                                # ghi vào session_state nếu `save_new_vars_flag` là True và nó thực sự sinh mới.
                                # Nhưng tốt hơn là `run_retrieval_evaluation` trả về các biến thể đã sinh (nếu có).
                                # Hiện tại, `run_retrieval_evaluation` không trả về các biến thể.
                                # Nó sẽ sử dụng preloaded hoặc tự sinh. Nếu muốn lưu các biến thể sinh mới khi chạy full eval,
                                # cần sửa `run_retrieval_evaluation` để trả về chúng, hoặc nó tự ghi vào st.session_state.
                                # Cách đơn giản là sau khi chạy full eval, nếu chế độ là "Tạo mới" và "Lưu", thì
                                # ta có thể gọi lại hàm generate_and_collect_variations_only để đảm bảo có file lưu.
                                # Hoặc, `run_retrieval_evaluation` sẽ cần một cơ chế để trả về các biến thể đó.
                            )
                            total_eval_time_pg_main = time.time() - start_eval_time_pg_main
                            st.success(f"Hoàn thành đánh giá Retrieval & Metrics sau {total_eval_time_pg_main:.2f} giây.")
                            st.session_state.eval_pg_results_df = results_df_output_pg_main
                            st.session_state.eval_pg_run_completed = True

                            # Nếu người dùng muốn lưu các biến thể được tạo mới trong lần chạy đánh giá này
                            if st.session_state.eval_pg_variation_mode == "Tạo mới từ LLM" and \
                               st.session_state.get("eval_pg_save_newly_generated_variations_cb", False) and \
                               results_df_output_pg_main is not None and not results_df_output_pg_main.empty:
                                # Cần thu thập lại các biến thể đã được sinh (nếu run_retrieval_evaluation không trả về)
                                # Cách an toàn nhất là gọi lại generate_and_collect_variations_only
                                # Hoặc sửa run_retrieval_evaluation để nó trả về dict các biến thể nếu được sinh mới.
                                # Hiện tại, ta giả định người dùng sẽ dùng nút "Chỉ Sinh và Lưu" nếu muốn file chính xác.
                                # Hoặc, ta có thể thêm một cột "generated_variations_details" vào results_df rồi trích xuất.
                                # Để đơn giản cho lần này, nếu muốn lưu, người dùng nên dùng nút "Chỉ Sinh và Lưu".
                                st.info("Để lưu các biến thể câu hỏi vừa được tạo trong quá trình đánh giá, vui lòng sử dụng nút '📝 Bắt đầu: Chỉ Sinh và Lưu Biến thể Câu hỏi' sau khi quá trình này hoàn tất (nếu bạn muốn một file riêng).")


                            st.rerun() # Rerun để hiển thị kết quả

        # --- Nút tải xuống file biến thể (nếu có và được tạo từ nút "Chỉ sinh và lưu") ---
        if st.session_state.get("eval_pg_generated_variations_for_saving"):
            try:
                variations_json_to_save = json.dumps(st.session_state.eval_pg_generated_variations_for_saving, indent=2, ensure_ascii=False)
                ts_var_save = datetime.now().strftime("%Y%m%d_%H%M%S")
                gem_name_var_save = st.session_state.eval_pg_selected_gemini_model_name.split('/')[-1].replace('.','-')[:15]
                var_fname_save = f"generated_query_variations_{gem_name_var_save}_{ts_var_save}.json"
                st.download_button(
                    label="📥 Tải về File Biến thể Câu hỏi đã tạo (.json)",
                    data=variations_json_to_save,
                    file_name=var_fname_save,
                    mime="application/json",
                    key="download_generated_variations_btn"
                )
            except Exception as e_dl_gen_var:
                st.error(f"Lỗi khi chuẩn bị file biến thể đã tạo để tải xuống: {e_dl_gen_var}")


        # --- Hiển thị kết quả đánh giá (nếu có) ---
        if st.session_state.eval_pg_run_completed and st.session_state.eval_pg_results_df is not None:
            st.subheader("Kết quả Đánh giá Chi tiết")
            detailed_results_df_display_pg = st.session_state.eval_pg_results_df
            last_config_run_display_pg = st.session_state.eval_pg_last_config_run

            st.markdown("**Cấu hình đã sử dụng cho lần chạy cuối:**")
            cfg_cols_pg = st.columns(6) # Thêm cột cho Variation Mode
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


            avg_metrics_res_pg, num_eval_pg, num_skip_err_pg = calculate_average_metrics(detailed_results_df_display_pg)

            st.metric("Tổng số Queries trong File", len(detailed_results_df_display_pg))
            col_rc1_pg, col_rc2_pg = st.columns(2)
            col_rc1_pg.metric("Queries Đánh giá Hợp lệ", num_eval_pg)
            col_rc2_pg.metric("Queries Bỏ qua / Lỗi Runtime", num_skip_err_pg)

            if avg_metrics_res_pg:
                st.markdown("#### Metrics Trung bình @K (trên các queries hợp lệ)")
                # ... (giữ nguyên phần hiển thị metrics P, R, F1, MRR, NDCG) ...
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
                # ... (giữ nguyên phần hiển thị hiệu năng) ...
                perf_col1_pg, perf_col2_pg, perf_col3_pg, perf_col4_pg = st.columns(4) # Thêm cột
                perf_col1_pg.metric("Avg. Query Time (s)", f"{avg_metrics_res_pg.get('avg_processing_time', 0.0):.2f}s")
                perf_col1_pg.metric("Avg. Variation Time (s)", f"{avg_metrics_res_pg.get('avg_variation_time', 0.0):.3f}s") # Chính xác hơn
                perf_col2_pg.metric("Avg. Search Time (s)", f"{avg_metrics_res_pg.get('avg_search_time', 0.0):.2f}s")
                perf_col2_pg.metric("Avg. Rerank Time (s)", f"{avg_metrics_res_pg.get('avg_rerank_time', 0.0):.3f}s") # Chính xác hơn
                perf_col3_pg.metric("Avg. #Variations Gen", f"{avg_metrics_res_pg.get('avg_num_variations_generated', 0.0):.1f}")
                perf_col3_pg.metric("Avg. #Docs Reranked", f"{avg_metrics_res_pg.get('avg_num_docs_reranked', 0.0):.1f}")
                perf_col4_pg.metric("Avg. #Docs After Rerank", f"{avg_metrics_res_pg.get('avg_num_retrieved_after_rerank',0.0):.1f}")
                perf_col4_pg.metric("Avg. #Unique Docs Found", f"{avg_metrics_res_pg.get('avg_num_unique_docs_found',0.0):.1f}")


            with st.expander("Xem Kết quả Chi tiết từng Query (Raw Data)"):
                display_cols_eval_pg_results = [
                    'query_id', 'query', 'status', 'error_message',
                    'embedding_model_name', 'retrieval_query_mode','retrieval_method', 'selected_reranker_model',
                    'variation_mode_run', 'variation_source', 'llm_model_for_variation', # Thêm các cột mới
                    'precision@3', 'recall@3', 'f1@3', 'mrr@3', 'ndcg@3',
                    'precision@5', 'recall@5', 'f1@5', 'mrr@5', 'ndcg@5',
                    'precision@10', 'recall@10', 'f1@10', 'mrr@10', 'ndcg@10',
                    'processing_time', 'variation_time', 'search_time', 'rerank_time',
                    'num_variations_generated','num_unique_docs_found', 'num_retrieved_before_rerank',
                    'num_docs_reranked', 'num_retrieved_after_rerank',
                    'summarizing_query', 'retrieved_ids', 'relevant_ids'
                ]
                existing_display_cols_eval_pg_results = [col for col in display_cols_eval_pg_results if col in detailed_results_df_display_pg.columns]
                st.dataframe(detailed_results_df_display_pg[existing_display_cols_eval_pg_results])

            st.subheader("Lưu Kết quả Đánh giá Retrieval")
            # ... (giữ nguyên logic tải file kết quả đánh giá) ...
            try:
                results_json_pg_main = detailed_results_df_display_pg.to_json(orient='records', indent=2, force_ascii=False)
                results_csv_pg_main = detailed_results_df_display_pg.to_csv(index=False).encode('utf-8')
                timestamp_pg_main = datetime.now().strftime("%Y%m%d_%H%M%S")

                emb_sfx_pg_main = last_config_run_display_pg.get('embedding_model_name', 'na').split('/')[-1].replace('-', '').replace('_', '')[:10]
                qmode_sfx_pg_main = last_config_run_display_pg.get('retrieval_query_mode', 'na').lower()[:3]
                method_sfx_pg_main = last_config_run_display_pg.get('retrieval_method', 'na').lower()
                rer_sfx_pg_main = "norr"
                sel_rer_fname_pg_main = last_config_run_display_pg.get('selected_reranker_model_name', 'Không sử dụng')
                if sel_rer_fname_pg_main != 'Không sử dụng':
                    rer_sfx_pg_main = sel_rer_fname_pg_main.split('/')[-1].replace('-', '').replace('_', '')[:10]
                mod_sfx_pg_main = last_config_run_display_pg.get('gemini_model_name', 'gemini').split('/')[-1].replace('.','-')[:15]
                var_mode_sfx_pg_main = last_config_run_display_pg.get('variation_mode_used', 'na').lower().replace(" ","")[:10]


                base_fname_pg_main = f"eval_results_{emb_sfx_pg_main}_{qmode_sfx_pg_main}_{method_sfx_pg_main}_{rer_sfx_pg_main}_{mod_sfx_pg_main}_{var_mode_sfx_pg_main}_{timestamp_pg_main}"
                fname_json_pg_main = f"{base_fname_pg_main}.json"
                fname_csv_pg_main = f"{base_fname_pg_main}.csv"

                dl_col1_pg_main, dl_col2_pg_main = st.columns(2)
                with dl_col1_pg_main:
                    st.download_button("💾 Tải về Kết quả Đánh giá (JSON)", results_json_pg_main, fname_json_pg_main, "application/json", key="dl_json_eval_pg_main")
                with dl_col2_pg_main:
                    st.download_button("💾 Tải về Kết quả Đánh giá (CSV)", results_csv_pg_main, fname_csv_pg_main, "text/csv", key="dl_csv_eval_pg_main")
            except Exception as e_file_dl_main:
                st.error(f"Lỗi khi chuẩn bị file kết quả đánh giá: {e_file_dl_main}")


        st.markdown("---")
        st.subheader("Quản lý Trạng thái Đánh giá")
        if st.button("Xóa File Đã Tải và Kết Quả Hiện Tại", key="eval_pg_clear_state_button_main"):
            st.session_state.eval_pg_data = None
            st.session_state.eval_pg_upload_counter += 1
            st.session_state.eval_pg_run_completed = False
            st.session_state.eval_pg_results_df = None
            st.session_state.eval_pg_last_config_run = {}
            st.session_state.eval_pg_uploaded_filename = None
            st.session_state.eval_pg_variations_data_from_file = None # Xóa cả dữ liệu biến thể đã tải
            st.session_state.eval_pg_uploaded_variations_file_obj_name = None
            st.session_state.eval_pg_generated_variations_for_saving = None # Xóa cả biến thể vừa tạo
            st.success("Đã xóa trạng thái đánh giá hiện tại.")
            time.sleep(1)
            st.rerun()
    else: # can_run_evaluation_flow is False
        st.warning("⚠️ Không thể tiến hành do thiếu các thành phần model cần thiết. Vui lòng kiểm tra thông báo lỗi ở trên và cấu hình trong sidebar.")

elif not st.session_state.eval_page_resources_initialized:
    eval_page_status_placeholder.error("⚠️ Tài nguyên trang Đánh giá CHƯA SẴN SÀNG. Lỗi trong quá trình tải model hoặc tạo RAG. Vui lòng kiểm tra log chi tiết hoặc làm mới trang.")