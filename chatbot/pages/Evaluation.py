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

# Hàm chạy đánh giá không thay đổi nhiều, chỉ cần đảm bảo nó nhận đúng các model objects
def run_retrieval_evaluation(
    eval_data: list,
    retriever_instance_for_eval, # Đã được chọn
    embedding_model_object_for_eval, # Đã được chọn
    reranking_model_object_for_eval, # Có thể None
    gemini_model_object_for_eval,    # Đã được chọn
    eval_config_params: dict # Chứa tên các model đã chọn cho lần chạy này
    ):
    results_list = []
    k_values_metrics = [3, 5, 10] # Đổi tên để tránh xung đột

    # Lấy các tham số cấu hình từ eval_config_params
    retrieval_query_mode_eval = eval_config_params.get('retrieval_query_mode', 'Tổng quát')
    retrieval_method_eval = eval_config_params.get('retrieval_method', 'hybrid')
    selected_reranker_name_eval_run = eval_config_params.get('selected_reranker_model_name', 'Không sử dụng')
    use_reranker_eval_run = reranking_model_object_for_eval is not None and selected_reranker_name_eval_run != 'Không sử dụng'

    progress_bar_eval = st.progress(0)
    status_text_area_eval = st.empty()
    total_items_eval = len(eval_data)
    # ... (logic tạm dừng giữa các batch giữ nguyên từ file gốc) ...
    queries_per_batch_eval = 15
    wait_time_seconds_eval = 60 # Giảm thời gian chờ nếu Gemini API cho phép

    for i, item_eval in enumerate(eval_data):
        if i > 0 and i % queries_per_batch_eval == 0:
            pause_msg_eval = f"Đã xử lý {i}/{total_items_eval} queries. Tạm dừng {wait_time_seconds_eval} giây để tránh rate limit..."
            status_text_area_eval.text(pause_msg_eval)
            time.sleep(wait_time_seconds_eval) # Tạm dừng

        query_id_eval = item_eval.get("query_id")
        original_query_eval = item_eval.get("query")
        relevant_chunk_ids_eval = set(str(cid) for cid in item_eval.get("relevant_chunk_ids", [])) # Đảm bảo là set của string

        # Hiển thị thông tin model đang dùng cho query này
        emb_name_disp_eval = eval_config_params.get('embedding_model_name', 'N/A').split('/')[-1]
        rer_name_disp_eval = selected_reranker_name_eval_run.split('/')[-1] if selected_reranker_name_eval_run != 'Không sử dụng' else "Tắt"
        status_text_area_eval.text(
            f"Đang xử lý query {i+1}/{total_items_eval}: {query_id_eval}\n"
            f"(Emb: {emb_name_disp_eval}, QueryMode: {retrieval_query_mode_eval}, "
            f"Method: {retrieval_method_eval}, Reranker: {rer_name_disp_eval})"
        )

        start_time_eval_q = time.time()
        query_metrics_dict = { #
            "query_id": query_id_eval, "query": original_query_eval,
            "embedding_model_name": eval_config_params.get('embedding_model_name', 'N/A'), #
            "retrieval_query_mode": retrieval_query_mode_eval, #
            "retrieval_method": retrieval_method_eval, #
            "selected_reranker_model": selected_reranker_name_eval_run, #
            "status": "error_default", "retrieved_ids": [], "relevant_ids": list(relevant_chunk_ids_eval), #
            "processing_time": 0.0, 'summarizing_query': '', #
            'variation_time': 0.0, 'search_time': 0.0, 'rerank_time': 0.0, #
            'num_variations_generated': 0, 'num_unique_docs_found': 0, 'num_docs_reranked': 0, #
            'num_retrieved_before_rerank': 0, 'num_retrieved_after_rerank': 0, 'error_message': '' #
        }
        for k_val_m in k_values_metrics: #
            query_metrics_dict[f'precision@{k_val_m}'] = 0.0; query_metrics_dict[f'recall@{k_val_m}'] = 0.0 #
            query_metrics_dict[f'f1@{k_val_m}'] = 0.0; query_metrics_dict[f'mrr@{k_val_m}'] = 0.0; query_metrics_dict[f'ndcg@{k_val_m}'] = 0.0 #

        try:
            var_start_eval = time.time()
            relevance_status_eval, _, all_queries_eval, summarizing_q_eval = generate_query_variations( #
                original_query=original_query_eval,
                gemini_model=gemini_model_object_for_eval,
                chat_history=None,
                num_variations=config.NUM_QUERY_VARIATIONS #
            )
            query_metrics_dict["variation_time"] = time.time() - var_start_eval #
            query_metrics_dict["summarizing_query"] = summarizing_q_eval #
            query_metrics_dict["num_variations_generated"] = len(all_queries_eval) -1 if isinstance(all_queries_eval, list) and len(all_queries_eval) > 0 else 0 #

            if relevance_status_eval == 'invalid': #
                query_metrics_dict["status"] = "skipped_irrelevant" #
            else:
                queries_to_search_eval = []
                if retrieval_query_mode_eval == 'Đơn giản': queries_to_search_eval = [original_query_eval] #
                elif retrieval_query_mode_eval == 'Tổng quát': queries_to_search_eval = [summarizing_q_eval] if summarizing_q_eval else [original_query_eval] #
                elif retrieval_query_mode_eval == 'Sâu': queries_to_search_eval = all_queries_eval if all_queries_eval else [original_query_eval] #

                collected_docs_data_eval = {}
                search_start_eval_q = time.time()
                for q_var_eval in queries_to_search_eval:
                    if not q_var_eval: continue
                    search_results_eval = retriever_instance_for_eval.search( #
                        q_var_eval,
                        embedding_model_object_for_eval,
                        method=retrieval_method_eval,
                        k=config.VECTOR_K_PER_QUERY #
                    )
                    for res_item_eval in search_results_eval: #
                        doc_idx_eval = res_item_eval.get('index') #
                        if isinstance(doc_idx_eval, int) and doc_idx_eval >= 0 and doc_idx_eval not in collected_docs_data_eval: #
                            collected_docs_data_eval[doc_idx_eval] = res_item_eval #
                query_metrics_dict["search_time"] = time.time() - search_start_eval_q #
                query_metrics_dict["num_unique_docs_found"] = len(collected_docs_data_eval) #

                retrieved_docs_list_eval = list(collected_docs_data_eval.values()) #
                sort_reverse_eval = (retrieval_method_eval != 'dense') #
                retrieved_docs_list_eval.sort(key=lambda x: x.get('score', 0 if sort_reverse_eval else float('inf')), reverse=sort_reverse_eval) #
                query_metrics_dict["num_retrieved_before_rerank"] = len(retrieved_docs_list_eval) #

                final_docs_for_metrics_eval = []
                rerank_start_eval_q_time = time.time()

                if use_reranker_eval_run and retrieved_docs_list_eval: #
                    query_for_reranking_eval = summarizing_q_eval if summarizing_q_eval else original_query_eval #
                    docs_to_rerank_input_eval = retrieved_docs_list_eval[:config.MAX_DOCS_FOR_RERANK] #
                    query_metrics_dict["num_docs_reranked"] = len(docs_to_rerank_input_eval) #
                    rerank_input_fmt_eval = [{'doc': item_d['doc'], 'index': item_d['index']} for item_d in docs_to_rerank_input_eval] #

                    reranked_results_eval = rerank_documents( #
                        query_for_reranking_eval,
                        rerank_input_fmt_eval,
                        reranking_model_object_for_eval
                    )
                    final_docs_for_metrics_eval = reranked_results_eval[:config.FINAL_NUM_RESULTS_AFTER_RERANK] #
                    query_metrics_dict["rerank_time"] = time.time() - rerank_start_eval_q_time #
                elif retrieved_docs_list_eval: #
                    temp_final_docs_eval = retrieved_docs_list_eval[:config.FINAL_NUM_RESULTS_AFTER_RERANK] #
                    final_docs_for_metrics_eval = [ #
                        {'doc': item['doc'], 'score': item.get('score'), 'original_index': item['index']}
                        for item in temp_final_docs_eval
                    ]
                    query_metrics_dict["rerank_time"] = 0.0 #
                    query_metrics_dict["num_docs_reranked"] = 0 #
                else: #
                    query_metrics_dict["rerank_time"] = 0.0 #
                    query_metrics_dict["num_docs_reranked"] = 0 #

                query_metrics_dict["num_retrieved_after_rerank"] = len(final_docs_for_metrics_eval) #
                retrieved_ids_eval_q = []
                for res_final_eval in final_docs_for_metrics_eval: #
                    doc_data_eval = res_final_eval.get('doc', {}) #
                    chunk_id_eval = None
                    if isinstance(doc_data_eval, dict): #
                        # Cố gắng lấy 'id' hoặc 'chunk_id' từ metadata hoặc trực tiếp từ chunk
                        chunk_id_eval = doc_data_eval.get('id') or \
                                        doc_data_eval.get('chunk_id') or \
                                        doc_data_eval.get('metadata', {}).get('id') or \
                                        doc_data_eval.get('metadata', {}).get('chunk_id')
                    if chunk_id_eval is not None: #
                        retrieved_ids_eval_q.append(str(chunk_id_eval)) #

                query_metrics_dict["retrieved_ids"] = retrieved_ids_eval_q #
                query_metrics_dict["status"] = "evaluated" #
                for k_val_m_calc in k_values_metrics: #
                    query_metrics_dict[f'precision@{k_val_m_calc}'] = precision_at_k(retrieved_ids_eval_q, relevant_chunk_ids_eval, k_val_m_calc) #
                    query_metrics_dict[f'recall@{k_val_m_calc}'] = recall_at_k(retrieved_ids_eval_q, relevant_chunk_ids_eval, k_val_m_calc) #
                    query_metrics_dict[f'f1@{k_val_m_calc}'] = f1_at_k(retrieved_ids_eval_q, relevant_chunk_ids_eval, k_val_m_calc) #
                    query_metrics_dict[f'mrr@{k_val_m_calc}'] = mrr_at_k(retrieved_ids_eval_q, relevant_chunk_ids_eval, k_val_m_calc) #
                    query_metrics_dict[f'ndcg@{k_val_m_calc}'] = ndcg_at_k(retrieved_ids_eval_q, relevant_chunk_ids_eval, k_val_m_calc) #

        except Exception as e_eval_q:
            query_metrics_dict["status"] = "error_runtime" #
            query_metrics_dict["error_message"] = f"{type(e_eval_q).__name__}: {str(e_eval_q)}" #
            # Ghi log chi tiết hơn vào đâu đó nếu cần, ví dụ st.sidebar.expander
            st.sidebar.error(f"Lỗi khi xử lý query {query_id_eval}: {e_eval_q}")
        finally:
            query_metrics_dict["processing_time"] = time.time() - start_time_eval_q #
            results_list.append(query_metrics_dict)
            progress_bar_eval.progress((i + 1) / total_items_eval)

    status_text_area_eval.text(f"Hoàn thành đánh giá {total_items_eval} queries!")
    return pd.DataFrame(results_list)

# --- Trang Streamlit cho Đánh giá ---
st.set_page_config(page_title="Đánh giá Retrieval", layout="wide")
st.title("📊 Đánh giá Hệ thống Retrieval")
st.markdown("Trang này cho phép bạn chạy đánh giá hiệu suất của hệ thống retrieval và reranking với các mô hình đã được tải trước.")

# --- Khởi tạo Session State cho trang Đánh giá ---
# Model selections (lưu tên model)
if "eval_pg_selected_embedding_model_name" not in st.session_state:
    st.session_state.eval_pg_selected_embedding_model_name = config.DEFAULT_EMBEDDING_MODEL #
if "eval_pg_selected_gemini_model_name" not in st.session_state:
    st.session_state.eval_pg_selected_gemini_model_name = config.DEFAULT_GEMINI_MODEL #
if "eval_pg_selected_reranker_model_name" not in st.session_state:
    st.session_state.eval_pg_selected_reranker_model_name = config.DEFAULT_RERANKER_MODEL #

# Other eval configs
if "eval_pg_retrieval_query_mode" not in st.session_state: st.session_state.eval_pg_retrieval_query_mode = 'Tổng quát' #
if "eval_pg_retrieval_method" not in st.session_state: st.session_state.eval_pg_retrieval_method = 'hybrid' #

# Dữ liệu và kết quả đánh giá
if 'eval_pg_data' not in st.session_state: st.session_state.eval_pg_data = None #
if 'eval_pg_results_df' not in st.session_state: st.session_state.eval_pg_results_df = None #
if 'eval_pg_run_completed' not in st.session_state: st.session_state.eval_pg_run_completed = False #
if 'eval_pg_uploaded_filename' not in st.session_state: st.session_state.eval_pg_uploaded_filename = None #
if "eval_pg_upload_counter" not in st.session_state: st.session_state.eval_pg_upload_counter = 0 #
if 'eval_pg_last_config_run' not in st.session_state: st.session_state.eval_pg_last_config_run = {} #

# --- Tải trước Models và RAG cho Trang Đánh giá ---
if "eval_pg_loaded_embedding_models" not in st.session_state:
    st.session_state.eval_pg_loaded_embedding_models = {}
if "eval_pg_loaded_reranker_models" not in st.session_state:
    st.session_state.eval_pg_loaded_reranker_models = {}
if "eval_pg_rag_components_per_embedding_model" not in st.session_state:
    st.session_state.eval_pg_rag_components_per_embedding_model = {}

def initialize_evaluation_page_resources():
    """Tải và chuẩn bị tài nguyên cho trang đánh giá."""
    eval_init_successful = True
    if not st.session_state.eval_pg_loaded_embedding_models:
        with st.status("Đang tải Embedding Models (Đánh giá)...", expanded=True) as emb_s:
            st.session_state.eval_pg_loaded_embedding_models = load_all_embedding_models() #
            if not st.session_state.eval_pg_loaded_embedding_models:
                emb_s.update(label="Lỗi tải Embedding Models (Đánh giá)!", state="error")
                eval_init_successful = False
            else:
                emb_s.update(label="Tải xong Embedding Models (Đánh giá).", state="complete")

    if not st.session_state.eval_pg_loaded_reranker_models:
        with st.status("Đang tải Reranker Models (Đánh giá)...", expanded=True) as rer_s:
            st.session_state.eval_pg_loaded_reranker_models = load_all_reranker_models() #
            # Không coi là lỗi nghiêm trọng nếu reranker không tải được hết
            rer_s.update(label="Tải xong Reranker Models (Đánh giá).", state="complete")

    if eval_init_successful and st.session_state.eval_pg_loaded_embedding_models:
        for model_n, emb_obj in st.session_state.eval_pg_loaded_embedding_models.items():
            if model_n not in st.session_state.eval_pg_rag_components_per_embedding_model:
                with st.status(f"Chuẩn bị RAG cho '{model_n.split('/')[-1]}' (Đánh giá)...", expanded=True) as rag_s:
                    rag_prefix = config.get_rag_data_prefix(model_n) #
                    try:
                        v_db, retr = load_or_create_rag_components(emb_obj, rag_prefix) #
                        if v_db and retr:
                            st.session_state.eval_pg_rag_components_per_embedding_model[model_n] = (v_db, retr)
                            rag_s.update(label=f"RAG cho '{model_n.split('/')[-1]}' (Đánh giá) sẵn sàng.", state="complete")
                        else:
                            rag_s.update(label=f"Lỗi RAG cho '{model_n.split('/')[-1]}' (Đánh giá).", state="error")
                            eval_init_successful = False; break
                    except Exception as e_rag:
                        rag_s.update(label=f"Exception RAG cho '{model_n.split('/')[-1]}' (Đánh giá): {e_rag}", state="error")
                        eval_init_successful = False; break
    elif not st.session_state.eval_pg_loaded_embedding_models:
        eval_init_successful = False
    return eval_init_successful

# --- Sidebar cho Trang Đánh giá ---
with st.sidebar:
    st.title("Tùy chọn Đánh giá")
    st.header("Mô hình")

    current_eval_emb_name_sb = st.session_state.eval_pg_selected_embedding_model_name
    current_eval_gem_name_sb = st.session_state.eval_pg_selected_gemini_model_name
    current_eval_rer_name_sb = st.session_state.eval_pg_selected_reranker_model_name

    # Danh sách các embedding model đã được tải cho trang đánh giá
    eval_pg_avail_emb_names = list(st.session_state.get("eval_pg_loaded_embedding_models", {}).keys())
    if not eval_pg_avail_emb_names: eval_pg_avail_emb_names = config.AVAILABLE_EMBEDDING_MODELS #

    eval_sel_emb_name_ui = st.selectbox(
        "Chọn mô hình Embedding (Đánh giá):", options=eval_pg_avail_emb_names,
        index=eval_pg_avail_emb_names.index(current_eval_emb_name_sb) if current_eval_emb_name_sb in eval_pg_avail_emb_names else 0,
        key="eval_pg_emb_select", help="Chọn embedding model đã tải trước cho đánh giá."
    )
    if eval_sel_emb_name_ui != st.session_state.eval_pg_selected_embedding_model_name:
        st.session_state.eval_pg_selected_embedding_model_name = eval_sel_emb_name_ui
        st.rerun()

    eval_sel_gem_name_ui = st.selectbox(
        "Chọn mô hình Gemini (Đánh giá Query Variations):", options=config.AVAILABLE_GEMINI_MODELS, #
        index=config.AVAILABLE_GEMINI_MODELS.index(current_eval_gem_name_sb) if current_eval_gem_name_sb in config.AVAILABLE_GEMINI_MODELS else 0, #
        key="eval_pg_gem_select", help="Chọn Gemini model cho đánh giá."
    )
    if eval_sel_gem_name_ui != st.session_state.eval_pg_selected_gemini_model_name:
        st.session_state.eval_pg_selected_gemini_model_name = eval_sel_gem_name_ui
        st.rerun()

    eval_pg_avail_rer_names = list(st.session_state.get("eval_pg_loaded_reranker_models", {}).keys())
    if not eval_pg_avail_rer_names: eval_pg_avail_rer_names = config.AVAILABLE_RERANKER_MODELS #

    eval_sel_rer_name_ui = st.selectbox(
        "Chọn mô hình Reranker (Đánh giá):", options=eval_pg_avail_rer_names,
        index=eval_pg_avail_rer_names.index(current_eval_rer_name_sb) if current_eval_rer_name_sb in eval_pg_avail_rer_names else 0,
        key="eval_pg_rer_select", help="Chọn reranker model đã tải trước. 'Không sử dụng' để tắt."
    )
    if eval_sel_rer_name_ui != st.session_state.eval_pg_selected_reranker_model_name:
        st.session_state.eval_pg_selected_reranker_model_name = eval_sel_rer_name_ui
        st.rerun()

    st.header("Cấu hình Retrieval (Đánh giá)")
    st.radio("Nguồn câu hỏi:", options=['Đơn giản', 'Tổng quát', 'Sâu'],
             index=['Đơn giản', 'Tổng quát', 'Sâu'].index(st.session_state.eval_pg_retrieval_query_mode),
             key="eval_pg_retrieval_query_mode", horizontal=True) #
    st.radio("Phương thức Retrieval:", options=['dense', 'sparse', 'hybrid'],
             index=['dense', 'sparse', 'hybrid'].index(st.session_state.eval_pg_retrieval_method),
             key="eval_pg_retrieval_method", horizontal=True) #


# --- Khởi tạo tài nguyên cho trang Đánh giá ---
eval_page_status_placeholder = st.empty()
if "eval_page_resources_initialized" not in st.session_state:
    st.session_state.eval_page_resources_initialized = False

if not st.session_state.eval_page_resources_initialized:
    with st.spinner("Đang khởi tạo tài nguyên cho trang Đánh giá..."):
        eval_resources_ready = initialize_evaluation_page_resources()
        st.session_state.eval_page_resources_initialized = eval_resources_ready

if st.session_state.eval_page_resources_initialized:
    eval_page_status_placeholder.success("✅ Tài nguyên trang Đánh giá đã sẵn sàng!")

    # Lấy các object model cần thiết DỰA TRÊN LỰA CHỌN HIỆN TẠI trên trang Đánh giá
    eval_pg_active_emb_name = st.session_state.eval_pg_selected_embedding_model_name
    eval_pg_active_rer_name = st.session_state.eval_pg_selected_reranker_model_name
    eval_pg_active_gem_name = st.session_state.eval_pg_selected_gemini_model_name

    eval_pg_active_emb_obj = st.session_state.eval_pg_loaded_embedding_models.get(eval_pg_active_emb_name)
    eval_pg_active_rag_comps = st.session_state.eval_pg_rag_components_per_embedding_model.get(eval_pg_active_emb_name)
    eval_pg_active_retriever = eval_pg_active_rag_comps[1] if eval_pg_active_rag_comps else None
    eval_pg_active_rer_obj = st.session_state.eval_pg_loaded_reranker_models.get(eval_pg_active_rer_name)
    eval_pg_active_gem_obj = load_gemini_model(eval_pg_active_gem_name) #

    # --- Kiểm tra các thành phần trước khi cho phép chạy đánh giá ---
    can_run_evaluation = True
    if not eval_pg_active_emb_obj:
        st.error(f"Lỗi: Embedding model '{eval_pg_active_emb_name.split('/')[-1]}' (Đánh giá) chưa tải.")
        can_run_evaluation = False
    if not eval_pg_active_retriever:
        st.error(f"Lỗi: Retriever cho '{eval_pg_active_emb_name.split('/')[-1]}' (Đánh giá) chưa sẵn sàng.")
        can_run_evaluation = False
    if not eval_pg_active_gem_obj:
        st.error(f"Lỗi: Gemini model '{eval_pg_active_gem_name}' (Đánh giá) chưa tải.")
        can_run_evaluation = False

    if can_run_evaluation:
        st.caption( #
            f"Đánh giá với: Embedding: `{eval_pg_active_emb_name.split('/')[-1]}` | "
            f"Gemini: `{eval_pg_active_gem_name}` | "
            f"Query Mode: `{st.session_state.eval_pg_retrieval_query_mode}` | " #
            f"Retrieval: `{st.session_state.eval_pg_retrieval_method}` | " #
            f"Reranker: `{eval_pg_active_rer_name.split('/')[-1] if eval_pg_active_rer_name != 'Không sử dụng' else 'Tắt'}`" #
        )

        uploader_key_eval_pg = f"eval_pg_file_uploader_{st.session_state.eval_pg_upload_counter}" #
        st.subheader("Tải Lên File Đánh giá (.json)") #
        uploaded_file_eval_pg = st.file_uploader("Chọn file JSON chứa dữ liệu đánh giá...", type=["json"], key=uploader_key_eval_pg) #

        if uploaded_file_eval_pg is not None: #
            if uploaded_file_eval_pg.name != st.session_state.eval_pg_uploaded_filename: #
                try:
                    eval_data_list_pg = json.loads(uploaded_file_eval_pg.getvalue().decode('utf-8')) #
                    st.session_state.eval_pg_data = eval_data_list_pg #
                    st.session_state.eval_pg_uploaded_filename = uploaded_file_eval_pg.name #
                    st.session_state.eval_pg_run_completed = False #
                    st.session_state.eval_pg_results_df = None #
                    st.session_state.eval_pg_last_config_run = {} #
                    st.success(f"Đã tải file '{uploaded_file_eval_pg.name}' ({len(eval_data_list_pg)} câu hỏi).") #
                except Exception as e_json: #
                    st.error(f"Lỗi xử lý file JSON: {e_json}") #
                    st.session_state.eval_pg_data = None; st.session_state.eval_pg_uploaded_filename = None #

        if st.session_state.eval_pg_data is not None: #
            st.info(f"Sẵn sàng đánh giá với: **{st.session_state.eval_pg_uploaded_filename}** ({len(st.session_state.eval_pg_data)} câu hỏi).") #
            if st.checkbox("Hiển thị dữ liệu mẫu (5 dòng đầu)", key="eval_pg_show_preview"): #
                st.dataframe(pd.DataFrame(st.session_state.eval_pg_data).head()) #

            if st.button("🚀 Bắt đầu Đánh giá", key="eval_pg_start_button"): #
                # Lưu cấu hình của lần chạy này
                eval_config_for_this_run_pg = { #
                    'embedding_model_name': eval_pg_active_emb_name, #
                    'retrieval_query_mode': st.session_state.eval_pg_retrieval_query_mode, #
                    'retrieval_method': st.session_state.eval_pg_retrieval_method, #
                    'selected_reranker_model_name': eval_pg_active_rer_name, #
                    'gemini_model_name': eval_pg_active_gem_name, #
                }
                st.session_state.eval_pg_last_config_run = eval_config_for_this_run_pg.copy() #

                with st.spinner("⏳ Đang chạy đánh giá... Thao tác này có thể mất nhiều thời gian."): #
                    start_eval_time_pg = time.time() #
                    results_df_output_pg = run_retrieval_evaluation( #
                        eval_data=st.session_state.eval_pg_data, #
                        retriever_instance_for_eval=eval_pg_active_retriever, #
                        embedding_model_object_for_eval=eval_pg_active_emb_obj, #
                        reranking_model_object_for_eval=eval_pg_active_rer_obj, #
                        gemini_model_object_for_eval=eval_pg_active_gem_obj, #
                        eval_config_params=st.session_state.eval_pg_last_config_run #
                    )
                    total_eval_time_pg = time.time() - start_eval_time_pg #
                    st.success(f"Hoàn thành đánh giá sau {total_eval_time_pg:.2f} giây.") #
                    st.session_state.eval_pg_results_df = results_df_output_pg #
                    st.session_state.eval_pg_run_completed = True #
                    st.rerun()

        if st.session_state.eval_pg_run_completed and st.session_state.eval_pg_results_df is not None: #
            st.subheader("Kết quả Đánh giá Chi tiết") #
            # ... (Phần hiển thị kết quả và tải về giữ nguyên như trong file Evaluation.py gốc)
            # Chỉ cần đảm bảo sử dụng đúng các biến của trang Đánh giá (có suffix _pg)
            # Ví dụ: st.session_state.eval_pg_results_df, st.session_state.eval_pg_last_config_run
            detailed_results_df_display_pg = st.session_state.eval_pg_results_df #
            last_config_run_display_pg = st.session_state.eval_pg_last_config_run #

            st.markdown("**Cấu hình đã sử dụng cho lần chạy cuối:**") #
            cfg_col1_pg, cfg_col2_pg, cfg_col3_pg, cfg_col4_pg, cfg_col5_pg = st.columns(5) #
            emb_n_disp_pg = last_config_run_display_pg.get('embedding_model_name', 'N/A').split('/')[-1] #
            cfg_col1_pg.metric("Embedding", emb_n_disp_pg) #
            cfg_col2_pg.metric("Query Mode", last_config_run_display_pg.get('retrieval_query_mode', 'N/A')) #
            cfg_col3_pg.metric("Ret. Method", last_config_run_display_pg.get('retrieval_method', 'N/A')) #
            rer_n_disp_pg = last_config_run_display_pg.get('selected_reranker_model_name', 'N/A').split('/')[-1] #
            if rer_n_disp_pg == "Không sử dụng".split('/')[-1]: rer_n_disp_pg = "Tắt" #
            cfg_col4_pg.metric("Reranker", rer_n_disp_pg) #
            gem_n_disp_pg = last_config_run_display_pg.get('gemini_model_name', 'N/A').split('/')[-1] #
            cfg_col5_pg.metric("Gemini (Var)", gem_n_disp_pg) #

            avg_metrics_res_pg, num_eval_pg, num_skip_err_pg = calculate_average_metrics(detailed_results_df_display_pg) #

            st.metric("Tổng số Queries trong File", len(detailed_results_df_display_pg)) #
            col_rc1_pg, col_rc2_pg = st.columns(2) #
            col_rc1_pg.metric("Queries Đánh giá Hợp lệ", num_eval_pg) #
            col_rc2_pg.metric("Queries Bỏ qua / Lỗi Runtime", num_skip_err_pg) #

            if avg_metrics_res_pg: #
                st.markdown("#### Metrics Trung bình @K (trên các queries hợp lệ)") #
                k_vals_disp_pg = [3, 5, 10] #
                cols_k_pg = st.columns(len(k_vals_disp_pg)) #
                for idx_k_pg, k_v_pg in enumerate(k_vals_disp_pg): #
                    with cols_k_pg[idx_k_pg]: #
                        st.markdown(f"**K = {k_v_pg}**") #
                        st.text(f"Precision: {avg_metrics_res_pg.get(f'avg_precision@{k_v_pg}', 0.0):.4f}") #
                        st.text(f"Recall:    {avg_metrics_res_pg.get(f'avg_recall@{k_v_pg}', 0.0):.4f}") #
                        st.text(f"F1:        {avg_metrics_res_pg.get(f'avg_f1@{k_v_pg}', 0.0):.4f}") #
                        st.text(f"MRR:       {avg_metrics_res_pg.get(f'avg_mrr@{k_v_pg}', 0.0):.4f}") #
                        st.text(f"NDCG:      {avg_metrics_res_pg.get(f'avg_ndcg@{k_v_pg}', 0.0):.4f}") #
                # Phần hiển thị hiệu năng giữ nguyên
                st.markdown("#### Thông tin Hiệu năng & Số lượng Trung bình (trên các queries hợp lệ)") #
                perf_col1_pg, perf_col2_pg, perf_col3_pg = st.columns(3) #
                perf_col1_pg.metric("Avg. Query Time (s)", f"{avg_metrics_res_pg.get('avg_processing_time', 0.0):.2f}s") #
                perf_col1_pg.metric("Avg. Variation Time (s)", f"{avg_metrics_res_pg.get('avg_variation_time', 0.0):.2f}s") #
                perf_col2_pg.metric("Avg. Search Time (s)", f"{avg_metrics_res_pg.get('avg_search_time', 0.0):.2f}s") #
                perf_col2_pg.metric("Avg. Rerank Time (s)", f"{avg_metrics_res_pg.get('avg_rerank_time', 0.0):.2f}s") #
                perf_col3_pg.metric("Avg. #Variations", f"{avg_metrics_res_pg.get('avg_num_variations_generated', 0.0):.1f}") #
                perf_col3_pg.metric("Avg. #Docs Reranked", f"{avg_metrics_res_pg.get('avg_num_docs_reranked', 0.0):.1f}") #
                perf_col3_pg.metric("Avg. #Docs After Rerank", f"{avg_metrics_res_pg.get('avg_num_retrieved_after_rerank',0.0):.1f}") #


            with st.expander("Xem Kết quả Chi tiết từng Query (Raw Data)"): #
                display_cols_eval_pg = [ #
                    'query_id', 'query', 'status', 'error_message', #
                    'embedding_model_name', 'retrieval_query_mode','retrieval_method', 'selected_reranker_model', #
                    'precision@3', 'recall@3', 'f1@3', 'mrr@3', 'ndcg@3', #
                    'precision@5', 'recall@5', 'f1@5', 'mrr@5', 'ndcg@5', #
                    'precision@10', 'recall@10', 'f1@10', 'mrr@10', 'ndcg@10', #
                    'processing_time', 'variation_time', 'search_time', 'rerank_time', #
                    'num_variations_generated','num_unique_docs_found', 'num_retrieved_before_rerank', #
                    'num_docs_reranked', 'num_retrieved_after_rerank', #
                    'summarizing_query', 'retrieved_ids', 'relevant_ids' #
                ]
                existing_display_cols_eval_pg = [col for col in display_cols_eval_pg if col in detailed_results_df_display_pg.columns] #
                st.dataframe(detailed_results_df_display_pg[existing_display_cols_eval_pg]) #

            st.subheader("Lưu Kết quả Chi tiết") #
            try: #
                results_json_pg = detailed_results_df_display_pg.to_json(orient='records', indent=2, force_ascii=False) #
                results_csv_pg = detailed_results_df_display_pg.to_csv(index=False).encode('utf-8') #
                timestamp_pg = datetime.now().strftime("%Y%m%d_%H%M%S") #

                emb_sfx_pg = last_config_run_display_pg.get('embedding_model_name', 'na').split('/')[-1].replace('-', '').replace('_', '')[:10] #
                qmode_sfx_pg = last_config_run_display_pg.get('retrieval_query_mode', 'na').lower()[:3] #
                method_sfx_pg = last_config_run_display_pg.get('retrieval_method', 'na').lower() #
                rer_sfx_pg = "norr" #
                sel_rer_fname_pg = last_config_run_display_pg.get('selected_reranker_model_name', 'Không sử dụng') #
                if sel_rer_fname_pg != 'Không sử dụng': #
                    rer_sfx_pg = sel_rer_fname_pg.split('/')[-1].replace('-', '').replace('_', '')[:10] #
                mod_sfx_pg = last_config_run_display_pg.get('gemini_model_name', 'gemini').split('/')[-1].replace('.','-')[:15] #

                base_fname_pg = f"eval_results_{emb_sfx_pg}_{qmode_sfx_pg}_{method_sfx_pg}_{rer_sfx_pg}_{mod_sfx_pg}_{timestamp_pg}" #
                fname_json_pg = f"{base_fname_pg}.json" #
                fname_csv_pg = f"{base_fname_pg}.csv" #

                dl_col1_pg, dl_col2_pg = st.columns(2) #
                with dl_col1_pg: #
                    st.download_button("💾 Tải về JSON", results_json_pg, fname_json_pg, "application/json", key="dl_json_eval_pg") #
                with dl_col2_pg: #
                    st.download_button("💾 Tải về CSV", results_csv_pg, fname_csv_pg, "text/csv", key="dl_csv_eval_pg") #
            except Exception as e_file_dl: #
                st.error(f"Lỗi khi chuẩn bị file kết quả: {e_file_dl}") #


        st.markdown("---") #
        st.subheader("Quản lý Trạng thái Đánh giá") #
        if st.button("Xóa File Đã Tải và Kết Quả Hiện Tại", key="eval_pg_clear_state"): #
            st.session_state.eval_pg_data = None #
            st.session_state.eval_pg_upload_counter += 1 #
            st.session_state.eval_pg_run_completed = False #
            st.session_state.eval_pg_results_df = None #
            st.session_state.eval_pg_last_config_run = {} #
            st.session_state.eval_pg_uploaded_filename = None #
            st.success("Đã xóa trạng thái đánh giá hiện tại.") #
            time.sleep(1) #
            st.rerun() #

    else: # can_run_evaluation is False
        st.error("⚠️ Trang Đánh giá không thể hoạt động do thiếu các thành phần model cần thiết. Vui lòng kiểm tra thông báo lỗi ở trên.")

elif not st.session_state.eval_page_resources_initialized:
    eval_page_status_placeholder.error("⚠️ Tài nguyên trang Đánh giá CHƯA SẴN SÀNG. Lỗi trong quá trình tải model hoặc tạo RAG. Vui lòng kiểm tra log chi tiết hoặc làm mới trang.")