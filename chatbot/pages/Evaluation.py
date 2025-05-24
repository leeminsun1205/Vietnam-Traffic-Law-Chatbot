# pages/Evaluation.py
import time
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Thêm thư mục gốc vào sys.path
import config
from model_loader import load_embedding_model, load_reranker_model, load_gemini_model
from data_loader import load_or_create_rag_components # Import từ thư mục gốc
from reranker import rerank_documents
from utils import (
    generate_query_variations,
    precision_at_k, recall_at_k, f1_at_k, mrr_at_k, ndcg_at_k,
    calculate_average_metrics
)

# @st.cache_resource # Không cache toàn bộ hàm run_retrieval_evaluation vì nó có st.progress, st.empty
def run_retrieval_evaluation(
    eval_data: list,
    # Thông tin về retriever chính
    primary_hybrid_retriever, # Đổi tên để rõ ràng đây là retriever chính
    primary_embedding_model_object, # Model object cho VDB chính của retriever
    # Thông tin về các nguồn dense phụ (nếu có)
    additional_dense_sources_for_eval: list, # List of (emb_obj, vdb_obj)
    # Các model khác
    reranking_model_selected_obj, # Đổi tên để rõ là object
    gemini_model_obj, # Đổi tên để rõ là object
    eval_config: dict # Chứa các tên model và cấu hình khác
    ):
    results_list = []
    k_values_metrics = [3, 5, 10] # Đổi tên biến để tránh trùng

    # Lấy cấu hình từ eval_config
    retrieval_query_mode_eval = eval_config.get('retrieval_query_mode', 'Tổng quát')
    retrieval_method_eval = eval_config.get('retrieval_method', 'hybrid')
    selected_reranker_name_for_eval_run = eval_config.get('selected_reranker_model', 'Không sử dụng') # Tên model reranker
    
    use_reranker_for_eval_run_flag = reranking_model_selected_obj is not None and selected_reranker_name_for_eval_run != 'Không sử dụng'

    progress_bar = st.progress(0)
    status_text_area = st.empty() # Sử dụng st.text_area để có thể hiển thị nhiều dòng hơn nếu cần
    total_items = len(eval_data)
    # Giảm batch size và tăng wait time nếu API Gemini có rate limit chặt
    queries_per_batch_limit = 15 
    wait_time_seconds_between_batches = 60 

    for i, item_data in enumerate(eval_data):
        # Xử lý batching và tạm dừng
        if i > 0 and i % queries_per_batch_limit == 0:
            pause_msg = f"Đã xử lý {i}/{total_items} queries. Tạm dừng {wait_time_seconds_between_batches} giây để tránh rate limit..."
            status_text_area.text(pause_msg)
            time.sleep(wait_time_seconds_between_batches)
        
        query_id_val = item_data.get("query_id", f"query_{i+1}")
        original_query_text = item_data.get("query")
        relevant_chunk_ids_set = set(map(str, item_data.get("relevant_chunk_ids", []))) # Đảm bảo là set của string

        # Hiển thị thông tin xử lý
        reranker_display_name_eval = selected_reranker_name_for_eval_run.split('/')[-1] if selected_reranker_name_for_eval_run != 'Không sử dụng' else "Tắt"
        primary_emb_name_display = eval_config.get('embedding_model_name', 'N/A').split('/')[-1]
        
        additional_emb_names_display_list = []
        if retrieval_method_eval == 'hybrid' and eval_config.get('additional_embedding_model_names_eval'):
            for name in eval_config.get('additional_embedding_model_names_eval', []):
                 additional_emb_names_display_list.append(name.split('/')[-1])
        additional_embs_str = f", Phụ: {', '.join(additional_emb_names_display_list)}" if additional_emb_names_display_list else ""
        
        status_text_area.text(
            f"Đang xử lý query {i+1}/{total_items}: {query_id_val}\n"
            f"Embedding Chính: {primary_emb_name_display}{additional_embs_str}\n"
            f"QueryMode: {retrieval_query_mode_eval}, Method: {retrieval_method_eval}, Reranker: {reranker_display_name_eval}"
        )

        # Khởi tạo dict lưu trữ metrics cho query hiện tại
        current_query_metrics = {
            "query_id": query_id_val, "query": original_query_text,
            "embedding_model_name": eval_config.get('embedding_model_name', 'N/A'),
            "additional_embedding_models_used": eval_config.get('additional_embedding_model_names_eval', []), # Lưu danh sách tên model phụ
            "retrieval_query_mode": retrieval_query_mode_eval,
            "retrieval_method": retrieval_method_eval,
            "selected_reranker_model": selected_reranker_name_for_eval_run,
            "status": "error_undefined", "retrieved_ids": [], "relevant_ids": list(relevant_chunk_ids_set),
            "processing_time": 0.0, 'summarizing_query': '',
            'variation_time': 0.0, 'search_time': 0.0, 'rerank_time': 0.0,
            'num_variations_generated': 0, 'num_unique_docs_found': 0, 
            'num_docs_reranked': 0, 'num_retrieved_before_rerank': 0, 'num_retrieved_after_rerank': 0,
            'error_message': ''
        }
        for k_m in k_values_metrics:
            current_query_metrics[f'precision@{k_m}'] = 0.0; current_query_metrics[f'recall@{k_m}'] = 0.0
            current_query_metrics[f'f1@{k_m}'] = 0.0; current_query_metrics[f'mrr@{k_m}'] = 0.0; current_query_metrics[f'ndcg@{k_m}'] = 0.0

        try:
            eval_query_start_time = time.time()
            # Tạo variations
            variation_start_t = time.time()
            relevance_status_eval, _, all_queries_eval, summarizing_q_eval = generate_query_variations(
                original_query=original_query_text, gemini_model=gemini_model_obj, chat_history=None,
                num_variations=config.NUM_QUERY_VARIATIONS
            )
            current_query_metrics["variation_time"] = time.time() - variation_start_t
            current_query_metrics["summarizing_query"] = summarizing_q_eval
            current_query_metrics["num_variations_generated"] = len(all_queries_eval) -1 if isinstance(all_queries_eval, list) and len(all_queries_eval) > 0 else 0
            
            if relevance_status_eval == 'invalid':
                current_query_metrics["status"] = "skipped_irrelevant_by_llm"
                current_query_metrics["processing_time"] = time.time() - eval_query_start_time
                results_list.append(current_query_metrics)
                progress_bar.progress((i + 1) / total_items); continue

            # Chọn queries để tìm kiếm
            queries_to_search_eval = []
            if retrieval_query_mode_eval == 'Đơn giản': queries_to_search_eval = [original_query_text]
            elif retrieval_query_mode_eval == 'Tổng quát': queries_to_search_eval = [summarizing_q_eval] if summarizing_q_eval else [original_query_text]
            elif retrieval_query_mode_eval == 'Sâu': queries_to_search_eval = all_queries_eval if all_queries_eval else [original_query_text]

            # Thực hiện Retrieval
            collected_docs_data_eval = {}
            search_start_t = time.time()
            for q_variant_eval in queries_to_search_eval:
                if not q_variant_eval: continue
                
                # Gọi search của retriever chính, truyền các nguồn phụ vào đây
                search_results_variant_eval = primary_hybrid_retriever.search(
                    q_variant_eval, 
                    primary_embedding_model_object, # Model object cho VDB chính
                    method=retrieval_method_eval,
                    k=config.VECTOR_K_PER_QUERY, # Lấy nhiều hơn để rerank/fusion
                    additional_dense_sources=additional_dense_sources_for_eval # Danh sách các (emb_obj, vdb_obj) phụ
                )
                for res_item_eval in search_results_variant_eval:
                    doc_idx_eval = res_item_eval.get('index')
                    if isinstance(doc_idx_eval, int) and doc_idx_eval >= 0:
                        if doc_idx_eval not in collected_docs_data_eval:
                            collected_docs_data_eval[doc_idx_eval] = res_item_eval
                        else: # Cập nhật score nếu tốt hơn
                            # Score RRF/BM25: cao hơn tốt hơn. Score L2 distance (dense thuần túy): thấp hơn tốt hơn.
                            current_score = collected_docs_data_eval[doc_idx_eval]['score']
                            new_score = res_item_eval['score']
                            if (retrieval_method_eval == 'hybrid' or retrieval_method_eval == 'sparse') and new_score > current_score:
                                collected_docs_data_eval[doc_idx_eval] = res_item_eval
                            elif retrieval_method_eval == 'dense' and new_score < current_score:
                                collected_docs_data_eval[doc_idx_eval] = res_item_eval


            current_query_metrics["search_time"] = time.time() - search_start_t
            current_query_metrics["num_unique_docs_found"] = len(collected_docs_data_eval)
            
            retrieved_docs_list_eval = list(collected_docs_data_eval.values())
            # Sắp xếp dựa trên phương thức retrieval cuối cùng
            sort_reverse_final_eval = (retrieval_method_eval == 'hybrid' or retrieval_method_eval == 'sparse')
            retrieved_docs_list_eval.sort(key=lambda x: x.get('score', 0 if sort_reverse_final_eval else float('inf')), reverse=sort_reverse_final_eval)
            current_query_metrics["num_retrieved_before_rerank"] = len(retrieved_docs_list_eval)

            # Rerank (Nếu được bật)
            final_docs_for_metrics_calc = []
            rerank_start_time_eval_step = time.time()
            if use_reranker_for_eval_run_flag and retrieved_docs_list_eval:
                query_for_reranking_eval = summarizing_q_eval if summarizing_q_eval else original_query_text
                docs_to_rerank_eval_input = retrieved_docs_list_eval[:config.MAX_DOCS_FOR_RERANK]
                current_query_metrics["num_docs_reranked"] = len(docs_to_rerank_eval_input)
                
                rerank_input_formatted_eval = [{'doc': item_doc_rr['doc'], 'index': item_doc_rr['index']} for item_doc_rr in docs_to_rerank_eval_input]
                
                reranked_results_eval = rerank_documents(
                    query_for_reranking_eval, rerank_input_formatted_eval, reranking_model_selected_obj
                )
                final_docs_for_metrics_calc = reranked_results_eval[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                current_query_metrics["rerank_time"] = time.time() - rerank_start_time_eval_step
            elif retrieved_docs_list_eval: # Không dùng reranker nhưng có kết quả retrieval
                final_docs_for_metrics_calc = retrieved_docs_list_eval[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                current_query_metrics["rerank_time"] = 0.0; current_query_metrics["num_docs_reranked"] = 0
            else: # Không có kết quả retrieval
                 current_query_metrics["rerank_time"] = 0.0; current_query_metrics["num_docs_reranked"] = 0
            
            current_query_metrics["num_retrieved_after_rerank"] = len(final_docs_for_metrics_calc)
            
            # Trích xuất ID từ kết quả cuối cùng để tính metrics
            retrieved_ids_for_metrics = []
            for res_final_item in final_docs_for_metrics_calc:
                doc_data_item = res_final_item.get('doc', {})
                chunk_id_val = None
                if isinstance(doc_data_item, dict): # Đảm bảo doc_data_item là dict
                    chunk_id_val = doc_data_item.get('id') # Ưu tiên trường 'id'
                    if not chunk_id_val: # Nếu không có, thử tìm trong metadata
                        metadata = doc_data_item.get('metadata', {})
                        if isinstance(metadata, dict): # Đảm bảo metadata là dict
                            chunk_id_val = metadata.get('id') or metadata.get('chunk_id') # Thử cả 'id' và 'chunk_id' trong metadata
                if chunk_id_val is not None: # Chỉ thêm nếu chunk_id_val có giá trị (không phải None)
                    retrieved_ids_for_metrics.append(str(chunk_id_val)) # Chuyển sang string để so sánh

            current_query_metrics["retrieved_ids"] = retrieved_ids_for_metrics
            current_query_metrics["status"] = "evaluated"
            
            # Tính toán metrics
            for k_metric_val in k_values_metrics:
                current_query_metrics[f'precision@{k_metric_val}'] = precision_at_k(retrieved_ids_for_metrics, relevant_chunk_ids_set, k_metric_val)
                current_query_metrics[f'recall@{k_metric_val}'] = recall_at_k(retrieved_ids_for_metrics, relevant_chunk_ids_set, k_metric_val)
                current_query_metrics[f'f1@{k_metric_val}'] = f1_at_k(retrieved_ids_for_metrics, relevant_chunk_ids_set, k_metric_val)
                current_query_metrics[f'mrr@{k_metric_val}'] = mrr_at_k(retrieved_ids_for_metrics, relevant_chunk_ids_set, k_metric_val)
                current_query_metrics[f'ndcg@{k_metric_val}'] = ndcg_at_k(retrieved_ids_for_metrics, relevant_chunk_ids_set, k_metric_val)

        except Exception as e_eval:
            current_query_metrics["status"] = "error_runtime_eval"
            current_query_metrics["error_message"] = f"{type(e_eval).__name__}: {str(e_eval)}"
        finally:
            current_query_metrics["processing_time"] = time.time() - eval_query_start_time
            results_list.append(current_query_metrics)
            progress_bar.progress((i + 1) / total_items)

    status_text_area.text(f"Hoàn thành đánh giá {total_items} queries!")
    return pd.DataFrame(results_list)

# --- UI STREAMLIT ---
st.set_page_config(page_title="Đánh giá Retrieval", layout="wide")
st.title("📊 Đánh giá Hệ thống Retrieval")
st.markdown("Trang này cho phép bạn chạy đánh giá hiệu suất của hệ thống retrieval (bao gồm cả hybrid đa nguồn) và reranking dựa trên một tập dữ liệu có gán nhãn.")

# --- Sidebar Cấu hình Đánh giá ---
with st.sidebar:
    st.title("Tùy chọn Đánh giá")
    
    # Khởi tạo session state cho trang đánh giá nếu chưa có
    DEFAULT_EVAL_CONFIG_STATE_PAGE = {
        "eval_selected_embedding_model": st.session_state.get("eval_selected_embedding_model", config.DEFAULT_EMBEDDING_MODEL),
        "eval_additional_hybrid_models": st.session_state.get("eval_additional_hybrid_models", []), # Cho model phụ
        "eval_selected_gemini_model": st.session_state.get("eval_selected_gemini_model", config.DEFAULT_GEMINI_MODEL),
        "eval_retrieval_query_mode": st.session_state.get("eval_retrieval_query_mode", 'Tổng quát'), # 'Đơn giản', 'Tổng quát', 'Sâu'
        "eval_retrieval_method": st.session_state.get("eval_retrieval_method", 'hybrid'), # 'dense', 'sparse', 'hybrid'
        "eval_selected_reranker_model": st.session_state.get("eval_selected_reranker_model", config.DEFAULT_RERANKER_MODEL),
    }
    for key_eval_ss, default_val_eval_ss in DEFAULT_EVAL_CONFIG_STATE_PAGE.items():
        if key_eval_ss not in st.session_state:
            st.session_state[key_eval_ss] = default_val_eval_ss

    st.header("Mô hình")
    # Chọn Embedding Model chính cho đánh giá
    st.selectbox(
        "Chọn mô hình Embedding chính:",
        options=config.AVAILABLE_EMBEDDING_MODELS,
        index=config.AVAILABLE_EMBEDDING_MODELS.index(st.session_state.eval_selected_embedding_model),
        key="eval_selected_embedding_model",
        help="Chọn mô hình embedding chính để đánh giá."
    )
    # Chọn Embedding Model phụ cho đánh giá (chỉ khi hybrid)
    if st.session_state.eval_retrieval_method == 'hybrid':
        available_for_additional_eval = [
            m_eval for m_eval in config.AVAILABLE_EMBEDDING_MODELS 
            if m_eval != st.session_state.eval_selected_embedding_model
        ]
        current_additional_selection_eval = st.session_state.get("eval_additional_hybrid_models", [])
        if not isinstance(current_additional_selection_eval, list): current_additional_selection_eval = []
        valid_current_additional_eval = [m for m in current_additional_selection_eval if m in available_for_additional_eval]

        if available_for_additional_eval:
            selected_additional_eval = st.multiselect(
                "Chọn thêm mô hình Embedding phụ cho Hybrid (tối đa 2):",
                options=available_for_additional_eval,
                default=valid_current_additional_eval,
                key="eval_additional_hybrid_models_multiselect", # Key riêng cho widget này
                help="Chọn tối đa 2 mô hình embedding phụ để kết hợp trong đánh giá hybrid."
            )
            if len(selected_additional_eval) > 2:
                st.warning("Tối đa 2 mô hình phụ. Sẽ lấy 2 lựa chọn đầu.")
                st.session_state.eval_additional_hybrid_models = selected_additional_eval[:2]
            else:
                st.session_state.eval_additional_hybrid_models = selected_additional_eval
        else:
            st.markdown("<p style='font-size:0.9em; font-style:italic;'>Không có mô hình phụ nào khác để chọn.</p>", unsafe_allow_html=True)
            st.session_state.eval_additional_hybrid_models = []
    else:
        st.session_state.eval_additional_hybrid_models = []


    st.selectbox("Chọn mô hình Gemini (tạo query variations):", options=config.AVAILABLE_GEMINI_MODELS,
        index=config.AVAILABLE_GEMINI_MODELS.index(st.session_state.eval_selected_gemini_model),
        key="eval_selected_gemini_model", help="Mô hình LLM để tạo biến thể câu hỏi."
    )
    st.selectbox("Chọn mô hình Reranker:", options=config.AVAILABLE_RERANKER_MODELS,
        index=config.AVAILABLE_RERANKER_MODELS.index(st.session_state.eval_selected_reranker_model),
        key="eval_selected_reranker_model", help="Mô hình Reranker để đánh giá (hoặc 'Không sử dụng')."
    )
    
    st.header("Cấu hình Retrieval")
    # Index cho radio eval_retrieval_query_mode
    query_mode_options_eval = ['Đơn giản', 'Tổng quát', 'Sâu']
    current_query_mode_idx_eval = query_mode_options_eval.index(st.session_state.eval_retrieval_query_mode) \
        if st.session_state.eval_retrieval_query_mode in query_mode_options_eval else 1 # Default 'Tổng quát'
    st.radio("Nguồn câu hỏi cho Retrieval:", options=query_mode_options_eval, index=current_query_mode_idx_eval,
        key="eval_retrieval_query_mode", horizontal=True, help="Cách tạo câu hỏi đầu vào cho hệ thống retrieval."
    )
    # Index cho radio eval_retrieval_method
    method_options_eval = ['dense', 'sparse', 'hybrid']
    current_method_idx_eval = method_options_eval.index(st.session_state.eval_retrieval_method) \
        if st.session_state.eval_retrieval_method in method_options_eval else 2 # Default 'hybrid'
    st.radio("Phương thức Retrieval:", options=method_options_eval, index=current_method_idx_eval,
        key="eval_retrieval_method", horizontal=True, help="Phương thức tìm kiếm."
    )

# --- Khởi tạo các biến session state cho trang đánh giá ---
if 'eval_data' not in st.session_state: st.session_state.eval_data = None
if 'eval_results_df' not in st.session_state: st.session_state.eval_results_df = None
if 'eval_run_completed' not in st.session_state: st.session_state.eval_run_completed = False
if 'eval_uploaded_filename' not in st.session_state: st.session_state.eval_uploaded_filename = None
if "upload_counter_eval" not in st.session_state: st.session_state.upload_counter_eval = 0 # Đổi tên key
if 'last_eval_config_run' not in st.session_state: st.session_state.last_eval_config_run = {} # Đổi tên key

# --- Khởi tạo tài nguyên cốt lõi cho đánh giá ---
st.subheader("Trạng thái Hệ thống Cơ bản cho Đánh giá")
init_ok_eval_page = False
# Primary components
g_primary_embedding_model_eval_obj = None
g_primary_retriever_instance_eval = None
# Reranker
g_reranking_model_eval_obj_loaded = None

with st.spinner("Kiểm tra và khởi tạo tài nguyên cho đánh giá..."):
    try:
        # 1. Tải Embedding Model chính
        g_primary_embedding_model_eval_obj = load_embedding_model(st.session_state.eval_selected_embedding_model)
        
        # 2. Tải Retriever chính (dựa trên model embedding chính)
        if g_primary_embedding_model_eval_obj:
            current_eval_rag_prefix_main = config.get_rag_data_prefix(st.session_state.eval_selected_embedding_model)
            # load_or_create_rag_components trả về (vector_db, retriever)
            # Chúng ta cần retriever chính ở đây. VectorDB chính đã nằm trong retriever này.
            _, g_primary_retriever_instance_eval = load_or_create_rag_components(
                g_primary_embedding_model_eval_obj, current_eval_rag_prefix_main
            )
        
        # 3. Tải Reranker Model (nếu được chọn)
        if st.session_state.eval_selected_reranker_model != 'Không sử dụng':
            g_reranking_model_eval_obj_loaded = load_reranker_model(st.session_state.eval_selected_reranker_model)
        
        # Kiểm tra điều kiện khởi tạo thành công
        if g_primary_embedding_model_eval_obj and g_primary_retriever_instance_eval:
            init_ok_eval_page = True
            st.success(f"Embedding model chính '{st.session_state.eval_selected_embedding_model.split('/')[-1]}' và Retriever chính sẵn sàng.")
            
            # Kiểm tra các model phụ (chỉ thông báo, không làm init thất bại)
            if st.session_state.eval_retrieval_method == 'hybrid' and st.session_state.eval_additional_hybrid_models:
                st.markdown("---")
                st.write("Kiểm tra các Embedding Model phụ:")
                for model_name_add_eval_init in st.session_state.eval_additional_hybrid_models:
                    emb_obj_add_eval_init = load_embedding_model(model_name_add_eval_init) # Cache gre normalen
                    if emb_obj_add_eval_init:
                        prefix_add_eval_init = config.get_rag_data_prefix(model_name_add_eval_init)
                        vdb_add_eval_init, _ = load_or_create_rag_components(emb_obj_add_eval_init, prefix_add_eval_init) # Cache gre normalen
                        if vdb_add_eval_init:
                             st.caption(f"- Model phụ '{model_name_add_eval_init.split('/')[-1]}' và VDB: Sẵn sàng.")
                        else: st.caption(f"- Model phụ '{model_name_add_eval_init.split('/')[-1]}': Lỗi tải VDB.")
                    else: st.caption(f"- Model phụ '{model_name_add_eval_init.split('/')[-1]}': Lỗi tải Embedding Model.")
                st.markdown("---")

            if st.session_state.eval_selected_reranker_model != 'Không sử dụng':
                if g_reranking_model_eval_obj_loaded:
                    st.success(f"Reranker model '{st.session_state.eval_selected_reranker_model.split('/')[-1]}' sẵn sàng.")
                else:
                    st.warning(f"⚠️ Không tải được Reranker Model ({st.session_state.eval_selected_reranker_model}).")
            else: st.info("Reranker không được chọn sử dụng cho đánh giá.")
        else:
            missing_components = []
            if not g_primary_embedding_model_eval_obj: missing_components.append(f"Embedding Model chính ({st.session_state.eval_selected_embedding_model})")
            if not g_primary_retriever_instance_eval: missing_components.append("Retriever/VectorDB chính")
            st.error(f"⚠️ Lỗi khởi tạo cho đánh giá: {', '.join(missing_components)} không sẵn sàng.")
            
    except Exception as e_init_eval:
        st.error(f"⚠️ Lỗi nghiêm trọng khi khởi tạo hệ thống cho đánh giá: {str(e_init_eval)}")

# --- Giao diện chính của trang đánh giá ---
if init_ok_eval_page:
    # Hiển thị cấu hình hiện tại cho đánh giá
    reranker_display_eval_caption_page = st.session_state.eval_selected_reranker_model.split('/')[-1] \
        if st.session_state.eval_selected_reranker_model != 'Không sử dụng' else "Tắt"
    
    additional_embs_caption_list = []
    if st.session_state.eval_retrieval_method == 'hybrid' and st.session_state.eval_additional_hybrid_models:
        additional_embs_caption_list = [name.split('/')[-1] for name in st.session_state.eval_additional_hybrid_models]
    additional_embs_caption_str = f", Phụ: {', '.join(additional_embs_caption_list)}" if additional_embs_caption_list else ""

    st.caption(
        f"Emb. Chính: `{st.session_state.eval_selected_embedding_model.split('/')[-1]}`{additional_embs_caption_str} | "
        f"LLM: `{st.session_state.eval_selected_gemini_model}` | Query: `{st.session_state.eval_retrieval_query_mode}` | "
        f"Retrieval: `{st.session_state.eval_retrieval_method}` | Reranker: `{reranker_display_eval_caption_page}`"
    )

    # Tải file đánh giá
    uploader_key_eval = f"eval_file_uploader_{st.session_state.upload_counter_eval}"
    st.subheader("1. Tải Lên File Dữ liệu Đánh giá")
    uploaded_file_eval = st.file_uploader("Chọn file JSON chứa các cặp (query, relevant_chunk_ids)...", type=["json"], key=uploader_key_eval)

    if uploaded_file_eval is not None:
        # Xử lý file mới tải lên
        if uploaded_file_eval.name != st.session_state.eval_uploaded_filename:
            try:
                eval_data_list_from_file = json.loads(uploaded_file_eval.getvalue().decode('utf-8'))
                # Kiểm tra cấu trúc cơ bản của file
                if isinstance(eval_data_list_from_file, list) and \
                   all(isinstance(item, dict) and "query" in item and "relevant_chunk_ids" in item for item in eval_data_list_from_file):
                    st.session_state.eval_data = eval_data_list_from_file
                    st.session_state.eval_uploaded_filename = uploaded_file_eval.name
                    st.session_state.eval_run_completed = False # Reset trạng thái chạy
                    st.session_state.last_eval_config_run = {} # Reset cấu hình chạy trước
                    st.success(f"Đã tải và xác thực file '{uploaded_file_eval.name}' ({len(eval_data_list_from_file)} câu hỏi).")
                else:
                    st.error("File JSON không đúng định dạng. Mỗi item phải là dict có key 'query' và 'relevant_chunk_ids'.")
                    st.session_state.eval_data = None; st.session_state.eval_uploaded_filename = None
            except json.JSONDecodeError:
                st.error("Lỗi giải mã JSON. Vui lòng kiểm tra lại định dạng file.")
                st.session_state.eval_data = None; st.session_state.eval_uploaded_filename = None
            except Exception as e_file:
                st.error(f"Lỗi xử lý file: {str(e_file)}")
                st.session_state.eval_data = None; st.session_state.eval_uploaded_filename = None
    
    # Nếu có dữ liệu đánh giá đã tải
    if st.session_state.eval_data is not None:
        st.info(f"Sẵn sàng đánh giá với dữ liệu từ: **{st.session_state.eval_uploaded_filename}** ({len(st.session_state.eval_data)} câu hỏi).")
        if st.checkbox("Hiển thị 5 dòng dữ liệu mẫu", key="show_eval_data_preview_checkbox"):
            st.dataframe(pd.DataFrame(st.session_state.eval_data).head())

        st.subheader("2. Chạy Đánh giá")
        if st.button("🚀 Bắt đầu Đánh giá ngay!", key="start_eval_button_main"):
            # --- Chuẩn bị các đối tượng model và cấu hình cho lần chạy đánh giá ---
            current_run_config = {
                'embedding_model_name': st.session_state.eval_selected_embedding_model, # Tên model chính
                'additional_embedding_model_names_eval': list(st.session_state.eval_additional_hybrid_models), # List tên model phụ
                'retrieval_query_mode': st.session_state.eval_retrieval_query_mode,
                'retrieval_method': st.session_state.eval_retrieval_method,
                'selected_reranker_model': st.session_state.eval_selected_reranker_model, # Tên model reranker
                'gemini_model_name': st.session_state.eval_selected_gemini_model, # Tên model Gemini
                # Tên model reranker thực sự được sử dụng (nếu có)
                'reranker_model_name_actually_used': st.session_state.eval_selected_reranker_model \
                    if g_reranking_model_eval_obj_loaded and st.session_state.eval_selected_reranker_model != 'Không sử dụng' \
                    else "Không sử dụng"
            }
            st.session_state.last_eval_config_run = current_run_config.copy()

            # Tải LLM (Gemini) cho việc tạo query variations
            with st.spinner(f"Đang tải model Gemini cho đánh giá: {st.session_state.eval_selected_gemini_model}..."):
                 g_gemini_model_for_eval_run_obj = load_gemini_model(st.session_state.eval_selected_gemini_model) # Cache gre normalen

            if not g_gemini_model_for_eval_run_obj:
                st.error(f"Không thể tải model Gemini '{st.session_state.eval_selected_gemini_model}' cho đánh giá. Vui lòng thử lại."); st.stop()

            # Chuẩn bị các nguồn dense phụ (embedding model objects và VDB objects)
            additional_dense_sources_for_current_eval_run = []
            if current_run_config['retrieval_method'] == 'hybrid' and current_run_config['additional_embedding_model_names_eval']:
                with st.spinner("Đang chuẩn bị các nguồn embedding phụ cho đánh giá..."):
                    for model_name_add_eval in current_run_config['additional_embedding_model_names_eval']:
                        emb_obj_add_eval = load_embedding_model(model_name_add_eval) # Cache gre normalen
                        if emb_obj_add_eval:
                            prefix_add_eval = config.get_rag_data_prefix(model_name_add_eval)
                            vdb_add_eval, _ = load_or_create_rag_components(emb_obj_add_eval, prefix_add_eval) # Cache gre normalen
                            if vdb_add_eval:
                                additional_dense_sources_for_current_eval_run.append((emb_obj_add_eval, vdb_add_eval))
                            else: st.warning(f"Không tải được VDB cho model phụ '{model_name_add_eval}' trong đánh giá. Bỏ qua.")
                        else: st.warning(f"Không tải được Embedding Model phụ '{model_name_add_eval}' trong đánh giá. Bỏ qua.")
            
            st.info(f"Model Gemini '{st.session_state.eval_selected_gemini_model}' và các tài nguyên khác đã sẵn sàng.")
            with st.spinner("⏳ Đang chạy đánh giá toàn bộ tập dữ liệu... Quá trình này có thể mất vài phút đến vài chục phút tùy thuộc vào số lượng câu hỏi và cấu hình."):
                eval_run_start_time = time.time()
                
                # Gọi hàm chạy đánh giá
                results_df_output = run_retrieval_evaluation(
                    eval_data=st.session_state.eval_data,
                    primary_hybrid_retriever=g_primary_retriever_instance_eval, # Retriever chính
                    primary_embedding_model_object=g_primary_embedding_model_eval_obj, # Model object chính
                    additional_dense_sources_for_eval=additional_dense_sources_for_current_eval_run, # List (emb_obj, vdb_obj) phụ
                    reranking_model_selected_obj=g_reranking_model_eval_obj_loaded, # Reranker object (hoặc None)
                    gemini_model_obj=g_gemini_model_for_eval_run_obj, # Gemini object
                    eval_config=st.session_state.last_eval_config_run # Dict cấu hình
                )
                total_eval_run_time = time.time() - eval_run_start_time
                st.success(f"🎉 Hoàn thành đánh giá toàn bộ dữ liệu sau {total_eval_run_time:.2f} giây.")
                st.session_state.eval_results_df = results_df_output
                st.session_state.eval_run_completed = True
                st.rerun() # Chạy lại để hiển thị kết quả

    # --- Hiển thị kết quả đánh giá ---
    if st.session_state.eval_run_completed and st.session_state.eval_results_df is not None:
        st.subheader("3. Kết quả Đánh giá")
        detailed_results_df_display = st.session_state.eval_results_df
        last_run_config_display = st.session_state.last_eval_config_run

        st.markdown("**Cấu hình đã sử dụng cho lần chạy cuối:**")
        # Sử dụng cột để hiển thị gọn gàng hơn
        cfg_col_main, cfg_col_add, cfg_col_ret, cfg_col_rerank = st.columns(4)
        
        primary_emb_name_res = last_run_config_display.get('embedding_model_name', 'N/A').split('/')[-1]
        cfg_col_main.metric("Embedding Chính", primary_emb_name_res)

        add_emb_names_res_list = []
        if last_run_config_display.get('additional_embedding_model_names_eval'):
            add_emb_names_res_list = [name.split('/')[-1] for name in last_run_config_display.get('additional_embedding_model_names_eval')]
        cfg_col_add.metric("Embeddings Phụ", ", ".join(add_emb_names_res_list) if add_emb_names_res_list else "Không")
        
        cfg_col_ret.metric("Retrieval Method", last_run_config_display.get('retrieval_method', 'N/A'))
        
        reranker_name_res = last_run_config_display.get('selected_reranker_model', 'N/A') # Tên model reranker từ config
        reranker_name_disp_res = reranker_name_res.split('/')[-1] if reranker_name_res != 'Không sử dụng' else "Tắt"
        cfg_col_rerank.metric("Reranker Config", reranker_name_disp_res)

        st.caption(
            f"LLM (Variations): `{last_run_config_display.get('gemini_model_name', 'N/A')}` | "
            f"Nguồn Query: `{last_run_config_display.get('retrieval_query_mode', 'N/A')}` | "
            f"Reranker thực tế dùng: `{last_run_config_display.get('reranker_model_name_actually_used', 'N/A').split('/')[-1] if last_run_config_display.get('reranker_model_name_actually_used') != 'Không sử dụng' else 'Tắt'}`"
        )
        
        # Tính toán metrics trung bình
        avg_metrics_results, num_evaluated_queries, num_skipped_or_error_queries = calculate_average_metrics(detailed_results_df_display)

        st.metric("Tổng số Queries trong File", len(detailed_results_df_display))
        col_res_count1, col_res_count2 = st.columns(2)
        col_res_count1.metric("Queries Đánh giá Hợp lệ (status='evaluated')", num_evaluated_queries)
        col_res_count2.metric("Queries Bỏ qua / Lỗi", num_skipped_or_error_queries)

        if avg_metrics_results:
            st.markdown("#### Metrics Trung bình @K (trên các queries hợp lệ)")
            k_values_for_display = [3, 5, 10] # Các giá trị K muốn hiển thị
            cols_k_metrics = st.columns(len(k_values_for_display))
            for idx_k_disp, k_val_disp in enumerate(k_values_for_display):
                with cols_k_metrics[idx_k_disp]:
                    st.markdown(f"**K = {k_val_disp}**")
                    st.text(f"Precision: {avg_metrics_results.get(f'avg_precision@{k_val_disp}', 0.0):.4f}")
                    st.text(f"Recall:    {avg_metrics_results.get(f'avg_recall@{k_val_disp}', 0.0):.4f}")
                    st.text(f"F1 Score:  {avg_metrics_results.get(f'avg_f1@{k_val_disp}', 0.0):.4f}")
                    st.text(f"MRR:       {avg_metrics_results.get(f'avg_mrr@{k_val_disp}', 0.0):.4f}")
                    st.text(f"NDCG:      {avg_metrics_results.get(f'avg_ndcg@{k_val_disp}', 0.0):.4f}")
            
            st.markdown("#### Thông tin Hiệu năng & Số lượng Trung bình (trên các queries hợp lệ)")
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            perf_col1.metric("Avg Process Time (s)", f"{avg_metrics_results.get('avg_processing_time', 0.0):.2f}s")
            perf_col1.metric("Avg Variation Time (s)", f"{avg_metrics_results.get('avg_variation_time', 0.0):.2f}s")
            perf_col2.metric("Avg Search Time (s)", f"{avg_metrics_results.get('avg_search_time', 0.0):.2f}s")
            perf_col2.metric("Avg Rerank Time (s)", f"{avg_metrics_results.get('avg_rerank_time', 0.0):.2f}s")
            
            perf_col3.metric("Avg Variations Gen.", f"{avg_metrics_results.get('avg_num_variations_generated', 0.0):.1f}")
            perf_col3.metric("Avg Docs Found (Unique)", f"{avg_metrics_results.get('avg_num_unique_docs_found', 0.0):.1f}")
            perf_col3.metric("Avg Docs to Rerank", f"{avg_metrics_results.get('avg_num_docs_reranked', 0.0):.1f}")
            perf_col3.metric("Avg Docs After Rerank/Limit", f"{avg_metrics_results.get('avg_num_retrieved_after_rerank', 0.0):.1f}")

        with st.expander("Xem Kết quả Chi tiết cho từng Query", expanded=False):
            cols_to_display_ordered = [
                'query_id', 'query', 'status', 'error_message',
                'embedding_model_name', 'additional_embedding_models_used', # Thêm cột model phụ đã dùng
                'retrieval_query_mode','retrieval_method', 'selected_reranker_model',
                'precision@3', 'recall@3', 'f1@3', 'mrr@3', 'ndcg@3',
                'precision@5', 'recall@5', 'f1@5', 'mrr@5', 'ndcg@5',
                'precision@10', 'recall@10', 'f1@10', 'mrr@10', 'ndcg@10',
                'processing_time', 'variation_time', 'search_time', 'rerank_time',
                'num_variations_generated','num_unique_docs_found', 
                'num_retrieved_before_rerank','num_docs_reranked', 'num_retrieved_after_rerank',
                'retrieved_ids', 'relevant_ids', 'summarizing_query'
            ]
            # Lọc ra các cột thực sự có trong DataFrame để tránh lỗi
            existing_cols_for_display = [col_disp for col_disp in cols_to_display_ordered if col_disp in detailed_results_df_display.columns]
            st.dataframe(detailed_results_df_display[existing_cols_for_display])

        st.subheader("4. Lưu Kết quả Chi tiết")
        try:
            results_json_output = detailed_results_df_display.to_json(orient='records', indent=2, force_ascii=False)
            results_csv_output = detailed_results_df_display.to_csv(index=False).encode('utf-8')
            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            emb_main_suffix = last_run_config_display.get('embedding_model_name', 'na').split('/')[-1].replace('-', '').replace('_', '')[:10]
            add_emb_suffix = "none"
            if last_run_config_display.get('additional_embedding_model_names_eval'):
                add_emb_suffix = "_".join([name.split('/')[-1].replace('-', '').replace('_', '')[:5] for name in last_run_config_display.get('additional_embedding_model_names_eval')])
            
            qmode_file_suffix = last_run_config_display.get('retrieval_query_mode', 'na').lower()[:3]
            method_file_suffix = last_run_config_display.get('retrieval_method', 'na').lower()
            
            reranker_file_suffix_used = "NoRerank"
            selected_reranker_for_filename_actual = last_run_config_display.get('reranker_model_name_actually_used', 'Không sử dụng')
            if selected_reranker_for_filename_actual != 'Không sử dụng':
                  reranker_file_suffix_used = selected_reranker_for_filename_actual.split('/')[-1].replace('-', '').replace('_', '')[:10]
            
            llm_file_suffix = last_run_config_display.get('gemini_model_name', 'gemini').split('/')[-1].replace('.','-')[:15]

            base_output_filename = f"evalRes_{emb_main_suffix}_add_{add_emb_suffix}_{qmode_file_suffix}_{method_file_suffix}_{reranker_file_suffix_used}_{llm_file_suffix}_{current_timestamp}"
            final_fname_json = f"{base_output_filename}.json"
            final_fname_csv = f"{base_output_filename}.csv"

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button("💾 Tải về JSON", results_json_output, final_fname_json, "application/json", key="download_json_button")
            with dl_col2:
                st.download_button("💾 Tải về CSV", results_csv_output, final_fname_csv, "text/csv", key="download_csv_button")
        except Exception as e_savefile:
            st.error(f"Lỗi khi chuẩn bị file kết quả để tải về: {str(e_savefile)}")
    
    st.markdown("---")
    st.subheader("Quản lý Trạng thái Đánh giá Hiện tại")
    if st.button("Xóa File Đã Tải và Kết Quả Đánh giá Hiện tại", key="clear_current_eval_state_button"):
        st.session_state.eval_data = None
        st.session_state.upload_counter_eval += 1 # Thay đổi key uploader để reset
        st.session_state.eval_run_completed = False
        st.session_state.eval_results_df = None
        st.session_state.last_eval_config_run = {}
        st.session_state.eval_uploaded_filename = None
        st.success("Đã xóa trạng thái của lần đánh giá hiện tại (file tải lên, kết quả).")
        time.sleep(1); st.rerun()
else:
    st.warning("⚠️ Hệ thống cơ bản cho trang đánh giá chưa sẵn sàng. Vui lòng kiểm tra lỗi và thử làm mới trang.")