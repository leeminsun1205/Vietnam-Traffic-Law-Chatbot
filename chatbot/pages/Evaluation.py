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
from model_loader import load_embedding_model, load_reranker_model, load_gemini_model
from data_loader import load_or_create_rag_components
from reranker import rerank_documents
from utils import (
    generate_query_variations,
    precision_at_k, recall_at_k, f1_at_k, mrr_at_k, ndcg_at_k,
    calculate_average_metrics
)

def run_retrieval_evaluation(
    eval_data: list,
    hybrid_retriever,
    embedding_model_object, # ++ Nhận object model embedding ++
    reranking_model_selected,
    gemini_model,
    eval_config: dict
    ):
    results_list = []
    k_values = [3, 5, 10]

    retrieval_query_mode = eval_config.get('retrieval_query_mode', 'Tổng quát')
    retrieval_method = eval_config.get('retrieval_method', 'hybrid')
    selected_reranker_name_for_eval = eval_config.get('selected_reranker_model', 'Không sử dụng')
    use_reranker_for_eval_run = reranking_model_selected is not None and selected_reranker_name_for_eval != 'Không sử dụng'

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_items = len(eval_data)
    queries_per_batch = 15 # Cân nhắc nếu API Gemini có rate limit
    wait_time_seconds = 60

    for i, item in enumerate(eval_data):
        if i > 0 and i % queries_per_batch == 0:
            pause_msg = f"Đã xử lý {i}/{total_items} queries. Tạm dừng {wait_time_seconds} giây..."
            status_text.text(pause_msg)
            time.sleep(wait_time_seconds)
            status_text.text(f"Tiếp tục xử lý query {i+1}/{total_items}...")

        query_id = item.get("query_id"); original_query = item.get("query")
        relevant_chunk_ids = set(item.get("relevant_chunk_ids", []))

        reranker_display_eval = selected_reranker_name_for_eval.split('/')[-1] if selected_reranker_name_for_eval != 'Không sử dụng' else "Tắt"
        # ++ Thêm tên embedding model vào status text ++
        current_embedding_model_name_for_eval_display = eval_config.get('embedding_model_name', 'N/A').split('/')[-1]
        status_text.text(
            f"Đang xử lý query {i+1}/{total_items}: {query_id} "
            f"(Embedding: {current_embedding_model_name_for_eval_display}, "
            f"QueryMode: {retrieval_query_mode}, Method: {retrieval_method}, Reranker: {reranker_display_eval})"
        )


        start_time = time.time()
        query_metrics = {
            "query_id": query_id, "query": original_query,
            "embedding_model_name": eval_config.get('embedding_model_name', 'N/A'), # ++ Lưu tên embedding model ++
            "retrieval_query_mode": retrieval_query_mode,
            "retrieval_method": retrieval_method,
            "selected_reranker_model": selected_reranker_name_for_eval,
            "status": "error", "retrieved_ids": [], "relevant_ids": list(relevant_chunk_ids),
            "processing_time": 0.0, 'summarizing_query': '',
            'variation_time': 0.0, 'search_time': 0.0, 'rerank_time': 0.0,
            'num_variations_generated': 0, 'num_unique_docs_found': 0, 'num_docs_reranked': 0,
            'num_retrieved_before_rerank': 0, 'num_retrieved_after_rerank': 0
        }
        for k_val in k_values: # Sửa tên biến k thành k_val
            query_metrics[f'precision@{k_val}'] = 0.0; query_metrics[f'recall@{k_val}'] = 0.0
            query_metrics[f'f1@{k_val}'] = 0.0; query_metrics[f'mrr@{k_val}'] = 0.0; query_metrics[f'ndcg@{k_val}'] = 0.0

        try:
            variation_start = time.time()
            relevance_status, _, all_queries, summarizing_query = generate_query_variations(
                original_query=original_query, gemini_model=gemini_model,
                chat_history=None,
                num_variations=config.NUM_QUERY_VARIATIONS
            )
            query_metrics["variation_time"] = time.time() - variation_start
            query_metrics["summarizing_query"] = summarizing_query
            query_metrics["num_variations_generated"] = len(all_queries) -1 if isinstance(all_queries, list) and len(all_queries) > 0 else 0

            if relevance_status == 'invalid':
                query_metrics["status"] = "skipped_irrelevant"
                query_metrics["processing_time"] = time.time() - start_time
                results_list.append(query_metrics)
                progress_bar.progress((i + 1) / total_items)
                continue

            queries_to_search = []
            if retrieval_query_mode == 'Đơn giản': queries_to_search = [original_query]
            elif retrieval_query_mode == 'Tổng quát': queries_to_search = [summarizing_query] if summarizing_query else [original_query]
            elif retrieval_query_mode == 'Sâu': queries_to_search = all_queries if all_queries else [original_query]

            collected_docs_data = {}
            search_start = time.time()
            for q_variant in queries_to_search:
                if not q_variant: continue
                search_results = hybrid_retriever.search(
                    q_variant, embedding_model_object, # ++ Truyền object model embedding ++
                    method=retrieval_method,
                    k=config.VECTOR_K_PER_QUERY
                )
                for res_item in search_results:
                    doc_index = res_item.get('index')
                    if isinstance(doc_index, int) and doc_index >= 0 and doc_index not in collected_docs_data:
                        collected_docs_data[doc_index] = res_item
            query_metrics["search_time"] = time.time() - search_start
            query_metrics["num_unique_docs_found"] = len(collected_docs_data)

            retrieved_docs_list = list(collected_docs_data.values())
            sort_reverse = (retrieval_method != 'dense')
            retrieved_docs_list.sort(key=lambda x: x.get('score', 0 if sort_reverse else float('inf')), reverse=sort_reverse)
            query_metrics["num_retrieved_before_rerank"] = len(retrieved_docs_list)

            final_docs_for_metrics = []
            rerank_start_time_eval = time.time()

            if use_reranker_for_eval_run and retrieved_docs_list:
                query_for_reranking = summarizing_query if summarizing_query else original_query
                docs_to_rerank = retrieved_docs_list[:config.MAX_DOCS_FOR_RERANK]
                query_metrics["num_docs_reranked"] = len(docs_to_rerank)
                rerank_input = [{'doc': item_doc['doc'], 'index': item_doc['index']} for item_doc in docs_to_rerank]
                reranked_results = rerank_documents(
                    query_for_reranking, rerank_input, reranking_model_selected
                )
                final_docs_for_metrics = reranked_results[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                query_metrics["rerank_time"] = time.time() - rerank_start_time_eval
            elif retrieved_docs_list:
                final_docs_for_metrics = retrieved_docs_list[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
                query_metrics["rerank_time"] = 0.0
                query_metrics["num_docs_reranked"] = 0
            else:
                 query_metrics["rerank_time"] = 0.0
                 query_metrics["num_docs_reranked"] = 0

            query_metrics["num_retrieved_after_rerank"] = len(final_docs_for_metrics)
            retrieved_ids = []
            for res_final in final_docs_for_metrics:
                doc_data = res_final.get('doc', {})
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
            query_metrics["status"] = "evaluated"
            for k_val_metrics in k_values: # Sửa tên biến k
                query_metrics[f'precision@{k_val_metrics}'] = precision_at_k(retrieved_ids, relevant_chunk_ids, k_val_metrics)
                query_metrics[f'recall@{k_val_metrics}'] = recall_at_k(retrieved_ids, relevant_chunk_ids, k_val_metrics)
                query_metrics[f'f1@{k_val_metrics}'] = f1_at_k(retrieved_ids, relevant_chunk_ids, k_val_metrics)
                query_metrics[f'mrr@{k_val_metrics}'] = mrr_at_k(retrieved_ids, relevant_chunk_ids, k_val_metrics)
                query_metrics[f'ndcg@{k_val_metrics}'] = ndcg_at_k(retrieved_ids, relevant_chunk_ids, k_val_metrics)

        except Exception as e:
            query_metrics["status"] = "error_runtime"
            query_metrics["error_message"] = str(e)
        finally:
            query_metrics["processing_time"] = time.time() - start_time
            results_list.append(query_metrics)
            progress_bar.progress((i + 1) / total_items)

    status_text.text(f"Hoàn thành đánh giá {total_items} queries!")
    return pd.DataFrame(results_list)

st.set_page_config(page_title="Đánh giá Retrieval", layout="wide")
st.title("📊 Đánh giá Hệ thống Retrieval")
st.markdown("Trang này cho phép bạn chạy đánh giá hiệu suất của hệ thống retrieval và reranking...")

with st.sidebar:
    st.title("Tùy chọn Đánh giá")
    DEFAULT_EVAL_CONFIG_STATE = {
        # ++ Thêm embedding model vào config mặc định của trang đánh giá ++
        "eval_selected_embedding_model": st.session_state.get("eval_selected_embedding_model", config.DEFAULT_EMBEDDING_MODEL),
        "eval_selected_gemini_model": st.session_state.get("eval_selected_gemini_model", config.DEFAULT_GEMINI_MODEL),
        "eval_retrieval_query_mode": st.session_state.get("eval_retrieval_query_mode", 'Tổng quát'),
        "eval_retrieval_method": st.session_state.get("eval_retrieval_method", 'hybrid'),
        "eval_selected_reranker_model": st.session_state.get("eval_selected_reranker_model", config.DEFAULT_RERANKER_MODEL),
    }
    for key, default_value in DEFAULT_EVAL_CONFIG_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    st.header("Mô hình")
    # ++ Selectbox cho Embedding Model trong trang Đánh giá ++
    st.selectbox(
        "Chọn mô hình Embedding:",
        options=config.AVAILABLE_EMBEDDING_MODELS,
        index=config.AVAILABLE_EMBEDDING_MODELS.index(st.session_state.eval_selected_embedding_model),
        key="eval_selected_embedding_model",
        help="Chọn mô hình embedding để đánh giá."
    )
    st.selectbox(
        "Chọn mô hình Gemini (để tạo query variations):",
        options=config.AVAILABLE_GEMINI_MODELS,
        index=config.AVAILABLE_GEMINI_MODELS.index(st.session_state.eval_selected_gemini_model),
        key="eval_selected_gemini_model",
        help="Chọn mô hình ngôn ngữ lớn để phân tích và tạo biến thể câu hỏi cho Retrieval."
    )
    st.selectbox(
        "Chọn mô hình Reranker:",
        options=config.AVAILABLE_RERANKER_MODELS,
        index=config.AVAILABLE_RERANKER_MODELS.index(st.session_state.eval_selected_reranker_model),
        key="eval_selected_reranker_model", help="..."
    )
    st.header("Cấu hình Retrieval")
    st.radio(
        "Nguồn câu hỏi cho Retrieval:",
        options=['Đơn giản', 'Tổng quát', 'Sâu'],
        index=['Đơn giản', 'Tổng quát', 'Sâu'].index(st.session_state.eval_retrieval_query_mode),
        key="eval_retrieval_query_mode", horizontal=True, help="..."
    )
    st.radio(
        "Phương thức Retrieval:",
        options=['dense', 'sparse', 'hybrid'],
        index=['dense', 'sparse', 'hybrid'].index(st.session_state.eval_retrieval_method),
        key="eval_retrieval_method", horizontal=True, help="..."
    )

if 'eval_data' not in st.session_state: st.session_state.eval_data = None
if 'eval_results_df' not in st.session_state: st.session_state.eval_results_df = None
if 'eval_run_completed' not in st.session_state: st.session_state.eval_run_completed = False
if 'eval_uploaded_filename' not in st.session_state: st.session_state.eval_uploaded_filename = None
if "upload_counter" not in st.session_state: st.session_state.upload_counter = 0
if 'last_eval_config' not in st.session_state: st.session_state.last_eval_config = {}

st.subheader("Trạng thái Hệ thống Cơ bản")
init_ok_eval = False
retriever_instance_eval = None
g_embedding_model_eval_object = None # ++ Đổi tên để rõ ràng là object ++
g_reranking_model_eval_loaded = None

with st.spinner("Kiểm tra và khởi tạo tài nguyên cốt lõi cho đánh giá..."):
    try:
        # ++ Tải embedding model đã chọn cho trang đánh giá ++
        g_embedding_model_eval_object = load_embedding_model(st.session_state.eval_selected_embedding_model)

        if st.session_state.eval_selected_reranker_model != 'Không sử dụng':
            g_reranking_model_eval_loaded = load_reranker_model(st.session_state.eval_selected_reranker_model)
        else:
            g_reranking_model_eval_loaded = None

        # ++ Sử dụng hàm get_rag_data_prefix và truyền object model ++
        current_eval_rag_prefix = config.get_rag_data_prefix(st.session_state.eval_selected_embedding_model)
        _, retriever_instance_eval = load_or_create_rag_components(g_embedding_model_eval_object, current_eval_rag_prefix)

        if retriever_instance_eval and g_embedding_model_eval_object:
            init_ok_eval = True
            st.success(f"Embedding model '{st.session_state.eval_selected_embedding_model.split('/')[-1]}' và Retriever sẵn sàng cho đánh giá.")
            if st.session_state.eval_selected_reranker_model != 'Không sử dụng' and not g_reranking_model_eval_loaded:
                st.warning(f"⚠️ Không tải được Reranker Model ({st.session_state.eval_selected_reranker_model}) cho đánh giá.")
            elif g_reranking_model_eval_loaded:
                 st.success(f"Reranker model '{st.session_state.eval_selected_reranker_model}' sẵn sàng cho đánh giá.")
            else:
                 st.info("Reranker không được chọn sử dụng cho đánh giá.")
        else:
            missing = []
            if not retriever_instance_eval: missing.append("Retriever/VectorDB")
            if not g_embedding_model_eval_object: missing.append(f"Embedding Model ({st.session_state.eval_selected_embedding_model})")
            st.error(f"⚠️ Lỗi khởi tạo cho đánh giá: {', '.join(missing)}.")
    except Exception as e:
        st.error(f"⚠️ Lỗi nghiêm trọng khi khởi tạo hệ thống cho đánh giá: {e}")

if init_ok_eval:
    reranker_display_eval_caption = st.session_state.eval_selected_reranker_model.split('/')[-1] if st.session_state.eval_selected_reranker_model != 'Không sử dụng' else "Tắt"
    # ++ Cập nhật caption cho trang đánh giá ++
    st.caption(
        f"Embedding: `{st.session_state.eval_selected_embedding_model.split('/')[-1]}` | "
        f"Mô hình Gemini: `{st.session_state.eval_selected_gemini_model}` | "
        f"Nguồn Query: `{st.session_state.eval_retrieval_query_mode}` | "
        f"Retrieval: `{st.session_state.eval_retrieval_method}` | "
        f"Reranker: `{reranker_display_eval_caption}`"
    )


    uploader_key = f"eval_file_uploader_{st.session_state.upload_counter}"
    st.subheader("Tải Lên File Đánh giá")
    uploaded_file = st.file_uploader("Chọn file JSON dữ liệu đánh giá...", type=["json"], key=uploader_key)

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.eval_uploaded_filename:
            try:
                eval_data_list = json.loads(uploaded_file.getvalue().decode('utf-8'))
                st.session_state.eval_data = eval_data_list
                st.session_state.eval_uploaded_filename = uploaded_file.name
                st.session_state.eval_run_completed = False
                st.session_state.last_eval_config = {}
                st.success(f"Đã tải file '{uploaded_file.name}' ({len(eval_data_list)} câu hỏi).")
            except Exception as e:
                st.error(f"Lỗi xử lý file JSON: {e}")
                st.session_state.eval_data = None; st.session_state.eval_uploaded_filename = None
                st.session_state.eval_run_completed = False

    if st.session_state.eval_data is not None:
        st.info(f"Sẵn sàng đánh giá với dữ liệu từ: **{st.session_state.eval_uploaded_filename}**.")
        if st.checkbox("Hiển thị dữ liệu mẫu (5 dòng)", key="show_eval_data_preview"):
            st.dataframe(pd.DataFrame(st.session_state.eval_data).head())

        if st.button("🚀 Bắt đầu Đánh giá", key="start_eval_button"):
            current_eval_config_for_run = {
                # ++ Thêm tên embedding model vào config chạy đánh giá ++
                'embedding_model_name': st.session_state.eval_selected_embedding_model,
                'retrieval_query_mode': st.session_state.eval_retrieval_query_mode,
                'retrieval_method': st.session_state.eval_retrieval_method,
                'selected_reranker_model': st.session_state.eval_selected_reranker_model,
                'gemini_model_name': st.session_state.eval_selected_gemini_model,
                'reranker_model_name_used': st.session_state.eval_selected_reranker_model if g_reranking_model_eval_loaded else "Không sử dụng"
            }
            st.session_state.last_eval_config = current_eval_config_for_run.copy()

            with st.spinner(f"Đang tải model Gemini cho đánh giá: {st.session_state.eval_selected_gemini_model}..."):
                 g_gemini_model_for_eval_run = load_gemini_model(st.session_state.eval_selected_gemini_model)

            if g_gemini_model_for_eval_run:
                st.info(f"Model Gemini '{st.session_state.eval_selected_gemini_model}' đã sẵn sàng cho đánh giá.")
                with st.spinner("⏳ Đang chạy đánh giá..."):
                    start_eval_time = time.time()
                    results_df = run_retrieval_evaluation(
                        eval_data=st.session_state.eval_data,
                        hybrid_retriever=retriever_instance_eval,
                        embedding_model_object=g_embedding_model_eval_object, # ++ Truyền object model embedding ++
                        reranking_model_selected=g_reranking_model_eval_loaded,
                        gemini_model=g_gemini_model_for_eval_run,
                        eval_config=st.session_state.last_eval_config
                    )
                    total_eval_time = time.time() - start_eval_time
                    st.success(f"Hoàn thành đánh giá sau {total_eval_time:.2f} giây.")
                    st.session_state.eval_results_df = results_df
                    st.session_state.eval_run_completed = True
                    st.rerun()
            else:
                st.error(f"Không thể tải model Gemini '{st.session_state.eval_selected_gemini_model}' cho đánh giá.")

    if st.session_state.eval_run_completed and st.session_state.eval_results_df is not None:
        st.subheader("Kết quả Đánh giá")
        detailed_results_df = st.session_state.eval_results_df
        last_config_run = st.session_state.last_eval_config

        st.markdown("**Cấu hình đã sử dụng cho lần chạy cuối:**")
        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4) # ++ Thêm cột cho Embedding ++
        # ++ Hiển thị Embedding model đã dùng ++
        embedding_name_display_result = last_config_run.get('embedding_model_name', 'N/A').split('/')[-1]
        cfg_col1.metric("Embedding Model", embedding_name_display_result)
        cfg_col2.metric("Nguồn Query", last_config_run.get('retrieval_query_mode', 'N/A'))
        cfg_col3.metric("Ret. Method", last_config_run.get('retrieval_method', 'N/A'))

        reranker_name_display_result = last_config_run.get('selected_reranker_model', 'N/A')
        reranker_name_display_result = reranker_name_display_result.split('/')[-1] if reranker_name_display_result != 'Không sử dụng' else "Tắt"
        cfg_col4.metric("Reranker", reranker_name_display_result)

        st.caption(f"Gemini: `{last_config_run.get('gemini_model_name', 'N/A')}`, Reranker thực tế: `{last_config_run.get('reranker_model_name_used', 'N/A').split('/')[-1] if last_config_run.get('reranker_model_name_used') != 'Không sử dụng' else 'Không sử dụng'}`")

        avg_metrics, num_eval, num_skipped_error = calculate_average_metrics(detailed_results_df)

        st.metric("Tổng số Queries", len(detailed_results_df))
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Queries Đánh giá Hợp lệ", num_eval)
        col_res2.metric("Queries Bỏ qua / Lỗi", num_skipped_error)

        if avg_metrics:
            st.markdown("#### Metrics Trung bình @K (trên các queries hợp lệ)")
            k_values_display = [3, 5, 10]
            cols_k = st.columns(len(k_values_display))
            for idx, k_val_display in enumerate(k_values_display): # Sửa tên biến
                with cols_k[idx]:
                    st.markdown(f"**K = {k_val_display}**")
                    st.text(f"Precision: {avg_metrics.get(f'avg_precision@{k_val_display}', 0.0):.4f}")
                    st.text(f"Recall:    {avg_metrics.get(f'avg_recall@{k_val_display}', 0.0):.4f}")
                    st.text(f"F1:        {avg_metrics.get(f'avg_f1@{k_val_display}', 0.0):.4f}")
                    st.text(f"MRR:       {avg_metrics.get(f'avg_mrr@{k_val_display}', 0.0):.4f}")
                    st.text(f"NDCG:      {avg_metrics.get(f'avg_ndcg@{k_val_display}', 0.0):.4f}")

            st.markdown("#### Thông tin Hiệu năng & Số lượng Trung bình")
            # ... ( giữ nguyên phần hiển thị hiệu năng và số lượng) ...

        with st.expander("Xem Kết quả Chi tiết cho từng Query"):
            display_columns = [
                'query_id', 'query', 'status',
                'embedding_model_name', # ++ Thêm cột embedding model ++
                'retrieval_query_mode','retrieval_method', 'selected_reranker_model',
                # ... (giữ nguyên các cột metrics khác) ...
                'processing_time', 'variation_time', 'search_time', 'rerank_time',
                'num_variations_generated','num_unique_docs_found', 'num_retrieved_before_rerank','num_docs_reranked', 'num_retrieved_after_rerank',
                'retrieved_ids', 'relevant_ids', 'summarizing_query', 'error_message'
            ]
            existing_display_columns = [col for col in display_columns if col in detailed_results_df.columns]
            st.dataframe(detailed_results_df[existing_display_columns])

        st.subheader("Lưu Kết quả Chi tiết")
        try:
            results_json = detailed_results_df.to_json(orient='records', indent=2, force_ascii=False)
            results_csv = detailed_results_df.to_csv(index=False).encode('utf-8')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # ++ Thêm embedding model vào tên file ++
            emb_suffix = last_config_run.get('embedding_model_name', 'na').split('/')[-1].replace('-', '').replace('_', '')[:10]
            qmode_suffix = last_config_run.get('retrieval_query_mode', 'na').lower()[:3]
            method_suffix = last_config_run.get('retrieval_method', 'na').lower()
            reranker_file_suffix = "norr"
            selected_reranker_for_filename = last_config_run.get('selected_reranker_model', 'Không sử dụng')
            if selected_reranker_for_filename != 'Không sử dụng':
                  reranker_file_suffix = selected_reranker_for_filename.split('/')[-1].replace('-', '').replace('_', '')[:10]
            model_suffix = last_config_run.get('gemini_model_name', 'gemini').split('/')[-1].replace('.','-')[:15]

            base_filename = f"eval_{emb_suffix}_{qmode_suffix}_{method_suffix}_{reranker_file_suffix}_{model_suffix}_{timestamp}"
            fname_json = f"{base_filename}.json"
            fname_csv = f"{base_filename}.csv"

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("💾 Tải về JSON", results_json, fname_json, "application/json", key="dl_json")
            with col_dl2:
                st.download_button("💾 Tải về CSV", results_csv, fname_csv, "text/csv", key="dl_csv")
        except Exception as e:
            st.error(f"Lỗi khi chuẩn bị file kết quả: {e}")

    st.markdown("---")
    st.subheader("Quản lý Trạng thái Đánh giá")
    if st.button("Xóa File Đã Tải và Kết Quả", key="clear_eval_state"):
        st.session_state.eval_data = None
        st.session_state.upload_counter += 1
        st.session_state.eval_run_completed = False
        st.session_state.eval_results_df = None
        st.session_state.last_eval_config = {}
        st.session_state.eval_uploaded_filename = None
        st.success("Đã xóa trạng thái đánh giá.")
        time.sleep(1)
        st.rerun()
else:
    st.warning("⚠️ Hệ thống cơ bản cho trang đánh giá chưa sẵn sàng. Vui lòng kiểm tra lỗi và khởi động lại.")