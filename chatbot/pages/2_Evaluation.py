# # pages/2_Evaluation.py

# import streamlit as st
# import pandas as pd
# import json
# import time
# import math
# import os
# import logging
# from datetime import datetime
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import config
# import utils
# import data_loader
# from vector_db import SimpleVectorDatabase # Cần lớp này để kiểm tra kiểu
# from retriever import HybridRetriever     # Cần lớp này để kiểm tra kiểu

# # --- Cấu hình Logging ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # --- Các hàm tính Metrics (Giống trong notebook) ---
# def precision_at_k(retrieved_ids, relevant_ids, k):
#     if k <= 0: return 0.0
#     retrieved_at_k = retrieved_ids[:k]; relevant_set = set(relevant_ids)
#     if not relevant_set: return 0.0
#     intersect = set(retrieved_at_k) & relevant_set
#     return len(intersect) / k

# def recall_at_k(retrieved_ids, relevant_ids, k):
#     relevant_set = set(relevant_ids)
#     if not relevant_set: return 1.0
#     retrieved_at_k = retrieved_ids[:k]
#     intersect = set(retrieved_at_k) & relevant_set
#     return len(intersect) / len(relevant_set)

# def f1_at_k(retrieved_ids, relevant_ids, k):
#     prec = precision_at_k(retrieved_ids, relevant_ids, k); rec = recall_at_k(retrieved_ids, relevant_ids, k)
#     return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

# def mrr_at_k(retrieved_ids, relevant_ids, k):
#     relevant_set = set(relevant_ids);
#     if not relevant_set: return 0.0
#     for rank, doc_id in enumerate(retrieved_ids[:k], 1):
#         if doc_id in relevant_set: return 1.0 / rank
#     return 0.0

# def ndcg_at_k(retrieved_ids, relevant_ids, k):
#     relevant_set = set(relevant_ids);
#     if not relevant_set: return 1.0
#     retrieved_at_k = retrieved_ids[:k]; dcg = 0.0; idcg = 0.0
#     for i, doc_id in enumerate(retrieved_at_k):
#         if doc_id in relevant_set: dcg += 1.0 / math.log2(i + 2)
#     for i in range(min(k, len(relevant_set))): idcg += 1.0 / math.log2(i + 2)
#     return dcg / idcg if idcg > 0 else 0.0

# # --- Hàm thực thi lõi đánh giá Retrieval ---
# def run_retrieval_evaluation(
#     eval_data: list,
#     hybrid_retriever: HybridRetriever,
#     embedding_model, # Model đã load
#     reranking_model, # Model đã load
#     gemini_model,    # Model đã load (có thể None)
#     eval_config: dict # Chứa các tùy chọn như retrieval_mode, use_history,...
#     ):
    
#     results_list = []
#     k_values = [3, 5, 10] # K values for evaluation metrics

#     # Lấy cấu hình từ dict
#     retrieval_mode = eval_config.get('retrieval_mode', 'Đơn giản') # Default nếu thiếu
#     use_history = eval_config.get('use_history_for_llm1', False)
#     dummy_history = [{"role": "user", "content": "..."}] if use_history else None # Simple dummy history

#     # Placeholder cho progress bar
#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     total_items = len(eval_data)
#     for i, item in enumerate(eval_data):
#         query_id = item.get("query_id"); original_query = item.get("query")
#         relevant_chunk_ids = set(item.get("relevant_chunk_ids", []))
#         if not query_id or not original_query: continue # Bỏ qua item không hợp lệ

#         status_text.text(f"Đang xử lý query {i+1}/{total_items}: {query_id}")
#         logging.info(f"Eval - Processing QID: {query_id}")

#         start_time = time.time()
#         query_metrics = {
#             "query_id": query_id, "query": original_query, "retrieval_mode": retrieval_mode,
#             "status": "error", "retrieved_ids": [], "relevant_ids": list(relevant_chunk_ids),
#             "processing_time": 0.0
#         }
#         # Init metrics columns
#         for k in k_values: query_metrics[f'precision@{k}'] = 0.0; query_metrics[f'recall@{k}'] = 0.0; query_metrics[f'f1@{k}'] = 0.0; query_metrics[f'mrr@{k}'] = 0.0; query_metrics[f'ndcg@{k}'] = 0.0
#         timing_keys = ['variation_time', 'search_time', 'rerank_time']
#         count_keys = ['num_variations_generated', 'num_unique_docs_found', 'num_docs_reranked']
#         for k in timing_keys + count_keys: query_metrics[k] = 0.0

#         try:
#             # 1. Generate Variations / Check Relevance
#             variation_start = time.time()
#             relevance_status, _, all_queries, summarizing_query = utils.generate_query_variations(
#                 original_query=original_query,
#                 gemini_model=gemini_model,
#                 chat_history=dummy_history
#             )
#             query_metrics["variation_time"] = time.time() - variation_start
#             query_metrics["summarizing_query"] = summarizing_query
#             query_metrics["num_variations_generated"] = len(all_queries)

#             if relevance_status == 'invalid':
#                 query_metrics["status"] = "skipped_irrelevant"
#                 query_metrics["processing_time"] = time.time() - start_time
#                 results_list.append(query_metrics)
#                 progress_bar.progress((i + 1) / total_items) # Cập nhật progress bar
#                 continue

#             # 2. Retrieval based on mode
#             collected_docs_data = {}
#             search_start = time.time()
#             if retrieval_mode == 'Đơn giản':
#                 variant_results = hybrid_retriever.hybrid_search(
#                     summarizing_query, embedding_model,
#                     vector_search_k=config.VECTOR_K_PER_QUERY, final_k=config.HYBRID_K_PER_QUERY
#                 )
#                 for res_item in variant_results:
#                     idx = res_item.get('index');
#                     if isinstance(idx, int) and idx >= 0: collected_docs_data[idx] = res_item
#             elif retrieval_mode == 'Sâu':
#                 for q_variant in all_queries:
#                     variant_results = hybrid_retriever.hybrid_search(
#                         q_variant, embedding_model,
#                         vector_search_k=config.VECTOR_K_PER_QUERY, final_k=config.HYBRID_K_PER_QUERY
#                     )
#                     for res_item in variant_results:
#                         idx = res_item.get('index')
#                         if isinstance(idx, int) and idx >= 0 and idx not in collected_docs_data:
#                              collected_docs_data[idx] = res_item
#             query_metrics["search_time"] = time.time() - search_start
#             query_metrics["num_unique_docs_found"] = len(collected_docs_data)

#             # 3. Prepare for Reranking
#             unique_docs_list = list(collected_docs_data.values())
#             unique_docs_list.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
#             docs_for_reranking_input = unique_docs_list[:config.MAX_DOCS_FOR_RERANK]
#             query_metrics["num_docs_reranked"] = len(docs_for_reranking_input)

#             # 4. Reranking
#             rerank_start = time.time()
#             reranked_results = utils.rerank_documents(
#                 summarizing_query, docs_for_reranking_input, reranking_model
#             )
#             query_metrics["rerank_time"] = time.time() - rerank_start

#             # 5. Get Final Retrieved IDs
#             final_retrieved_docs = reranked_results[:config.FINAL_NUM_RESULTS_AFTER_RERANK]
#             retrieved_ids = []
#             for res in final_retrieved_docs:
#                 doc_data = res.get('doc', {}); chunk_id = None
#                 if isinstance(doc_data, dict):
#                     chunk_id = doc_data.get('id') or doc_data.get('metadata', {}).get('id') or doc_data.get('metadata', {}).get('chunk_id')
#                 if chunk_id: retrieved_ids.append(str(chunk_id))
#             query_metrics["retrieved_ids"] = retrieved_ids

#             # 6. Calculate Metrics
#             query_metrics["status"] = "evaluated"
#             for k in k_values:
#                 query_metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, relevant_chunk_ids, k)
#                 query_metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, relevant_chunk_ids, k)
#                 query_metrics[f'f1@{k}'] = f1_at_k(retrieved_ids, relevant_chunk_ids, k)
#                 query_metrics[f'mrr@{k}'] = mrr_at_k(retrieved_ids, relevant_chunk_ids, k)
#                 query_metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved_ids, relevant_chunk_ids, k)

#         except Exception as e:
#             logging.exception(f"Error evaluating QID {query_id}: {e}")
#             query_metrics["status"] = "error_runtime"
#             query_metrics["error_message"] = str(e)
#         finally:
#             query_metrics["processing_time"] = time.time() - start_time
#             results_list.append(query_metrics)
#             progress_bar.progress((i + 1) / total_items) # Cập nhật progress bar

#     status_text.text(f"Hoàn thành đánh giá {total_items} queries!")
#     return pd.DataFrame(results_list)

# # --- Hàm tính toán tổng hợp Metrics ---
# def calculate_average_metrics(df_results: pd.DataFrame):
#     """Tính toán metrics trung bình từ DataFrame kết quả chi tiết."""
#     evaluated_df = df_results[df_results['status'] == 'evaluated']
#     num_evaluated = len(evaluated_df)
#     if num_evaluated == 0:
#         return None, num_evaluated, len(df_results) - num_evaluated

#     avg_metrics = {}
#     k_values = [3, 5, 10]
#     metric_keys_k = [f'{m}@{k}' for k in k_values for m in ['precision', 'recall', 'f1', 'mrr', 'ndcg']]
#     timing_keys = ['processing_time', 'variation_time', 'search_time', 'rerank_time']
#     count_keys = ['num_variations_generated', 'num_unique_docs_found', 'num_docs_reranked']

#     for key in metric_keys_k + timing_keys + count_keys:
#         # Tính tổng, bỏ qua NaN nếu có
#         total = evaluated_df[key].sum(skipna=True)
#         avg_metrics[f'avg_{key}'] = total / num_evaluated

#     return avg_metrics, num_evaluated, len(df_results) - num_evaluated


# # --- Giao diện Streamlit cho Trang Đánh giá ---
# st.set_page_config(page_title="Đánh giá Retrieval", layout="wide")
# st.title("📊 Đánh giá Hệ thống Retrieval")

# st.markdown("""
# Trang này cho phép bạn chạy đánh giá hiệu suất của hệ thống retrieval (tìm kiếm + xếp hạng lại)
# dựa trên một tập dữ liệu có chứa các câu hỏi và các chunk tài liệu liên quan (ground truth).
# """)

# # --- Kiểm tra và Hiển thị Trạng thái Hệ thống ---
# st.subheader("Trạng thái Hệ thống Cơ bản")
# init_ok = False
# models_ready = False
# retriever_instance = None

# # Cố gắng lấy các thành phần đã cache từ app chính (có thể cần điều chỉnh cách truy cập)
# # Streamlit không dễ dàng chia sẻ @st.cache_resource giữa các trang một cách trực tiếp.
# # Cách tiếp cận đơn giản là tải lại trên trang này nếu cần.
# # Hoặc lưu trữ instance vào st.session_state trong app.py nếu có thể.

# # Tạm thời, chúng ta sẽ tải lại model và retriever trên trang này khi cần đánh giá.
# # Điều này đảm bảo trang đánh giá hoạt động độc lập hơn nhưng không tận dụng cache từ trang chính.

# with st.spinner("Kiểm tra và khởi tạo tài nguyên..."):
#     # Tải models (sẽ dùng cache nội bộ của hàm nếu chưa force reload)
#     g_embedding_model = utils.load_embedding_model(config.embedding_model_name)
#     g_reranking_model = utils.load_reranker_model(config.reranking_model_name)

#     # Tải vector DB và retriever
#     try:
#         # Sử dụng hàm gốc từ data_loader để đảm bảo nhất quán
#         # Hàm này cũng có cache riêng (@st.cache_resource trong app.py)
#         # nhưng gọi lại ở đây sẽ tạo instance mới nếu không có cache session
#         # TODO: Tìm cách chia sẻ instance từ app.py hiệu quả hơn nếu cần tối ưu
#         # Hiện tại, tải lại để đảm bảo trang độc lập
#         vector_db, retriever_instance = data_loader.load_or_create_rag_components(g_embedding_model)
#         if vector_db and retriever_instance and g_embedding_model and g_reranking_model:
#             init_ok = True
#             models_ready = True # Assume Gemini load happens later
#             st.success("✅ VectorDB, Retriever, Embedding, Reranker đã sẵn sàng.")
#         else:
#             st.error("⚠️ Lỗi khi khởi tạo VectorDB hoặc Retriever.")
#     except Exception as e:
#         st.error(f"⚠️ Lỗi nghiêm trọng khi khởi tạo hệ thống: {e}")

# if init_ok:
#     st.subheader("Cấu hình Đánh giá")
#     st.markdown("Đánh giá sẽ được chạy với các cấu hình **hiện tại** được chọn trong Sidebar của trang Chatbot chính.")

#     # Lấy cấu hình hiện tại từ session_state (được thiết lập ở app.py)
#     current_gemini_model = st.session_state.get('selected_gemini_model', config.DEFAULT_GEMINI_MODEL)
#     current_retrieval_mode = st.session_state.get('retrieval_mode', 'Đơn giản')
#     current_use_history = st.session_state.get('use_history_for_llm1', True)

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.info(f"**Chế độ Retrieval:** `{current_retrieval_mode}`")
#     with col2:
#         st.info(f"**Model Gemini (Variations):** `{current_gemini_model}`")
#     with col3:
#         st.info(f"**Sử dụng Lịch sử (LLM1):** `{current_use_history}`")

#     eval_config_dict = {
#         'retrieval_mode': current_retrieval_mode,
#         'use_history_for_llm1': current_use_history,
#         # Thêm các config khác nếu cần
#     }

#     st.subheader("Tải Lên File Đánh giá")
#     uploaded_file = st.file_uploader(
#         "Chọn file JSON chứa dữ liệu đánh giá (định dạng: [{'query_id': ..., 'query': ..., 'relevant_chunk_ids': [...]}, ...])",
#         type=["json"]
#     )

#     if uploaded_file is not None:
#         try:
#             # Đọc nội dung file
#             eval_data = json.load(uploaded_file)
#             st.success(f"Đã tải lên thành công file '{uploaded_file.name}' chứa {len(eval_data)} câu hỏi.")

#             # Hiển thị một phần dữ liệu để kiểm tra
#             if st.checkbox("Hiển thị dữ liệu mẫu (5 dòng đầu)"):
#                 st.dataframe(pd.DataFrame(eval_data).head())

#             # Nút bắt đầu đánh giá
#             if st.button("🚀 Bắt đầu Đánh giá", key="start_eval_button"):
#                 # Tải model Gemini được chọn
#                 with st.spinner(f"Đang tải model Gemini: {current_gemini_model}..."):
#                     g_gemini_model = utils.load_gemini_model(current_gemini_model) # Dùng hàm gốc để tải

#                 if not g_gemini_model and config.GOOGLE_API_KEY: # Chỉ báo lỗi nếu có key mà tải thất bại
#                      st.error(f"Lỗi tải model Gemini '{current_gemini_model}'. Vui lòng kiểm tra API Key và cấu hình.")
#                 elif not config.GOOGLE_API_KEY:
#                      st.warning("Không tìm thấy Google API Key. Bước tạo biến thể/kiểm tra liên quan sẽ bị bỏ qua.")
#                      # Đánh giá vẫn có thể chạy nếu không cần bước này

#                 # Chạy đánh giá
#                 with st.spinner("⏳ Đang chạy đánh giá... Quá trình này có thể mất vài phút."):
#                     start_eval_time = time.time()
#                     detailed_results_df = run_retrieval_evaluation(
#                         eval_data=eval_data,
#                         hybrid_retriever=retriever_instance,
#                         embedding_model=g_embedding_model,
#                         reranking_model=g_reranking_model,
#                         gemini_model=g_gemini_model,
#                         eval_config=eval_config_dict
#                     )
#                     total_eval_time = time.time() - start_eval_time
#                     st.info(f"Hoàn thành đánh giá sau {total_eval_time:.2f} giây.")

#                 # --- Hiển thị Kết quả ---
#                 st.subheader("Kết quả Đánh giá")

#                 # Tính toán metrics trung bình
#                 avg_metrics, num_eval, num_skipped_error = calculate_average_metrics(detailed_results_df)

#                 st.metric("Tổng số Queries", len(detailed_results_df))
#                 col_res1, col_res2, col_res3 = st.columns(3)
#                 col_res1.metric("Queries Đánh giá Hợp lệ", num_eval)
#                 col_res2.metric("Queries Bỏ qua/Lỗi", num_skipped_error)

#                 if avg_metrics:
#                     st.markdown("#### Metrics Trung bình (trên các queries hợp lệ)")
#                     k_values_display = [3, 5, 10]
#                     cols_k = st.columns(len(k_values_display))
#                     for idx, k in enumerate(k_values_display):
#                         with cols_k[idx]:
#                             st.markdown(f"**K = {k}**")
#                             st.text(f"Precision: {avg_metrics.get(f'avg_precision@{k}', 0.0):.4f}")
#                             st.text(f"Recall:    {avg_metrics.get(f'avg_recall@{k}', 0.0):.4f}")
#                             st.text(f"F1:        {avg_metrics.get(f'avg_f1@{k}', 0.0):.4f}")
#                             st.text(f"MRR:       {avg_metrics.get(f'avg_mrr@{k}', 0.0):.4f}")
#                             st.text(f"NDCG:      {avg_metrics.get(f'avg_ndcg@{k}', 0.0):.4f}")

#                     st.markdown("#### Thông tin Hiệu năng Trung bình")
#                     col_perf1, col_perf2, col_perf3 = st.columns(3)
#                     col_perf1.metric("Avg Total Time/Query (s)", f"{avg_metrics.get('avg_processing_time', 0.0):.3f}")
#                     col_perf2.metric("Avg Variation Time (s)", f"{avg_metrics.get('avg_variation_time', 0.0):.3f}")
#                     col_perf3.metric("Avg Search Time (s)", f"{avg_metrics.get('avg_search_time', 0.0):.3f}")
#                     # Có thể thêm các timing khác nếu muốn

#                 # Hiển thị bảng kết quả chi tiết
#                 with st.expander("Xem Kết quả Chi tiết cho từng Query"):
#                     # Chọn các cột cần hiển thị để bảng gọn hơn
#                     display_columns = [
#                         'query_id', 'query', 'status', 'retrieval_mode',
#                         'precision@3', 'recall@10', 'mrr@10', 'ndcg@10',
#                         'processing_time', 'retrieved_ids' # Giữ lại retrieved_ids để tham khảo
#                     ]
#                     # Lọc các cột tồn tại trong df
#                     existing_display_columns = [col for col in display_columns if col in detailed_results_df.columns]
#                     st.dataframe(detailed_results_df[existing_display_columns])

#                 # --- Lưu Kết quả ---
#                 st.subheader("Lưu Kết quả Chi tiết")
#                 try:
#                     # Chuyển DataFrame thành JSON lines hoặc CSV để tải về
#                     results_json = detailed_results_df.to_json(orient='records', indent=2, force_ascii=False)
#                     results_csv = detailed_results_df.to_csv(index=False).encode('utf-8')

#                     # Tạo tên file động
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     mode_suffix = current_retrieval_mode.lower().replace(' ', '_')
#                     fname_json = f"evaluation_results_{mode_suffix}_{timestamp}.json"
#                     fname_csv = f"evaluation_results_{mode_suffix}_{timestamp}.csv"

#                     col_dl1, col_dl2 = st.columns(2)
#                     with col_dl1:
#                         st.download_button(
#                             label="💾 Tải về JSON",
#                             data=results_json,
#                             file_name=fname_json,
#                             mime="application/json",
#                         )
#                     with col_dl2:
#                         st.download_button(
#                             label="💾 Tải về CSV",
#                             data=results_csv,
#                             file_name=fname_csv,
#                             mime="text/csv",
#                         )
#                 except Exception as e:
#                     st.error(f"Lỗi khi chuẩn bị file để tải về: {e}")

#         except json.JSONDecodeError:
#             st.error("Lỗi: File tải lên không phải là định dạng JSON hợp lệ.")
#         except Exception as e:
#             st.error(f"Lỗi không xác định khi xử lý file hoặc chạy đánh giá: {e}")
#             logging.exception("Unhandled error during evaluation page processing.")

# else:
#     st.warning("⚠️ Hệ thống cơ bản chưa sẵn sàng. Vui lòng kiểm tra lại trang Chatbot chính hoặc khởi động lại ứng dụng.")