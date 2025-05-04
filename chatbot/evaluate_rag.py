# evaluate_rag.py
import argparse
import json
import time
import logging
import pandas as pd
import os

# Import các module cần thiết từ dự án của bạn
import config
import utils
import data_loader
# Giả định retriever.py và vector_db.py cũng cần thiết gián tiếp qua data_loader
# from utils import get_chunk_id # Đảm bảo hàm này tồn tại và đúng

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Hàm lấy ID từ chunk (QUAN TRỌNG: PHẢI GIỐNG HÀM TRONG APP.PY VÀ ĐÚNG VỚI DỮ LIỆU) ---
def get_chunk_id(doc_object, default_prefix="index_"):
    """Lấy ID duy nhất từ object chunk. Ưu tiên metadata['id']."""
    # === BẠN CẦN ĐẢM BẢO LOGIC NÀY ĐÚNG VỚI DỮ LIỆU CỦA BẠN ===
    if isinstance(doc_object, dict):
        metadata = doc_object.get('metadata', {})
        if 'id' in metadata: # Ưu tiên hàng đầu
            return metadata['id']
        # Thêm các logic lấy ID khác nếu cần (ví dụ từ key cấp cao hơn, hoặc tạo ID hash)
        # Ví dụ fallback (không khuyến khích dùng index):
        # if 'index' in item: return f"index_{item['index']}" # Nếu có index từ retrieval result
    logging.warning(f"Không thể lấy ID đáng tin cậy từ chunk: {str(doc_object)[:100]}...")
    return None # Trả về None nếu không có ID rõ ràng

# --- Hàm tính toán chỉ số retrieval ---
def calculate_retrieval_metrics(retrieved_ids_set, ground_truth_ids_set):
    """Tính Precision, Recall, F1 cho retrieval."""
    if not isinstance(retrieved_ids_set, set): retrieved_ids_set = set(retrieved_ids_set)
    if not isinstance(ground_truth_ids_set, set): ground_truth_ids_set = set(ground_truth_ids_set)

    true_positives = len(retrieved_ids_set.intersection(ground_truth_ids_set))
    retrieved_count = len(retrieved_ids_set)
    ground_truth_count = len(ground_truth_ids_set)

    precision = true_positives / retrieved_count if retrieved_count > 0 else 0
    recall = true_positives / ground_truth_count if ground_truth_count > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "true_positives": true_positives, "retrieved_count": retrieved_count, "ground_truth_count": ground_truth_count}

# --- Hàm chính để chạy đánh giá ---
def run_evaluation(eval_data_path, results_output_path, retrieval_mode='Sâu'):
    """Chạy đánh giá RAG trên bộ dữ liệu."""
    logging.info("--- Bắt đầu quá trình đánh giá RAG ---")

    # --- 1. Tải dữ liệu đánh giá ---
    logging.info(f"Đang tải dữ liệu đánh giá từ: {eval_data_path}")
    try:
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)
        logging.info(f"Đã tải {len(evaluation_data)} mẫu đánh giá.")
        # Thêm query_id nếu chưa có
        for i, item in enumerate(evaluation_data):
            item['query_id'] = item.get('query_id', f"query_{i+1}")
    except Exception as e:
        logging.error(f"Lỗi khi tải hoặc xử lý tệp đánh giá: {e}", exc_info=True)
        return

    # --- 2. Khởi tạo các thành phần RAG ---
    logging.info("Đang khởi tạo các thành phần RAG (models, retriever)...")
    try:
        # Tải models (không cần LLM nếu chỉ đánh giá retrieval/rerank)
        embedding_model = utils.load_embedding_model(config.embedding_model_name)
        reranker_model = utils.load_reranker_model(config.reranking_model_name)
        # Tải retriever (bao gồm Vector DB)
        _, hybrid_retriever = data_loader.load_or_create_rag_components(embedding_model)

        # (Tùy chọn) Tải LLM nếu muốn đánh giá cả bước generation
        # gemini_model = utils.load_gemini_model(config.DEFAULT_GEMINI_MODEL) # Hoặc model cụ thể

        if not all([embedding_model, reranker_model, hybrid_retriever]):
            raise RuntimeError("Không thể khởi tạo tất cả các thành phần RAG cần thiết.")
        logging.info("Khởi tạo thành phần RAG thành công.")
    except Exception as e:
        logging.error(f"Lỗi khi khởi tạo thành phần RAG: {e}", exc_info=True)
        return

    # --- 3. Chạy Pipeline và Thu thập Kết quả ---
    all_results = []
    start_time_total = time.time()

    logging.info(f"Bắt đầu xử lý {len(evaluation_data)} câu hỏi...")
    for i, eval_item in enumerate(evaluation_data):
        query_id = eval_item['query_id']
        query = eval_item['query']
        ground_truth_ids = set(eval_item['relevant_chunk_ids']) # Dùng set để dễ so sánh
        item_results = {"query_id": query_id, "query": query}
        start_time_item = time.time()

        logging.info(f"Đang xử lý câu hỏi {i+1}/{len(evaluation_data)}: '{query_id}'")

        try:
            # --- Bước A (Tùy chọn): Chạy LLM 1 để lấy summary/variants ---
            # Để đơn giản, có thể bỏ qua bước này và dùng query gốc cho retrieval/rerank
            # Hoặc chạy nó nếu muốn mô phỏng chính xác app
            # _, _, all_queries, summarizing_q = utils.generate_query_variations(query, gemini_model, ...)
            # Nếu bỏ qua LLM1:
            summarizing_q = query # Dùng query gốc làm summary tạm
            all_queries = [query] # Chỉ dùng query gốc nếu không chạy LLM1

            # --- Bước B: Retrieval ---
            collected_docs_data_by_id = {} # Dùng ID làm key
            item_results["retrieval_mode_used"] = retrieval_mode

            if retrieval_mode == 'Đơn giản':
                variant_results = hybrid_retriever.hybrid_search(
                    summarizing_q, embedding_model,
                    vector_search_k=config.VECTOR_K_PER_QUERY,
                    final_k=config.HYBRID_K_PER_QUERY
                )
                for item in variant_results:
                     doc = item.get('doc')
                     doc_id = get_chunk_id(doc)
                     if doc_id and doc_id not in collected_docs_data_by_id: collected_docs_data_by_id[doc_id] = doc
            else: # 'Sâu'
                for query_variant in all_queries:
                     variant_results = hybrid_retriever.hybrid_search(
                         query_variant, embedding_model,
                         vector_search_k=config.VECTOR_K_PER_QUERY,
                         final_k=config.HYBRID_K_PER_QUERY
                     )
                     for item in variant_results:
                         doc = item.get('doc')
                         doc_id = get_chunk_id(doc)
                         if doc_id and doc_id not in collected_docs_data_by_id: collected_docs_data_by_id[doc_id] = doc

            retrieved_ids = set(collected_docs_data_by_id.keys())
            item_results["retrieved_ids"] = list(retrieved_ids) # Lưu dạng list vào kết quả

            # Tính retrieval metrics
            retrieval_metrics = calculate_retrieval_metrics(retrieved_ids, ground_truth_ids)
            item_results.update(retrieval_metrics) # Thêm các chỉ số vào results

            # --- Bước C & D: Reranking ---
            retrieved_chunks = list(collected_docs_data_by_id.values())
            unique_docs_for_reranking_input_eval = []
            final_relevant_documents_eval = []
            reranked_docs_info_list = [] # Lưu thông tin rerank

            if retrieved_chunks:
                # Giới hạn số lượng rerank
                docs_to_rerank = retrieved_chunks[:config.MAX_DOCS_FOR_RERANK]
                # Input cho reranker cần là list các dict có key 'doc'
                # Hàm rerank_documents cần trả về cả score và index gốc hoặc ID nếu có
                reranker_input = [{'doc': doc} for doc in docs_to_rerank] # Cần đảm bảo rerank_documents xử lý input này

                reranked_results_with_scores = utils.rerank_documents(
                    summarizing_q, # Dùng summarizing_q để rerank
                    reranker_input, # Chỉ truyền doc
                    reranker_model
                ) # Hàm này cần trả về list dict {'doc': ..., 'score': ...}

                # Lấy ID và score sau rerank
                reranked_docs_info_list = []
                for rank, item in enumerate(reranked_results_with_scores):
                     doc = item.get('doc')
                     score = item.get('score')
                     doc_id = get_chunk_id(doc)
                     if doc_id:
                         reranked_docs_info_list.append({'id': doc_id, 'rerank_score': score, 'rerank_pos': rank + 1})

                final_relevant_documents_eval = reranked_results_with_scores[:config.FINAL_NUM_RESULTS_AFTER_RERANK]

            item_results["reranked_ids_scores"] = reranked_docs_info_list # Lưu list ID và score sau rerank
            final_context_ids = {get_chunk_id(item['doc']) for item in final_relevant_documents_eval if get_chunk_id(item['doc'])}
            item_results["final_context_ids"] = list(final_context_ids)

            # --- (Tùy chọn) Bước E: Generation ---
            # if gemini_model:
            #     generated_answer = utils.generate_answer_with_gemini(
            #         query, final_relevant_documents_eval, gemini_model,
            #         mode='Đầy đủ', # Hoặc mode khác
            #         chat_history=None # Không dùng history trong eval batch
            #     )
            #     item_results["generated_answer"] = generated_answer
            # else:
            #     item_results["generated_answer"] = None

            item_results["processing_time"] = time.time() - start_time_item
            logging.info(f"Hoàn thành câu hỏi {i+1}. Thời gian: {item_results['processing_time']:.2f}s. Recall: {item_results['recall']:.2f}")

        except Exception as item_error:
            logging.error(f"Lỗi khi xử lý câu hỏi '{query_id}': {item_error}", exc_info=True)
            item_results["error"] = str(item_error)
            item_results["processing_time"] = time.time() - start_time_item

        all_results.append(item_results)

    # --- 4. Tổng hợp và Báo cáo Kết quả ---
    logging.info("--- Hoàn thành xử lý các câu hỏi ---")
    total_time = time.time() - start_time_total
    logging.info(f"Tổng thời gian xử lý: {total_time:.2f} giây")
    logging.info(f"Thời gian trung bình mỗi câu hỏi: {total_time / len(evaluation_data):.2f} giây")

    # Tạo DataFrame từ kết quả
    results_df = pd.DataFrame(all_results)

    # Tính toán các chỉ số trung bình
    avg_precision = results_df['precision'].mean()
    avg_recall = results_df['recall'].mean()
    avg_f1 = results_df['f1'].mean()
    # Có thể tính thêm các chỉ số khác (ví dụ: MRR từ reranked_ids_scores nếu cần)

    logging.info("--- Kết quả Đánh giá Retrieval Trung bình ---")
    logging.info(f"Precision (Trung bình): {avg_precision:.4f}")
    logging.info(f"Recall (Trung bình):    {avg_recall:.4f}")
    logging.info(f"F1-Score (Trung bình):  {avg_f1:.4f}")

    # --- 5. Lưu kết quả chi tiết ---
    try:
        results_df.to_csv(results_output_path, index=False, encoding='utf-8-sig')
        logging.info(f"Đã lưu kết quả chi tiết vào: {results_output_path}")
    except Exception as save_error:
        logging.error(f"Lỗi khi lưu kết quả vào tệp CSV: {save_error}", exc_info=True)

    logging.info("--- Quá trình đánh giá kết thúc ---")


# --- Entry point để chạy script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy đánh giá hàng loạt cho hệ thống RAG Luật GTĐB.")
    parser.add_argument("eval_data_path", type=str, help="Đường dẫn đến tệp JSON chứa dữ liệu đánh giá (query, relevant_chunk_ids).")
    parser.add_argument("-o", "--output", type=str, default="evaluation_results.csv", help="Đường dẫn để lưu kết quả đánh giá chi tiết (CSV). Mặc định: evaluation_results.csv")
    parser.add_argument("-r", "--retrieval_mode", type=str, default="Sâu", choices=['Đơn giản', 'Sâu'], help="Chế độ truy vấn để đánh giá ('Đơn giản' hoặc 'Sâu'). Mặc định: Sâu")
    # Thêm các argument khác nếu muốn tùy chỉnh config khi chạy (ví dụ: model name, K values...)

    args = parser.parse_args()

    run_evaluation(args.eval_data_path, args.output, args.retrieval_mode)