# utils.py
import json
import os
import re
import logging
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from kaggle_secrets import UserSecretsClient # Cần thiết nếu chạy trên Kaggle
import streamlit as st
# Import config (đảm bảo config.py cùng cấp hoặc trong sys.path)
import config

# --- Model Loading Functions ---
# (Giữ lại ở app.py để dùng với cache Streamlit)
# Các hàm load_..._model() sẽ nằm trong app.py

# --- Query Augmentation ---
def generate_query_variations(original_query, gemini_model, num_variations=config.NUM_QUERY_VARIATIONS):
    """Tạo biến thể và câu hỏi tổng hợp từ câu hỏi gốc."""
    logging.info(f"Đang tạo {num_variations} biến thể và tóm tắt cho query: \"{original_query[:50]}...\"")
    prompt = f"""Bạn là một chuyên gia về luật giao thông đường bộ Việt Nam... (Giữ nguyên prompt chi tiết như code gốc)...

Hãy trả lời THEO ĐÚNG ĐỊNH DẠNG JSON sau... (Giữ nguyên yêu cầu JSON) ...
{{
  "variations": [...],
  "summarizing_query": "..."
}}""" # Giữ nguyên prompt như code gốc

    all_queries = [original_query]
    summarizing_query = original_query
    if gemini_model is None:
        logging.warning("Gemini model chưa load, không thể tạo biến thể query.")
        return all_queries, summarizing_query

    try:
        response = gemini_model.generate_content(prompt)
        if hasattr(response, 'text') and response.text:
            generated_text = response.text
            json_match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                if json_str:
                    try:
                        parsed_data = json.loads(json_str.strip())
                        if isinstance(parsed_data, dict):
                            variations = parsed_data.get('variations', [])
                            parsed_summary = parsed_data.get('summarizing_query', '')
                            if isinstance(variations, list) and variations:
                                all_queries.extend(variations[:num_variations])
                                all_queries = list(set(all_queries))
                            if parsed_summary and isinstance(parsed_summary, str):
                                summarizing_query = parsed_summary
                    except json.JSONDecodeError as e:
                        logging.warning(f"Lỗi parse JSON từ LLM (generate_query_variations): {e}")
        else:
             logging.warning(f"Không tìm thấy khối JSON trong phản hồi LLM (generate_query_variations).")
    except Exception as e:
        logging.error(f"Lỗi khi gọi Gemini API (generate_query_variations): {e}")

    if not summarizing_query: summarizing_query = original_query
    logging.info(f"Tạo xong {len(all_queries)} queries. Summarizing query: \"{summarizing_query[:50]}...\"")
    return all_queries, summarizing_query

# --- Data Processing ---
def embed_legal_chunks_return_valid(file_paths, model):
    
    all_chunks_read = []

    print(f"Bắt đầu xử lý dữ liệu.")
    print(f"Đang đọc dữ liệu từ {len(file_paths)} file JSON...")

    # Đọc tất cả các chunk từ các file JSON
    for file_path in file_paths:
        # Kiểm tra xem file có tồn tại không
        if os.path.exists(file_path):
            print(f"Đang đọc: {os.path.basename(file_path)}")
            try:
                # Mở và đọc file JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Kiểm tra xem dữ liệu có phải là một list không
                    if isinstance(data, list):
                        all_chunks_read.extend(data)
                        print(f"  -> Đã đọc {len(data)} chunks. Tổng cộng tạm thời: {len(all_chunks_read)}")
                    else:
                        # Cảnh báo nếu file không chứa list JSON
                        print(f"  -> Cảnh báo: File không chứa danh sách JSON. Bỏ qua.")
            except json.JSONDecodeError:
                print(f"  -> Lỗi: File không phải là JSON hợp lệ. Bỏ qua.")
            except Exception as e:
                 print(f"  -> Lỗi khi đọc file: {e}. Bỏ qua.")
        else:
            print(f"Cảnh báo: File không tồn tại: '{file_path}'. Bỏ qua.")


    # Kiểm tra xem có chunk nào được đọc không
    if not all_chunks_read:
        print("Không đọc được chunk nào từ các file đã cung cấp. Trả về danh sách rỗng và None.")
        return [], None # Trả về list rỗng và None

    print(f"\nTổng số chunks đã đọc từ các file: {len(all_chunks_read)}")

    # Lọc ra các chunk hợp lệ và văn bản tương ứng
    texts_to_embed = []
    valid_chunks = []

    # Lặp qua các chunk đã đọc để lấy text hợp lệ
    for chunk in all_chunks_read:
        text = chunk.get('text')
        # Kiểm tra xem 'text' có tồn tại, là string và không rỗng (sau khi loại bỏ khoảng trắng)
        if text and isinstance(text, str) and text.strip():
            texts_to_embed.append(text.strip()) # Thêm text đã được làm sạch
            valid_chunks.append(chunk) # Thêm chunk hợp lệ vào danh sách

    # Kiểm tra xem có text nào hợp lệ để embed không
    if not texts_to_embed:
        print("Không tìm thấy nội dung text hợp lệ trong các chunk để embedding. Trả về danh sách rỗng và None.")
        return [], None # Trả về list rỗng và None

    print(f"Tìm thấy {len(valid_chunks)} chunks hợp lệ có text.")
    print(f"Đang tạo embeddings cho {len(texts_to_embed)} đoạn text hợp lệ...")

    # Tạo embeddings sử dụng model được cung cấp
    try:
        # show_progress_bar=True để hiển thị thanh tiến trình
        # convert_to_numpy=True để đảm bảo kết quả là mảng NumPy
        embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
        print("Đã tạo embeddings thành công.")

        # Đảm bảo kiểu dữ liệu là float32 như hàm gốc
        embeddings_float32 = embeddings.astype(np.float32)

        print(f"--- Quá trình xử lý hoàn tất ---")
        # Trả về danh sách các chunk hợp lệ và mảng embeddings tương ứng
        return valid_chunks, embeddings_float32

    except Exception as e:
        print(f"Lỗi trong quá trình tạo embeddings: {e}")
        # Trả về danh sách chunk hợp lệ đã lọc nhưng không có embedding
        return valid_chunks, None

# --- Retrieval ---
def retrieve_relevant_chunks(query_text, embedding_model, vector_db, k=5):
    """Embed query và tìm kiếm trong vector_db."""
    if embedding_model is None: logging.error("Embedding model chưa load."); return [], []
    if vector_db is None or vector_db.index is None: logging.error("Vector DB chưa sẵn sàng."); return [], []
    # logging.info(f"Embedding query: \"{query_text[:50]}...\"") # Giảm log
    try:
        query_embedding = embedding_model.encode(query_text, convert_to_numpy=True).astype('float32')
        distances, indices = vector_db.search(query_embedding, k=k)
        # logging.info(f"Vector search trả về {len(indices)} indices.") # Giảm log
        return distances, indices
    except Exception as e:
        logging.error(f"Lỗi khi embedding hoặc tìm kiếm vector: {e}")
        return [], []

# --- Re-ranking ---
def rerank_documents(query_text, documents_with_indices, reranking_model):
    """Xếp hạng lại documents dựa trên query bằng CrossEncoder."""
    original_docs = [item['doc'] for item in documents_with_indices]
    original_indices = [item['index'] for item in documents_with_indices]
    if not original_docs: return []
    if reranking_model is None:
        logging.warning("Reranker model chưa load, trả về không xếp hạng.")
        return [{"doc": doc, "score": None, "original_index": idx} for doc, idx in zip(original_docs, original_indices)]

    # logging.info(f"Re-ranking {len(original_docs)} docs cho query: \"{query_text[:50]}...\"") # Giảm log
    sentence_pairs = [[query_text, doc.get('text', '')] for doc in original_docs]
    try:
        relevance_scores = reranking_model.predict(sentence_pairs, show_progress_bar=False) # Tắt progress bar
        scored_documents = [{'doc': doc, 'score': relevance_scores[i], 'original_index': original_indices[i]}
                            for i, doc in enumerate(original_docs)]
        scored_documents.sort(key=lambda x: x['score'], reverse=True)
        # logging.info("Re-ranking hoàn tất.") # Giảm log
        return scored_documents
    except Exception as e:
        logging.error(f"Lỗi trong quá trình re-ranking: {e}")
        # Trả về danh sách gốc nếu lỗi
        return [{"doc": doc, "score": None, "original_index": idx} for doc, idx in zip(original_docs, original_indices)]


# --- Generation ---
def generate_answer_with_gemini(query_text, relevant_documents, gemini_model):
    """Tạo câu trả lời cuối cùng bằng Gemini dựa trên context."""
    if gemini_model is None: return "Lỗi: Mô hình Gemini chưa sẵn sàng."

    context_str_parts = []
    unique_urls = set()
    # ... (Copy logic chuẩn bị context_text và unique_urls như trong code gốc) ...
    if not relevant_documents:
        context_str_parts.append("Không có thông tin ngữ cảnh nào được cung cấp.")
    else:
        for i, item in enumerate(relevant_documents):
            doc = item.get('doc'); text = doc.get('text', '').strip(); metadata = doc.get('metadata', {})
            if not doc or not text: continue
            source = metadata.get('source', 'N/A'); url = metadata.get('url'); context_meta = metadata.get('context', {})
            chuong = context_meta.get('chuong'); dieu = context_meta.get('dieu'); khoan = context_meta.get('khoan'); diem = context_meta.get('diem'); muc = context_meta.get('muc')
            source_parts = [f"Văn bản: {source}"]
            if chuong: source_parts.append(f"Chương {chuong}")
            if muc: source_parts.append(f"Mục {muc}")
            if dieu: source_parts.append(f"Điều {dieu}")
            if khoan: source_parts.append(f"Khoản {khoan}")
            if diem: source_parts.append(f"Điểm {diem}")
            source_ref = ", ".join(source_parts)
            context_str_parts.append(f"--- Nguồn tham khảo {i+1}: [{source_ref}] ---\n{text}\n---")
            if url: unique_urls.add(url)
        if not context_str_parts: context_str_parts.append("Không có thông tin ngữ cảnh nào được cung cấp.")
    context_text = "\n".join(context_str_parts)
    urls_string = "\n".join(f"- {url}" for url in unique_urls)

    # --- Prompt chi tiết ---
    prompt = f"""Bạn là một trợ lý chuyên gia về luật giao thông đường bộ Việt Nam... (Giữ nguyên prompt chi tiết như code gốc, bao gồm các yêu cầu 1-10)...

**Ngữ cảnh được cung cấp (Mỗi đoạn có kèm nguồn tham khảo):**
{context_text}

**Câu hỏi của người dùng:** {query_text}

**Yêu cầu trả lời:**
... (Giữ nguyên các yêu cầu 1-10) ...
**Trả lời:**
"""
    # --- Gọi API và xử lý kết quả ---
    final_answer = "Lỗi khi tạo câu trả lời từ Gemini."
    try:
        response = gemini_model.generate_content(prompt)
        # Kiểm tra xem có bị block không
        if response.parts:
             final_answer = response.text
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             final_answer = f"Không thể tạo câu trả lời do bị chặn bởi bộ lọc an toàn: {response.prompt_feedback.block_reason}"
             logging.warning(f"Gemini response blocked: {response.prompt_feedback}")
        else:
             # Trường hợp không có text và không có block reason rõ ràng
             logging.warning(f"Gemini response không có text và không rõ lý do bị chặn: {response}")
             final_answer = "Không nhận được phản hồi hợp lệ từ mô hình ngôn ngữ."

    except Exception as e:
        logging.error(f"Lỗi khi gọi API Gemini (generate_answer): {e}")
        final_answer = f"Đã xảy ra lỗi khi kết nối với mô hình ngôn ngữ: {e}"

    # Thêm URL nếu phù hợp
    if unique_urls and "không tìm thấy nội dung phù hợp" not in final_answer and "bị chặn bởi bộ lọc an toàn" not in final_answer and "Lỗi khi" not in final_answer:
        final_answer += "\n\n**Bạn có thể tham khảo thêm tại:**\n" + urls_string

    return final_answer