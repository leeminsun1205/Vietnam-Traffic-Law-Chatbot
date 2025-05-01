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
    print(f"\nGenerating {num_variations} variations AND summary for query: \"{original_query}\"")

    prompt = f"""Bạn là một chuyên gia về luật giao thông đường bộ Việt Nam. Nhiệm vụ của bạn là:
1.  Diễn đạt lại câu hỏi gốc sau đây theo {num_variations} cách khác nhau, giữ nguyên ý nghĩa cốt lõi, đa dạng về từ ngữ và cấu trúc, ưu tiên từ khóa luật giao thông và loại phương tiện liên quan (ô tô, xe máy, xe tải,...), sử dụng từ đồng nghĩa (phạt, xử phạt, mức phạt,...).
2.  Tạo ra MỘT câu hỏi tổng hợp DUY NHẤT, đại diện cho ý định tìm kiếm chung của câu hỏi gốc và các biến thể, giữ lại từ khóa quan trọng, diễn đạt tự nhiên và bao quát.
3.  Ưu tiên 1 biến thể có chứa cụm từ "không tuân thủ" nếu nó là 1 câu hỏi về lỗi vi phạm.

Câu hỏi gốc: "{original_query}"

Hãy trả lời THEO ĐÚNG ĐỊNH DẠNG JSON sau, không thêm bất kỳ lời giải thích hay giới thiệu nào khác:
{{
  "variations": [
    "[Biến thể 1]",
    "[Biến thể 2]",
    "[Biến thể {num_variations}]"
  ],
  "summarizing_query": "[Câu hỏi tổng hợp duy nhất]"
}}
"""

    all_queries = [original_query]
    summarizing_query = original_query # Default fallback

    if gemini_model is None:
        print("Warning: Gemini model not loaded. Cannot generate query variations.")
        return all_queries, summarizing_query

    response = gemini_model.generate_content(prompt)

    if hasattr(response, 'text') and response.text:
        generated_text = response.text
        # Attempt to extract JSON block
        json_match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", generated_text, re.DOTALL)
        parsed_data = None
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            if json_str:
                # Basic cleanup before parsing
                json_str = json_str.strip()
                # Attempt parsing
                parsing_successful = False
                try: # Use try-except block specifically for JSON parsing robustness
                    parsed_data = json.loads(json_str)
                    parsing_successful = True
                except json.JSONDecodeError as e:
                     print(f"Warning: Could not parse JSON response from LLM: {e}")
                     print(f"Raw response part for JSON: {json_str}")


                if parsing_successful and isinstance(parsed_data, dict):
                    variations = parsed_data.get('variations', [])
                    parsed_summary = parsed_data.get('summarizing_query', '')

                    if isinstance(variations, list) and variations:
                        all_queries.extend(variations[:num_variations])
                        all_queries = list(set(all_queries)) # Remove duplicates
                    else:
                         print("Warning: 'variations' key not found or empty in parsed JSON.")

                    if parsed_summary and isinstance(parsed_summary, str):
                        summarizing_query = parsed_summary
                    else:
                        print("Warning: 'summarizing_query' key not found or empty/invalid in parsed JSON. Falling back.")
                # else if parsing successful but not dict: handle this case if needed
                # else: handled by parsing_successful check

            else:
                 print("Warning: Extracted JSON string is empty.")
        else:
            print("Warning: Could not find JSON block in LLM response.")
    else:
        print(f"Warning: Gemini response did not contain text. Using original query only. Response: {response}")

    # Ensure summarizing_query is never empty
    if not summarizing_query:
        summarizing_query = original_query

    print(f"Final list of queries ({len(all_queries)}): {[q[:80] + '...' if len(q) > 80 else q for q in all_queries]}")
    print(f"Final summarizing query for rerank: \"{summarizing_query}\"")

    return all_queries, summarizing_query

# --- Data Processing ---
def embed_legal_chunks(file_paths, model):
    """Đọc các file JSON, trích xuất text và tạo embeddings."""
    all_chunks_read = []
    logging.info(f"Bắt đầu xử lý dữ liệu từ {len(file_paths)} file JSON...")
    for file_path in file_paths:

        if os.path.exists(file_path):
            logging.info(f"Đang đọc: {os.path.basename(file_path)}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list): all_chunks_read.extend(data)
                    else: logging.warning(f"File '{file_path}' không phải list JSON.")
            except json.JSONDecodeError: logging.error(f"File '{file_path}' không phải JSON hợp lệ.")
            except Exception as e: logging.error(f"Lỗi đọc file '{file_path}': {e}")
        else: logging.warning(f"File không tồn tại: '{file_path}'")

    if not all_chunks_read: logging.error("Không đọc được chunk nào."); return [], None
    logging.info(f"Đọc thành công tổng cộng: {len(all_chunks_read)} chunks.")

    texts_to_embed, valid_chunks = [], []
    for chunk in all_chunks_read:
        text = chunk.get('text')
        if text and isinstance(text, str) and text.strip():
            texts_to_embed.append(text)
            valid_chunks.append(chunk)

    if not texts_to_embed: logging.error("Không tìm thấy text hợp lệ để embed."); return [], None
    logging.info(f"Đang tạo embeddings cho {len(texts_to_embed)} chunks...")
    try:
        # Tắt progress bar vì có thể không hiển thị tốt ở mọi môi trường
        embeddings = model.encode(texts_to_embed, show_progress_bar=False, convert_to_numpy=True)
        logging.info("Tạo embeddings thành công.")
        return valid_chunks, embeddings.astype('float32')
    except Exception as e:
        logging.error(f"Lỗi trong quá trình tạo embeddings: {e}")
        return [], None

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
    prompt = f"""Bạn là một trợ lý giao thông đường bộ Việt Nam hữu ích. Hãy sử dụng các đoạn văn bản sau để trả lời câu hỏi của người dùng. Nếu các đoạn văn bản không chứa thông tin liên quan đến câu hỏi, chỉ cần nói rằng bạn không tìm thấy thông tin phù hợp dựa trên ngữ cảnh được cung cấp. Đừng cố gắng tạo ra câu trả lời từ kiến thức của riêng bạn. 
    Tuy nhiên bạn cũng phải có khả năng liên tưởng những trường hợp đồng nghĩa ví dụ Rượu, bia, Đèn đỏ thì liên quan tới đèn tín hiệu, xe mô tô thì là xe 2 bánh, ... chứ không phải khớp hoàn toàn thì mới được

    Đoạn văn bản: (Chỉ rõ Thông tin: Chương, Mục, Điều, Khoản, Điểm cho mỗi thông tin (nếu có))
    ---
    {context_text}
    ---

    Câu hỏi: {query_text}

    Trả lời:
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