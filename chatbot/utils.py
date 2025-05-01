# utils.py
import json
import os
import re
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from kaggle_secrets import UserSecretsClient 
import streamlit as st
import config


# --- Model Loading Functions ---
@st.cache_resource
def load_embedding_model(model_name):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Lỗi tải Embedding Model ({model_name}): {e}")
        return None

@st.cache_resource
def load_reranker_model(model_name):
    try:
        model = CrossEncoder(model_name)
        return model
    except Exception as e:
        st.error(f"Lỗi tải Reranker Model ({model_name}): {e}")
        return None

@st.cache_resource
def load_gemini_model(model_name):
    user_secrets = UserSecretsClient()
    google_api_key = user_secrets.get_secret("GOOGLE_API_KEY")

    if google_api_key:
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel(model_name)
        return model
    else:
        st.error("Không tìm thấy GOOGLE_API_KEY.")
        return None

# --- Query Augmentation ---
def generate_query_variations(original_query, gemini_model, num_variations=config.NUM_QUERY_VARIATIONS):
    """
    Phân loại mức độ liên quan, tạo câu trả lời trực tiếp nếu không liên quan,
    hoặc tạo các biến thể nếu liên quan.

    Returns:
        tuple: (relevance_status, direct_answer_if_invalid, all_queries, summarizing_query)
               relevance_status: 'valid' hoặc 'invalid'
               direct_answer_if_invalid: Câu trả lời được tạo sẵn nếu invalid, ngược lại là chuỗi rỗng.
               all_queries: List các câu hỏi nếu valid, [original_query] nếu invalid.
               summarizing_query: Câu hỏi tổng hợp nếu valid, original_query nếu invalid.
    """

    # --- Cập nhật Prompt ---
    prompt = f"""Bạn là một trợ lý AI chuyên về Luật Giao thông Đường bộ Việt Nam. Nhiệm vụ của bạn là xử lý câu hỏi sau: "{original_query}"

**Bước 1: Phân loại Mức độ Liên quan**
Xác định câu hỏi có liên quan trực tiếp đến Luật GTĐB Việt Nam không (quy tắc, biển báo, phạt, giấy phép, đăng ký xe,...).

**Bước 2: Tạo Phản hồi JSON**
Dựa vào Bước 1, tạo phản hồi JSON **CHÍNH XÁC** theo định dạng sau:

* **Nếu câu hỏi KHÔNG liên quan:** Hãy tạo một câu trả lời **ngắn gọn, trực tiếp** phù hợp với câu hỏi đó (ví dụ: nếu hỏi "Bạn là ai?", trả lời "Tôi là chatbot về Luật GTĐB Việt Nam."). Đặt câu trả lời này vào trường `invalid_answer`. Nếu không thể tạo câu trả lời phù hợp, để trống trường này.
    ```json
    {{
      "relevance": "invalid",
      "invalid_answer": "[Câu trả lời trực tiếp cho câu hỏi không liên quan hoặc chuỗi rỗng]",
      "variations": [],
      "summarizing_query": ""
    }}
    ```
* **Nếu câu hỏi CÓ liên quan:** Hãy thực hiện các yêu cầu sau:
    1.  Tạo {num_variations} biến thể câu hỏi (ưu tiên từ khóa luật, phương tiện, từ đồng nghĩa).
    2.  Tạo MỘT câu hỏi tổng hợp bao quát.
    3.  Ưu tiên biến thể chứa "không tuân thủ" nếu hỏi về lỗi.
    ```json
    {{
      "relevance": "valid",
      "invalid_answer": "",
      "variations": [
        "[Biến thể 1]",
        "[Biến thể 2]",
        "[Biến thể {num_variations}]"
      ],
      "summarizing_query": "[Câu hỏi tổng hợp duy nhất]"
    }}
    ```

**Lưu ý:** Chỉ trả về DUY NHẤT một khối JSON hợp lệ, không thêm giải thích nào khác.
"""

    # --- Xử lý kết quả từ Gemini ---
    relevance_status = 'valid' # Mặc định
    direct_answer_if_invalid = ""
    all_queries = [original_query]
    summarizing_query = original_query

    try:
        response = gemini_model.generate_content(prompt)

        if hasattr(response, 'text') and response.text:
            generated_text = response.text
            json_match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", generated_text, re.DOTALL)
            parsed_data = None
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                if json_str:
                    json_str = json_str.strip()
                    try:
                        parsed_data = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON response from LLM: {e}")
                        print(f"Raw response part for JSON: {json_str}")

            if isinstance(parsed_data, dict):
                relevance_status = parsed_data.get('relevance', 'valid')

                if relevance_status == 'invalid':
                    direct_answer_if_invalid = parsed_data.get('invalid_answer', "") # Lấy câu trả lời trực tiếp
                    # Giữ nguyên giá trị mặc định cho all_queries, summarizing_query
                else: # relevance_status == 'valid'
                    variations = parsed_data.get('variations', [])
                    parsed_summary = parsed_data.get('summarizing_query', '')

                    if isinstance(variations, list) and variations:
                        valid_variations = [v for v in variations if isinstance(v, str) and v.strip()]
                        all_queries.extend(valid_variations[:num_variations])
                        all_queries = list(set(all_queries))

                    if parsed_summary and isinstance(parsed_summary, str) and parsed_summary.strip():
                        summarizing_query = parsed_summary.strip()
                    else:
                         summarizing_query = original_query # Fallback

        # Đảm bảo summarizing_query không bao giờ rỗng nếu relevance là 'valid'
        if relevance_status == 'valid' and not summarizing_query:
            summarizing_query = original_query

    except Exception as e:
        print(f"Error during Gemini API call or processing: {e}")
        # Giữ nguyên giá trị mặc định nếu có lỗi

    # Đảm bảo trả về đúng thứ tự
    if relevance_status == 'invalid':
        all_queries = [original_query]
        summarizing_query = original_query

    return relevance_status, direct_answer_if_invalid, all_queries, summarizing_query

# --- Data Processing ---
def embed_legal_chunks(file_paths, model):
    """Đọc các file JSON, trích xuất text và tạo embeddings."""
    all_chunks_read = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list): all_chunks_read.extend(data)

    if not all_chunks_read: return [], None

    texts_to_embed, valid_chunks = [], []
    for chunk in all_chunks_read:
        text = chunk.get('text')
        if text and isinstance(text, str) and text.strip():
            texts_to_embed.append(text)
            valid_chunks.append(chunk)

    if not texts_to_embed: return [], None
    try:
        embeddings = model.encode(texts_to_embed, show_progress_bar=False, convert_to_numpy=True)
        return valid_chunks, embeddings.astype('float32')
    except Exception as e:
        return [], None

# --- Re-ranking ---
def rerank_documents(query_text, documents_with_indices, reranking_model):
    """Xếp hạng lại documents dựa trên query bằng CrossEncoder."""
    original_docs = [item['doc'] for item in documents_with_indices]
    original_indices = [item['index'] for item in documents_with_indices]
    if not original_docs: return []
    if reranking_model is None:
        return [{"doc": doc, "score": None, "original_index": idx} for doc, idx in zip(original_docs, original_indices)]

    sentence_pairs = [[query_text, doc.get('text', '')] for doc in original_docs]
    try:
        relevance_scores = reranking_model.predict(sentence_pairs, show_progress_bar=False) 
        scored_documents = [{'doc': doc, 'score': relevance_scores[i], 'original_index': original_indices[i]}
                            for i, doc in enumerate(original_docs)]
        scored_documents.sort(key=lambda x: x['score'], reverse=True)
        return scored_documents
    except Exception as e:
        return [{"doc": doc, "score": None, "original_index": idx} for doc, idx in zip(original_docs, original_indices)]


# --- Generation ---
def generate_answer_with_gemini(query_text, relevant_documents, gemini_model):
    """Tạo câu trả lời cuối cùng bằng Gemini dựa trên context."""

    context_str_parts = []
    unique_urls = set()

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

    prompt = f"""Bạn là trợ lý chuyên về luật giao thông Việt Nam.
    Nhiệm vụ: Trả lời câu hỏi người dùng (`{query_text}`) một cách **NGẮN GỌN** và chính xác, **CHỈ DÙNG** thông tin từ ngữ cảnh pháp lý được cung cấp (`{context_text}`).

    **Ngữ cảnh được cung cấp (Mỗi đoạn có kèm nguồn tham khảo):**
    {context_text}

    **Câu hỏi của người dùng:** {query_text}

    **Yêu cầu trả lời:**
    1.  **Chỉ dùng ngữ cảnh:** Tuyệt đối không suy diễn hay thêm kiến thức ngoài.
    2.  **Tổng hợp và trích dẫn:**
        * Kết hợp thông tin từ nhiều đoạn nếu cần, **diễn đạt lại mạch lạc**, tránh lặp lại nguyên văn dài.
        * Sau mỗi ý hoặc nhóm ý, **nêu rõ nguồn gốc** dùng thông tin trong dấu `[...]`.
        * **Gom nhóm nguồn** hợp lý: Trích dẫn một lần cho cùng một Điều/Khoản/Điểm; trích dẫn Điều chung nếu các Khoản/Điểm khác nhau trong cùng Điều; trích dẫn một lần nếu chỉ dùng một nguồn. Ưu tiên sự súc tích. Ví dụ: `(Theo Điều 5, Khoản 2, Điểm a, Văn bản: 36/2024/QH15)`.
    3.  **Trình bày rõ ràng:** Dùng gạch đầu dòng `-`, số thứ tự `1., 2.`, **in đậm** (`** **`) cho điểm chính/mức phạt/kết luận.
    4.  **Hiểu ngữ nghĩa:** Tìm thông tin liên quan ngay cả khi từ ngữ không khớp hoàn toàn (ví dụ: "nồng độ cồn" vs "rượu bia", "đèn đỏ" vs "tín hiệu giao thông", "xe máy" vs "xe mô tô/gắn máy").
    5.  **Thiếu thông tin:** Nếu ngữ cảnh không có thông tin, trả lời: "**Dựa trên thông tin được cung cấp, tôi không tìm thấy nội dung phù hợp để trả lời câu hỏi này.**"
    6.  **Thông tin liên quan (không trực tiếp):** Nếu không có câu trả lời trực tiếp nhưng tìm thấy thông tin có thể liên quan, hãy nêu rõ điều đó sau khi báo không tìm thấy câu trả lời chính xác (ví dụ: "Tôi không tìm thấy quy định trực tiếp về X, tuy nhiên có thông tin về Y như sau:... (Nguồn:...)").
    7.  *Tham khảo thêm:** Cuối câu trả lời, nếu có URL trong ngữ cảnh, thêm phần "Nguồn:" và liệt kê URL thuộc về thông tin mà bạn sử dụng.

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
        else:
             final_answer = "Không nhận được phản hồi hợp lệ từ mô hình ngôn ngữ."

    except Exception as e:
        final_answer = f"Đã xảy ra lỗi khi kết nối với mô hình ngôn ngữ: {e}"

    if unique_urls and "không tìm thấy nội dung phù hợp" not in final_answer and "bị chặn bởi bộ lọc an toàn" not in final_answer and "Lỗi khi" not in final_answer:
        final_answer += "\n\n**Nguồn:**\n" + urls_string

    return final_answer