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
    summarizing_query = original_query 

    response = gemini_model.generate_content(prompt)

    if hasattr(response, 'text') and response.text:
        generated_text = response.text
        json_match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", generated_text, re.DOTALL)
        parsed_data = None
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            if json_str:
                json_str = json_str.strip()
                parsing_successful = False
                try: 
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

                    if parsed_summary and isinstance(parsed_summary, str):
                        summarizing_query = parsed_summary

    # Ensure summarizing_query is never empty
    if not summarizing_query:
        summarizing_query = original_query

    return all_queries, summarizing_query

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