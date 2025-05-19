# utils.py
import json
import os
import re
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from kaggle_secrets import UserSecretsClient 
import streamlit as st
import config
import math
import pandas as pd
from datetime import datetime
import base64

# --- Model Loading Functions ---
@st.cache_resource
def load_embedding_model(model_name):
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
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

# @st.cache_resource
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
def generate_query_variations(original_query, gemini_model, chat_history=None, num_variations=config.NUM_QUERY_VARIATIONS):
    history_prefix = ""
    if chat_history:
        history_prefix = "**Lịch sử trò chuyện gần đây (để tham khảo ngữ cảnh):**\n"
        limited_history = chat_history[-(config.MAX_HISTORY_TURNS * 2):] 
        for msg in limited_history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "").strip()
            # Giới hạn độ dài content mỗi tin nhắn nếu cần
            # content = content[:150] + '...' if len(content) > 150 else content
            if role and content:
                history_prefix += f"{role}: {content}\n"
        history_prefix += "---\n" 
    # --- Cập nhật Prompt ---
    prompt = f"""Bạn là một trợ lý AI chuyên về Luật Giao thông Đường bộ Việt Nam. Nhiệm vụ của bạn là xử lý câu hỏi sau: "{history_prefix}".
    Có xem xét lịch sử trò chuyện (nếu được cung cấp ở trên) để hiểu ngữ cảnh. 
    **Câu hỏi HIỆN TẠI:** "{original_query}"

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
        2.  Tạo MỘT câu hỏi tổng hợp bao quát (summarizing_query), câu hỏi này phải đảm bảo chứa tất cả các keyword của câu hỏi gốc (để tránh làm sai lệch quá nhiều).
        3.  Ưu tiên biến thể chứa "không tuân thủ, không chấp hành" nếu hỏi về lỗi, vi phạm.
        4.  Nếu gặp các câu hỏi về "vượt đèn đỏ/đèn vàng" thì tất cả biến thể chuyển thành "không chấp hành hiệu lệnh của đèn tín hiệu".
        5.  Nếu câu hỏi có **liên quan đến phương tiện** mà hỏi 1 cách tổng quát (không chỉ rõ loại xe nào) phải ưu tiên 1 câu có "xe mô tô, xe gắn máy, các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy" và 1 câu có "xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô".
        6.  Cần phân biệt rõ xe máy và xe máy chuyên dùng. Nếu câu hỏi chỉ nói là xe máy thì phải hiểu là xe hai bánh, xe gắn máy, ... Xe máy chuyên dùng gồm xe máy thi công, xe máy nông nghiệp, lâm nghiệp và các loại xe đặc chủng khác sử dụng vào mục đích quốc phòng, an ninh có tham gia giao thông đường bộ.
        7.  Nếu câu hỏi dùng từ xe máy phải thay thế bằng từ xe gắn máy, xe hai bánh, ...
        8.  Đối với summarizing_query cũng phải áp dụng tương tự quy tắc 5.
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
    relevance_status = 'valid' 
    direct_answer_if_invalid = ""
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
                try:
                    parsed_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON response from LLM: {e}")
                    print(f"Raw response part for JSON: {json_str}")

        if isinstance(parsed_data, dict):
            relevance_status = parsed_data.get('relevance', 'valid')

            if relevance_status == 'invalid':
                direct_answer_if_invalid = parsed_data.get('invalid_answer', "")
            else: 
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

    if relevance_status == 'valid' and not summarizing_query:
        summarizing_query = original_query

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

# --- Hàm tải tệp mapping URL ---
@st.cache_data 
def load_document_url_mapping(filepath="/kaggle/working/CS431.P22/loader/vanban_url_map.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
        return mapping 
    
def extract_and_normalize_document_key(citation_text):
    citation_text = citation_text.strip() 
    match1 = re.search(r'(\d+)\s*[/_]\s*(\d{4})\s*[/_]\s*([A-ZĐ]+(?:-[A-ZĐ]+)*)', citation_text, re.IGNORECASE)
    if match1:
        number = match1.group(1)
        year = match1.group(2)
        identifier = match1.group(3)
        if identifier == 'qh' or identifier == 'QH':
            identifier = 'QH15'
        key = f"{number}_{year}_{identifier}".upper()
        return key

    match2 = re.search(r'(\d+)\s*[/_]\s*(\d{4})\s*[/_]\s*([A-Z]+\d+)', citation_text, re.IGNORECASE)
    if match2:
        number = match2.group(1)
        year = match2.group(2)
        identifier = match2.group(3)
        if identifier == 'qh' or identifier == 'QH':
            identifier = 'QH15'
        key = f"{number}_{year}_{identifier}".upper()
        return key
    
    match3 = re.search(r'(\d+)\s*[_/]\s*([A-ZĐ]+(?:-[A-ZĐ]+)*)', citation_text, re.IGNORECASE)
    if match3:
        number = match3.group(1)
        identifier = match3.group(2)
        key = f"{number}_{identifier}".upper()
        return key
    return None

# --- Generation ---
# def generate_answer_with_gemini(query_text, relevant_documents, gemini_model, mode='Đầy đủ', chat_history=None):
#     """Tạo câu trả lời cuối cùng bằng Gemini dựa trên context."""

#     url_mapping_dict = load_document_url_mapping()
#     context_for_prompt = "..."
#     history_prefix = "..."
#     context_str_parts = []
#     source_details_for_prompt = []

#     if not relevant_documents:
#         context_str_parts.append("Không có thông tin ngữ cảnh nào được cung cấp.")
#     else:
#         for i, item in enumerate(relevant_documents):
#             doc = item.get('doc'); 
#             text = doc.get('text', '').strip(); 
#             metadata = doc.get('metadata', {})
#             if not doc or not text: continue

#             source_name = metadata.get('source', 'N/A'); 
#             url = metadata.get('url'); 
#             context_meta = metadata.get('context', {})
#             chuong = context_meta.get('chuong')
#             muc = context_meta.get('muc')
#             dieu = context_meta.get('dieu')
#             khoan = context_meta.get('khoan')
#             diem = context_meta.get('diem')
            
#             source_parts = [f"Văn bản: {source_name}"]
#             if chuong: source_parts.append(f"Chương {chuong}")
#             if muc: source_parts.append(f"Mục {muc}")
#             if dieu: source_parts.append(f"Điều {dieu}")
#             if khoan: source_parts.append(f"Khoản {khoan}")
#             if diem: source_parts.append(f"Điểm {diem}")
#             source_ref_full = ", ".join(source_parts)
#             source_details_for_prompt.append(f"Nguồn {i+1}: [{source_ref_full}]\nNội dung: {text}\n---")

#         if not source_details_for_prompt:
#              context_for_prompt = "Không có thông tin ngữ cảnh nào được cung cấp."
#         else:
#              context_for_prompt = "\n".join(source_details_for_prompt) 

#     # --- Xây dựng chuỗi lịch sử chat gần đây ---
#     if chat_history: # chat_history là list các dict {"role": "user/assistant", "content": ...}
#         history_prefix = "**Lịch sử trò chuyện gần đây:**\n"
#         for msg in chat_history:
#             # Đảm bảo role và content tồn tại và là string
#             role = msg.get("role", "unknown").capitalize()
#             content = msg.get("content", "").strip()
#             if role and content:
#                  history_prefix += f"{role}: {content}\n"
#         history_prefix += "---\n"

#     full_prompt_template = f"""Bạn là trợ lý chuyên về luật giao thông Việt Nam.
#     {history_prefix}
#     Nhiệm vụ: Dựa vào Lịch sử trò chuyện gần đây (nếu có) và Ngữ cảnh được cung cấp, trả lời câu hỏi HIỆN TẠI của người dùng (`{query_text}`) một cách **CHI TIẾT** và chính xác. **CHỈ DÙNG** thông tin từ ngữ cảnh pháp lý được cung cấp để trả lời các câu hỏi về luật. Đối với các câu hỏi khác, có thể dựa vào lịch sử trò chuyện.

#     **Ngữ cảnh được cung cấp (Dùng để trả lời câu hỏi về luật):**
#     {context_for_prompt}

#     **Câu hỏi HIỆN TẠI của người dùng:** {query_text}

#     **Yêu cầu trả lời:**
#     1.  **Chỉ dùng ngữ cảnh:** Tuyệt đối không suy diễn hay thêm kiến thức ngoài. 
#     2.  * **Gom nhóm nguồn** hợp lý: Trích dẫn một lần cho cùng một Văn Bản/Chương/Mục/Điều/Khoản/Điểm; trích dẫn Điều chung nếu các Khoản/Điểm khác nhau trong cùng Điều; trích dẫn một lần nếu chỉ dùng một nguồn. Ưu tiên sự súc tích. Phải nêu đầy đủ theo cấu trúc sau: `(Theo Điều 5, Khoản 2, Điểm a, Văn bản: 36/2024/QH15)`.
#         **Tổng hợp và query_text trích dẫn:**
#         * Kết hợp thông tin từ nhiều đoạn nếu cần, đảm bảo không **bỏ sót** hoặc **dư thừa** thông tin, **diễn đạt lại mạch lạc**, tránh lặp lại nguyên văn dài.
#         * Sau mỗi ý hoặc nhóm ý, **nêu rõ nguồn gốc** dùng thông tin trong dấu `(...)`.
#     3.  **Trình bày rõ ràng:** Dùng gạch đầu dòng `-`, số thứ tự `1., 2.`, **in đậm** (`** **`) cho nội dung chính/mức phạt/kết luận.
#     4.  **Hiểu ngữ nghĩa:** Tìm thông tin liên quan ngay cả khi từ ngữ không khớp hoàn toàn (ví dụ: "rượu, bia" sẽ liên quan tới "nồng độ cồn"; "đèn đỏ", "đèn vàng" là "đèn tín hiệu", "xe máy" vs "xe mô tô/gắn máy/xe hai bánh", ...và từ ngữ giao thông khác).
#     5.  **Thiếu thông tin:** Nếu ngữ cảnh không có thông tin, trả lời: "**Dựa trên thông tin được cung cấp, tôi không tìm thấy nội dung phù hợp để trả lời câu hỏi này.**"
#     6.  **Thông tin liên quan (nếu có):** Nếu không có câu trả lời trực tiếp nhưng có thông tin liên quan (phải liên quan đến ý nghĩa **chuẩn xác** của câu hỏi), có thể đề cập sau khi báo không tìm thấy câu trả lời chính xác. Nhưng chỉ đề cập những câu gần ý nghĩa nhất.
#     7.  Thứ tự ưu tiên khi câu hỏi mang tính so sánh là:
#         - Điểm thứ a trong Điều/Khoản (ưu tiên cao nhất)
#         - Điểm thứ b trong Điều/Khoản
#         - Điểm thứ c trong Điều/Khoản
#         - Điểm thứ d trong Điều/Khoản
#         - Điểm thứ đ trong Điều/Khoản
#         - ... 
#     8.  Cần phân biệt rõ xe máy và xe máy chuyên dùng (Xe máy chuyên dùng gồm xe máy thi công, xe máy nông nghiệp, lâm nghiệp và các loại xe đặc chủng khác sử dụng vào mục đích quốc phòng, an ninh có tham gia giao thông đường bộ). Nếu câu hỏi chỉ nói là xe máy thì câu trả lời phải hiểu là xe máy thông thường, xe hai bánh, xe gắn máy, ...
#     9.  Phải thể hiện được tính logic từ câu hỏi sang câu trả lời ví dụ câu hỏi yêu cầu cao như so sánh, tính tổng, tình huống, ...
#     **Trả lời:**
#     """

#     # Prompt Ngắn Gọn 
#     brief_prompt_template = f"""Bạn là trợ lý luật giao thông Việt Nam.
#     {history_prefix}
#     Nhiệm vụ: Dựa vào Lịch sử trò chuyện (nếu có) và Ngữ cảnh, trả lời câu hỏi HIỆN TẠI (`{query_text}`) **CỰC KỲ NGẮN GỌN**, đi thẳng vào trọng tâm. **CHỈ DÙNG** ngữ cảnh để trả lời về luật.

#     **Ngữ cảnh (Dùng để trả lời câu hỏi về luật):**
#     {context_for_prompt}

#     **Câu hỏi HIỆN TẠI:** {query_text}

#     **Yêu cầu trả lời NGẮN GỌN:**
#     1.  **Chỉ dùng ngữ cảnh.** 
#     2.  **Súc tích:** Trả lời trực tiếp, dùng gạch đầu dòng (-) nếu cần. **In đậm** điểm chính/mức phạt.
#     3.  **Trích dẫn tối thiểu:** Chỉ nêu nguồn chính yếu nếu thực sự cần. Phải nêu đầy đủ theo cấu trúc sau `(Theo Đ.5, K.2, Điểm a, Văn bản: 36/2024/QH15)`.
#     4.  **Hiểu ngữ nghĩa:** Tìm thông tin liên quan ngay cả khi từ ngữ không khớp hoàn toàn (ví dụ: "rượu, bia" sẽ liên quan tới "nồng độ cồn"; "đèn đỏ", "đèn vàng" là "đèn tín hiệu", "xe máy" vs "xe mô tô/gắn máy/xe hai bánh", ...và từ ngữ giao thông khác).
#     5.  **Thiếu thông tin:** Nếu không có, nói: "**Không tìm thấy thông tin phù hợp.**"
#     6.  Thứ tự ưu tiên khi câu hỏi mang tính so sánh là:
#         - Điểm thứ a trong Điều/Khoản (ưu tiên cao nhất)
#         - Điểm thứ b trong Điều/Khoản
#         - Điểm thứ c trong Điều/Khoản
#         - Điểm thứ d trong Điều/Khoản
#         - Điểm thứ đ trong Điều/Khoản
#         - ...
#     7.  Cần phân biệt rõ xe máy và xe máy chuyên dùng (Xe máy chuyên dùng gồm xe máy thi công, xe máy nông nghiệp, lâm nghiệp và các loại xe đặc chủng khác sử dụng vào mục đích quốc phòng, an ninh có tham gia giao thông đường bộ). Nếu câu hỏi chỉ nói là xe máy thì câu trả lời phải hiểu là xe máy thông thường, xe hai bánh, xe gắn máy, ...
#     8.  Phải thể hiện được tính logic từ câu hỏi sang câu trả lời ví dụ câu hỏi yêu cầu cao như so sánh, tính tổng, tình huống, ...
#     **Trả lời NGẮN GỌN:**
#     """

#     # --- Chọn Prompt dựa trên chế độ ---
#     if mode == 'Ngắn gọn':
#         prompt = brief_prompt_template
#     else: 
#         prompt = full_prompt_template

#     # --- Gọi API và xử lý kết quả ---
#     final_answer_display = "Lỗi khi tạo câu trả lời từ Gemini."
#     try:
#         if not gemini_model: raise ValueError("...")
#         response = gemini_model.generate_content(prompt)
#         if response.parts:
#             final_answer_display = response.text
#         else: final_answer_display = "..."
#     except Exception as e:
#         final_answer_display = f"Lỗi: {e}"

#     # --- Bước 3 & 4: Phân tích trích dẫn và Tra cứu URL từ mapping ---
#     found_urls = set()
    
#     citations_found = re.findall(r'\((.*?)\)', final_answer_display)
#     for citation in citations_found:
#         # Trích xuất và chuẩn hóa khóa
#         doc_key = extract_and_normalize_document_key(citation)
#         if doc_key:
#             url = url_mapping_dict.get(doc_key) 
#             if url:
#                 found_urls.add(url)

#     # --- Nối chuỗi URL vào câu trả lời (nếu tìm thấy) ---
#     if found_urls:
#         sorted_urls = sorted(list(found_urls))
#         urls_string = "\n".join(f"- [{url}]({url})" for url in sorted_urls)
#         final_answer_display += f"\n\n**Nguồn:**\n{urls_string}"

#     return final_answer_display
# --- Generation (Phần chính cần sửa) ---
def generate_answer_with_gemini(query_text, relevant_documents, gemini_model, mode='Đầy đủ', chat_history=None):
    """Tạo câu trả lời cuối cùng bằng Gemini, xử lý hiển thị biển báo."""

    url_mapping_dict = load_document_url_mapping() # Tải mapping URL
    context_str_parts = []
    source_details_for_prompt = [] # Đây sẽ là list các chuỗi, mỗi chuỗi mô tả một nguồn

    if not relevant_documents:
        context_for_prompt = "Không có thông tin ngữ cảnh nào được cung cấp."
    else:
        for i, item in enumerate(relevant_documents): # relevant_documents là list các dict {'doc': ..., 'score': ..., 'original_index': ...}
            doc_content = item.get('doc') # doc_content là dict {'id':..., 'text':..., 'metadata':...}
            if not isinstance(doc_content, dict):
                # st.warning(f"Bỏ qua item không hợp lệ trong relevant_documents ở vị trí {i}: item không phải dict hoặc thiếu 'doc'.")
                continue

            text = doc_content.get('text', '').strip()
            metadata = doc_content.get('metadata', {})
            if not text: # Bỏ qua nếu không có nội dung text
                continue

            source_name = metadata.get('source', 'N/A')
            context_meta = metadata.get('context', {}) # metadata['context'] là một dict
            chuong = context_meta.get('chuong')
            muc = context_meta.get('muc')
            dieu = context_meta.get('dieu')
            khoan = context_meta.get('khoan')
            diem = context_meta.get('diem')

            source_parts = [f"Văn bản: {source_name}"]
            if chuong: source_parts.append(f"Chương {chuong}")
            if muc: source_parts.append(f"Mục {muc}")
            if dieu: source_parts.append(f"Điều {dieu}")
            if khoan: source_parts.append(f"Khoản {khoan}")
            if diem: source_parts.append(f"Điểm {diem}")
            source_ref_full = ", ".join(source_parts)

            # --- THAY ĐỔI: Thêm thông tin về biển báo vào prompt cho LLM ---
            traffic_sign_filename = metadata.get('traffic_sign')
            traffic_sign_info_for_llm = ""
            if traffic_sign_filename:
                # Thông báo cho LLM biết nguồn này có liên quan đến biển báo
                traffic_sign_info_for_llm = (
                    f" (LƯU Ý QUAN TRỌNG: Nội dung này có liên quan đến biển báo '{traffic_sign_filename}'. "
                    f"Nếu sử dụng thông tin từ đây để trả lời, hãy nhớ dùng placeholder chỉ mục biển báo.)"
                )
            # -------------------------------------------------------------
            source_details_for_prompt.append(f"Nguồn {i+1}: [{source_ref_full}]{traffic_sign_info_for_llm}\nNội dung: {text}\n---")

        if not source_details_for_prompt:
             context_for_prompt = "Không có thông tin ngữ cảnh nào được cung cấp để xây dựng prompt."
        else:
             context_for_prompt = "\n".join(source_details_for_prompt)

    # --- Xây dựng chuỗi lịch sử chat gần đây ---
    history_prefix = ""
    if chat_history:
        history_prefix = "**Lịch sử trò chuyện gần đây (dùng để tham khảo ngữ cảnh):**\n"
        # Giới hạn số lượt chat trong history để không làm prompt quá dài
        limited_history = chat_history[-(config.MAX_HISTORY_TURNS * 2):] # Lấy tối đa MAX_HISTORY_TURNS cặp user-assistant
        for msg in limited_history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "").strip()
            # Giới hạn độ dài từng tin nhắn nếu cần thiết
            # content = content[:200] + '...' if len(content) > 200 else content
            if role and content:
                 history_prefix += f"{role}: {content}\n"
        history_prefix += "---\n"

    # --- THAY ĐỔI: Cập nhật Prompt với yêu cầu placeholder cho biển báo ---
    placeholder_instruction = (
        "10. **QUAN TRỌNG - HIỂN THỊ BIỂN BÁO**: Nếu bạn sử dụng thông tin từ một 'Nguồn' (được đánh số thứ tự 1, 2, 3,... trong phần 'Ngữ cảnh được cung cấp') "
        "VÀ nguồn đó có ghi chú '(LƯU Ý QUAN TRỌNG: Nội dung này có liên quan đến biển báo ...)', "
        "thì NGAY SAU KHI bạn trình bày xong phần nội dung văn bản lấy từ nguồn đó (ví dụ: sau một gạch đầu dòng, hoặc cuối một đoạn giải thích), "
        "bạn PHẢI đặt một placeholder đặc biệt có dạng: `[DISPLAY_TRAFFIC_SIGN_INDEX_{index_nguồn}]`. "
        "Trong đó, `{index_nguồn}` CHÍNH LÀ số thứ tự của 'Nguồn' đó (ví dụ: 1, 2, 3,...). "
        "Ví dụ: Nếu 'Nguồn 3' trong ngữ cảnh có thông tin về biển báo và bạn dùng nó để trả lời, thì bạn phải viết: "
        "`...nội dung bạn trích từ Nguồn 3... [DISPLAY_TRAFFIC_SIGN_INDEX_3]`."
        "Đảm bảo placeholder này nằm tách biệt và dễ dàng được tìm thấy."
    )

    common_requirements = f"""
    1.  **Chỉ dùng ngữ cảnh:** Tuyệt đối không suy diễn hay thêm kiến thức ngoài luồng được cung cấp.
    2.  **Gom nhóm nguồn và trích dẫn:**
        * Khi trích dẫn, hãy tham chiếu đến cấu trúc văn bản (Điều, Khoản, Điểm, Chương, Mục, tên Văn bản) một cách rõ ràng nhất có thể dựa trên thông tin `[{source_ref_full}]` đã cung cấp cho mỗi nguồn. Ví dụ: `(Theo Điều X, Khoản Y, Điểm z, Văn bản ABC)`.
        * Kết hợp thông tin từ nhiều đoạn nếu cần, đảm bảo không **bỏ sót** hoặc **dư thừa** thông tin, **diễn đạt lại mạch lạc**, tránh lặp lại nguyên văn dài dòng.
        * Sau mỗi ý hoặc nhóm ý chính, **nêu rõ nguồn gốc** của thông tin đó bằng cách trích dẫn như trên.
    3.  **Trình bày rõ ràng:** Sử dụng gạch đầu dòng (`-`), đánh số (`1., 2.`), **in đậm** (`** **`) cho các nội dung quan trọng như mức phạt, kết luận, hoặc các điểm chính.
    4.  **Hiểu ngữ nghĩa:** Cố gắng tìm thông tin liên quan ngay cả khi từ ngữ trong câu hỏi không khớp hoàn toàn với từ ngữ trong văn bản (ví dụ: "rượu, bia" có thể liên quan đến "nồng độ cồn"; "đèn đỏ", "đèn vàng" là "đèn tín hiệu giao thông"; "xe máy" có thể là "xe mô tô", "xe gắn máy", "xe hai bánh", v.v.). Hãy dựa vào các quy tắc đã được cung cấp trong prompt tạo query variations nếu có.
    5.  **Trường hợp thiếu thông tin:** Nếu ngữ cảnh được cung cấp không chứa thông tin để trả lời câu hỏi, hãy trả lời một cách trung thực, ví dụ: "**Dựa trên thông tin được cung cấp, tôi không tìm thấy nội dung phù hợp để trả lời câu hỏi này.**"
    6.  **Thông tin liên quan (nếu có và phù hợp):** Nếu không có câu trả lời trực tiếp nhưng có thông tin liên quan chặt chẽ đến ý nghĩa của câu hỏi, bạn có thể đề cập sau khi đã thông báo không tìm thấy câu trả lời chính xác. Tuy nhiên, chỉ nên đề cập những thông tin thực sự gần với câu hỏi.
    7.  **Thứ tự ưu tiên (nếu câu hỏi mang tính so sánh hoặc liệt kê):**
        - Điểm (ví dụ: Điểm a, Điểm b...)
        - Khoản (ví dụ: Khoản 1, Khoản 2...)
        - Điều (ví dụ: Điều 5, Điều 6...)
        - Mục, Chương, Văn bản.
    8.  **Phân biệt loại xe:** Cần phân biệt rõ "xe máy" (thường là xe mô tô, xe gắn máy) và "xe máy chuyên dùng" (xe máy thi công, nông nghiệp, lâm nghiệp, v.v.) nếu câu hỏi hoặc ngữ cảnh đề cập.
    9.  **Logic và suy luận:** Phải thể hiện được tính logic từ câu hỏi đến câu trả lời, đặc biệt với các câu hỏi yêu cầu so sánh, tính toán đơn giản (nếu có thể từ ngữ cảnh), hoặc phân tích tình huống.
    {placeholder_instruction}
    """

    full_prompt_template = f"""Bạn là một trợ lý AI chuyên sâu về Luật Giao thông Đường bộ Việt Nam.
    {history_prefix}
    **Nhiệm vụ:** Dựa vào Lịch sử trò chuyện (nếu có) và thông tin chi tiết trong phần 'Ngữ cảnh được cung cấp' dưới đây, hãy trả lời câu hỏi HIỆN TẠI của người dùng một cách **CHI TIẾT, ĐẦY ĐỦ** và chính xác nhất có thể.

    **Ngữ cảnh được cung cấp (Đây là nguồn thông tin duy nhất bạn được phép sử dụng để trả lời các câu hỏi về luật):**
    {context_for_prompt}
    ---
    **Câu hỏi HIỆN TẠI của người dùng:** {query_text}
    ---
    **Yêu cầu trả lời CHI TIẾT:**
    {common_requirements}
    **Câu trả lời của bạn (chi tiết):**
    """

    brief_prompt_template = f"""Bạn là một trợ lý AI chuyên sâu về Luật Giao thông Đường bộ Việt Nam.
    {history_prefix}
    **Nhiệm vụ:** Dựa vào Lịch sử trò chuyện (nếu có) và thông tin trong 'Ngữ cảnh được cung cấp', trả lời câu hỏi HIỆN TẠI của người dùng (`{query_text}`) một cách **CỰC KỲ NGẮN GỌN**, đi thẳng vào trọng tâm vấn đề.

    **Ngữ cảnh được cung cấp (Nguồn thông tin duy nhất để trả lời về luật):**
    {context_for_prompt}
    ---
    **Câu hỏi HIỆN TẠI của người dùng:** {query_text}
    ---
    **Yêu cầu trả lời NGẮN GỌN:**
    {common_requirements}
    **Câu trả lời của bạn (ngắn gọn):**
    """
    # -------------------------------------------------------------------

    prompt_to_use = brief_prompt_template if mode == 'Ngắn gọn' else full_prompt_template

    final_answer_display = "Xin lỗi, tôi chưa thể tạo câu trả lời vào lúc này." # Default error message
    try:
        if not gemini_model:
            raise ValueError("Mô hình Gemini chưa được tải hoặc không khả dụng.")

        # st.info(f"Prompt gửi đến Gemini:\n```\n{prompt_to_use}\n```") # Để debug prompt

        response = gemini_model.generate_content(prompt_to_use)

        if hasattr(response, 'text') and response.text:
            final_answer_display = response.text.strip()
        elif response.parts:
            final_answer_display = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
            if not final_answer_display: # Nếu sau khi join vẫn rỗng
                final_answer_display = "Không nhận được nội dung văn bản từ Gemini."
        else:
            # Ghi lại lỗi hoặc thông tin response để debug nếu cần
            # st.warning(f"Gemini không trả về 'text' hoặc 'parts' có thể đọc được. Response: {response}")
            final_answer_display = "Phản hồi từ Gemini không có nội dung văn bản."

    except Exception as e:
        # st.error(f"Lỗi nghiêm trọng khi tạo câu trả lời từ Gemini: {e}")
        final_answer_display = f"Đã xảy ra sự cố khi xử lý yêu cầu của bạn. Chi tiết lỗi: {str(e)[:200]}..." # Giới hạn độ dài lỗi hiển thị

    # --- THAY ĐỔI: Xử lý placeholder và nhúng ảnh Base64 ---
    # `relevant_documents` là danh sách các dict {'doc': document_content, 'score': ..., 'original_index': ...}
    # `document_content` là dict {'id': ..., 'text': ..., 'metadata': ...}
    if relevant_documents and isinstance(relevant_documents, list) and final_answer_display:
        processed_answer_parts = []
        last_idx = 0
        # Tìm tất cả các placeholder trong câu trả lời của LLM
        for match in re.finditer(r"\[DISPLAY_TRAFFIC_SIGN_INDEX_(\d+)]", final_answer_display):
            placeholder_full = match.group(0)
            source_idx_one_based = int(match.group(1)) # Index LLM trả về (1-based)
            source_idx_zero_based = source_idx_one_based - 1 # Chuyển về 0-based để truy cập list

            processed_answer_parts.append(final_answer_display[last_idx:match.start()]) # Phần text trước placeholder

            image_markdown_to_insert = "" # Mặc định là chuỗi rỗng nếu có lỗi

            if 0 <= source_idx_zero_based < len(relevant_documents):
                doc_item = relevant_documents[source_idx_zero_based]
                doc_content = doc_item.get('doc')
                if isinstance(doc_content, dict):
                    metadata = doc_content.get('metadata', {})
                    traffic_sign_filename = metadata.get('traffic_sign')

                    if traffic_sign_filename:
                        image_full_path = os.path.join(config.TRAFFIC_SIGN_IMAGES_ROOT_DIR, traffic_sign_filename)
                        if os.path.exists(image_full_path):
                            try:
                                with open(image_full_path, "rb") as img_file:
                                    b64_string = base64.b64encode(img_file.read()).decode()
                                file_ext = os.path.splitext(traffic_sign_filename)[1][1:].lower()
                                if not file_ext: file_ext = "png" # Mặc định

                                image_markdown_to_insert = (
                                    f"\n<div style='text-align: left; margin-top: 8px; margin-bottom: 8px; padding-left: 20px;'>" # Căn lề trái, thụt vào
                                    f"<img src='data:image/{file_ext};base64,{b64_string}' "
                                    f"alt='Biển báo: {traffic_sign_filename}' "
                                    f"style='max-width: 150px; max-height:150px; height: auto; border: 1px solid #ddd; padding: 2px; border-radius: 4px;'/>"
                                    f"<p style='font-size: 0.8em; margin-top: 2px; font-style: italic;'>{traffic_sign_filename}</p>"
                                    f"</div>\n"
                                )
                            except Exception as e_img:
                                # st.warning(f"Lỗi khi mã hóa ảnh {traffic_sign_filename}: {e_img}")
                                image_markdown_to_insert = f"\n<p style='color: red; font-style: italic; padding-left: 20px;'>[Lỗi hiển thị ảnh: {traffic_sign_filename}]</p>"
                        else:
                            # st.warning(f"Không tìm thấy tệp ảnh: {image_full_path} (cho placeholder {placeholder_full})")
                            image_markdown_to_insert = f"\n<p style='color: orange; font-style: italic; padding-left: 20px;'>[Không tìm thấy ảnh: {traffic_sign_filename}]</p>"
                    # else: LLM dùng placeholder nhưng chunk đó lại không có 'traffic_sign' trong metadata (lỗi của LLM hoặc dữ liệu)
                    #     st.warning(f"LLM đã dùng placeholder {placeholder_full} nhưng Nguồn {source_idx_one_based} không có 'traffic_sign' trong metadata.")
            # else: Index từ LLM không hợp lệ
            #     st.warning(f"LLM trả về index không hợp lệ trong placeholder: {placeholder_full}")

            processed_answer_parts.append(image_markdown_to_insert)
            last_idx = match.end()

        processed_answer_parts.append(final_answer_display[last_idx:]) # Phần text còn lại sau placeholder cuối cùng
        final_answer_display = "".join(processed_answer_parts)
    # -------------------------------------------------------

    # --- Xử lý trích dẫn URL (giữ nguyên logic, nhưng áp dụng cho `final_answer_display` đã được xử lý) ---
    found_urls = set()
    # Regex tìm trích dẫn có thể cần điều chỉnh nếu ảnh làm ảnh hưởng (nhưng không nên vì ảnh là HTML)
    citations_found = re.findall(r'\((?:[Tt]heo\s)?([^)]+?)\)', final_answer_display)

    for citation_text in citations_found:
        # Bỏ qua nếu citation_text chứa tag HTML (có thể là phần của ảnh)
        if '<' in citation_text and '>' in citation_text:
            continue
        doc_key = extract_and_normalize_document_key(citation_text)
        if doc_key:
            url = url_mapping_dict.get(doc_key)
            if url:
                found_urls.add(url)

    if found_urls:
        sorted_urls = sorted(list(found_urls))
        # Cải thiện cách hiển thị URL, có thể chỉ lấy tên miền
        urls_display_list = []
        for url in sorted_urls:
            display_name = url
            try:
                # Lấy phần domain + path, bỏ query string/fragment
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                display_name = parsed_url.netloc + parsed_url.path
                if display_name.endswith('/'): display_name = display_name[:-1]
            except ImportError: # dự phòng nếu urllib.parse không có sẵn
                pass
            urls_display_list.append(f"- [{display_name}]({url})")

        urls_string = "\n".join(urls_display_list)
        final_answer_display += f"\n\n**Nguồn tham khảo (Văn bản gốc):**\n{urls_string}"

    return final_answer_display.strip()

# --- Các hàm tính toán metrics ---

def precision_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0: return 0.0
    retrieved_at_k = retrieved_ids[:k]; relevant_set = set(relevant_ids)
    if not relevant_set: return 0.0
    intersect = set(retrieved_at_k) & relevant_set
    return len(intersect) / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    if not relevant_set: return 1.0
    retrieved_at_k = retrieved_ids[:k]
    intersect = set(retrieved_at_k) & relevant_set
    return len(intersect) / len(relevant_set)

def f1_at_k(retrieved_ids, relevant_ids, k):
    prec = precision_at_k(retrieved_ids, relevant_ids, k); rec = recall_at_k(retrieved_ids, relevant_ids, k)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def mrr_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    if not relevant_set: return 0.0
    retrieved_at_k = retrieved_ids[:k]
    for rank, doc_id in enumerate(retrieved_at_k, 1):
        if doc_id in relevant_set: return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    if not relevant_set: return 1.0
    retrieved_at_k = retrieved_ids[:k]; dcg = 0.0; idcg = 0.0
    for i, doc_id in enumerate(retrieved_at_k):
        relevance = 1.0 if doc_id in relevant_set else 0.0
        dcg += relevance / math.log2(i + 2)
    num_relevant_in_total = len(relevant_set)
    for i in range(min(k, num_relevant_in_total)):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

def calculate_average_metrics(df_results: pd.DataFrame):
    evaluated_df = df_results[df_results['status'] == 'evaluated'].copy()
    num_evaluated = len(evaluated_df)
    num_skipped_error = len(df_results) - num_evaluated

    if num_evaluated == 0:
        return None, num_evaluated, num_skipped_error

    avg_metrics = {}
    k_values = [3, 5, 10]
    metric_keys_k = [f'{m}@{k}' for k in k_values for m in ['precision', 'recall', 'f1', 'mrr', 'ndcg']]
    timing_keys = ['processing_time', 'variation_time', 'search_time', 'rerank_time']
    count_keys = ['num_variations_generated', 'num_unique_docs_found', 'num_docs_reranked', 'num_retrieved_before_rerank', 'num_retrieved_after_rerank']

    all_keys_to_average = metric_keys_k + timing_keys + count_keys

    for key in all_keys_to_average:
        if key in evaluated_df.columns:
            evaluated_df[key] = pd.to_numeric(evaluated_df[key], errors='coerce')
            total = evaluated_df[key].sum(skipna=True)
            valid_count = evaluated_df[key].notna().sum()
            avg_metrics[f'avg_{key}'] = total / valid_count if valid_count > 0 else 0.0
        else:
            avg_metrics[f'avg_{key}'] = 0.0

    return avg_metrics, num_evaluated, num_skipped_error

def log_qa_to_json(user_query, chatbot_response, filepath=None):
    """
    Lưu trữ câu hỏi của người dùng và câu trả lời của chatbot vào một tệp JSON.
    Mỗi mục sẽ bao gồm timestamp, câu hỏi và câu trả lời.
    """
    if filepath is None:
        filepath = config.QA_LOG_FILE

    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": user_query,
        "chatbot_response": chatbot_response
    }

    # Đọc dữ liệu hiện có từ tệp JSON (nếu tệp tồn tại và hợp lệ)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list): # Đảm bảo dữ liệu là một danh sách
                    data = []
            except json.JSONDecodeError: # Xử lý trường hợp tệp rỗng hoặc bị hỏng
                data = []
    else:
        data = [] # Nếu tệp không tồn tại, bắt đầu với một danh sách rỗng

    # Thêm mục mới vào danh sách
    data.append(new_entry)

    # Ghi lại toàn bộ danh sách vào tệp JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

