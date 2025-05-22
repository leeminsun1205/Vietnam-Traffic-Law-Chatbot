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
        6.  Nếu câu hỏi chỉ có đề "xe máy" thì phải thay thế bằng "xe mô tô, xe gắn máy", trừ cụm từ "xe máy chuyên dùng".
        7.  Đối với summarizing_query cũng phải áp dụng đầy đủ các quy tắc 2, 3, 5, 6.
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
def load_document_url_mapping(filepath):
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

def generate_answer_with_gemini(query_text, relevant_documents, gemini_model, mode='Đầy đủ', chat_history=None):
    url_mapping_dict = load_document_url_mapping(config.MAP_URL_PATH)
    source_details_for_prompt = []

    if not relevant_documents:
        context_for_prompt = "Không có thông tin ngữ cảnh nào được cung cấp."
    else:
        for i, item in enumerate(relevant_documents):
            doc_content = item.get('doc')
            text = doc_content.get('text', '').strip()
            metadata = doc_content.get('metadata', {})
            if not doc_content or not text: continue

            source_name = metadata.get('source', 'N/A')
            context_meta = metadata.get('context', {})
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

            # Lấy giá trị traffic_sign, có thể là string hoặc list
            traffic_sign_value = metadata.get('traffic_sign')
            traffic_sign_info_for_llm = ""
            if traffic_sign_value:
                if isinstance(traffic_sign_value, list):
                    display_filenames = ", ".join(traffic_sign_value[:3])
                    if len(traffic_sign_value) > 3:
                        display_filenames += "..."
                    traffic_sign_info_for_llm = (
                        f" (LƯU Ý QUAN TRỌNG: Nội dung này có liên quan đến các biển báo: '{display_filenames}'. "
                        f"Nếu sử dụng thông tin từ đây để trả lời, hãy nhớ dùng placeholder chỉ mục biển báo.)"
                    )
                elif isinstance(traffic_sign_value, str):
                    traffic_sign_info_for_llm = (
                        f" (LƯU Ý QUAN TRỌNG: Nội dung này có liên quan đến biển báo '{traffic_sign_value}'. "
                        f"Nếu sử dụng thông tin từ đây để trả lời, hãy nhớ dùng placeholder chỉ mục biển báo.)"
                    )
            source_details_for_prompt.append(f"Nguồn {i+1}: [{source_ref_full}]{traffic_sign_info_for_llm}\nNội dung: {text}\n---")

        if not source_details_for_prompt:
             context_for_prompt = "Không có thông tin ngữ cảnh nào được cung cấp để xây dựng prompt."
        else:
             context_for_prompt = "\n".join(source_details_for_prompt)

    history_prefix = ""
    if chat_history:
        history_prefix = "**Lịch sử trò chuyện gần đây (dùng để tham khảo ngữ cảnh):**\n"
        limited_history = chat_history[-(config.MAX_HISTORY_TURNS * 2):]
        for msg in limited_history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "").strip()
            if role and content:
                 history_prefix += f"{role}: {content}\n"
        history_prefix += "---\n"

    placeholder_instruction = (
        "9. **QUAN TRỌNG - HIỂN THỊ BIỂN BÁO VÀ TRÍCH DẪN NGUỒN**: "
        "Khi bạn sử dụng thông tin từ một 'Nguồn' (được đánh số thứ tự 1, 2, 3,... trong phần 'Ngữ cảnh được cung cấp') "
        "VÀ nguồn đó có ghi chú đặc biệt về việc nội dung liên quan đến một hoặc nhiều biển báo (thường bắt đầu bằng cụm từ như '(LƯU Ý QUAN TRỌNG: Nội dung này có liên quan đến (các) biển báo...'), "
        "hãy tuân theo THỨ TỰ sau cho mỗi phần thông tin bạn lấy từ nguồn đó:\n"
        "    a. Đầu tiên, trình bày **nội dung văn bản** mà bạn trích xuất hoặc diễn giải.\n"
        "    b. **Ngay sau nội dung văn bản đó**, nếu bạn muốn hoặc cần trích dẫn nguồn chi tiết cho phần văn bản này, hãy viết **trích dẫn nguồn** (ví dụ: `(Theo Chương A, Mục B, Điều X, Khoản Y, Điểm z, Văn bản ABC)`).\n"
        "    c. **CUỐI CÙNG, và NGAY SAU trích dẫn nguồn (hoặc ngay sau nội dung văn bản nếu không có trích dẫn nguồn cho đoạn đó)**, bạn PHẢI đặt một placeholder đặc biệt có dạng: `[DISPLAY_TRAFFIC_SIGN_INDEX_{index_nguồn}]`. "
        "`{index_nguồn}` CHÍNH LÀ số thứ tự của 'Nguồn' đó.\n"
        "    * **Ví dụ THỨ TỰ ĐÚNG**: `...nội dung từ Nguồn 3... (Theo Điều A, QCVN XYZ) [DISPLAY_TRAFFIC_SIGN_INDEX_3]`\n"
        "    * Hoặc nếu không có trích dẫn cụ thể cho đoạn văn bản đó: `...nội dung từ Nguồn 3... [DISPLAY_TRAFFIC_SIGN_INDEX_3]`\n"
        "    * **TUYỆT ĐỐI KHÔNG** đặt placeholder ảnh trước trích dẫn nguồn của nó. Placeholder ảnh phải là yếu tố cuối cùng liên quan đến khối thông tin của nguồn đó."
    )
    common_requirements = f"""
    1.  **Chỉ dùng ngữ cảnh:** Tuyệt đối không suy diễn kiến thức ngoài luồng.
    2.  **Gom nhóm nguồn và trích dẫn:**
        * Khi trích dẫn, hãy tham chiếu đến cấu trúc văn bản (Điều, Khoản, Điểm, tên Văn bản) một cách rõ ràng nhất đã. Ví dụ: `(Theo Điều X, Khoản Y, Điểm z, Văn bản ABC)`.
        * Những đoạn cùng cấu trúc nguồn (Văn bản, Điều, Khoản) **BẮT BUỘC** nhóm nguồn lại với nhau để gọn gàng và dễ nhìn cho người dùng.
        * Kết hợp thông tin từ nhiều đoạn nếu cần, **diễn đạt lại mạch lạc**, tránh lặp lại nguyên văn dài dòng. Nhưng **tuyệt đối không pha trộn thông tin 1 cách tùy tiện gây sai lệch nghiêm trọng thông tin**.
        * Sau mỗi ý hoặc nhóm ý chính, **nêu rõ nguồn gốc** dùng thông tin trong dấu "`(...)`".
    3.  **Trình bày súc tích:** Sử dụng gạch đầu dòng (`-`) nếu cần hoặc đánh số (`1., 2.`), **in đậm** (`** **`) cho các nội dung quan trọng như mức phạt, kết luận, hoặc các điểm chính.
    4.  **Hiểu ngữ nghĩa:** Tìm thông tin liên quan ngay cả khi từ ngữ không khớp hoàn toàn (ví dụ: "rượu, bia" sẽ liên quan tới "nồng độ cồn"; "đèn đỏ", "đèn vàng" là "đèn tín hiệu", "xe máy" vs "xe mô tô/gắn máy/xe hai bánh", ... (xe máy không phải là xe máy chuyên dùng) và từ ngữ giao thông khác).
    5.  **Trường hợp thiếu thông tin:** Nếu ngữ cảnh được cung cấp không chứa thông tin để trả lời câu hỏi, hãy trả lời một cách trung thực, ví dụ: "**Dựa trên thông tin được cung cấp, tôi không tìm thấy nội dung phù hợp để trả lời câu hỏi này.**"
    6.  Thứ tự ưu tiên khi câu hỏi mang tính so sánh là:
        - Điểm thứ a trong Điều/Khoản (ưu tiên cao nhất)
        - Điểm thứ b trong Điều/Khoản
        - Điểm thứ c trong Điều/Khoản
        - Điểm thứ d trong Điều/Khoản
        - Điểm thứ đ trong Điều/Khoản
        - ...
    7.  **Phân biệt loại xe:** Cần phân biệt rõ "xe máy" (xe mô tô, xe gắn máy, ...) và "xe máy chuyên dùng" (xe máy thi công, nông nghiệp, lâm nghiệp, v.v.). Nếu câu hỏi về "xe máy" thì **CHỈ** trả lời "xe mô tô, xe gắn máy". 
    8.  **Logic và suy luận:** Phải thể hiện được tính logic từ câu hỏi đến câu trả lời, đặc biệt với các câu hỏi yêu cầu so sánh, tính toán đơn giản (nếu có thể từ ngữ cảnh), hoặc phân tích tình huống.
    {placeholder_instruction}
    """
    full_prompt_template = f"""Bạn là một trợ lý chuyên sâu về Luật Giao thông Đường bộ Việt Nam.
    {history_prefix}
    **Nhiệm vụ:** Dựa vào Lịch sử trò chuyện gầy đây (nếu có) và thông tin chi tiết trong phần 'Ngữ cảnh được cung cấp' dưới đây, hãy trả lời câu hỏi HIỆN TẠI của người dùng một cách **CHI TIẾT, ĐẦY ĐỦ** và chính xác nhất có thể.

    **Ngữ cảnh được cung cấp (Đây là nguồn thông tin duy nhất bạn được phép sử dụng để trả lời các câu hỏi về luật):**
    {context_for_prompt}
    ---
    **Câu hỏi HIỆN TẠI của người dùng:** {query_text}
    ---
    **Yêu cầu trả lời CHI TIẾT:**
    {common_requirements}
    **Câu trả lời của bạn (chi tiết):**
    """
    brief_prompt_template = f"""Bạn là một trợ lý chuyên sâu về Luật Giao thông Đường bộ Việt Nam.
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
    prompt_to_use = brief_prompt_template if mode == 'Ngắn gọn' else full_prompt_template
    final_answer_display = "Xin lỗi, tôi chưa thể tạo câu trả lời vào lúc này."
    try:
        if not gemini_model:
            raise ValueError("Mô hình Gemini chưa được tải hoặc không khả dụng.")
        response = gemini_model.generate_content(prompt_to_use)
        if hasattr(response, 'text') and response.text:
            final_answer_display = response.text.strip()
        elif response.parts:
            final_answer_display = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
            if not final_answer_display:
                final_answer_display = "Không nhận được nội dung văn bản từ Gemini."
        else:
            final_answer_display = "Phản hồi từ Gemini không có nội dung văn bản."
    except Exception as e:
        final_answer_display = f"Đã xảy ra sự cố khi xử lý yêu cầu của bạn. Chi tiết lỗi: {str(e)[:200]}..."

    # if relevant_documents and isinstance(relevant_documents, list) and final_answer_display:
    #     processed_answer_parts = []
    #     last_idx = 0
        # displayed_sign_filenames = set()

        # for match in re.finditer(r"\[DISPLAY_TRAFFIC_SIGN_INDEX_(\d+)]", final_answer_display):
        #     # placeholder_full = match.group(0)
        #     source_idx_one_based = int(match.group(1))
        #     source_idx_zero_based = source_idx_one_based - 1

        #     processed_answer_parts.append(final_answer_display[last_idx:match.start()])
            
        #     # --- BẮT ĐẦU THAY ĐỔI LOGIC XỬ LÝ NHIỀU ẢNH ---
        #     images_html_for_current_placeholder_list = [] # Lưu HTML cho từng ảnh của placeholder này

        #     if 0 <= source_idx_zero_based < len(relevant_documents):
        #         doc_item = relevant_documents[source_idx_zero_based]
        #         doc_content = doc_item.get('doc')
        #         if isinstance(doc_content, dict):
        #             metadata = doc_content.get('metadata', {})
        #             traffic_sign_value = metadata.get('traffic_sign') # Đây có thể là string hoặc list

        #             filenames_to_process_for_chunk = []
        #             if isinstance(traffic_sign_value, str):
        #                 filenames_to_process_for_chunk = [traffic_sign_value]
        #             elif isinstance(traffic_sign_value, list):
        #                 filenames_to_process_for_chunk = traffic_sign_value
                    
        #             for traffic_sign_filename_original in filenames_to_process_for_chunk: # Đổi tên biến để tránh nhầm lẫn
        #                 if traffic_sign_filename_original and traffic_sign_filename_original not in displayed_sign_filenames:
        #                     image_full_path = os.path.join(config.TRAFFIC_SIGN_IMAGES_ROOT_DIR, traffic_sign_filename_original)
        #                     if os.path.exists(image_full_path):
        #                         try:
        #                             with open(image_full_path, "rb") as img_file:
        #                                 b64_string = base64.b64encode(img_file.read()).decode()
        #                             file_ext = os.path.splitext(traffic_sign_filename_original)[1][1:].lower()
        #                             if not file_ext: file_ext = "png"
                                    
        #                             # --- THAY ĐỔI CÁCH HIỂN THỊ TÊN BIỂN BÁO ---
        #                             base_name_no_ext = os.path.splitext(traffic_sign_filename_original)[0]
        #                             display_sign_name = base_name_no_ext.replace("_", ".")
        #                             # ------------------------------------------
                                    
        #                             single_image_html = (
        #                                 f"<div style='flex: 1 0 23%; max-width: 24%; margin: 5px; text-align: center;'>"
        #                                 f"<img src='data:image/{file_ext};base64,{b64_string}' "
        #                                 f"alt='Biển báo: {display_sign_name}' " # Alt text cũng có thể dùng tên đã xử lý
        #                                 f"style='width: 100%; max-width: 150px; height: auto; border: 1px solid #ddd; padding: 2px; border-radius: 4px;'/>"
        #                                 f"<p style='font-size: 0.75em; margin-top: 2px; font-style: italic; word-wrap: break-word;'>{display_sign_name}</p>" # Sử dụng tên đã xử lý
        #                                 f"</div>"
        #                             )
        #                             images_html_for_current_placeholder_list.append(single_image_html)
        #                             # Vẫn dùng tên file gốc để kiểm tra trùng lặp
        #                             displayed_sign_filenames.add(traffic_sign_filename_original) 
        #                         except Exception as e_img:
        #                             images_html_for_current_placeholder_list.append(f"<div style='color: red; font-style: italic; padding-left: 20px; flex-basis:100%'>[Lỗi hiển thị ảnh: {traffic_sign_filename_original}]</div>")
        #                     else:
        #                         images_html_for_current_placeholder_list.append(f"<div style='color: orange; font-style: italic; padding-left: 20px; flex-basis:100%'>[Không tìm thấy ảnh: {traffic_sign_filename_original}]</div>")
        #                 elif traffic_sign_filename_original in displayed_sign_filenames:
        #                      pass

        #     # Tạo container grid cho các ảnh của placeholder này nếu có ảnh
        #     image_markdown_to_insert = ""
        #     if images_html_for_current_placeholder_list:
        #         image_grid_html_content = "".join(images_html_for_current_placeholder_list)
        #         image_markdown_to_insert = (
        #             f"\n<div style='display: flex; flex-wrap: wrap; justify-content: flex-start; align-items: flex-start; margin-top: 10px; margin-bottom: 10px; padding-left: 15px; border-left: 3px solid #eee;'>"
        #             f"{image_grid_html_content}"
        #             f"</div>\n"
        #         )
        #     # --- KẾT THÚC THAY ĐỔI LOGIC XỬ LÝ NHIỀU ẢNH ---
            
        #     processed_answer_parts.append(image_markdown_to_insert)
        #     last_idx = match.end()

        # processed_answer_parts.append(final_answer_display[last_idx:])
        # final_answer_display = "".join(processed_answer_parts)

    found_urls = set()
    citations_found = re.findall(r'\((?:[Tt]heo\s)?([^)]+?)\)', final_answer_display)
    for citation_text in citations_found:
        if '<' in citation_text and '>' in citation_text:
            continue
        doc_key = extract_and_normalize_document_key(citation_text)
        if doc_key:
            url = url_mapping_dict.get(doc_key)
            if url:
                found_urls.add(url)
    if found_urls:
        sorted_urls = sorted(list(found_urls))
        urls_display_list = []
        for url in sorted_urls:
            display_name = url
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                display_name = parsed_url.netloc + parsed_url.path
                if display_name.endswith('/'): display_name = display_name[:-1]
            except ImportError:
                pass
            urls_display_list.append(f"- [{display_name}]({url})")
        urls_string = "\n".join(urls_display_list)
        final_answer_display += f"\n\n**Nguồn tham khảo (Văn bản gốc):**\n{urls_string}"

    return final_answer_display.strip()

def render_html_for_assistant_message(text_content, relevant_documents):
    if not relevant_documents or not isinstance(relevant_documents, list) or not text_content:
        return text_content
    st.write('START')
    processed_answer_parts = []
    last_idx = 0
    displayed_sign_filenames = set()

    for match in re.finditer(r"\[DISPLAY_TRAFFIC_SIGN_INDEX_(\d+)]", text_content):
        source_idx_one_based = int(match.group(1))
        source_idx_zero_based = source_idx_one_based - 1
        processed_answer_parts.append(text_content[last_idx:match.start()])

        images_html_for_current_placeholder_list = []
        if 0 <= source_idx_zero_based < len(relevant_documents):
            doc_item = relevant_documents[source_idx_zero_based]
            doc_content = doc_item.get('doc')
            if isinstance(doc_content, dict):
                metadata = doc_content.get('metadata', {})
                traffic_sign_value = metadata.get('traffic_sign')
                filenames_to_process_for_chunk = []
                if isinstance(traffic_sign_value, str):
                    filenames_to_process_for_chunk = [traffic_sign_value]
                elif isinstance(traffic_sign_value, list):
                    filenames_to_process_for_chunk = traffic_sign_value

                for traffic_sign_filename_original in filenames_to_process_for_chunk:
                    if traffic_sign_filename_original and traffic_sign_filename_original not in displayed_sign_filenames:
                        image_full_path = os.path.join(config.TRAFFIC_SIGN_IMAGES_ROOT_DIR, traffic_sign_filename_original)
                        if os.path.exists(image_full_path):
                            try:
                                with open(image_full_path, "rb") as img_file:
                                    b64_string = base64.b64encode(img_file.read()).decode()
                                file_ext = os.path.splitext(traffic_sign_filename_original)[1][1:].lower()
                                if not file_ext: file_ext = "png"
                                base_name_no_ext = os.path.splitext(traffic_sign_filename_original)[0]
                                display_sign_name = base_name_no_ext.replace("_", ".")
                                single_image_html = (
                                    f"<div style='flex: 1 0 23%; max-width: 24%; margin: 5px; text-align: center;'>"
                                    f"<img src='data:image/{file_ext};base64,{b64_string}' "
                                    f"alt='Biển báo: {display_sign_name}' "
                                    f"style='width: 100%; max-width: 150px; height: auto; border: 1px solid #ddd; padding: 2px; border-radius: 4px;'/>"
                                    f"<p style='font-size: 0.75em; margin-top: 2px; font-style: italic; word-wrap: break-word;'>{display_sign_name}</p>"
                                    f"</div>"
                                )
                                images_html_for_current_placeholder_list.append(single_image_html)
                                displayed_sign_filenames.add(traffic_sign_filename_original)
                            except Exception as e_img:
                                images_html_for_current_placeholder_list.append(f"<div style='color: red; font-style: italic; padding-left: 20px; flex-basis:100%'>[Lỗi hiển thị ảnh: {traffic_sign_filename_original} - {e_img}]</div>")
                        else:
                            images_html_for_current_placeholder_list.append(f"<div style='color: orange; font-style: italic; padding-left: 20px; flex-basis:100%'>[Không tìm thấy ảnh: {traffic_sign_filename_original}]</div>")
                    elif traffic_sign_filename_original in displayed_sign_filenames:
                         pass

        image_markdown_to_insert = ""
        if images_html_for_current_placeholder_list:
            image_grid_html_content = "".join(images_html_for_current_placeholder_list)
            image_markdown_to_insert = (
                f"\n<div style='display: flex; flex-wrap: wrap; justify-content: flex-start; align-items: flex-start; margin-top: 10px; margin-bottom: 10px; padding-left: 15px; border-left: 3px solid #eee;'>"
                f"{image_grid_html_content}"
                f"</div>\n"
            )
        processed_answer_parts.append(image_markdown_to_insert)
        last_idx = match.end()

    processed_answer_parts.append(text_content[last_idx:])
    return "".join(processed_answer_parts)

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

