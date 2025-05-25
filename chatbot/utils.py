# utils.py
import json
import os
import re
import streamlit as st
import config
import math
import pandas as pd
from datetime import datetime
import base64

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
    Xác định câu hỏi có bị mơ hồ hoặc liên quan trực tiếp đến Luật GTĐB Việt Nam không (quy tắc, biển báo, phạt, giấy phép, phương tiện,...) (chấp nhận các câu hỏi về đường sắt, loại đường ngang).

    **Bước 2: Tạo Phản hồi JSON**
    Dựa vào Bước 1, tạo phản hồi JSON **CHÍNH XÁC** theo định dạng sau:

    * **Nếu câu hỏi KHÔNG liên quan hoặc có thông tin MƠ HỒ (ngữ nghĩa quá ngắn gọn để hiểu hoặc gây khó  hiểu):** Hãy tạo một câu trả lời **ngắn gọn, trực tiếp** phù hợp với câu hỏi đó (ví dụ: nếu hỏi "Bạn là ai?", trả lời "Tôi là chatbot về Luật GTĐB Việt Nam."). Đặt câu trả lời này vào trường `invalid_answer`. Nếu không thể tạo câu trả lời phù hợp, để trống trường này.
        ```json
        {{
        "relevance": "invalid",
        "invalid_answer": "[Câu trả lời trực tiếp cho câu hỏi không liên quan, hoặc câu hỏi ngược lại ( có thể dựa vào lịch sử để gợi ý nếu có) người dùng nhằm xác minh chi tiết hơn ý đồ của người dùng muốn hỏi]",
        "variations": [],
        "summarizing_query": ""
        }}
        ```
    * **Nếu câu hỏi CÓ liên quan:** Hãy thực hiện các yêu cầu sau:
        1.  Tạo {num_variations} biến thể câu hỏi (ưu tiên từ khóa luật, phương tiện, từ đồng nghĩa).
        2.  Tạo MỘT câu hỏi tổng hợp bao quát (summarizing_query), câu hỏi này phải đảm bảo chứa tất cả các keyword của câu hỏi gốc (để tránh làm sai lệch quá nhiều).
        3.  Ưu tiên biến thể chứa "không tuân thủ, không chấp hành" nếu hỏi về lỗi, vi phạm.
        4.  Nếu câu hỏi có **liên quan đến phương tiện** mà hỏi 1 cách tổng quát (không chỉ rõ loại xe nào) phải ưu tiên 1 câu có "xe mô tô, xe gắn máy, các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy" và 1 câu có "xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô".
        5.  Nếu câu hỏi chứa từ "xe máy" thì phải thay thế bằng "xe mô tô, xe gắn máy", ngoại trừ cụm từ "xe máy chuyên dùng".
        6.  Nếu câu hỏi về có từ "lề", "lề đường" phải chuyển thành "vỉa hè". 
        7.  Đối với summarizing_query cũng phải áp dụng đầy đủ các quy tắc 2, 3, 4, 5, 6.
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

# --- Generation utils ---
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

# --- Render traffic sign in chat utils ---
def render_html_for_assistant_message(text_content, relevant_documents):
    if not relevant_documents or not isinstance(relevant_documents, list) or not text_content:
        return text_content
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

# --- Calculate metrics ---

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

    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list): 
                data = []
    else: data = [] 

    data.append(new_entry)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

