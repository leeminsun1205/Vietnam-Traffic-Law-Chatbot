from urllib.parse import urlparse
from utils import load_document_url_mapping, extract_and_normalize_document_key
import config

def generate_answer_with_gemini(query_text, relevant_documents, gemini_model, mode='Đầy đủ', chat_history=None):
    url_mapping_dict = load_document_url_mapping(config.MAP_URL_PATH)
    source_details_for_prompt = []
    key_to_original_source_name_map = {}

    if not relevant_documents:
        context_for_prompt = "Không có thông tin ngữ cảnh nào được cung cấp."
    else:
        for i, item in enumerate(relevant_documents):
            doc_content = item.get('doc')
            text = doc_content.get('text', '').strip()
            metadata = doc_content.get('metadata', {})
            source_name_from_metadata = metadata.get('source', 'N/A')
            normalized_key_from_meta = extract_and_normalize_document_key(source_name_from_metadata)
            if normalized_key_from_meta and normalized_key_from_meta not in key_to_original_source_name_map:
                key_to_original_source_name_map[normalized_key_from_meta] = source_name_from_metadata
            if not doc_content or not text: continue

            context_meta = metadata.get('context', {})
            chuong = context_meta.get('chuong')
            muc = context_meta.get('muc')
            dieu = context_meta.get('dieu')
            khoan = context_meta.get('khoan')
            diem = context_meta.get('diem')

            source_parts = [f"Văn bản: {source_name_from_metadata}"]
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
        "11. **QUAN TRỌNG - HIỂN THỊ BIỂN BÁO VÀ TRÍCH DẪN NGUỒN**: "
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
    9.  **Trả lời trọng tâm**: Đảm bảo câu trả lời **đủ để giải đáp cho câu hỏi**, không cần trả lời lan man, thông tin không liên quan, trừ khi thật sự cần thiết, nhất là các câu hỏi về phương tiện, biển báo. Với các phương tiện, thực thể nằm sau từ "trừ" thì tức là không có tác dụng vi phạm cho điều luật đó.
    10. Kiểm tra lại cú pháp Markdown của câu trả lời trước khi hoàn thành.
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

    collected_citations = {} 
    citations_found_in_llm_text = re.findall(r'\((?:[Tt]heo\s)?([^)]+?)\)', final_answer_display)

    for citation_text_from_llm in citations_found_in_llm_text:
        if '<' in citation_text_from_llm and '>' in citation_text_from_llm: 
            continue
        
        doc_key_from_citation = extract_and_normalize_document_key(citation_text_from_llm)
        
        if doc_key_from_citation:
            url_from_mapping = url_mapping_dict.get(doc_key_from_citation)
            if url_from_mapping:
                current_name_for_link = None
                source_is_from_metadata = False

                if doc_key_from_citation in key_to_original_source_name_map:
                    raw_metadata_source_name = key_to_original_source_name_map[doc_key_from_citation]
                    if raw_metadata_source_name and raw_metadata_source_name.strip():
                        current_name_for_link = raw_metadata_source_name.replace("_", "/")
                        source_is_from_metadata = True
                
                if not current_name_for_link:
                    formatted_doc_key = doc_key_from_citation.replace("_", "/")
                    if formatted_doc_key and formatted_doc_key.strip():
                        current_name_for_link = formatted_doc_key
                
                if current_name_for_link: 
                    existing_entry = collected_citations.get(url_from_mapping)
                    if not existing_entry or \
                       (source_is_from_metadata and not existing_entry['is_metadata_source']) or \
                       (not existing_entry['name'] and current_name_for_link): 
                        collected_citations[url_from_mapping] = {
                            'name': current_name_for_link,
                            'is_metadata_source': source_is_from_metadata
                        }
    
    if collected_citations:
        link_display_items = []
        for url_key, data_val in collected_citations.items():
            name_to_display_in_link = data_val['name']
            if not name_to_display_in_link or not name_to_display_in_link.strip():
                parsed_url_obj = urlparse(url_key)
                name_to_display_in_link = parsed_url_obj.netloc + parsed_url_obj.path
                if name_to_display_in_link.endswith('/'): 
                    name_to_display_in_link = name_to_display_in_link[:-1]
                if not name_to_display_in_link.strip(): 
                    name_to_display_in_link = url_key
            link_display_items.append({'url': url_key, 'name': name_to_display_in_link})
        
        link_display_items.sort(key=lambda item: item['name'])
        
        markdown_links_list = [f"- [{item['name']}]({item['url']})" for item in link_display_items]
        markdown_links_string = "\n".join(markdown_links_list)
        final_answer_display += f"\n\n**Nguồn tham khảo (Văn bản pháp luật):**\n{markdown_links_string}"

    return final_answer_display.strip()