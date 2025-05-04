import re
import json
from datetime import datetime

TYPE_SOURCE = "Quy chuẩn kỹ thuật số"
DATE_EFF = "1-1-2025"
CHANGE_URL = "https://luatvietnam.vn/giao-thong/quy-chuan-qcvn-412024-bgtvt-bao-hieu-duong-bo-376856-d3.html"
NAME_FILE = "legal_32"

# --- Cập nhật Regex cho cấu trúc Phụ lục / Mục / Điểm ---
REGEX_PHU_LUC = re.compile(r"^\s*Phụ lục\s+([A-Z][0-9]*)\s*(.*)", re.IGNORECASE)
REGEX_MUC_PHU_LUC = re.compile(r"^\s*([A-Z][0-9]*)\.(\d+)\s*(.*)", re.IGNORECASE) # Ví dụ: A.1, B.10
REGEX_DIEM = re.compile(r"^\s*([a-zđ])\)\s*(.*)", re.IGNORECASE) # Giữ nguyên: Ví dụ a), b)

# --- Regex khác (Giữ nguyên) ---
REGEX_SO = re.compile(r"^\s*(?:Số|Luật số):\s*(\S+)", re.IGNORECASE)
REGEX_NGAY_THANG = re.compile(r"^\s*.*?ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", re.IGNORECASE)
REGEX_DOC_TYPE = re.compile(r"^\s*(NGHỊ ĐỊNH|LUẬT|THÔNG TƯ|QUY CHUẨN KỸ THUẬT)", re.IGNORECASE) # Thêm Quy chuẩn kỹ thuật
REGEX_CAN_CU = re.compile(r"^\s*Căn cứ", re.IGNORECASE)

# --- Cập nhật cấp độ Node ---
NODE_TYPE_LEVELS = {'root': 4, 'phu_luc': 3, 'muc_phu_luc': 2, 'diem': 1}

def parse_date_vietnamese(date_str):
    match = re.search(r'ngày (\d{1,2}) tháng (\d{1,2}) năm (\d{4})', date_str, re.IGNORECASE)
    if match:
        day, month, year = map(int, match.groups())
        try:
            datetime(year, month, day)
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError: return None
    return None

def format_symbol_string(symbol_str):
    if not symbol_str: return None
    formatted_symbol = symbol_str.strip().replace('/', '_').replace(':','_') # Thay ':' bằng '_'
    formatted_symbol = re.sub(r'[^\w\.-]', '_', formatted_symbol, flags=re.UNICODE)
    formatted_symbol = re.sub(r'__+', '_', formatted_symbol)
    formatted_symbol = formatted_symbol.strip('_.')
    return formatted_symbol

def generate_structured_id(doc_symbol, chunk_number):
    formatted_symbol = format_symbol_string(doc_symbol)
    prefix = formatted_symbol if formatted_symbol else "unknown_doc"
    return f"{prefix}_chunk_{chunk_number}"

class LegalNode:
    def __init__(self, node_type, identifier=None, title="", full_line=""):
        self.node_type = node_type
        self.identifier = identifier
        self.title = title.strip()
        self.full_line = full_line.strip()
        self.direct_text_lines = []
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def add_text(self, text_line):
        cleaned_line = text_line.strip()
        if cleaned_line:
             self.direct_text_lines.append(cleaned_line)

    def is_structural_leaf(self):
         return not bool(self.children)

    def get_context_dict(self):
        context = {}
        curr = self
        while curr and curr.node_type != 'root':
            if curr.identifier:
                context_key = curr.node_type
                context[context_key] = curr.identifier
            curr = curr.parent
        # Sắp xếp context theo cấp độ giảm dần (tùy chọn, để dễ đọc hơn)
        # sorted_context = {k: context[k] for k in ['phu_luc', 'muc_phu_luc', 'diem'] if k in context}
        # return sorted_context
        return context # Trả về context gốc nếu không cần sắp xếp

    def get_ancestor_headings_and_self(self):
        headings = []
        temp_stack = []
        curr = self

        while curr and curr.node_type != 'root':
            line_to_add = None
            node_type = curr.node_type
            is_self = (curr == self)
            defining_line = curr.full_line # Sử dụng dòng gốc để tái tạo tiêu đề

            # Tái tạo/sử dụng dòng định nghĩa cho từng loại node
            if node_type == 'phu_luc':
                 # Đảm bảo có tiêu đề, nếu không thì tạo từ identifier
                 if not defining_line: defining_line = f"Phụ lục {curr.identifier}. {curr.title}".strip()
                 line_to_add = defining_line
            elif node_type == 'muc_phu_luc':
                 # Đảm bảo có tiêu đề, nếu không thì tạo từ identifier
                 if not defining_line: defining_line = f"{curr.identifier}. {curr.title}".strip()
                 line_to_add = defining_line
            elif node_type == 'diem':
                 # Đối với điểm, chỉ lấy dòng đầy đủ nếu nó là node hiện tại
                 if is_self:
                     if defining_line:
                        # Điều chỉnh định dạng nếu cần (ví dụ: bỏ phần mục cha nếu có)
                        # Giả sử full_line đã có dạng "a) Nội dung..."
                        line_to_add = defining_line
                     # Nếu không có full_line, tạo từ identifier và title
                     # elif curr.identifier and curr.title:
                     #     line_to_add = f"{curr.identifier}) {curr.title}"

            if line_to_add:
                 temp_stack.append(line_to_add)

            curr = curr.parent
        return temp_stack[::-1] # Đảo ngược để có thứ tự từ cao xuống thấp

def process_legal_text(input_filepath, output_filepath):
    chunks = []
    document_metadata = {
        "symbol": None, "type": None, "title": None,
        "issue_date": None, "effective_date": DATE_EFF, "url": CHANGE_URL,
    }
    title_lines = []

    with open(input_filepath, 'r', encoding='utf-8') as f: lines = f.readlines()

    content_start_line_index = 0
    search_limit = min(len(lines), 50) # Giới hạn tìm kiếm metadata
    for i, line in enumerate(lines[:search_limit]):
        line_stripped = line.strip();
        if not line_stripped: continue

        if not document_metadata["symbol"]:
            m_so = REGEX_SO.match(line_stripped)
            if m_so:
                document_metadata["symbol"] = m_so.group(1).strip()

        if not document_metadata["issue_date"]:
            m_ngay = REGEX_NGAY_THANG.match(line_stripped)
            if m_ngay:
                date = parse_date_vietnamese(line_stripped)
                if date:
                    document_metadata["issue_date"] = date

        if not document_metadata["type"]:
            m_type = REGEX_DOC_TYPE.match(line_stripped)
            if m_type:
                # Ưu tiên lấy 'QUY CHUẨN KỸ THUẬT' nếu khớp
                doc_type_match = m_type.group(1).strip().upper()
                if doc_type_match == "QUY CHUẨN KỸ THUẬT":
                     document_metadata["type"] = "Quy chuẩn kỹ thuật"
                else:
                     document_metadata["type"] = doc_type_match.title()
                content_start_line_index = i + 1 # Bắt đầu tìm tiêu đề sau dòng loại văn bản

        # Tìm tiêu đề sau khi đã xác định loại văn bản
        if document_metadata["type"] and not document_metadata["title"]:
             # Cập nhật điều kiện bắt đầu nội dung
             is_content_start = (REGEX_CAN_CU.match(line_stripped) or
                                 REGEX_PHU_LUC.match(line_stripped) or
                                 REGEX_MUC_PHU_LUC.match(line_stripped))
             if is_content_start and i >= content_start_line_index :
                 document_metadata["title"] = " ".join(l.strip() for l in title_lines if l.strip());
                 content_start_line_index = i; # Nội dung bắt đầu từ đây
                 break
             elif line_stripped and i >= content_start_line_index:
                 # Chỉ thêm vào title_lines nếu nó nằm sau dòng loại văn bản
                 title_lines.append(line_stripped)


    # Xử lý nếu không tìm thấy tiêu đề trong search_limit
    if not document_metadata.get("title"):
        if title_lines: document_metadata["title"] = " ".join(l.strip() for l in title_lines if l.strip())
        else: # Tạo tiêu đề mặc định nếu không có
            doc_type_part = document_metadata.get("type", "Văn bản"); symbol_part = document_metadata.get("symbol", "");
            document_metadata["title"] = f"{doc_type_part} {symbol_part}".strip()

        # Tìm lại dòng bắt đầu nội dung chính xác hơn nếu cần
        first_structure_found = False;
        potential_start = content_start_line_index if content_start_line_index > 0 else 0
        for j in range(potential_start, len(lines)):
            s_check = lines[j].strip()
            # Cập nhật điều kiện tìm cấu trúc
            if REGEX_CAN_CU.match(s_check) or REGEX_PHU_LUC.match(s_check) or REGEX_MUC_PHU_LUC.match(s_check):
                 # Nếu tìm thấy cấu trúc sau giới hạn tìm metadata ban đầu, hoặc nếu chưa từng tìm thấy title
                 if j >= search_limit or content_start_line_index < search_limit:
                     content_start_line_index = j
                 first_structure_found = True; break
        # Nếu không tìm thấy cấu trúc nào và index vẫn trong vùng metadata, dời index ra sau vùng metadata
        if not first_structure_found and content_start_line_index < search_limit : content_start_line_index = search_limit

    # --- Bắt đầu chunk ID từ 491 ---
    global_chunk_counter = 491

    root_node = LegalNode('root')
    node_stack = [root_node]

    for i in range(content_start_line_index, len(lines)):
        line = lines[i]
        s = line.strip()
        if not s: continue

        # --- Cập nhật logic khớp Regex ---
        m_phuluc = REGEX_PHU_LUC.match(s)
        m_muc_phuluc = REGEX_MUC_PHU_LUC.match(s)
        m_diem = REGEX_DIEM.match(s)

        new_node = None
        current_level = 0

        if m_phuluc:
            current_level = NODE_TYPE_LEVELS['phu_luc']
            identifier = m_phuluc.group(1).strip() # Ví dụ: 'A', 'B1'
            title = m_phuluc.group(2).strip()
            new_node = LegalNode('phu_luc', identifier, title, s)
        elif m_muc_phuluc:
            current_level = NODE_TYPE_LEVELS['muc_phu_luc']
            phuluc_id = m_muc_phuluc.group(1).strip() # Ví dụ: 'A'
            muc_num = m_muc_phuluc.group(2).strip() # Ví dụ: '1'
            identifier = f"{phuluc_id}.{muc_num}" # Tạo ID đầy đủ: 'A.1'
            title = m_muc_phuluc.group(3).strip()
            new_node = LegalNode('muc_phu_luc', identifier, title, s)
        elif m_diem:
            # Chỉ coi là 'diem' nếu node cha gần nhất là 'muc_phu_luc'
            if node_stack[-1].node_type == 'muc_phu_luc':
                current_level = NODE_TYPE_LEVELS['diem']
                identifier = m_diem.group(1).lower() # Ví dụ: 'a', 'b'
                title = m_diem.group(2).strip()
                # Không cần thêm prefix mục vào full_line của điểm nữa vì get_ancestor_headings_and_self sẽ xử lý
                # full_diem_line = f"{identifier}) {title}"
                new_node = LegalNode('diem', identifier, title, s) # Lưu dòng gốc 'a) ...'

        if new_node:
            # --- Cập nhật logic đẩy/pop stack dựa trên cấp độ mới ---
            while len(node_stack) > 1 and NODE_TYPE_LEVELS[node_stack[-1].node_type] <= current_level:
                 node_stack.pop()
            parent_node = node_stack[-1]
            parent_node.add_child(new_node)
            node_stack.append(new_node)
        else:
             # Gán text cho node hiện tại trên stack nếu không khớp cấu trúc nào
             if node_stack:
                  node_stack[-1].add_text(s)

    def traverse_and_collect(node):
        nonlocal global_chunk_counter, chunks, document_metadata

        is_leaf = node.is_structural_leaf()

        # Chỉ tạo chunk cho các node lá (không có con cấu trúc) VÀ không phải root
        if node.node_type != 'root' and is_leaf:
            # Lấy các tiêu đề cha và dòng định nghĩa của chính node đó
            heading_lines = node.get_ancestor_headings_and_self()

            full_text_lines = heading_lines[:] # Bắt đầu với các dòng tiêu đề

            # Thêm text trực tiếp của node lá (nếu có)
            if node.direct_text_lines:
                # Thêm dòng trống nếu có tiêu đề và có text trực tiếp
                # if full_text_lines and full_text_lines[-1].strip():
                #      full_text_lines.append("")
                full_text_lines.extend(node.direct_text_lines)

            final_text = "\n".join(full_text_lines).strip()
            # Loại bỏ các dòng trống thừa (nếu có)
            final_text_cleaned_lines = [line for line in final_text.split('\n') if line.strip()]
            final_text = "\n".join(final_text_cleaned_lines)


            if final_text: # Chỉ thêm chunk nếu có nội dung
                metadata_context = node.get_context_dict()
                metadata_output = {
                    "source": f"{TYPE_SOURCE} {format_symbol_string(document_metadata.get('symbol', ''))}".strip(),
                    "effective_date": document_metadata.get("effective_date"),
                    "url": document_metadata.get("url", ""),
                    "context": metadata_context # Sử dụng context đã lấy
                }
                chunk_id_val = generate_structured_id(document_metadata.get("symbol"), global_chunk_counter)
                global_chunk_counter += 1
                chunks.append({"id": chunk_id_val, "text": final_text, "metadata": metadata_output})

        # Nếu không phải node lá, duyệt tiếp các node con
        elif not is_leaf :
            for child in node.children:
                traverse_and_collect(child)

    # --- Thêm chunk tiêu đề văn bản ---
    title_parts = [p for p in [document_metadata.get("type"), document_metadata.get("symbol"),
                     document_metadata.get("title"),
                     f"Ngày ban hành: {document_metadata['issue_date']}" if document_metadata.get("issue_date") else None] if p]
    doc_title_text = "\n".join(title_parts)
    if doc_title_text:
        title_chunk_id = generate_structured_id(document_metadata.get("symbol"), global_chunk_counter)
        chunks.append({
            "id": title_chunk_id,
            "text": doc_title_text,
            "metadata": {
                "source": f"{TYPE_SOURCE} {format_symbol_string(document_metadata.get('symbol', ''))}".strip(),
                "effective_date": document_metadata.get("effective_date"),
                "url": document_metadata.get("url", ""),
                "context": {} # Chunk tiêu đề không có context cấu trúc
            }
        })
        global_chunk_counter += 1

    traverse_and_collect(root_node)

    with open(output_filepath, 'w', encoding='utf-8') as fw:
        json.dump(chunks, fw, ensure_ascii=False, indent=2)
    print(f"Đã lưu kết quả vào {output_filepath}")


if __name__ == '__main__':
    input_file = f'text/{NAME_FILE}.txt'
    output_file = f'datasets/{NAME_FILE}.json'
    process_legal_text(input_file, output_file)