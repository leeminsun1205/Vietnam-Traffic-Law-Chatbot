import re
import json
from datetime import datetime

TYPE_SOURCE = "Quy chuẩn kỹ thuật số"
DATE_EFF = "1-1-2025"
CHANGE_URL = "https://luatvietnam.vn/giao-thong/quy-chuan-qcvn-412024-bgtvt-bao-hieu-duong-bo-376856-d3.html"
NAME_FILE = "legal_32"

# --- Regex cho cấu trúc mới ---
REGEX_PHU_LUC = re.compile(r"^\s*Phụ lục\s+([A-Z][0-9]*)\s*(.*)", re.IGNORECASE) # Phụ lục G
REGEX_MUC_CAP1_DOTTED = re.compile(r"^\s*([A-Z]\d*)\.(\d+)\.?\s+(.*)") # A.1 Tiêu đề / A.1. Tiêu đề
REGEX_MUC_CAP1_NO_DOT = re.compile(r"^\s*([A-Z]\d*)(\d+)\.?\s+(.*)")   # G1 Tiêu đề / G1. Tiêu đề
REGEX_MUC_CAP2 = re.compile(r"^\s*([A-Z]\d*\.\d+)\.(\d+)\.?\s+(.*)")  # G1.1 Tiêu đề / G1.1. Tiêu đề
REGEX_DIEM = re.compile(r"^\s*([a-zđ])\)\s*(.*)", re.IGNORECASE)     # a) Tiêu đề

# --- Regex khác (Giữ nguyên) ---
REGEX_SO = re.compile(r"^\s*(?:Số|Luật số):\s*(\S+)", re.IGNORECASE)
REGEX_NGAY_THANG = re.compile(r"^\s*.*?ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", re.IGNORECASE)
REGEX_DOC_TYPE = re.compile(r"^\s*(NGHỊ ĐỊNH|LUẬT|THÔNG TƯ|QUY CHUẨN KỸ THUẬT)", re.IGNORECASE)
REGEX_CAN_CU = re.compile(r"^\s*Căn cứ", re.IGNORECASE)

# --- Cập nhật cấp độ Node ---
NODE_TYPE_LEVELS = {'root': 5, 'phu_luc': 4, 'muc_1': 3, 'muc_2': 2, 'diem': 1}

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
    formatted_symbol = symbol_str.strip().replace('/', '_').replace(':','_')
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
                # Đảm bảo không ghi đè nếu có nhiều node cùng loại trong tổ tiên (ít xảy ra)
                if curr.node_type not in context:
                     context[curr.node_type] = curr.identifier
            curr = curr.parent
        return context

    def get_ancestor_headings_and_self(self):
        headings = []
        temp_stack = []
        curr = self

        while curr and curr.node_type != 'root':
            line_to_add = None
            node_type = curr.node_type
            is_self = (curr == self)
            defining_line = curr.full_line # Ưu tiên dùng dòng gốc

            if not defining_line and curr.identifier: # Tạo dòng nếu dòng gốc trống
                 if node_type == 'phu_luc':
                     defining_line = f"Phụ lục {curr.identifier}. {curr.title}".strip()
                 elif node_type == 'muc_1':
                     # Xử lý cả dạng A.1 và G1
                     defining_line = f"{curr.identifier}. {curr.title}".strip()
                 elif node_type == 'muc_2':
                     defining_line = f"{curr.identifier}. {curr.title}".strip() # Ví dụ: G1.1. Tiêu đề
                 elif node_type == 'diem':
                     defining_line = f"{curr.identifier}) {curr.title}".strip()

            # Chỉ thêm dòng định nghĩa nếu nó tồn tại
            if defining_line:
                 # Đối với điểm, chỉ thêm dòng định nghĩa nếu nó là node hiện tại
                 if node_type == 'diem' and not is_self:
                      pass # Bỏ qua dòng của điểm cha/ông...
                 else:
                      line_to_add = defining_line

            if line_to_add:
                 temp_stack.append(line_to_add)

            curr = curr.parent
        return temp_stack[::-1]

def process_legal_text(input_filepath, output_filepath):
    chunks = []
    document_metadata = {
        "symbol": None, "type": None, "title": None,
        "issue_date": None, "effective_date": DATE_EFF, "url": CHANGE_URL,
    }
    title_lines = []

    with open(input_filepath, 'r', encoding='utf-8') as f: lines = f.readlines()

    content_start_line_index = 0
    search_limit = min(len(lines), 50)
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
                doc_type_match = m_type.group(1).strip().upper()
                if doc_type_match == "QUY CHUẨN KỸ THUẬT":
                     document_metadata["type"] = "Quy chuẩn kỹ thuật"
                else:
                     document_metadata["type"] = doc_type_match.title()
                content_start_line_index = i + 1

        if document_metadata["type"] and not document_metadata["title"]:
             # Cập nhật điều kiện bắt đầu nội dung với các regex mới
             is_content_start = (REGEX_CAN_CU.match(line_stripped) or
                                 REGEX_PHU_LUC.match(line_stripped) or
                                 REGEX_MUC_CAP1_DOTTED.match(line_stripped) or
                                 REGEX_MUC_CAP1_NO_DOT.match(line_stripped) or
                                 REGEX_MUC_CAP2.match(line_stripped)
                                 )
             if is_content_start and i >= content_start_line_index :
                 document_metadata["title"] = " ".join(l.strip() for l in title_lines if l.strip());
                 content_start_line_index = i;
                 break
             elif line_stripped and i >= content_start_line_index:
                 title_lines.append(line_stripped)

    if not document_metadata.get("title"):
        if title_lines: document_metadata["title"] = " ".join(l.strip() for l in title_lines if l.strip())
        else:
            doc_type_part = document_metadata.get("type", "Văn bản"); symbol_part = document_metadata.get("symbol", "");
            document_metadata["title"] = f"{doc_type_part} {symbol_part}".strip()

        first_structure_found = False;
        potential_start = content_start_line_index if content_start_line_index > 0 else 0
        for j in range(potential_start, len(lines)):
            s_check = lines[j].strip()
            # Cập nhật điều kiện tìm cấu trúc
            if (REGEX_CAN_CU.match(s_check) or REGEX_PHU_LUC.match(s_check) or
                REGEX_MUC_CAP1_DOTTED.match(s_check) or REGEX_MUC_CAP1_NO_DOT.match(s_check) or
                REGEX_MUC_CAP2.match(s_check)):
                 if j >= search_limit or content_start_line_index < search_limit:
                     content_start_line_index = j
                 first_structure_found = True; break
        if not first_structure_found and content_start_line_index < search_limit : content_start_line_index = search_limit

    global_chunk_counter = 491

    root_node = LegalNode('root')
    node_stack = [root_node]

    for i in range(content_start_line_index, len(lines)):
        line = lines[i]
        s = line.strip()
        if not s: continue

        # Khớp Regex theo thứ tự và ngữ cảnh
        m_phuluc = REGEX_PHU_LUC.match(s)
        m_muc1_dot = REGEX_MUC_CAP1_DOTTED.match(s) # A.1 Title
        m_muc1_nodot = REGEX_MUC_CAP1_NO_DOT.match(s) # G1 Title
        m_muc2 = REGEX_MUC_CAP2.match(s) # G1.1 Title
        m_diem = REGEX_DIEM.match(s) # a) Title

        new_node = None
        current_level = 0
        node_type = None
        identifier = None
        title = ""

        # Xác định loại node và cấp độ dựa trên regex khớp
        # Ưu tiên khớp các mẫu cụ thể trước
        if m_phuluc:
            node_type = 'phu_luc'
            current_level = NODE_TYPE_LEVELS[node_type]
            identifier = m_phuluc.group(1).strip() # ID: G
            title = m_phuluc.group(2).strip()
        elif m_muc2: # Regex này khá đặc trưng (ví dụ G1.1), kiểm tra trước muc_1
            node_type = 'muc_2'
            current_level = NODE_TYPE_LEVELS[node_type]
            # ID được ghép từ group(1) (G1) và group(2) (1)
            identifier = f"{m_muc2.group(1).strip()}.{m_muc2.group(2).strip()}" # ID: G1.1
            title = m_muc2.group(3).strip()
        elif m_muc1_dot:
            node_type = 'muc_1'
            current_level = NODE_TYPE_LEVELS[node_type]
            identifier = f"{m_muc1_dot.group(1).strip()}.{m_muc1_dot.group(2).strip()}" # ID: A.1
            title = m_muc1_dot.group(3).strip()
        elif m_muc1_nodot:
            node_type = 'muc_1'
            current_level = NODE_TYPE_LEVELS[node_type]
            identifier = f"{m_muc1_nodot.group(1).strip()}{m_muc1_nodot.group(2).strip()}" # ID: G1
            title = m_muc1_nodot.group(3).strip()
        elif m_diem:
            node_type = 'diem'
            current_level = NODE_TYPE_LEVELS[node_type]
            identifier = m_diem.group(1).lower() # ID: a
            title = m_diem.group(2).strip()

        # Nếu khớp một cấu trúc (node_type đã được xác định)
        if node_type:
            # 1. Tạo node mới
            new_node = LegalNode(node_type, identifier, title, s)

            # 2. Xác định node cha đúng bằng cách pop stack
            # Lưu lại node top cũ để xử lý trường hợp ngữ cảnh sai
            original_top_node = node_stack[-1] if len(node_stack) > 1 else root_node
            while len(node_stack) > 1 and NODE_TYPE_LEVELS[node_stack[-1].node_type] <= current_level:
                node_stack.pop()
            parent_node = node_stack[-1] # Node cha thực sự sau khi pop

            # 3. Kiểm tra ngữ cảnh của node cha
            valid_parent = False
            if node_type == 'phu_luc' and parent_node.node_type == 'root':
                valid_parent = True
            elif node_type == 'muc_1' and parent_node.node_type == 'phu_luc':
                valid_parent = True
            elif node_type == 'muc_2' and parent_node.node_type == 'muc_1':
                 # Kiểm tra xem ID của muc_2 có bắt đầu bằng ID của muc_1 cha không
                 if identifier and parent_node.identifier and identifier.startswith(parent_node.identifier + "."):
                      valid_parent = True
                 # Fallback nếu ID không khớp hoàn hảo (ít xảy ra nếu cấu trúc đúng)
                 # elif identifier and parent_node.identifier: # Logic dự phòng đơn giản hơn
                 #      valid_parent = True
            elif node_type == 'diem' and parent_node.node_type in ['muc_1', 'muc_2']:
                valid_parent = True

            # 4. Thêm node con nếu ngữ cảnh cha hợp lệ
            if valid_parent:
                parent_node.add_child(new_node)
                node_stack.append(new_node)
            else:
                # Nếu ngữ cảnh không hợp lệ (ví dụ: tìm thấy 'a)' dưới 'phu_luc')
                # Coi dòng này là text của node *trước khi pop* (original_top_node)
                if original_top_node.node_type != 'root':
                     original_top_node.add_text(s)
                # Hoặc có thể báo lỗi/bỏ qua dòng không hợp lệ tùy yêu cầu
                # print(f"Cảnh báo: Ngữ cảnh không hợp lệ cho dòng {i+1}: '{s}'. Cha dự kiến: {parent_node.node_type}, Node mới: {node_type}")

        # Nếu không khớp cấu trúc nào và stack không rỗng/root
        elif node_stack and node_stack[-1].node_type != 'root':
             # Thêm dòng này làm text cho node hiện tại trên stack
             node_stack[-1].add_text(s)


    def traverse_and_collect(node):
        nonlocal global_chunk_counter, chunks, document_metadata

        is_leaf = node.is_structural_leaf()

        # Tạo chunk cho node lá (không phải root)
        if node.node_type != 'root' and is_leaf:
            heading_lines = node.get_ancestor_headings_and_self()
            full_text_lines = heading_lines[:]

            if node.direct_text_lines:
                # if full_text_lines and full_text_lines[-1].strip():
                #      full_text_lines.append("") # Thêm dòng trống nếu cần tách biệt
                full_text_lines.extend(node.direct_text_lines)

            final_text = "\n".join(full_text_lines).strip()
            final_text_cleaned_lines = [line for line in final_text.split('\n') if line.strip()]
            final_text = "\n".join(final_text_cleaned_lines)

            if final_text:
                metadata_context = node.get_context_dict()
                metadata_output = {
                    "source": f"{TYPE_SOURCE} {format_symbol_string(document_metadata.get('symbol', ''))}".strip(),
                    "effective_date": document_metadata.get("effective_date"),
                    "url": document_metadata.get("url", ""),
                    "context": metadata_context
                }
                chunk_id_val = generate_structured_id(document_metadata.get("symbol"), global_chunk_counter)
                global_chunk_counter += 1
                chunks.append({"id": chunk_id_val, "text": final_text, "metadata": metadata_output})
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
                "context": {}
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