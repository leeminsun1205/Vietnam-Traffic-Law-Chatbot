import re
import json
from datetime import datetime

# --- Constants and Regex ---
TYPE_SOURCE = "Nghị định số"
DATE_EFF = "10-10-2024"
CHANGE_URL = "https://thuvienphapluat.vn/van-ban/Thue-Phi-Le-Phi/Nghi-dinh-130-2024-ND-CP-quy-dinh-thu-phi-su-dung-duong-bo-cao-toc-thuoc-so-huu-toan-dan-621998.aspx"
NAME_FILE = "legal_48"

REGEX_CHUONG = re.compile(r"^\s*Chương\s+([IVXLCDM]+)\s*(.*)", re.IGNORECASE)
REGEX_MUC = re.compile(r"^\s*Mục\s+(\d+)\.?\s*(.*)", re.IGNORECASE)
REGEX_DIEU = re.compile(r"^\s*Điều\s+(\d+)\.?\s*(.*)", re.IGNORECASE)
REGEX_KHOAN = re.compile(r"^\s*(\d+)\.\s*(.*)")
REGEX_DIEM = re.compile(r"^\s*([a-zđ])\)\s*(.*)", re.IGNORECASE)
REGEX_SO = re.compile(r"^\s*(?:Số|Luật số):\s*(\S+)", re.IGNORECASE)
REGEX_NGAY_THANG = re.compile(r"^\s*.*?ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", re.IGNORECASE)
REGEX_DOC_TYPE = re.compile(r"^\s*(NGHỊ ĐỊNH|LUẬT|THÔNG TƯ)", re.IGNORECASE)
REGEX_CAN_CU = re.compile(r"^\s*Căn cứ", re.IGNORECASE)

NODE_TYPE_LEVELS = {'root': 6, 'chuong': 5, 'muc': 4, 'dieu': 3, 'khoan': 2, 'diem': 1}

# --- Helper Functions ---
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
    formatted_symbol = symbol_str.strip().replace('/', '_')
    formatted_symbol = re.sub(r'[^\w\.-]', '_', formatted_symbol, flags=re.UNICODE)
    formatted_symbol = re.sub(r'__+', '_', formatted_symbol)
    formatted_symbol = formatted_symbol.strip('_.')
    return formatted_symbol

def generate_structured_id(doc_symbol, chunk_number):
    formatted_symbol = format_symbol_string(doc_symbol)
    prefix = formatted_symbol if formatted_symbol else "unknown_doc"
    return f"{prefix}_chunk_{chunk_number}"

# --- Node Structure ---
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
        return context

    def get_ancestor_headings_and_self(self):
        headings = []
        temp_stack = []
        curr = self
        processed_khoan_for_diem = False

        while curr and curr.node_type != 'root':
            line_to_add = None
            node_type = curr.node_type
            is_self = (curr == self)
            defining_line = curr.full_line

            if node_type == 'chuong':
                if not defining_line: defining_line = f"Chương {curr.identifier}. {curr.title}".strip()
                line_to_add = defining_line
            elif node_type == 'muc':
                if not defining_line: defining_line = f"Mục {curr.identifier}. {curr.title}".strip()
                line_to_add = defining_line
            elif node_type == 'dieu':
                if not defining_line: defining_line = f"Điều {curr.identifier}. {curr.title}".strip()
                line_to_add = defining_line
            elif node_type == 'khoan':
                if is_self or (self.node_type == 'diem' and not processed_khoan_for_diem):
                     if defining_line:
                         line_to_add = defining_line
                         processed_khoan_for_diem = True
            elif node_type == 'diem':
                 if is_self:
                      if defining_line:
                          line_to_add = defining_line

            if line_to_add:
                 temp_stack.append(line_to_add)

            curr = curr.parent
        return temp_stack[::-1]

# --- Main Processing Function (Tree Logic) ---
def process_legal_text(input_filepath, output_filepath):
    chunks = []
    document_metadata = {
        "symbol": None, "type": None, "title": None,
        "issue_date": None, "effective_date": DATE_EFF, "url": CHANGE_URL,
    }
    title_lines = []

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError: print(f"Lỗi: Không tìm thấy tệp '{input_filepath}'"); return
    except Exception as e: print(f"Lỗi khi đọc tệp '{input_filepath}': {e}"); return

    # --- Extract Metadata and Title ---
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
                # *** SỬA LỖI: Chuyển kiểm tra 'date' vào trong khối 'if m_ngay:' ***
                if date:
                    document_metadata["issue_date"] = date

        if not document_metadata["type"]:
            m_type = REGEX_DOC_TYPE.match(line_stripped)
            if m_type:
                document_metadata["type"] = m_type.group(1).strip().title()
                content_start_line_index = i + 1

        if document_metadata["type"] and not document_metadata["title"]:
             is_content_start = (REGEX_CAN_CU.match(line_stripped) or REGEX_CHUONG.match(line_stripped) or REGEX_DIEU.match(line_stripped))
             if is_content_start and i >= content_start_line_index :
                 document_metadata["title"] = " ".join(l.strip() for l in title_lines if l.strip()); content_start_line_index = i; break
             elif line_stripped and i >= content_start_line_index:
                 title_lines.append(line_stripped)

    if not document_metadata.get("title"):
        if title_lines: document_metadata["title"] = " ".join(l.strip() for l in title_lines if l.strip())
        else: doc_type_part = document_metadata.get("type", "Văn bản"); symbol_part = document_metadata.get("symbol", ""); document_metadata["title"] = f"{doc_type_part} {symbol_part}".strip()
        first_structure_found = False; potential_start = content_start_line_index if content_start_line_index > 0 else 0
        for j in range(potential_start, len(lines)):
            s_check = lines[j].strip()
            if REGEX_CAN_CU.match(s_check) or REGEX_CHUONG.match(s_check) or REGEX_DIEU.match(s_check):
                 if j >= search_limit or content_start_line_index < search_limit: content_start_line_index = j
                 first_structure_found = True; break
        if not first_structure_found and content_start_line_index < search_limit : content_start_line_index = search_limit

    global_chunk_counter = 1

    # --- Pass 1: Build Tree ---
    root_node = LegalNode('root')
    node_stack = [root_node]
    expecting_chapter_title = False

    for i in range(content_start_line_index, len(lines)):
        line = lines[i]
        s = line.strip()

        current_node_on_stack = node_stack[-1]
        if expecting_chapter_title and s:
             if current_node_on_stack.node_type == 'chuong':
                 is_new_structure = (REGEX_CHUONG.match(s) or REGEX_MUC.match(s) or REGEX_DIEU.match(s) or
                                     REGEX_KHOAN.match(s) or REGEX_DIEM.match(s) or REGEX_CAN_CU.match(s))
                 if not is_new_structure:
                     current_node_on_stack.title = s
                     current_node_on_stack.full_line = f"{current_node_on_stack.full_line} {s}".strip()
                     expecting_chapter_title = False
                     continue
                 else:
                     expecting_chapter_title = False

        if not s: continue

        m_chuong=REGEX_CHUONG.match(s); m_muc=REGEX_MUC.match(s); m_dieu=REGEX_DIEU.match(s)
        m_khoan=REGEX_KHOAN.match(s); m_diem=REGEX_DIEM.match(s)

        new_node = None
        current_level = 0

        if m_chuong:
            current_level = NODE_TYPE_LEVELS['chuong']; identifier = m_chuong.group(1); title = m_chuong.group(2)
            new_node = LegalNode('chuong', identifier, title, s)
            if not title.strip(): expecting_chapter_title = True
        elif m_muc:
            current_level = NODE_TYPE_LEVELS['muc']; identifier = m_muc.group(1); title = m_muc.group(2)
            new_node = LegalNode('muc', identifier, title, s)
        elif m_dieu:
            current_level = NODE_TYPE_LEVELS['dieu']; identifier = m_dieu.group(1); title = m_dieu.group(2)
            new_node = LegalNode('dieu', identifier, title, s)
        elif m_khoan:
            current_level = NODE_TYPE_LEVELS['khoan']; identifier = m_khoan.group(1); title = m_khoan.group(2)
            new_node = LegalNode('khoan', identifier, title, s)
        elif m_diem:
            current_level = NODE_TYPE_LEVELS['diem']; identifier = m_diem.group(1).lower(); title = m_diem.group(2)
            parent_khoan_id = node_stack[-1].identifier if node_stack and node_stack[-1].node_type == 'khoan' else None
            diem_prefix = f"{parent_khoan_id}. " if parent_khoan_id else ""
            full_diem_line = f"{diem_prefix}{identifier}) {title}"
            new_node = LegalNode('diem', identifier, title, full_diem_line)

        if new_node:
            while len(node_stack) > 1 and NODE_TYPE_LEVELS[node_stack[-1].node_type] <= current_level:
                 node_stack.pop()
            parent_node = node_stack[-1]
            parent_node.add_child(new_node)
            node_stack.append(new_node)
        else:
             if node_stack:
                  node_stack[-1].add_text(s)


    # --- Pass 2: Traverse Tree and Generate Chunks ---
    def traverse_and_collect(node):
        nonlocal global_chunk_counter, chunks, document_metadata

        is_leaf = node.is_structural_leaf()

        if node.node_type != 'root' and is_leaf:
            heading_lines = node.get_ancestor_headings_and_self()

            full_text_lines = heading_lines[:]
            if node.direct_text_lines:
                if full_text_lines and full_text_lines[-1]:
                    full_text_lines.append("")
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

    # --- Add title chunk ---
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

    # --- Start Traversal from root ---
    traverse_and_collect(root_node)

    # --- Write Output ---
    try:
        with open(output_filepath, 'w', encoding='utf-8') as fw:
            json.dump(chunks, fw, ensure_ascii=False, indent=2)
        print(f"Đã lưu kết quả vào {output_filepath}")
    except Exception as e:
        print(f"Lỗi khi ghi tệp JSON '{output_filepath}': {e}")

# --- Main Execution Guard ---
if __name__ == '__main__':
    input_file = f'text/{NAME_FILE}.txt'
    output_file = f'datasets/{NAME_FILE}.json'
    process_legal_text(input_file, output_file)