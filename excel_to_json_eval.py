import pandas as pd
import json
import argparse
import os
import re # Thêm thư viện regex để parse list string an toàn hơn

def parse_chunk_id_string(id_string):
    """
    Phân tích chuỗi chứa danh sách Chunk ID (ví dụ: '[id1, id2, id3]')
    thành một list các chuỗi ID sạch.
    """
    if not isinstance(id_string, str):
        return [] # Trả về list rỗng nếu không phải string

    # Loại bỏ dấu ngoặc vuông ở đầu/cuối và khoảng trắng thừa
    cleaned_string = id_string.strip().strip('[]').strip()

    if not cleaned_string: # Nếu chuỗi rỗng sau khi làm sạch
        return []

    # Tách chuỗi bằng dấu phẩy, sau đó loại bỏ khoảng trắng thừa của từng ID
    # Dùng regex để tìm các ID, an toàn hơn split nếu có khoảng trắng bất thường
    # Regex này tìm các chuỗi ký tự không phải là dấu phẩy, cách, ngoặc vuông, hoặc ngoặc kép/đơn
    # Hoặc một cách đơn giản hơn nếu định dạng luôn là ", ":
    # ids = [item.strip() for item in cleaned_string.split(',')]
    # Cách dùng regex:
    ids = re.findall(r"[^,\s\'\"\[\]]+", cleaned_string) # Tìm tất cả các chuỗi ký tự hợp lệ làm ID

    # Lọc bỏ các phần tử rỗng có thể xuất hiện sau khi tách/tìm
    return [item for item in ids if item]


def convert_excel_to_json_eval_new(excel_filepath, json_output_path, sheet_name=0, question_col="Questions", chunk_col="ChunkIDs"):
    """
    Chuyển đổi tệp Excel (Câu hỏi, Chuỗi_List_ChunkID) sang định dạng JSON đánh giá RAG.
    """
    try:
        print(f"Đang đọc tệp Excel: {excel_filepath} (Sheet: {sheet_name})")
        df = pd.read_excel(excel_filepath, sheet_name=sheet_name)
        print(f"Đã đọc {len(df)} dòng.")

        if question_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột '{question_col}' trong tệp Excel.")
        if chunk_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột '{chunk_col}' trong tệp Excel.")

        # Chỉ giữ lại các cột cần thiết và loại bỏ hàng NaN ở cột Question
        df = df[[question_col, chunk_col]].dropna(subset=[question_col])
        df[question_col] = df[question_col].astype(str).str.strip()

        # Xử lý cột ChunkIDs - giữ lại NaN hoặc chuyển thành string và strip
        df[chunk_col] = df[chunk_col].apply(lambda x: str(x).strip() if pd.notna(x) else '')

        print(f"Số dòng hợp lệ sau khi xử lý NaN và strip: {len(df)}")
        if df.empty:
            print("Không có dữ liệu hợp lệ để xử lý.")
            return

        final_json_data = []
        # Lặp qua từng hàng, mỗi hàng là một câu hỏi
        for index, row in df.iterrows():
            question = row[question_col]
            chunk_ids_string = row[chunk_col]

            if not question: # Bỏ qua nếu câu hỏi rỗng sau khi strip
                continue

            # Phân tích chuỗi ChunkIDs thành list
            relevant_ids = parse_chunk_id_string(chunk_ids_string)

            final_json_data.append({
                "query_id": f"eval_{index+1:03d}", # Tạo query_id dựa trên index hàng
                "query": question,
                "relevant_chunk_ids": relevant_ids # Danh sách ID đã được parse
            })

        print(f"Đã xử lý và tạo {len(final_json_data)} mục JSON.")

        print(f"Đang ghi kết quả vào tệp JSON: {json_output_path}")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_data, f, ensure_ascii=False, indent=2)

        print("Chuyển đổi thành công!")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp Excel tại đường dẫn '{excel_filepath}'")
    except ValueError as ve:
        print(f"Lỗi dữ liệu: {ve}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")

# --- Thiết lập Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi Excel (Questions, Chuỗi_List_ChunkID) sang JSON đánh giá RAG.")
    parser.add_argument("excel_file", help="Đường dẫn đến tệp Excel đầu vào (.xlsx hoặc .xls).")
    parser.add_argument("json_output", help="Đường dẫn để lưu tệp JSON đầu ra.")
    parser.add_argument("-s", "--sheet", default=0, help="Tên hoặc chỉ số sheet trong Excel (mặc định: 0).")
    parser.add_argument("-q", "--qcol", default="Questions", help="Tên cột chứa câu hỏi (mặc định: 'Questions').")
    # Đổi tên cột chunk mặc định cho phù hợp với mô tả của bạn
    parser.add_argument("-c", "--ccol", default="ChunkIDs", help="Tên cột chứa chuỗi list Chunk ID (mặc định: 'ChunkIDs').")

    args = parser.parse_args()

    convert_excel_to_json_eval_new(args.excel_file, args.json_output, args.sheet, args.qcol, args.ccol)