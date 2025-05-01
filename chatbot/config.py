# config.py
import os
import streamlit as st
# --- Cấu hình Mô hình ---
embedding_model_name = 'truro7/vn-law-embedding' 
reranking_model_name = 'namdp-ptit/ViRanker' 
gemini_model_name = 'gemini-2.0-flash' 

# --- Cấu hình Đường dẫn ---
KAGGLE_INPUT_PATH = '/kaggle/working/CS431.P22/datasets'
LOCAL_DATA_PATH = './' 
VNCORENLP_SAVE_DIR = 'kaggle/working/' 
SAVED_DATA_DIR = 'loader'
SAVED_DATA_PREFIX = os.path.join(SAVED_DATA_DIR, 'legal_rag_data') 

# Xác định đường dẫn dữ liệu JSON
JSON_DATA_PATH = KAGGLE_INPUT_PATH 
JSON_FILE_PATTERN = os.path.join(JSON_DATA_PATH, 'legal_{i}.json')
NUM_FILES = 31 
NUMBERS_TO_SKIP = {29, 30}

# --- Cấu hình Tokenizer & Word Segmentation ---
REQUIRES_WORD_SEGMENTATION = False 
# Ví dụ kiểm tra tên model để tự động xác định
if "bkai" in embedding_model_name.lower() or "phobert" in embedding_model_name.lower():
     REQUIRES_WORD_SEGMENTATION = True # Tự động bật nếu là model cần tách từ

DEFAULT_MAX_LEN = 512 
LARGE_PLACEHOLDER_THRESHOLD = 1_000_000_000 

# --- Cấu hình RAG ---
NUM_QUERY_VARIATIONS = 3
VECTOR_K_PER_QUERY = 20
HYBRID_K_PER_QUERY = 15
MAX_DOCS_FOR_RERANK = 40
FINAL_NUM_RESULTS_AFTER_RERANK = 15
RRF_K = 60 

# --- Danh sách Stop Words ---
VIETNAMESE_STOP_WORDS = {
    'bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 
    'chỉ', 'chiếc', 'cho', 'chứ', 'chưa', 'có', 'có_thể', 
    'cứ', 'cùng', 'cũng', 'đã', 'đang', 'đây', 'để', 'đến_nỗi', 
    'đều', 'điều', 'do', 'đó', 'được', 'gì', 'khi', 'không', 
    'là', 'lại', 'lên', 'lúc', 'mà', 'mỗi', 'một_cách', 
    'này', 'nên', 'nếu', 'ngay', 'nhiều', 'như', 'nhưng', 
    'những', 'nơi', 'nữa', 'phải', 'qua', 'ra', 'rằng', 'rất', 
    'rồi', 'sau', 'sẽ', 'so', 'sự', 'tại', 'theo', 'thì', 'trên', 
    'trước', 'từ', 'từng', 'và', 'vẫn', 'vào', 'vậy', 'vì', 'việc', 'với', 'vừa'
}

# --- Cấu hình Khác ---
# Tạo thư mục lưu dữ liệu nếu chưa có
if not os.path.exists(SAVED_DATA_DIR):
    os.makedirs(SAVED_DATA_DIR)
if not os.path.exists(VNCORENLP_SAVE_DIR) and REQUIRES_WORD_SEGMENTATION:
     # Chỉ tạo nếu thực sự cần VnCoreNLP
     os.makedirs(VNCORENLP_SAVE_DIR)