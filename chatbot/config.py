# config.py
import os

# --- Cấu hình Mô hình ---
embedding_model_name = 'truro7/vn-law-embedding' 
# embedding_model_name = 'dangvantuan/vietnamese-document-embedding'
reranking_model_name = 'namdp-ptit/ViRanker' 

# --- Danh sách các mô hình Gemini có sẵn ---
AVAILABLE_GEMINI_MODELS = [
    'gemini-2.5-flash-preview-04-17',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-1.5-flash', 
    'gemini-1.5-flash-8b' 
]

DEFAULT_GEMINI_MODEL = 'gemini-2.0-flash'

# --- Cấu hình Đường dẫn ---
SAVED_DATA_DIR = '/kaggle/working/CS431.P22/loader'
SAVED_DATA_PREFIX = os.path.join(SAVED_DATA_DIR, 'legal_rag_data') 
QA_LOG_FILE = os.path.join(SAVED_DATA_DIR, 'chatbot_qa_log.json')
MAP_URL_PATH = os.path.join(SAVED_DATA_DIR, 'vanban_url_map.json')
TRAFFIC_SIGN_IMAGES_ROOT_DIR = '/kaggle/working/CS431.P22/traffic_sign'

# Xác định đường dẫn dữ liệu JSON
JSON_DATA_PATH = '/kaggle/working/CS431.P22/datasets'
JSON_FILE_PATTERN = os.path.join(JSON_DATA_PATH, 'legal_{i}.json')
NUM_FILES = 51 
NUMBERS_TO_SKIP = {29, 30, 37, 38, 39, 40}
MAX_HISTORY_TURNS = 4

# --- Cấu hình RAG ---
NUM_QUERY_VARIATIONS = 4
VECTOR_K_PER_QUERY = 50
HYBRID_K_PER_QUERY = 40
MAX_DOCS_FOR_RERANK = 40
FINAL_NUM_RESULTS_AFTER_RERANK = 15
RRF_K = 60 

# --- Danh sách Stop Words ---
VIETNAMESE_STOP_WORDS = {
    'bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 
    'chỉ', 'chiếc', 'cho', 'chứ', 'có', 'có_thể', 
    'cứ', 'cùng', 'cũng', 'đã', 'đang', 'đây', 'để', 'đến_nỗi', 
    'đều', 'điều', 'do', 'đó', 'được', 'gì', 'khi', 
    'là', 'lại', 'lên', 'lúc', 'mà', 'mỗi', 'một_cách', 
    'này', 'nên', 'nếu', 'ngay', 'nhiều', 'như', 'nhưng', 
    'những', 'nữa', 'phải', 'qua', 'ra', 'rằng', 'rất', 
    'rồi', 'sau', 'sẽ', 'so', 'tại', 'theo', 'thì', 'trên', 
    'trước', 'từ', 'từng', 'và', 'vẫn', 'vào', 'vậy', 'vì', 'với', 'vừa'
}