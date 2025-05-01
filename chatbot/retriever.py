# retriever.py
import re
import os
import numpy as np
import pickle
import config
from rank_bm25 import BM25Okapi
# Import ViTokenizer từ pyvi
from pyvi import ViTokenizer
from config import VIETNAMESE_STOP_WORDS

# --- Retrieval ---
def retrieve_relevant_chunks(query_text, embedding_model, vector_db, k=5):
    """Embed query và tìm kiếm trong vector_db."""
    query_embedding = embedding_model.encode(query_text, convert_to_numpy=True).astype('float32')
    distances, indices = vector_db.search(query_embedding, k=k)
    return distances, indices

class HybridRetriever:
    """Kết hợp Vector Search và BM25 Search."""
    def __init__(self, vector_db, bm25_save_path):
        self.vector_db = vector_db
        self.documents = getattr(vector_db, 'documents', [])
        self.document_texts = [doc.get('text', '') for doc in self.documents] if self.documents else []
        self.bm25_index_path = bm25_save_path
        self.bm25 = None

        # Khởi tạo hoặc tải BM25
        if not self.load_bm25(self.bm25_index_path):
            self._initialize_bm25() # Khởi tạo nếu không tải được
            if self.bm25:
                 self.save_bm25(self.bm25_index_path)

    def _initialize_bm25(self):
        """Khởi tạo BM25 index."""
        if not self.documents:
            return
        tokenized_corpus = [self._tokenize_vi(text) for text in self.document_texts]
        if not any(tokenized_corpus):
             self.bm25 = None
        else:
             self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize_vi(self, text):
        """Tokenize dùng pyvi.ViTokenizer, làm sạch, bỏ stop words."""
        if not text or not isinstance(text, str): return []
        text_lower = text.lower()
        # Sử dụng ViTokenizer.tokenize, nó trả về chuỗi các từ nối bằng '_'
        # ví dụ: "Trường_đại_học Công_nghệ Thông_tin"
        tokenized_text = ViTokenizer.tokenize(text_lower)
        # Tách chuỗi thành các token dựa trên khoảng trắng
        word_tokens = tokenized_text.split()

        final_tokens = []
        for token in word_tokens:
            # Loại bỏ dấu câu và ký tự đặc biệt (giữ lại '_')
            cleaned_token = re.sub(r'[^\w_]', '', token, flags=re.UNICODE)
            # Chỉ giữ lại token không rỗng và không phải là stop word
            if cleaned_token and cleaned_token not in VIETNAMESE_STOP_WORDS:
                final_tokens.append(cleaned_token)
        return final_tokens


    def hybrid_search(self, query_text, embedding_model, vector_search_k=20, final_k=10):
        """Thực hiện tìm kiếm kết hợp."""
        if self.vector_db is None or self.vector_db.index is None or self.vector_db.index.ntotal == 0:
            return []

        # --- 1. Vector Search ---
        vec_distances, vec_indices = retrieve_relevant_chunks(
            query_text, embedding_model, self.vector_db, k=vector_search_k
        )

        # --- 2. BM25 Search ---
        bm25_search_indices = []
        if self.bm25:
             # Sử dụng hàm tokenize mới với ViTokenizer
             tokenized_query = self._tokenize_vi(query_text)
             if tokenized_query:
                bm25_scores = self.bm25.get_scores(tokenized_query)
                # Lấy các index có score > 0 và sắp xếp
                bm25_scored_indices = [(score, i) for i, score in enumerate(bm25_scores) if score > 0]
                bm25_scored_indices.sort(key=lambda x: x[0], reverse=True)
                # Lấy top K index từ BM25
                bm25_search_indices = [index for score, index in bm25_scored_indices[:vector_search_k]]

        # --- 3. Rank Fusion (Reciprocal Rank Fusion - RRF) ---
        rank_lists_to_fuse = []
        if isinstance(vec_indices, (list, np.ndarray)) and len(vec_indices) > 0:
            # Đảm bảo vec_indices là list các số nguyên
            if isinstance(vec_indices, np.ndarray):
                vec_indices_list = vec_indices.flatten().tolist() # Đảm bảo là 1D list
            else:
                vec_indices_list = vec_indices # Giả sử đã là list
            rank_lists_to_fuse.append([int(i) for i in vec_indices_list if isinstance(i, (int, np.integer))])


        if isinstance(bm25_search_indices, list) and len(bm25_search_indices) > 0:
            rank_lists_to_fuse.append([int(i) for i in bm25_search_indices if isinstance(i, (int, np.integer))])


        fused_indices = []
        fused_scores_dict = {}
        if rank_lists_to_fuse:
            fused_indices, fused_scores_dict = self._rank_fusion_indices(rank_lists_to_fuse, k=config.RRF_K)
        # Fallback: Nếu chỉ có kết quả vector search, sử dụng nó
        elif isinstance(vec_indices, (list, np.ndarray)) and len(vec_indices) > 0:
             fused_indices = vec_indices.tolist() if isinstance(vec_indices, np.ndarray) else vec_indices
             fused_indices = [int(i) for i in fused_indices if isinstance(i, (int, np.integer))] # Đảm bảo kiểu int


        # --- 4. Get Top K Documents ---
        hybrid_results = []
        retrieved_indices = set() # Để tránh trùng lặp document
        for rank, idx in enumerate(fused_indices):
            # Đảm bảo idx là số nguyên hợp lệ và chưa được lấy
            if isinstance(idx, (int, np.integer)) and 0 <= idx < len(self.documents) and idx not in retrieved_indices:
                score = fused_scores_dict.get(idx, 1 / (rank + 1)) # Lấy score từ RRF hoặc tính score dựa trên rank
                hybrid_results.append({'doc': self.documents[idx], 'hybrid_score': score, 'index': idx})
                retrieved_indices.add(idx)
                if len(hybrid_results) >= final_k: # Dừng khi đủ K kết quả
                    break

        return hybrid_results

    def _rank_fusion_indices(self, rank_lists, k=60):
        """Thực hiện RRF để kết hợp các danh sách rank."""
        fused_scores = {}
        # Duyệt qua từng danh sách kết quả (từ vector search, bm25)
        for rank_list in rank_lists:
            # Đảm bảo rank_list là list
            if isinstance(rank_list, np.ndarray):
                rank_list = rank_list.flatten().tolist()
            if not isinstance(rank_list, list): continue

            # Duyệt qua từng document index trong danh sách
            for rank, doc_index in enumerate(rank_list):
                 # Đảm bảo doc_index là số nguyên
                 if isinstance(doc_index, (int, np.integer)):
                     rank_ = rank + 1 # Rank bắt đầu từ 1
                     # Tính điểm RRF và cộng dồn
                     fused_scores[doc_index] = fused_scores.get(doc_index, 0) + (1 / (rank_ + k)) # Thêm hằng số k

        # Sắp xếp các document index dựa trên điểm RRF giảm dần
        sorted_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return sorted_indices, fused_scores # Trả về cả dict điểm để tham khảo

    def save_bm25(self, filepath):
        """Lưu trạng thái BM25 vào file pickle."""
        if self.bm25:
            with open(filepath, 'wb') as f:
                pickle.dump(self.bm25, f)

    def load_bm25(self, filepath):
        """Tải trạng thái BM25 từ file pickle."""
        if os.path.exists(filepath):
             with open(filepath, 'rb') as f:
                 self.bm25 = pickle.load(f)
             # Kiểm tra sơ bộ xem object load được có hợp lệ không
             if not hasattr(self.bm25, 'get_scores'):
                 self.bm25 = None; return False
             return True
        else:
            return False