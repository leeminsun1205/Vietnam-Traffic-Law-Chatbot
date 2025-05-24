# retriever.py
import re
import os
import numpy as np
import pickle
import config
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
from config import VIETNAMESE_STOP_WORDS

# --- Hàm retrieve_relevant_chunks gốc ---
def retrieve_relevant_chunks(query_text, embedding_model, vector_db, k=5):
    """Embed query và tìm kiếm trong vector_db."""
    if not query_text or embedding_model is None or vector_db is None or vector_db.index is None or vector_db.index.ntotal == 0:
        return np.array([]), np.array([]) # Trả về mảng rỗng
    
    query_embedding = embedding_model.encode(query_text, convert_to_numpy=True).astype('float32')

    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    if query_embedding.shape[1] != vector_db.embedding_dimension:
        return np.array([]), np.array([])
    
    actual_k = min(k, vector_db.index.ntotal)
    if actual_k <= 0:
        return np.array([]), np.array([])
    distances, indices = vector_db.search(query_embedding, k=actual_k)
    # search trả về 2D arrays, lấy phần tử đầu tiên
    return distances, indices
class Retriever:
    """Kết hợp Vector Search (Dense), BM25 Search (Sparse), hoặc cả hai (Hybrid)."""
    def __init__(self, vector_db, bm25_save_path):
        self.vector_db = vector_db
        self.documents = getattr(vector_db, 'documents', [])
        self.document_texts = [doc.get('text', '') if isinstance(doc, dict) else '' for doc in self.documents]
        self.bm25_index_path = bm25_save_path
        self.bm25 = None

        # Khởi tạo hoặc tải BM25
        if not self.load_bm25(self.bm25_index_path):
            if self.document_texts: # Chỉ khởi tạo nếu có text
                self._initialize_bm25()
                if self.bm25:
                    self.save_bm25(self.bm25_index_path)

    def _initialize_bm25(self):
        """Khởi tạo BM25 index."""
        if not self.document_texts: return
        try:
            # Tokenize corpus, bỏ qua các document rỗng
            tokenized_corpus = [self._tokenize_vi(text) for text in self.document_texts if text and isinstance(text, str)]
            if not any(tokenized_corpus): # Kiểm tra xem có token nào không
                 self.bm25 = None
            else:
                 self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            self.bm25 = None

    def _tokenize_vi(self, text):
        """Tokenize dùng pyvi.ViTokenizer, làm sạch, bỏ stop words."""
        if not text or not isinstance(text, str): return []
        try:
            text_lower = text.lower()
            tokenized_text = ViTokenizer.tokenize(text_lower)
            word_tokens = tokenized_text.split()

            final_tokens = []
            for token in word_tokens:
                # Giữ lại dấu gạch dưới, loại bỏ các ký tự đặc biệt khác
                cleaned_token = re.sub(r'[^\w\s_]', '', token, flags=re.UNICODE).strip()
                if cleaned_token and cleaned_token not in VIETNAMESE_STOP_WORDS:
                    final_tokens.append(cleaned_token)
            return final_tokens
        except Exception as e:
            return []


    def search(self, query_text, embedding_model, method='hybrid', k=20):
        if not query_text: return []
        results = []
        indices_set = set() 

        if method == 'dense':
            distances, indices = retrieve_relevant_chunks(query_text, embedding_model, self.vector_db, k=k)
            if indices is not None and len(indices) > 0:
                for i, idx in enumerate(indices):
                    if isinstance(idx, (int, np.integer)) and 0 <= idx < len(self.documents) and idx not in indices_set:
                        results.append({
                            'doc': self.documents[idx],
                            'score': float(distances[i]), 
                            'index': int(idx)
                        })
                        indices_set.add(idx)
                # Sắp xếp theo distance tăng dần (score nhỏ hơn là tốt hơn)
                results.sort(key=lambda x: x['score'])

        elif method == 'sparse':
            if self.bm25:
                tokenized_query = self._tokenize_vi(query_text)
                if tokenized_query:    
                    bm25_scores = self.bm25.get_scores(tokenized_query)
                    bm25_scored_indices = [(bm25_scores[i], i) for i in range(len(bm25_scores)) if bm25_scores[i] > 0]
                    bm25_scored_indices.sort(key=lambda x: x[0], reverse=True)
                    for score, idx in bm25_scored_indices[:k]:
                        if idx not in indices_set: 
                            results.append({
                                'doc': self.documents[idx],
                                'score': float(score), 
                                'index': int(idx)
                            })
                            indices_set.add(idx)

        elif method == 'hybrid':
            # --- 1. Vector Search (Dense) ---
            _, vec_indices = retrieve_relevant_chunks(
                query_text, embedding_model, self.vector_db, k=config.VECTOR_K_PER_QUERY
            )
            vec_indices_list = []
            if vec_indices is not None and len(vec_indices) > 0:
                vec_indices_list = [int(i) for i in vec_indices.flatten().tolist() if isinstance(i, (int, np.integer))]

            # --- 2. BM25 Search (Sparse) ---
            bm25_indices_list = []
            if self.bm25:
                tokenized_query = self._tokenize_vi(query_text)
                if tokenized_query:
                    bm25_scores = self.bm25.get_scores(tokenized_query)
                    bm25_scored_indices = [(bm25_scores[i], i) for i in range(len(bm25_scores)) if bm25_scores[i] > 0]
                    bm25_scored_indices.sort(key=lambda x: x[0], reverse=True)
                    bm25_indices_list = [int(index) for _, index in bm25_scored_indices[:config.VECTOR_K_PER_QUERY]]

            # --- 3. Rank Fusion (RRF) ---
            rank_lists_to_fuse = []
            if vec_indices_list: rank_lists_to_fuse.append(vec_indices_list)
            if bm25_indices_list: rank_lists_to_fuse.append(bm25_indices_list)

            fused_indices = []
            fused_scores_dict = {}
            if rank_lists_to_fuse:
                fused_indices, fused_scores_dict = self._rank_fusion_indices(rank_lists_to_fuse, k=config.RRF_K) # Dùng RRF_K từ config
            elif vec_indices_list: # Fallback: Nếu chỉ có kết quả dense
                 fused_indices = vec_indices_list
                 # Tạo dict score giả dựa trên rank (score cao hơn cho rank thấp hơn)
                 fused_scores_dict = {idx: 1.0 / (rank + 1) for rank, idx in enumerate(fused_indices)}

            # --- 4. Get Top K Documents ---
            for rank, idx in enumerate(fused_indices):
                 # Lấy top K kết quả cuối cùng
                if len(results) >= k: break
                # Đảm bảo idx hợp lệ và chưa được thêm
                if isinstance(idx, (int, np.integer)) and 0 <= idx < len(self.documents) and idx not in indices_set:
                    score = fused_scores_dict.get(idx, 1.0 / (rank + 1 + config.RRF_K)) # Lấy score RRF hoặc rank-based
                    results.append({'doc': self.documents[idx], 'score': float(score), 'index': int(idx)})
                    indices_set.add(idx)

        else: return []
        return results 

    def _rank_fusion_indices(self, rank_lists, k=60):
        """Thực hiện RRF để kết hợp các danh sách rank."""
        fused_scores = {}
        for rank_list in rank_lists:
            for rank, doc_index in enumerate(rank_list):
                 if isinstance(doc_index, (int, np.integer)):
                     rank_ = rank + 1
                     fused_scores[doc_index] = fused_scores.get(doc_index, 0) + (1 / (rank_ + k))

        sorted_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return sorted_indices, fused_scores

    def save_bm25(self, filepath):
        """Lưu trạng thái BM25 vào file pickle."""
        if self.bm25:
            with open(filepath, 'wb') as f:
                pickle.dump(self.bm25, f)

    def load_bm25(self, filepath):
        """Tải trạng thái BM25 từ file pickle."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    self.bm25 = pickle.load(f)
                 # Kiểm tra xem object load được có phương thức cần thiết không
                if not hasattr(self.bm25, 'get_scores'):
                    self.bm25 = None
                    return False
                return True
            except Exception as e:
                self.bm25 = None
                return False
        else: return False