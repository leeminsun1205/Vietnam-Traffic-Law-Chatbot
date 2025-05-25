# retriever.py
import re
import os
import numpy as np
import pickle
import config
import streamlit as st
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
from config import VIETNAMESE_STOP_WORDS

# --- Hàm retrieve_relevant_chunks gốc ---
def retrieve_relevant_chunks(query_text, embedding_model, vector_db, k=5):
    """Embed query và tìm kiếm trong vector_db."""
    if not query_text or embedding_model is None or vector_db is None or vector_db.index is None or vector_db.index.ntotal == 0:
        return np.array([]), np.array([]) # Trả về mảng rỗng
    st.write('B')
    query_embedding = embedding_model.encode(query_text, convert_to_numpy=True).astype('float32')
    st.write('C')
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
            if self.document_texts: 
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


    def search(self, query_text,
               primary_embedding_model, # Model cho dense retriever chính
               method='Kết hợp', k=20,
               # Tham số cho dense retriever phụ (nếu dùng)
               secondary_embedding_model=None,
               secondary_vector_db=None
              ):
        if not query_text: return []
        results = []
        indices_set = set()

        if method == 'Ngữ nghĩa':
            distances, indices = retrieve_relevant_chunks(query_text, primary_embedding_model, self.primary_vector_db, k=k)
            if indices is not None and len(indices) > 0:
                for i, idx_val in enumerate(indices):
                    idx = int(idx_val) # Đảm bảo idx là int
                    if 0 <= idx < len(self.documents) and idx not in indices_set:
                        results.append({
                            'doc': self.documents[idx],
                            'score': float(distances[i]), # L2 distance, nhỏ hơn là tốt hơn
                            'index': idx
                        })
                        indices_set.add(idx)
                results.sort(key=lambda x: x['score']) # Sắp xếp theo distance tăng dần

        elif method == 'Từ khóa':
            if self.bm25:
                tokenized_query = self._tokenize_vi(query_text)
                if tokenized_query:
                    bm25_scores = self.bm25.get_scores(tokenized_query)
                    bm25_scored_indices = [(bm25_scores[i], i) for i in range(len(bm25_scores)) if bm25_scores[i] > 0]
                    bm25_scored_indices.sort(key=lambda x: x[0], reverse=True) # BM25 score, lớn hơn là tốt hơn
                    for score, idx_val in bm25_scored_indices[:k]:
                        idx = int(idx_val)
                        if idx not in indices_set:
                            results.append({
                                'doc': self.documents[idx],
                                'score': float(score),
                                'index': idx
                            })
                            indices_set.add(idx)

        elif method == 'Kết hợp':
            rank_lists_with_weights = []

            # --- 1. Primary Dense Search (Dense 1) ---
            _, vec1_indices_np = retrieve_relevant_chunks(
                query_text, primary_embedding_model, self.primary_vector_db, k=config.HYBRID_K_PER_QUERY
            )
            vec1_indices_list = []
            if vec1_indices_np is not None and len(vec1_indices_np) > 0:
                vec1_indices_list = [int(i) for i in vec1_indices_np.flatten().tolist() if isinstance(i, (int, np.integer))]

            # --- 2. Secondary Dense Search (Dense 2) - nếu được kích hoạt và cung cấp đủ tham số ---
            vec2_indices_list = []
            run_secondary_dense = (config.HYBRID_MODE == "2_dense_1_sparse" and
                                   secondary_embedding_model is not None and
                                   secondary_vector_db is not None)
            if run_secondary_dense:
                _, vec2_indices_np = retrieve_relevant_chunks(
                    query_text, secondary_embedding_model, secondary_vector_db, k=config.HYBRID_K_PER_QUERY
                )
                if vec2_indices_np is not None and len(vec2_indices_np) > 0:
                    vec2_indices_list = [int(i) for i in vec2_indices_np.flatten().tolist() if isinstance(i, (int, np.integer))]

            # --- 3. BM25 Search (Sparse) ---
            bm25_indices_list = []
            if self.bm25:
                tokenized_query = self._tokenize_vi(query_text)
                if tokenized_query:
                    bm25_scores = self.bm25.get_scores(tokenized_query)
                    bm25_scored_indices = [(bm25_scores[i], i) for i in range(len(bm25_scores)) if bm25_scores[i] > 0]
                    bm25_scored_indices.sort(key=lambda x: x[0], reverse=True)
                    bm25_indices_list = [int(idx) for _, idx in bm25_scored_indices[:config.HYBRID_K_PER_QUERY]]

            # --- Gán trọng số và thêm vào danh sách để fuse ---
            if run_secondary_dense: # Chế độ 2 dense + 1 sparse
                if vec1_indices_list:
                    rank_lists_with_weights.append((vec1_indices_list, config.DENSE1_WEIGHT_HYBRID_3COMP))
                if vec2_indices_list:
                    rank_lists_with_weights.append((vec2_indices_list, config.DENSE2_WEIGHT_HYBRID_3COMP))
                if bm25_indices_list:
                    rank_lists_with_weights.append((bm25_indices_list, config.SPARSE_WEIGHT_HYBRID_3COMP))
            else: # Chế độ 1 dense + 1 sparse (mặc định)
                if vec1_indices_list:
                    rank_lists_with_weights.append((vec1_indices_list, config.DENSE_WEIGHT_HYBRID_2COMP))
                if bm25_indices_list:
                    rank_lists_with_weights.append((bm25_indices_list, config.SPARSE_WEIGHT_HYBRID_2COMP))

            # --- 4. Rank Fusion (RRF) ---
            fused_indices = []
            fused_scores_dict = {}
            if rank_lists_with_weights:
                fused_indices, fused_scores_dict = self._rank_fusion_indices(
                    rank_lists_with_weights,
                    rrf_k_constant=config.RRF_K
                )

            # --- 5. Lấy Top K Documents từ kết quả RRF ---
            # Score từ RRF, lớn hơn là tốt hơn
            for rank, idx_val in enumerate(fused_indices):
                idx = int(idx_val)
                if len(results) >= k: break
                if 0 <= idx < len(self.documents) and idx not in indices_set:
                    score = fused_scores_dict.get(idx, 0.0)
                    results.append({'doc': self.documents[idx], 'score': float(score), 'index': idx})
                    indices_set.add(idx)
        else:
            # print(f"Phương thức truy vấn không hợp lệ: {method}")
            return []
        return results

    def _rank_fusion_indices(self, rank_lists_with_weights, rrf_k_constant=config.RRF_K):
        """Thực hiện RRF để kết hợp các danh sách rank."""
        fused_scores = {}
        for rank_list, weight in rank_lists_with_weights:
            for rank, doc_index in enumerate(rank_list):
                 if isinstance(doc_index, (int, np.integer)):
                     rank_ = rank + 1
                     score_contribution = weight * (1 / (rank_ + rrf_k_constant))
                     fused_scores[doc_index] = fused_scores.get(doc_index, 0) + score_contribution
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
                if not hasattr(self.bm25, 'get_scores'):
                    self.bm25 = None
                    return False
                return True
            except Exception as e:
                self.bm25 = None
                return False
        else: return False