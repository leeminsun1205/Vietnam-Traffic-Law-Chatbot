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
        # st.warning(f"Dimension mismatch: Query emb {query_embedding.shape[1]} vs DB emb {vector_db.embedding_dimension}")
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
        self.vector_db = vector_db # Đây là vector_db chính, dùng cho BM25 và là một trong các nguồn dense
        self.documents = getattr(vector_db, 'documents', [])
        self.document_texts = [doc.get('text', '') if isinstance(doc, dict) else '' for doc in self.documents]
        self.bm25_index_path = bm25_save_path
        self.bm25 = None

        if not self.load_bm25(self.bm25_index_path):
            if self.document_texts:
                self._initialize_bm25()
                if self.bm25:
                    self.save_bm25(self.bm25_index_path)

    def _initialize_bm25(self):
        """Khởi tạo BM25 index."""
        if not self.document_texts: return
        try:
            tokenized_corpus = [self._tokenize_vi(text) for text in self.document_texts if text and isinstance(text, str)]
            if not any(tokenized_corpus):
                 self.bm25 = None
            else:
                 self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            # st.error(f"Lỗi khởi tạo BM25: {e}")
            self.bm25 = None

    def _tokenize_vi(self, text):
        """Tokenize dùng pyvi.ViTokenizer, làm sạch, bỏ stop words."""
        if not text or not isinstance(text, str): return []
        try:
            text_lower = text.lower()
            tokenized_text = ViTokenizer.tokenize(text_lower)
            word_tokens = tokenized_text.split()
            final_tokens = [
                re.sub(r'[^\w\s_]', '', token, flags=re.UNICODE).strip()
                for token in word_tokens
                if token and token not in VIETNAMESE_STOP_WORDS
            ]
            return [token for token in final_tokens if token] # Loại bỏ token rỗng sau khi clean
        except Exception as e:
            # st.warning(f"Lỗi tokenize: {e}")
            return []

    def search(self, query_text, primary_embedding_model_object, method='hybrid', k=20, additional_dense_sources=None):
        """
        Thực hiện tìm kiếm.
        Args:
            query_text (str): Câu truy vấn.
            primary_embedding_model_object: Model embedding cho vector_db chính.
            method (str): 'dense', 'sparse', hoặc 'hybrid'.
            k (int): Số lượng kết quả cuối cùng trả về.
            additional_dense_sources (list, optional): 
                Danh sách các tuple (embedding_model_object, vector_db_object) cho các nguồn dense phụ.
        Returns:
            list: Danh sách các dictionary kết quả.
        """
        if not query_text: return []
        
        results = []
        indices_set = set() 
        if additional_dense_sources is None:
            additional_dense_sources = []

        if method == 'dense':
            distances, indices = retrieve_relevant_chunks(query_text, primary_embedding_model_object, self.vector_db, k=k)
            if indices is not None and len(indices) > 0:
                for i, idx in enumerate(indices):
                    idx_int = int(idx)
                    if isinstance(idx_int, int) and 0 <= idx_int < len(self.documents) and idx_int not in indices_set:
                        results.append({
                            'doc': self.documents[idx_int],
                            'score': float(distances[i]), 
                            'index': idx_int
                        })
                        indices_set.add(idx_int)
                results.sort(key=lambda x: x['score']) # Sắp xếp theo distance tăng dần

        elif method == 'sparse':
            if self.bm25:
                tokenized_query = self._tokenize_vi(query_text)
                if tokenized_query:    
                    bm25_scores = self.bm25.get_scores(tokenized_query)
                    bm25_scored_indices = [(bm25_scores[i], i) for i in range(len(bm25_scores)) if bm25_scores[i] > 0]
                    # Sắp xếp theo score giảm dần
                    bm25_scored_indices.sort(key=lambda x: x[0], reverse=True) 
                    for score, idx in bm25_scored_indices[:k]: # Lấy top k
                        if idx not in indices_set: 
                            results.append({
                                'doc': self.documents[idx],
                                'score': float(score), 
                                'index': int(idx)
                            })
                            indices_set.add(idx)

        elif method == 'hybrid':
            rank_lists_to_fuse = []

            # 1. Vector Search (Dense) - Nguồn chính
            _, vec_indices_primary = retrieve_relevant_chunks(
                query_text, primary_embedding_model_object, self.vector_db, k=config.VECTOR_K_PER_QUERY
            )
            if vec_indices_primary is not None and len(vec_indices_primary) > 0:
                rank_lists_to_fuse.append([int(i) for i in vec_indices_primary.flatten().tolist() if isinstance(i, (int, np.integer))])

            # 2. BM25 Search (Sparse)
            if self.bm25:
                tokenized_query = self._tokenize_vi(query_text)
                if tokenized_query:
                    bm25_scores = self.bm25.get_scores(tokenized_query)
                    bm25_scored_indices = sorted(
                        [(bm25_scores[i], i) for i in range(len(bm25_scores)) if bm25_scores[i] > 0],
                        key=lambda x: x[0], reverse=True
                    )
                    if bm25_scored_indices:
                        rank_lists_to_fuse.append([int(idx) for _, idx in bm25_scored_indices[:config.VECTOR_K_PER_QUERY]])
            
            # 3. Vector Search (Dense) - Các nguồn phụ
            for emb_model_add, vdb_add in additional_dense_sources:
                if emb_model_add and vdb_add:
                    _, vec_indices_add = retrieve_relevant_chunks(
                        query_text, emb_model_add, vdb_add, k=config.VECTOR_K_PER_QUERY
                    )
                    if vec_indices_add is not None and len(vec_indices_add) > 0:
                         rank_lists_to_fuse.append([int(i) for i in vec_indices_add.flatten().tolist() if isinstance(i, (int, np.integer))])

            # 4. Rank Fusion (RRF)
            fused_indices = []
            fused_scores_dict = {} 
            if rank_lists_to_fuse:
                fused_indices, fused_scores_dict = self._rank_fusion_indices(rank_lists_to_fuse, k_constant_rrf=config.RRF_K)
            
            # 5. Get Top K Documents từ fused_indices
            for rank, idx_int in enumerate(fused_indices):
                if len(results) >= k: break # Đã đủ k kết quả
                if isinstance(idx_int, int) and 0 <= idx_int < len(self.documents) and idx_int not in indices_set:
                    # Score RRF càng cao càng tốt
                    score = fused_scores_dict.get(idx_int, 1.0 / (rank + 1 + config.RRF_K)) 
                    results.append({'doc': self.documents[idx_int], 'score': float(score), 'index': idx_int})
                    indices_set.add(idx_int)
        
        return results 

    def _rank_fusion_indices(self, rank_lists, k_constant_rrf=60):
        """
        Thực hiện Reciprocal Rank Fusion (RRF) để kết hợp các danh sách rank.
        Args:
            rank_lists (list of lists): Danh sách các danh sách ID tài liệu đã được xếp hạng.
            k_constant_rrf (int): Hằng số k trong công thức RRF.
        Returns:
            tuple: (sorted_indices, fused_scores_dict)
                   sorted_indices là danh sách ID tài liệu đã được fusion và sắp xếp.
                   fused_scores_dict là dict {doc_index: rrf_score}.
        """
        fused_scores = {}
        for single_rank_list in rank_lists:
            for rank, doc_index in enumerate(single_rank_list):
                 if isinstance(doc_index, (int, np.integer)): 
                     current_rank_for_rrf = rank + 1
                     fused_scores[doc_index] = fused_scores.get(doc_index, 0) + (1.0 / (current_rank_for_rrf + k_constant_rrf))

        sorted_indices_after_fusion = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return sorted_indices_after_fusion, fused_scores

    def save_bm25(self, filepath):
        """Lưu trạng thái BM25 vào file pickle."""
        if self.bm25:
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(self.bm25, f)
            except Exception as e:
                pass

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