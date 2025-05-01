# retriever.py
import re
import os
import numpy as np
import pickle
import config
from rank_bm25 import BM25Okapi
import streamlit as st
from config import VIETNAMESE_STOP_WORDS, VNCORENLP_SAVE_DIR 
# Import py_vncorenlp một cách an toàn
try:
    import py_vncorenlp
except ImportError:
    logging.warning("Thư viện py_vncorenlp chưa được cài đặt. BM25 sẽ dùng split().")
    py_vncorenlp = None

# --- Retrieval ---
def retrieve_relevant_chunks(query_text, embedding_model, vector_db, k=5):
    """Embed query và tìm kiếm trong vector_db."""
    try:
        query_embedding = embedding_model.encode(query_text, convert_to_numpy=True).astype('float32')
        distances, indices = vector_db.search(query_embedding, k=k)
        return distances, indices
    except Exception as e:
        return [], []
class HybridRetriever:
    """Kết hợp Vector Search và BM25 Search."""
    def __init__(self, vector_db, bm25_save_path):
        self.vector_db = vector_db
        self.documents = getattr(vector_db, 'documents', [])
        self.document_texts = [doc.get('text', '') for doc in self.documents] if self.documents else []
        self.bm25_index_path = bm25_save_path
        self.rdrsegmenter = None
        self.bm25 = None

        # Khởi tạo VnCoreNLP
        self._initialize_vncorenlp(VNCORENLP_SAVE_DIR)

        # Khởi tạo hoặc tải BM25
        if not self.load_bm25(self.bm25_index_path):
            self._initialize_bm25() # Khởi tạo nếu không tải được
            if self.bm25:
                 self.save_bm25(self.bm25_index_path)

    def _initialize_vncorenlp(self, vncorenlp_dir):
        """Khởi tạo VnCoreNLP."""
        if py_vncorenlp:
            try:
                if not os.path.exists(vncorenlp_dir): os.makedirs(vncorenlp_dir)
                # py_vncorenlp.download_model(save_dir=vncorenlp_dir) # Nên chạy riêng
                self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)
            except Exception as e:
                self.rdrsegmenter = None

    def _initialize_bm25(self):
        """Khởi tạo BM25 index."""
        if not self.documents:
            return
        try:
            tokenized_corpus = [self._simple_tokenize_vncorenlp(text) for text in self.document_texts]
            if not any(tokenized_corpus):
                 self.bm25 = None 
            else:
                 self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            self.bm25 = None

    def _simple_tokenize_vncorenlp(self, text):
        """Tokenize dùng VnCoreNLP (nếu có), làm sạch, bỏ stop words."""
        if not text or not isinstance(text, str): return []
        text_lower = text.lower()
        word_tokens = []
        if self.rdrsegmenter:
            try:
                segmented_sentences = self.rdrsegmenter.word_segment(text_lower)
                word_tokens = [word for sentence in segmented_sentences for word in sentence]
                st.write('TESTTT')
            except Exception as e:
                word_tokens = text_lower.split()
        else:
            word_tokens = text_lower.split()
        final_tokens = []
        for token in word_tokens:
            cleaned_token = re.sub(r'[^\w_]', '', token, flags=re.UNICODE)
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
             tokenized_query = self._simple_tokenize_vncorenlp(query_text)
             if tokenized_query:
                bm25_scores = self.bm25.get_scores(tokenized_query)
                bm25_scored_indices = [(score, i) for i, score in enumerate(bm25_scores) if score > 0]
                bm25_scored_indices.sort(key=lambda x: x[0], reverse=True)
                bm25_search_indices = [index for score, index in bm25_scored_indices[:vector_search_k]]

        # --- 3. Rank Fusion ---
        rank_lists_to_fuse = []
        if isinstance(vec_indices, (list, np.ndarray)) and len(vec_indices) > 0:
            rank_lists_to_fuse.append(vec_indices)
        if isinstance(bm25_search_indices, list) and len(bm25_search_indices) > 0:
            rank_lists_to_fuse.append(bm25_search_indices)

        fused_indices, fused_scores_dict = [], {}
        if rank_lists_to_fuse:
            fused_indices, fused_scores_dict = self._rank_fusion_indices(rank_lists_to_fuse, k=config.RRF_K) 
        elif isinstance(vec_indices, (list, np.ndarray)) and len(vec_indices) > 0:
             fused_indices = vec_indices.tolist() if isinstance(vec_indices, np.ndarray) else vec_indices

        # --- 4. Get Top K ---
        hybrid_results = []
        for rank, idx in enumerate(fused_indices[:final_k]):
            if isinstance(idx, (int, np.integer)) and 0 <= idx < len(self.documents):
                score = fused_scores_dict.get(idx, 1 / (rank + 1))
                hybrid_results.append({'doc': self.documents[idx], 'hybrid_score': score, 'index': idx})

        return hybrid_results

    def _rank_fusion_indices(self, rank_lists, k=60):
        """Thực hiện RRF."""
        fused_scores = {}
        for rank_list in rank_lists:
            if isinstance(rank_list, np.ndarray): rank_list = rank_list.tolist()
            if not isinstance(rank_list, list): continue
            for rank, doc_index in enumerate(rank_list):
                 if isinstance(doc_index, (int, np.integer)):
                     rank_ = rank + 1
                     fused_scores[doc_index] = fused_scores.get(doc_index, 0) + (1 / (rank_ + k))
        sorted_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return sorted_indices, fused_scores

    def save_bm25(self, filepath):
        """Lưu trạng thái BM25."""
        if self.bm25: 
            with open(filepath, 'wb') as f: 
                pickle.dump(self.bm25, f)

    def load_bm25(self, filepath):
        """Tải trạng thái BM25."""
        if os.path.exists(filepath):
            try:
                 with open(filepath, 'rb') as f: self.bm25 = pickle.load(f)
                 if not hasattr(self.bm25, 'get_scores'): raise ValueError("Invalid BM25 object")
                 return True
            except Exception as e:
                 self.bm25 = None; return False
        else:
            # logging.info(f"BM25 state file '{filepath}' not found. Will initialize.") # Giảm log
            return False