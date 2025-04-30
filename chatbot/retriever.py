# retriever.py
import re
import os
import logging
import numpy as np
import pickle
from rank_bm25 import BM25Okapi

# Cần import các thành phần phụ thuộc
# from vector_db import SimpleVectorDatabase # Không cần trực tiếp, nhận qua __init__
from utils import retrieve_relevant_chunks # Cần hàm này
from config import VIETNAMESE_STOP_WORDS, VNCORENLP_SAVE_DIR # Lấy stop words và đường dẫn từ config

# Import py_vncorenlp một cách an toàn
try:
    import py_vncorenlp
except ImportError:
    logging.warning("Thư viện py_vncorenlp chưa được cài đặt. BM25 sẽ dùng split().")
    py_vncorenlp = None

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
            # logging.info("Initializing VnCoreNLP for Retriever...") # Giảm log
            try:
                if not os.path.exists(vncorenlp_dir): os.makedirs(vncorenlp_dir)
                # py_vncorenlp.download_model(save_dir=vncorenlp_dir) # Nên chạy riêng
                self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)
                logging.info("Retriever: VnCoreNLP (wseg) initialized.")
            except Exception as e:
                logging.error(f"Retriever: FAILED to initialize VnCoreNLP: {e}. BM25 will use simple split.")
                self.rdrsegmenter = None
        else:
            logging.warning("Retriever: py_vncorenlp not available. BM25 uses simple split.")

    def _initialize_bm25(self):
        """Khởi tạo BM25 index."""
        if not self.documents:
            logging.error("Retriever: No documents to initialize BM25.")
            return
        logging.info("Retriever: Initializing BM25 index...")
        try:
            tokenized_corpus = [self._simple_tokenize_vncorenlp(text) for text in self.document_texts]
            # Kiểm tra xem corpus có rỗng không sau khi tokenize
            if not any(tokenized_corpus): # Check if all lists inside are empty
                 logging.warning("Retriever: Corpus is empty after tokenization. BM25 index will be empty.")
                 self.bm25 = None # Không khởi tạo nếu corpus rỗng
            else:
                 self.bm25 = BM25Okapi(tokenized_corpus)
                 logging.info(f"Retriever: BM25 index initialized successfully with {len(tokenized_corpus)} documents.")
        except Exception as e:
            logging.error(f"Retriever: Error initializing BM25: {e}")
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
            except Exception as e:
                 logging.warning(f"Retriever: VnCoreNLP failed for text: '{text_lower[:50]}...'. Fallback. Error: {e}")
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
            logging.warning("Retriever: Vector DB not ready. Cannot hybrid search.")
            return []

        # --- 1. Vector Search ---
        # logging.info("Performing vector search...") # Giảm log
        vec_distances, vec_indices = retrieve_relevant_chunks(
            query_text, embedding_model, self.vector_db, k=vector_search_k
        )

        # --- 2. BM25 Search ---
        bm25_search_indices = []
        if self.bm25: # Chỉ chạy nếu BM25 đã khởi tạo thành công
             # logging.info("Performing BM25 search...") # Giảm log
             tokenized_query = self._simple_tokenize_vncorenlp(query_text)
             if tokenized_query:
                 try:
                     bm25_scores = self.bm25.get_scores(tokenized_query)
                     bm25_scored_indices = [(score, i) for i, score in enumerate(bm25_scores) if score > 0]
                     bm25_scored_indices.sort(key=lambda x: x[0], reverse=True)
                     bm25_search_indices = [index for score, index in bm25_scored_indices[:vector_search_k]]
                     # logging.info(f"BM25 found {len(bm25_search_indices)} results > 0.") # Giảm log
                 except Exception as bm25_err:
                      logging.error(f"Retriever: BM25 scoring error: {bm25_err}")
             # else: logging.warning("Retriever: Query empty after tokenization for BM25.") # Giảm log
        else:
             logging.warning("Retriever: BM25 not available, skipping sparse search.")

        # --- 3. Rank Fusion ---
        rank_lists_to_fuse = []
        if isinstance(vec_indices, (list, np.ndarray)) and len(vec_indices) > 0:
            rank_lists_to_fuse.append(vec_indices)
        if isinstance(bm25_search_indices, list) and len(bm25_search_indices) > 0:
            rank_lists_to_fuse.append(bm25_search_indices)

        fused_indices, fused_scores_dict = [], {}
        if rank_lists_to_fuse:
            fused_indices, fused_scores_dict = self._rank_fusion_indices(rank_lists_to_fuse, k=config.RRF_K) # Lấy K từ config
            # logging.info(f"Rank Fusion: {len(fused_indices)} unique docs.") # Giảm log
        elif isinstance(vec_indices, (list, np.ndarray)) and len(vec_indices) > 0:
             logging.warning("Retriever: Only vector results available, no fusion.")
             fused_indices = vec_indices.tolist() if isinstance(vec_indices, np.ndarray) else vec_indices
        # else: logging.warning("Retriever: No results from vector or BM25.") # Giảm log

        # --- 4. Get Top K ---
        hybrid_results = []
        for rank, idx in enumerate(fused_indices[:final_k]):
            if isinstance(idx, (int, np.integer)) and 0 <= idx < len(self.documents):
                score = fused_scores_dict.get(idx, 1 / (rank + 1))
                hybrid_results.append({'doc': self.documents[idx], 'hybrid_score': score, 'index': idx})
            else: logging.warning(f"Retriever: Invalid fused index {idx}, skipping.")

        # logging.info(f"Hybrid search returning {len(hybrid_results)} results.") # Giảm log
        return hybrid_results

    def _rank_fusion_indices(self, rank_lists, k=60):
        """Thực hiện RRF."""
        # ... (Copy implementation) ...
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
        # ... (Copy implementation) ...
        if self.bm25:
             # logging.info(f"Saving BM25 state to '{filepath}'...") # Giảm log
             try:
                 with open(filepath, 'wb') as f: pickle.dump(self.bm25, f)
                 # logging.info("BM25 state saved.") # Giảm log
             except Exception as e: logging.error(f"Error saving BM25 state: {e}")
        # else: logging.warning("BM25 not initialized. Cannot save.") # Giảm log

    def load_bm25(self, filepath):
        """Tải trạng thái BM25."""
        # ... (Copy implementation) ...
        if os.path.exists(filepath):
            # logging.info(f"Loading BM25 state from '{filepath}'...") # Giảm log
            try:
                 with open(filepath, 'rb') as f: self.bm25 = pickle.load(f)
                 # logging.info("BM25 state loaded successfully.") # Giảm log
                 if not hasattr(self.bm25, 'get_scores'): raise ValueError("Invalid BM25 object")
                 return True
            except Exception as e:
                 logging.error(f"Error loading BM25 state from {filepath}: {e}. Will re-initialize.")
                 self.bm25 = None; return False
        else:
            # logging.info(f"BM25 state file '{filepath}' not found. Will initialize.") # Giảm log
            return False