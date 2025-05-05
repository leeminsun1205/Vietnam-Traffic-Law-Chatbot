# retriever.py
import re
import os
import numpy as np
import pickle
import config
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
from config import VIETNAMESE_STOP_WORDS
import logging # Thêm logging
import streamlit as st

# --- Hàm retrieve_relevant_chunks gốc ---
def retrieve_relevant_chunks(query_text, embedding_model, vector_db, k=5):
    """Embed query và tìm kiếm trong vector_db."""
    if not query_text or embedding_model is None or vector_db is None or vector_db.index is None or vector_db.index.ntotal == 0:
        logging.warning("Vector search skipped due to missing query, model, or empty DB.")
        return np.array([]), np.array([]) # Trả về mảng rỗng
    st.write(embedding_model)
    try:
        st.write(query_text)
        query_embedding = embedding_model.encode(query_text, convert_to_numpy=True).astype('float32')
        st.write('dddđ')
        st.write(query_embedding)
        st.write('cccccc')
        # Đảm bảo query_embedding là 2D array
        if query_embedding.ndim == 1:
            st.write('hahaha')
            query_embedding = np.expand_dims(query_embedding, axis=0)

        # Kiểm tra dimension khớp với index
        if query_embedding.shape[1] != vector_db.embedding_dimension:
            st.write('kkakakak')
            logging.error(f"Query embedding dimension ({query_embedding.shape[1]}) mismatch with index dimension ({vector_db.embedding_dimension}).")
            return np.array([]), np.array([])
        st.write('hahaha')
        actual_k = min(k, vector_db.index.ntotal)
        if actual_k <= 0:
            return np.array([]), np.array([])
        st.write('query')
        st.write(query_embedding)
        st.write('query')
        distances, indices = vector_db.search(query_embedding, k=actual_k)
        # st.write(distances, indices)
        # search trả về 2D arrays, lấy phần tử đầu tiên
        return distances, indices
    except Exception as e:
        logging.exception(f"Error during vector search for query '{query_text[:50]}...': {e}")
        return np.array([]), np.array([]) # Trả về mảng rỗng khi có lỗi


class HybridRetriever:
    """Kết hợp Vector Search (Dense), BM25 Search (Sparse), hoặc cả hai (Hybrid)."""
    def __init__(self, vector_db, bm25_save_path):
        self.vector_db = vector_db
        # Lấy documents từ vector_db nếu có, nếu không khởi tạo list rỗng
        self.documents = getattr(vector_db, 'documents', [])
        # Lấy text từ documents, xử lý trường hợp document không phải dict hoặc thiếu key 'text'
        self.document_texts = [doc.get('text', '') if isinstance(doc, dict) else '' for doc in self.documents]
        self.bm25_index_path = bm25_save_path
        self.bm25 = None

        # Khởi tạo hoặc tải BM25
        if not self.load_bm25(self.bm25_index_path):
            if self.document_texts: # Chỉ khởi tạo nếu có text
                self._initialize_bm25()
                if self.bm25:
                    self.save_bm25(self.bm25_index_path)
            else:
                logging.warning("No document texts found to initialize BM25.")

    def _initialize_bm25(self):
        """Khởi tạo BM25 index."""
        if not self.document_texts:
            logging.warning("Cannot initialize BM25: No document texts available.")
            return
        logging.info("Initializing BM25 index...")
        try:
            # Tokenize corpus, bỏ qua các document rỗng
            tokenized_corpus = [self._tokenize_vi(text) for text in self.document_texts if text and isinstance(text, str)]
            if not any(tokenized_corpus): # Kiểm tra xem có token nào không
                 logging.warning("BM25 initialization failed: No valid tokens found after tokenization.")
                 self.bm25 = None
            else:
                 self.bm25 = BM25Okapi(tokenized_corpus)
                 logging.info(f"BM25 index initialized with {len(tokenized_corpus)} documents.")
        except Exception as e:
            logging.exception(f"Error initializing BM25: {e}")
            self.bm25 = None

    def _tokenize_vi(self, text):
        """Tokenize dùng pyvi.ViTokenizer, làm sạch, bỏ stop words."""
        if not text or not isinstance(text, str): return []
        try:
            text_lower = text.lower()
            # Sử dụng word_tokenize=True để tách từ tốt hơn trước khi split
            tokenized_text = ViTokenizer.tokenize(text_lower)
            word_tokens = tokenized_text.split()

            final_tokens = []
            for token in word_tokens:
                # Giữ lại dấu gạch dưới, loại bỏ các ký tự đặc biệt khác
                cleaned_token = re.sub(r'[^\w\s_]', '', token, flags=re.UNICODE).strip()
                # Kiểm tra token hợp lệ và không nằm trong stop words
                if cleaned_token and cleaned_token not in VIETNAMESE_STOP_WORDS:
                    final_tokens.append(cleaned_token)
            return final_tokens
        except Exception as e:
            logging.error(f"Error tokenizing text: '{text[:50]}...'. Error: {e}")
            return []


    def search(self, query_text, embedding_model, method='hybrid', k=20):
        """
        Thực hiện tìm kiếm theo phương thức được chỉ định.

        Args:
            query_text (str): Câu truy vấn.
            embedding_model: Mô hình embedding đã tải.
            method (str): Phương thức tìm kiếm ('dense', 'sparse', 'hybrid'). Mặc định là 'hybrid'.
            k (int): Số lượng kết quả mong muốn cuối cùng.

        Returns:
            list: Danh sách các dict {'doc': document, 'score': score, 'index': index}.
                  Score có thể là distance (dense), BM25 score (sparse), hoặc RRF score (hybrid).
                  Danh sách được sắp xếp theo score giảm dần (hybrid, sparse) hoặc distance tăng dần (dense).
        """
        if not query_text:
            logging.warning("Search skipped: Empty query text.")
            return []

        results = []
        indices_set = set() # Dùng để tránh trùng lặp index khi lấy document

        if method == 'dense':
            logging.debug(f"Performing DENSE search for: '{query_text[:50]}...' with k={k}")
            distances, indices = retrieve_relevant_chunks(query_text, embedding_model, self.vector_db, k=k)
            st.write(distances, indices)
            # st.write(indices.type)
            st.write(len(indices))
            st.write('xin chào nè')
            if indices is not None and len(indices) > 0:
                st.write('xin chào')
                for i, idx in enumerate(indices):
                     # Đảm bảo idx hợp lệ
                    if isinstance(idx, (int, np.integer)) and 0 <= idx < len(self.documents) and idx not in indices_set:
                        results.append({
                            'doc': self.documents[idx],
                            'score': float(distances[i]), # Lưu distance làm score
                            'index': int(idx)
                        })
                        indices_set.add(idx)
                # Sắp xếp theo distance tăng dần (score nhỏ hơn là tốt hơn)
                results.sort(key=lambda x: x['score'])
            st.write('khóc luôn rồi')
            logging.debug(f"Dense search found {len(results)} results.")


        elif method == 'sparse':
            logging.debug(f"Performing SPARSE search for: '{query_text[:50]}...' with k={k}")
            if self.bm25:
                tokenized_query = self._tokenize_vi(query_text)
                if tokenized_query:
                    try:
                        bm25_scores = self.bm25.get_scores(tokenized_query)
                        # Lấy index và score của các document có score > 0
                        bm25_scored_indices = [(bm25_scores[i], i) for i in range(len(bm25_scores)) if bm25_scores[i] > 0]
                        # Sắp xếp theo score giảm dần
                        bm25_scored_indices.sort(key=lambda x: x[0], reverse=True)
                        # Lấy top K kết quả
                        for score, idx in bm25_scored_indices[:k]:
                             if idx not in indices_set: # Chỉ thêm nếu chưa có
                                results.append({
                                    'doc': self.documents[idx],
                                    'score': float(score), # Lưu BM25 score
                                    'index': int(idx)
                                })
                                indices_set.add(idx)
                        logging.debug(f"Sparse search found {len(results)} results.")
                    except Exception as e:
                         logging.exception(f"Error during BM25 search: {e}")
                else:
                     logging.warning("Sparse search skipped: Could not tokenize query.")
            else:
                 logging.warning("Sparse search skipped: BM25 index not available.")


        elif method == 'hybrid':
            logging.debug(f"Performing HYBRID search for: '{query_text[:50]}...' with final_k={k}")
            # --- 1. Vector Search (Dense) ---
            vec_distances, vec_indices = retrieve_relevant_chunks(
                query_text, embedding_model, self.vector_db, k=config.VECTOR_K_PER_QUERY # Lấy nhiều hơn cho fusion
            )
            st.write(vec_indices)
            vec_indices_list = []
            if vec_indices is not None and len(vec_indices) > 0:
                # Chuyển numpy array thành list các số nguyên
                vec_indices_list = [int(i) for i in vec_indices.flatten().tolist() if isinstance(i, (int, np.integer))]

            # --- 2. BM25 Search (Sparse) ---
            bm25_indices_list = []
            if self.bm25:
                tokenized_query = self._tokenize_vi(query_text)
                if tokenized_query:
                    try:
                        bm25_scores = self.bm25.get_scores(tokenized_query)
                        bm25_scored_indices = [(bm25_scores[i], i) for i in range(len(bm25_scores)) if bm25_scores[i] > 0]
                        bm25_scored_indices.sort(key=lambda x: x[0], reverse=True)
                        # Lấy index từ kết quả BM25, số lượng bằng vector_search_k
                        bm25_indices_list = [int(index) for score, index in bm25_scored_indices[:config.VECTOR_K_PER_QUERY]]
                    except Exception as e:
                         logging.exception(f"Error during BM25 part of hybrid search: {e}")

            # --- 3. Rank Fusion (RRF) ---
            rank_lists_to_fuse = []
            if vec_indices_list: rank_lists_to_fuse.append(vec_indices_list)
            if bm25_indices_list: rank_lists_to_fuse.append(bm25_indices_list)

            fused_indices = []
            fused_scores_dict = {}
            if rank_lists_to_fuse:
                fused_indices, fused_scores_dict = self._rank_fusion_indices(rank_lists_to_fuse, k=config.RRF_K) # Dùng RRF_K từ config
                st.write(fused_indices, fused_scores_dict)
                st.write('BÁ CHÚ')
                logging.debug(f"Hybrid search fused {len(fused_indices)} indices.")
            elif vec_indices_list: # Fallback: Nếu chỉ có kết quả dense
                 fused_indices = vec_indices_list
                 # Tạo dict score giả dựa trên rank (score cao hơn cho rank thấp hơn)
                 fused_scores_dict = {idx: 1.0 / (rank + 1) for rank, idx in enumerate(fused_indices)}
                 logging.debug("Hybrid search using dense results only (fallback).")
            else:
                 logging.debug("Hybrid search found no results from dense or sparse.")


            # --- 4. Get Top K Documents ---
            for rank, idx in enumerate(fused_indices):
                 # Lấy top K kết quả cuối cùng
                if len(results) >= k: break
                # Đảm bảo idx hợp lệ và chưa được thêm
                if isinstance(idx, (int, np.integer)) and 0 <= idx < len(self.documents) and idx not in indices_set:
                    score = fused_scores_dict.get(idx, 1.0 / (rank + 1 + config.RRF_K)) # Lấy score RRF hoặc rank-based
                    results.append({'doc': self.documents[idx], 'score': float(score), 'index': int(idx)})
                    indices_set.add(idx)
            logging.debug(f"Hybrid search returning {len(results)} final results.")


        else:
            logging.error(f"Invalid search method specified: {method}")
            return []

        return results # Trả về danh sách kết quả đã được sắp xếp (hoặc không cần sắp xếp nếu rerank)

    def _rank_fusion_indices(self, rank_lists, k=60):
        """Thực hiện RRF để kết hợp các danh sách rank."""
        fused_scores = {}
        for rank_list in rank_lists:
            if not isinstance(rank_list, list): continue
            for rank, doc_index in enumerate(rank_list):
                 if isinstance(doc_index, (int, np.integer)):
                     rank_ = rank + 1
                     fused_scores[doc_index] = fused_scores.get(doc_index, 0) + (1 / (rank_ + k))

        sorted_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return sorted_indices, fused_scores

    def save_bm25(self, filepath):
        """Lưu trạng thái BM25 vào file pickle."""
        if self.bm25:
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(self.bm25, f)
                logging.info(f"BM25 index saved to {filepath}")
            except Exception as e:
                logging.exception(f"Error saving BM25 index to {filepath}: {e}")

    def load_bm25(self, filepath):
        """Tải trạng thái BM25 từ file pickle."""
        if os.path.exists(filepath):
             try:
                 with open(filepath, 'rb') as f:
                     self.bm25 = pickle.load(f)
                 # Kiểm tra xem object load được có phương thức cần thiết không
                 if not hasattr(self.bm25, 'get_scores'):
                     logging.error(f"Loaded object from {filepath} is not a valid BM25 index.")
                     self.bm25 = None; return False
                 logging.info(f"BM25 index loaded successfully from {filepath}")
                 return True
             except Exception as e:
                 logging.exception(f"Error loading BM25 index from {filepath}: {e}")
                 self.bm25 = None
                 return False
        else:
            logging.info(f"BM25 index file not found at {filepath}. Will attempt to initialize.")
            return False