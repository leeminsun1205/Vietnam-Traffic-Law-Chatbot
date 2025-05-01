# vector_db.py
import faiss
import numpy as np
import json
import os
import logging
import streamlit as st
class SimpleVectorDatabase:
    """Quản lý Faiss index và các document chunks tương ứng."""
    def __init__(self):
        self.index = None
        self.documents = []
        self.embedding_dimension = None
        logging.info("SimpleVectorDatabase initialized.")

    def add_documents(self, valid_chunks, embeddings_array):
        """Thêm documents và embeddings vào database."""
        if not valid_chunks or embeddings_array is None:
            logging.warning("Không có chunks hoặc embeddings hợp lệ để thêm vào DB.")
            return
        num_docs = len(valid_chunks)
        num_embeddings = embeddings_array.shape[0]
        if num_docs != num_embeddings:
            logging.error(f"Lỗi: Số lượng document ({num_docs}) và embedding ({num_embeddings}) không khớp.")
            return

        logging.info(f"Đang thêm {num_docs} documents vào database...")
        if self.index is None:
            self.embedding_dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            logging.info(f"Đã tạo Faiss index (IndexFlatL2) với chiều: {self.embedding_dimension}")
        if embeddings_array.shape[1] != self.embedding_dimension:
            logging.error(f"Lỗi: Chiều embedding không khớp. Mong đợi {self.embedding_dimension}, nhận được {embeddings_array.shape[1]}.")
            return

        self.index.add(embeddings_array)
        self.documents.extend(valid_chunks) # Lưu trữ các dict chunk gốc
        logging.info(f"Đã thêm {num_embeddings} embeddings. Tổng số docs: {len(self.documents)}. Tổng index: {self.index.ntotal}")

    def search(self, query_embedding, k=5):
        """Tìm kiếm k documents gần nhất với query_embedding."""
        if self.index is None or self.index.ntotal == 0:
            logging.warning("DB trống hoặc chưa khởi tạo. Không thể tìm kiếm.")
            return [], []
        if query_embedding is None:
            logging.warning("Query embedding là None. Không thể tìm kiếm.")
            return [], []

        query_embedding_array = np.array([query_embedding]).astype('float32')
        if query_embedding_array.shape[1] != self.embedding_dimension:
            logging.error(f"Lỗi: Chiều query embedding không khớp. Mong đợi {self.embedding_dimension}, nhận được {query_embedding_array.shape[1]}.")
            return [], []

        actual_k = min(k, self.index.ntotal)
        if actual_k == 0: return [], []
        # logging.info(f"Đang tìm kiếm top {actual_k} documents...") # Giảm bớt log
        distances, indices = self.index.search(query_embedding_array, actual_k)
        return distances[0], indices[0]

    def save(self, filepath_prefix):
        """Lưu trạng thái database (index, docs, meta)."""
        if self.index is None:
            logging.warning("DB trống. Không có gì để lưu.")
            return
        index_path = f"{filepath_prefix}_faiss.index"
        docs_path = f"{filepath_prefix}_docs.json"
        meta_path = f"{filepath_prefix}_meta.json"
        try:
            logging.info(f"Đang lưu Faiss index vào '{index_path}'...")
            faiss.write_index(self.index, index_path)
            logging.info(f"Đang lưu danh sách documents vào '{docs_path}'...")
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            logging.info(f"Đang lưu metadata vào '{meta_path}'...")
            metadata = {'embedding_dimension': self.embedding_dimension, 'doc_count': len(self.documents)}
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
            logging.info("Lưu database thành công.")
        except Exception as e:
            logging.error(f"Lỗi khi lưu database tại prefix '{filepath_prefix}': {e}")

    def load(self, filepath_prefix):
        """Tải trạng thái database từ các file đã lưu."""
        index_path = f"{filepath_prefix}_faiss.index"
        docs_path = f"{filepath_prefix}_docs.json"
        meta_path = f"{filepath_prefix}_meta.json"
        if not all(os.path.exists(p) for p in [index_path, docs_path, meta_path]):
            logging.warning(f"Không tìm thấy đủ file để tải database từ prefix '{filepath_prefix}'.")
            return False
        try:
            logging.info(f"Đang tải database từ prefix '{filepath_prefix}'...")
            self.index = faiss.read_index(index_path)
            with open(docs_path, 'r', encoding='utf-8') as f: self.documents = json.load(f)
            with open(meta_path, 'r', encoding='utf-8') as f: metadata = json.load(f)
            self.embedding_dimension = metadata.get('embedding_dimension')
            loaded_doc_count = metadata.get('doc_count')

            if self.embedding_dimension is None or loaded_doc_count is None:
                raise ValueError("Metadata file thiếu thông tin cần thiết.")
            if len(self.documents) != self.index.ntotal or len(self.documents) != loaded_doc_count:
                logging.warning("Số lượng docs/index không khớp khi tải database state.")

            logging.info(f"Tải database thành công. Index: {self.index.ntotal}, Docs: {len(self.documents)}, Dim: {self.embedding_dimension}")
            return True
        except Exception as e:
            logging.error(f"Lỗi khi tải database từ prefix '{filepath_prefix}': {e}")
            self.index = None
            self.documents = []
            self.embedding_dimension = None
            return False