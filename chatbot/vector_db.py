# vector_db.py
import faiss
import numpy as np
import json
import os
import streamlit as st
class SimpleVectorDatabase:
    """Quản lý Faiss index và các document chunks tương ứng."""
    def __init__(self):
        self.index = None
        self.documents = []
        self.embedding_dimension = None

    def add_documents(self, valid_chunks, embeddings_array):
        """Thêm documents và embeddings vào database."""
        if not valid_chunks or embeddings_array is None:
            return
        num_docs = len(valid_chunks)
        num_embeddings = embeddings_array.shape[0]
        if num_docs != num_embeddings:
            return

        if self.index is None:
            self.embedding_dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
        if embeddings_array.shape[1] != self.embedding_dimension:
            return

        self.index.add(embeddings_array)
        self.documents.extend(valid_chunks)

    def search(self, query_embedding, k=5):
        """Tìm kiếm k documents gần nhất với query_embedding."""
        st.write('START')
        if self.index is None or self.index.ntotal == 0:
            return [], []
        st.write('MIDDLE')
        if query_embedding is None:
            return [], []
        st.write('END')
        st.write('ffff')
        st.write(query_embedding)
        # query_embedding_array = np.array([query_embedding]).astype('float32')
        query_embedding_array = query_embedding
        st.write(query_embedding_array)
        st.write('bbbbb')
        if query_embedding_array.shape[1] != self.embedding_dimension:
            st.write('CHECK')
            return [], []
        st.write('END2')
        st.write('aaaa')
        st.write(query_embedding_array)
        st.write('aaaa')
        
        actual_k = min(k, self.index.ntotal)
        if actual_k == 0: return [], []
        st.write('aaaa')
        st.write(actual_k)
        st.write('aaaa')
        distances, indices = self.index.search(query_embedding_array, actual_k)
        st.write(distances, indices)
        return distances[0], indices[0]

    def save(self, filepath_prefix):
        """Lưu trạng thái database (index, docs, meta)."""
        if self.index is None:
            return
        index_path = f"{filepath_prefix}_faiss.index"
        docs_path = f"{filepath_prefix}_docs.json"
        meta_path = f"{filepath_prefix}_meta.json"
        try:
            faiss.write_index(self.index, index_path)
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            metadata = {'embedding_dimension': self.embedding_dimension, 'doc_count': len(self.documents)}
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
        except Exception:
            pass

    def load(self, filepath_prefix):
        """Tải trạng thái database từ các file đã lưu."""
        index_path = f"{filepath_prefix}_faiss.index"
        docs_path = f"{filepath_prefix}_docs.json"
        meta_path = f"{filepath_prefix}_meta.json"
        if not all(os.path.exists(p) for p in [index_path, docs_path, meta_path]):
            return False
        try:
            self.index = faiss.read_index(index_path)
            with open(docs_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.embedding_dimension = metadata.get('embedding_dimension')
            loaded_doc_count = metadata.get('doc_count')

            if self.embedding_dimension is None or loaded_doc_count is None:
                raise ValueError("Metadata file thiếu thông tin cần thiết.")
            if len(self.documents) != self.index.ntotal or len(self.documents) != loaded_doc_count:
                pass  

            return True
        except Exception:
            self.index = None
            self.documents = []
            self.embedding_dimension = None
            return False
