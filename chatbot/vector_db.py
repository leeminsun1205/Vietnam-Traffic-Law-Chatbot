# vector_db.py
import faiss
import json
import os
class SimpleVectorDatabase:
    """Quản lý Faiss index và các document chunks tương ứng."""
    def __init__(self):
        self.index = None
        self.documents = []
        self.embedding_dimension = None

    def add_documents(self, valid_chunks, embeddings_array):
        """Thêm documents và embeddings vào database."""
        if not valid_chunks or embeddings_array is None: return

        num_docs = len(valid_chunks)
        num_embeddings = embeddings_array.shape[0]
        if num_docs != num_embeddings: return

        if self.index is None:
            self.embedding_dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(self.embedding_dimension)        # Tìm kiếm chính xác, không tăng tốc (duyệt toàn bộ)
            # self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)  # Dùng phân cụm IVF để lọc trước, nhanh hơn nhưng không tuyệt đối
            # self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dimension, nlist, m, nbits)  # Dùng phân cụm + lượng hóa PQ, rất nhanh nhưng mất độ chính xác
            # self.index = faiss.IndexHNSWFlat(self.embedding_dimension, M)  # Dùng đồ thị HNSW cho tìm kiếm gần đúng, nhanh và hiệu quả trên tập lớn

        if embeddings_array.shape[1] != self.embedding_dimension: return

        self.index.add(embeddings_array)
        self.documents.extend(valid_chunks)

    def search(self, query_embedding, k=5):
        """Tìm kiếm k documents gần nhất với query_embedding."""
        if self.index is None or self.index.ntotal == 0: return [], []
        if query_embedding is None: return [], []

        query_embedding_array = query_embedding.astype('float32')
        if query_embedding_array.shape[1] != self.embedding_dimension: return [], []

        actual_k = min(k, self.index.ntotal)
        if actual_k == 0: return [], []
        distances, indices = self.index.search(query_embedding_array, actual_k)
        return distances[0], indices[0]

    def save(self, filepath_prefix):
        """Lưu trạng thái database (index, docs, meta)."""
        if self.index is None: return
        index_path = f"{filepath_prefix}_faiss.index"
        docs_path = f"{filepath_prefix}_docs.json"
        meta_path = f"{filepath_prefix}_meta.json"

        faiss.write_index(self.index, index_path)
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        metadata = {'embedding_dimension': self.embedding_dimension, 'doc_count': len(self.documents)}
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)

    def load(self, filepath_prefix):
        """Tải trạng thái database từ các file đã lưu."""
        index_path = f"{filepath_prefix}_faiss.index"
        docs_path = f"{filepath_prefix}_docs.json"
        meta_path = f"{filepath_prefix}_meta.json"
        if not all(os.path.exists(p) for p in [index_path, docs_path, meta_path]): return False
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
            return True
        
        except Exception:
            self.index = None
            self.documents = []
            self.embedding_dimension = None
            return False
