# data_loader.py
import config # Sẽ chứa hàm get_rag_data_prefix
from vector_db import SimpleVectorDatabase
from retriever import Retriever
from utils import embed_legal_chunks

# ++ Thêm tham số rag_data_prefix ++
def load_or_create_rag_components(embedding_model_object, rag_data_prefix: str):
    vector_db_instance = SimpleVectorDatabase()
    hybrid_retriever_instance = None

    # ++ Sử dụng rag_data_prefix để tải/lưu ++
    if vector_db_instance.load(rag_data_prefix):
        hybrid_retriever_instance = Retriever(vector_db_instance, bm25_save_path=f"{rag_data_prefix}_bm25.pkl")
    else:
        json_files = [
            config.JSON_FILE_PATTERN.format(i=i)
            for i in range(1, config.NUM_FILES + 1)
            if i not in config.NUMBERS_TO_SKIP
        ]

        if not json_files: return None, None

        valid_chunks, embeddings_array = embed_legal_chunks(json_files, embedding_model_object)
        if valid_chunks and embeddings_array is not None:
            vector_db_instance.add_documents(valid_chunks, embeddings_array)
            # ++ Sử dụng rag_data_prefix để tải/lưu ++
            vector_db_instance.save(rag_data_prefix)
            hybrid_retriever_instance = Retriever(vector_db_instance, bm25_save_path=f"{rag_data_prefix}_bm25.pkl")
        else: return None, None

    return vector_db_instance, hybrid_retriever_instance