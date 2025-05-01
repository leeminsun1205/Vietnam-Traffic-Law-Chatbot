# data_loader.py
import config
from vector_db import SimpleVectorDatabase 
from retriever import HybridRetriever 
from utils import embed_legal_chunks

def load_or_create_rag_components(embedding_model):

    vector_db_instance = SimpleVectorDatabase()
    hybrid_retriever_instance = None

    # Cố gắng tải Vector DB trước
    if vector_db_instance.load(config.SAVED_DATA_PREFIX):
        hybrid_retriever_instance = HybridRetriever(vector_db_instance, bm25_save_path=f"{config.SAVED_DATA_PREFIX}_bm25.pkl")
    else:
        json_files = [
            config.JSON_FILE_PATTERN.format(i=i)
            for i in range(1, config.NUM_FILES + 1)
            if i not in config.NUMBERS_TO_SKIP
        ]

        if not json_files: return None, None

        # Embed dữ liệu
        valid_chunks, embeddings_array = embed_legal_chunks(json_files, embedding_model)
        if valid_chunks and embeddings_array is not None:
            vector_db_instance.add_documents(valid_chunks, embeddings_array)
            vector_db_instance.save(config.SAVED_DATA_PREFIX)

            hybrid_retriever_instance = HybridRetriever(vector_db_instance, bm25_save_path=f"{config.SAVED_DATA_PREFIX}_bm25.pkl")
        else: return None, None 

    return vector_db_instance, hybrid_retriever_instance