# data_loader.py
import os
import logging
import config # Import cấu hình
from vector_db import SimpleVectorDatabase # Import class DB
from retriever import HybridRetriever # Import class Retriever
from utils import embed_legal_chunks # Import hàm embed
import streamlit as st

def load_or_create_rag_components(embedding_model):
    """
    Tải VectorDB, Retriever từ file đã lưu nếu có,
    hoặc tạo mới nếu không có (bao gồm embedding).
    Trả về (vector_db, hybrid_retriever) hoặc (None, None) nếu lỗi.
    """
    logging.info("--- Đang kiểm tra/khởi tạo Vector DB và Retriever ---")
    vector_db_instance = SimpleVectorDatabase()
    hybrid_retriever_instance = None

    # Cố gắng tải Vector DB trước
    if vector_db_instance.load(config.SAVED_DATA_PREFIX):
        logging.info("Đã tải Vector DB từ file.")
        # Nếu tải DB thành công, khởi tạo Retriever (nó sẽ tự tải/tạo BM25)
        hybrid_retriever_instance = HybridRetriever(vector_db_instance, bm25_save_path=f"{config.SAVED_DATA_PREFIX}_bm25.pkl")
    else:
        # Nếu không tải được DB -> Tạo mới
        logging.info("Không tìm thấy Vector DB đã lưu hoặc tải lỗi. Tiến hành tạo mới...")
        # Tạo danh sách file từ config
        json_files = [
            config.JSON_FILE_PATTERN.format(i=i)
            for i in range(1, config.NUM_FILES + 1)
            if i not in config.NUMBERS_TO_SKIP
        ]

        logging.info(f"Sẽ xử lý {len(json_files)} file JSON để tạo DB.")
        if not json_files:
             logging.error("Không có file JSON nào được chỉ định để xử lý.")
             return None, None

        # Embed dữ liệu
        valid_chunks, embeddings_array = embed_legal_chunks(json_files, embedding_model)
        st.write('check')
        st.write(valid_chunks, embeddings_array)
        st.write('check')
        if valid_chunks and embeddings_array is not None:
            logging.info(f"Embed thành công {len(valid_chunks)} chunks.")
            # Thêm vào DB và lưu DB
            vector_db_instance.add_documents(valid_chunks, embeddings_array)
            vector_db_instance.save(config.SAVED_DATA_PREFIX)

            # Khởi tạo Retriever (nó sẽ tự tạo và lưu BM25)
            hybrid_retriever_instance = HybridRetriever(vector_db_instance, bm25_save_path=f"{config.SAVED_DATA_PREFIX}_bm25.pkl")
            st.write('fasjdfaskdg')
            logging.info("Đã tạo và lưu Vector DB & trạng thái BM25.")
        else:
            logging.error("Lỗi: Không embed được dữ liệu. Không thể tạo DB/Retriever.")
            return None, None # Trả về None nếu lỗi
    st.write("END")
    # Kiểm tra lại lần cuối trước khi trả về
    if hybrid_retriever_instance and hybrid_retriever_instance.vector_db and hybrid_retriever_instance.vector_db.index:
        logging.info(f"--- Khởi tạo DB & Retriever thành công. Index size: {hybrid_retriever_instance.vector_db.index.ntotal} ---")
    else:
        logging.error("--- Khởi tạo DB &/hoặc Retriever thất bại. ---")
        return None, None

    return vector_db_instance, hybrid_retriever_instance