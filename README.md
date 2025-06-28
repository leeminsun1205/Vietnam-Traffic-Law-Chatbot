# Vietnam Traffic Law Chatbot

## 📑 Mục lục

- [Tính năng nổi bật](#tính-năng-nổi-bật)
- [Kiến trúc hệ thống (RAG Pipeline)](#kiến-trúc-hệ-thống-rag-pipeline)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Hướng dẫn cài đặt và sử dụng](#hướng-dẫn-cài-đặt-và-sử-dụng)
  - [Yêu cầu](#yêu-cầu)
  - [Cài đặt](#cài-đặt)
  - [Sử dụng](#sử-dụng)
- [Nguồn dữ liệu](#nguồn-dữ-liệu)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)

---

## 🌟 Tính năng nổi bật

- **Hệ thống RAG mạnh mẽ**: Kết hợp LLM (Google Gemini) với hệ thống truy xuất thông tin để trả lời câu hỏi dựa trên văn bản luật.
- **Truy vấn đa dạng (Hybrid Retrieval)**:
  - *Dense Retrieval*: Tìm kiếm ngữ nghĩa bằng vector embedding (FAISS).
  - *Sparse Retrieval*: Tìm kiếm từ khóa bằng BM25.
  - *Hybrid Retrieval*: Kết hợp cả hai bằng Rank-Fusion (RRF).
- **Mở rộng câu hỏi (Query Expansion)**: Tự động sinh biến thể câu hỏi bằng LLM, giúp cải thiện truy xuất.
- **Xếp hạng lại kết quả (Reranking)**: Sử dụng Cross-Encoder để đánh giá và sắp xếp tài liệu liên quan.
- **Hiển thị biển báo giao thông**: Nhận diện và hiển thị hình ảnh biển báo nếu được đề cập trong văn bản luật.
- **Giao diện tùy chỉnh linh hoạt**: Xây dựng bằng Streamlit, cho phép:
  - Tùy chọn mô hình Embedding, Reranker, Generative.
  - Thay đổi phương thức truy vấn, chế độ trả lời.
- **Trang đánh giá hiệu suất**: Đánh giá hệ thống với các metric: Precision@k, Recall@k, F1@k, MRR@k, NDCG@k.
- **Kiến trúc module hóa**: Dễ bảo trì, mở rộng.

---

## 🧠 Kiến trúc hệ thống (RAG Pipeline)

1. **Query Expansion**: Sử dụng Gemini để tạo các biến thể câu hỏi nếu hợp lệ.
2. **Information Retrieval**: Truy xuất tài liệu từ FAISS (Dense) và BM25 (Sparse).
3. **Reranking**: Dùng Cross-Encoder để đánh giá độ liên quan và sắp xếp lại tài liệu.
4. **Answer Generation**: Dựa trên tài liệu đã sắp xếp, sinh câu trả lời bằng Gemini.

---

## 🛠️ Công nghệ sử dụng

- **Framework**: Streamlit
- **LLM**: Google Gemini (`google-generativeai`)
- **Embedding / Reranking**: `Sentence Transformers`, `BAAI/bge-m3`, `bge-reranker-v2-m3`
- **Vector DB**: FAISS
- **Sparse Retrieval**: `rank_bm25`
- **NLP Tools**: `pyvi`, `py-vncorenlp`
- **Khác**: `NumPy`, `Pandas`

---

## 📁 Cấu trúc thư mục

```
Vietnam-Traffic-Law-Chatbot/
├── chatbot/              # Mã nguồn chính
│   ├── pages/            # Trang con (vd: Evaluation.py)
│   ├── Chatbot.py        # File chạy chính
│   ├── config.py         # Cấu hình mô hình, tham số
│   ├── data_loader.py    # Tải dữ liệu
│   ├── generation.py     # Sinh câu trả lời
│   ├── model_loader.py   # Tải các mô hình
│   ├── reranker.py       # Xếp hạng lại
│   ├── retriever.py      # Truy xuất Dense/Sparse/Hybrid
│   ├── utils.py          # Tiện ích, metrics
│   └── vector_db.py      # Quản lý FAISS DB
├── datasets/             # Bộ dữ liệu đã xử lý
├── loader/               # Tải dữ liệu thô
├── make_datasets/        # Xử lý dữ liệu gốc
├── notebook/             # Thử nghiệm Jupyter
├── text/                 # Văn bản luật gốc
├── traffic_sign/         # Ảnh biển báo
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🧪 Hướng dẫn cài đặt và sử dụng

### Yêu cầu

- Python 3.9+
- Pip
- Git

### Cài đặt

```bash
# Clone repository
git clone https://github.com/[Tên tài khoản của bạn]/Vietnam-Traffic-Law-Chatbot.git
cd Vietnam-Traffic-Law-Chatbot

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### Cấu hình API Key

Dự án sử dụng mô hình Gemini của Google. Cung cấp API key bằng một trong hai cách:

- **Biến môi trường**:

```bash
# macOS/Linux
export GOOGLE_API_KEY="your_api_key_here"

# Windows (Command Prompt)
set GOOGLE_API_KEY="your_api_key_here"
```

- **Kaggle Secrets**: Đặt key với tên `GOOGLE_API_KEY`. File `model_loader.py` sẽ tự động tìm.

---

### Sử dụng

#### Chạy ứng dụng Chatbot

```bash
streamlit run chatbot/Chatbot.py
```

> Trình duyệt sẽ mở giao diện trò chuyện. Có thể tùy chỉnh mô hình trong sidebar.

#### Chạy trang đánh giá hệ thống

```bash
streamlit run chatbot/pages/Evaluation.py
```

> Cho phép đánh giá hệ thống retrieval với bộ câu hỏi và tài liệu thực tế.

---

## 📚 Nguồn dữ liệu

Hệ thống sử dụng các văn bản pháp luật giao thông đường bộ Việt Nam đã được số hóa và xử lý trước.

- Các thư mục liên quan: `text/`, `make_datasets/`, `loader/`

> ⚠️ **Lưu ý**: Thông tin từ chatbot chỉ mang tính tham khảo, không thay thế cho văn bản pháp luật chính thức hoặc tư vấn pháp lý chuyên sâu.

---

## 🤝 Đóng góp

Mọi đóng góp được hoan nghênh! Hãy tạo Pull Request hoặc Issue nếu bạn muốn cải tiến dự án.

---

## 📄 Giấy phép

[MIT License](LICENSE)
