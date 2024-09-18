# import sys
# import os

# # Thêm đường dẫn đến thư mục 'src' vào Python Path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
from src.embeddings.embedding_model import get_embedding_model, get_embedding
from src.mongodb.mongodb_connection import get_mongo_client, ingest_data
from src.search.bm25_search import get_bm25_scores
from src.search.vector_search import get_vector_scores, combine_scores
from src.llm.llm_response import configure_llm, get_llm_response
from src.utils.config import MONGO_URI, LLM_API_KEY, DATA_PATH
import numpy as np

# Tải dữ liệu và chuẩn bị nhúng
st.title("Hệ thống hỏi đáp về pháp luật")

@st.cache(allow_output_mutation=True)
def load_data():
    dataset_df = pd.read_csv(DATA_PATH)
    embedding_model = get_embedding_model()
    dataset_df["embedding"] = dataset_df["Điều Luật"].apply(lambda text: get_embedding(text, embedding_model))
    return dataset_df, embedding_model

dataset_df, embedding_model = load_data()

# Kết nối MongoDB
mongo_client = get_mongo_client(MONGO_URI)
db = mongo_client['sample_mflix']
collection = db['movie_collection_2']

# Lưu dữ liệu vào MongoDB
if st.button("Lưu dữ liệu vào MongoDB"):
    ingest_data(collection, dataset_df.to_dict("records"))
    st.success("Dữ liệu đã được lưu trữ thành công vào MongoDB.")

# Truy vấn người dùng
query = st.text_input("Nhập câu hỏi của bạn:")

if query:
    # Lấy dữ liệu từ MongoDB
    data = collection.find()
    df = pd.DataFrame(list(data)).drop(['_id', 'Unnamed: 0'], axis=1)

    # Tính điểm BM25 và vector
    bm25_scores = get_bm25_scores(df['Điều Luật'], query)
    corpus_embeddings = embedding_model.encode(df['Điều Luật'].tolist(), convert_to_tensor=True)
    vector_scores = get_vector_scores(query, corpus_embeddings, embedding_model)
    combined_scores = combine_scores(bm25_scores, vector_scores)

    # Chọn tài liệu có điểm cao nhất
    max_score_index = np.argmax(combined_scores)
    best_document = df.iloc[max_score_index]
    st.write(f"Điều Luật liên quan nhất: {best_document['Điều Luật']}")
    st.write(f"Điểm kết hợp: {combined_scores[max_score_index]}")

    # Sử dụng mô hình LLM
    llm_model = configure_llm(LLM_API_KEY)
    if combined_scores[max_score_index] < 6:
        response = get_llm_response(llm_model, query)
        st.write("Câu trả lời từ LLM:")
        st.write(response)
    else:
        response = get_llm_response(llm_model, query, best_document['Điều Luật'])
        st.write("Câu trả lời từ tài liệu:")
        st.write(response)
