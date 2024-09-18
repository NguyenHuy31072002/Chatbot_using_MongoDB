from sentence_transformers import SentenceTransformer

def get_embedding_model():
    return SentenceTransformer("keepitreal/vietnamese-sbert")

def get_embedding(text, embedding_model):
    if not text.strip():
        return []
    return embedding_model.encode(text).tolist()
