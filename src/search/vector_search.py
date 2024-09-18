from sentence_transformers import util
import numpy as np

def get_vector_scores(query, corpus_embeddings, model):
    query_embedding = model.encode(query, convert_to_tensor=True)
    return util.pytorch_cos_sim(query_embedding, corpus_embeddings).cpu().numpy().flatten()

def combine_scores(bm25_scores, vector_scores):
    return np.array(bm25_scores) * vector_scores
