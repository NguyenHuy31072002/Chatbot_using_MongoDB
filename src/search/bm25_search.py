from rank_bm25 import BM25Okapi

def tokenize_corpus(corpus):
    return [doc.split() for doc in corpus]

def get_bm25_scores(corpus, query):
    tokenized_corpus = tokenize_corpus(corpus)
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25.get_scores(query.split())
