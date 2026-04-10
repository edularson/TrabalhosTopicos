import re
import math
from collections import Counter

corpus = {
    "Doc 1": "O Flamengo joga no Maracanã e tem a maior torcida do futebol brasileiro.",
    "Doc 2": "O Palmeiras disputa seus jogos no Allianz Parque e tem muitos títulos nacionais.",
    "Doc 3": "O São Paulo é conhecido mundialmente pelos seus três títulos intercontinentais no Morumbi.",
    "Doc 4": "O Corinthians joga na Neo Química Arena e possui uma torcida muito fiel.",
    "Doc 5": "O Santos revelou Pelé na Vila Belmiro e é um celeiro de craques históricos.",
}

STOPWORDS = {"o", "a", "e", "de", "do", "da", "no", "na", "um", "uma", "os", "as",
             "em", "por", "com", "se", "que", "é", "seus", "seus", "pelo", "pelos",
             "para", "mais", "ao"}

QUERY = "torcida estádio"


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\sàáâãäéêëíîïóôõöúûü]", "", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]


def build_term_doc_matrix(docs):
    processed = {doc: preprocess(text) for doc, text in docs.items()}
    vocab = sorted(set(t for tokens in processed.values() for t in tokens))
    matrix = {
        doc: {term: tokens.count(term) for term in vocab}
        for doc, tokens in processed.items()
    }
    return matrix, vocab, processed


def compute_tfidf(matrix, vocab, doc_names):
    N = len(doc_names)
    idf = {}
    for term in vocab:
        df = sum(1 for doc in doc_names if matrix[doc][term] > 0)
        idf[term] = math.log(N / df) if df > 0 else 0

    tfidf = {
        doc: {term: matrix[doc][term] * idf[term] for term in vocab}
        for doc in doc_names
    }
    return tfidf, idf


def cosine_similarity(vec_a, vec_b, vocab):
    dot = sum(vec_a[t] * vec_b.get(t, 0) for t in vocab)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    return dot / (norm_a * norm_b) if norm_a * norm_b != 0 else 0


def query_to_tfidf(query_tokens, idf, vocab):
    tf = Counter(query_tokens)
    return {term: tf.get(term, 0) * idf.get(term, 0) for term in vocab}


def run():
    doc_names = list(corpus.keys())
    matrix, vocab, processed = build_term_doc_matrix(corpus)
    tfidf, idf = compute_tfidf(matrix, vocab, doc_names)

    query_tokens = preprocess(QUERY)
    query_vec = query_to_tfidf(query_tokens, idf, vocab)

    scores = {
        doc: cosine_similarity(tfidf[doc], query_vec, vocab)
        for doc in doc_names
    }
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "processed": processed,
        "vocab": vocab,
        "matrix": matrix,
        "tfidf": tfidf,
        "idf": idf,
        "query_tokens": query_tokens,
        "query_vec": query_vec,
        "scores": scores,
        "ranking": ranking,
    }


if __name__ == "__main__":
    results = run()
    print("=== Ranking ===")
    for doc, score in results["ranking"]:
        print(f"  {doc}: {score:.4f}")