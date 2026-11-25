import numpy as np

from .llm import EMBED_MODEL


def embed(texts, client=None):
    """Unit-length vectors, one row per text.

    With a client: OpenAI embeddings (meaning-based).
    Without one: TF-IDF (word-overlap only), so the pipeline still runs offline.
    """
    if client is not None:
        resp = client.embeddings.create(model=EMBED_MODEL, input=[t[:2000] for t in texts])
        X = np.array([d.embedding for d in resp.data], dtype=float)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        X = TfidfVectorizer(stop_words="english").fit_transform(texts).toarray()
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X / norms
