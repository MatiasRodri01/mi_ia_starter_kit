from typing import List, Tuple
from pathlib import Path
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data" / "docs"
INDEX_DIR = Path(__file__).parent / "data"
VEC_PATH = INDEX_DIR / "tfidf_vectorizer.joblib"
MAT_PATH = INDEX_DIR / "tfidf_matrix.joblib"
TXT_PATH = INDEX_DIR / "tfidf_texts.joblib"

def _read_docs() -> List[Tuple[str, str]]:
    """
    Lee archivos .txt/.md dentro de data/docs y los divide en párrafos.
    Devuelve una lista de (doc_id, texto).
    """
    docs = []
    if not DATA_DIR.exists():
        return docs
    for p in DATA_DIR.glob("**/*"):
        if p.is_file() and p.suffix.lower() in [".txt", ".md"]:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            # Separamos por párrafos (línea en blanco)
            parts = re.split(r"\n\s*\n", content)
            for i, part in enumerate(parts):
                clean = part.strip()
                if clean:
                    docs.append((f"{p.name}#p{i+1}", clean))
    return docs

def build_or_load_index():
    """
    Carga el índice TF-IDF si existe, sino lo construye desde data/docs.
    Guarda vectorizador, matriz y lista de textos con joblib.
    """
    if VEC_PATH.exists() and MAT_PATH.exists() and TXT_PATH.exists():
        vectorizer = joblib.load(VEC_PATH)
        matrix = joblib.load(MAT_PATH)
        texts = joblib.load(TXT_PATH)
        return vectorizer, matrix, texts

    texts = _read_docs()
    corpus = [t for _, t in texts]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=1)
    if len(corpus) == 0:
        # índice vacío
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import csr_matrix
        matrix = csr_matrix((0, 0))
    else:
        matrix = vectorizer.fit_transform(corpus)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(matrix, MAT_PATH)
    joblib.dump(texts, TXT_PATH)
    return vectorizer, matrix, texts

def top_k(query: str, k: int = 4) -> List[Tuple[str, str, float]]:
    """
    Devuelve hasta k párrafos más similares a la consulta.
    Cada item: (doc_id, texto, score).
    """
    vectorizer, matrix, texts = build_or_load_index()
    if matrix.shape[0] == 0:
        return []
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, matrix).ravel()
    idxs = sims.argsort()[::-1][:k]
    out = []
    for i in idxs:
        doc_id, text = texts[i]
        out.append((doc_id, text, float(sims[i])))
    return out
