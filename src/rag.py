"""Part D tasks 1-4: chunking, TF-IDF retrieval, grounded answering, evaluation."""
import glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICY_DIR = os.path.join(ROOT, "data", "policies")

def chunk_words(text, size=40, overlap=10):
    """Windows of `size` words, each starting size-overlap words after the previous one.
    Stops after the first window whose end reaches or passes the last word."""
    words, step, chunks, i = text.split(), size - overlap, [], 0
    while True:
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += step
    return chunks

def build_index(size=40, overlap=10):
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        src = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            for c in chunk_words(f.read(), size, overlap):
                chunks.append({"source": src, "text": c})
    vec = TfidfVectorizer()
    mat = vec.fit_transform([c["text"] for c in chunks])
    return {"chunks": chunks, "vectorizer": vec, "matrix": mat}

_DEFAULT = {}
def default_index():
    if "idx" not in _DEFAULT:
        _DEFAULT["idx"] = build_index(40, 10)
    return _DEFAULT["idx"]

def retrieve(query, k=3, index=None):
    idx = index or default_index()
    sims = cosine_similarity(idx["vectorizer"].transform([query]), idx["matrix"])[0]
    order = sims.argsort()[::-1][:k]
    return [{"source": idx["chunks"][i]["source"], "score": round(float(sims[i]), 4),
             "text": idx["chunks"][i]["text"]} for i in order]

SYSTEM_RAG = ("You are a POLICY ASSISTANT for Instamart support. Answer only from the supplied CONTEXT. "
              "If the context does not contain the answer, say that you could not find the information "
              "in the policy documents.")

def answer(question, index=None):
    hits = retrieve(question, 3, index)
    ctx = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM_RAG},
                {"role": "user", "content": f"CONTEXT:\n{ctx}\nQUESTION:\n{question}"}]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [h["source"] for h in hits]

def evaluate(index=None, verbose=True):
    ev = pd.read_csv(os.path.join(ROOT, "data", "raw", "policy_eval.csv"), keep_default_na=False)
    doc_ok = fact_ok = fb_ok = n_ans = n_unans = 0
    failures = []
    for q in ev.itertuples(index=False):
        text, sources = answer(q.question, index)
        if q.expected_doc == "NONE":
            n_unans += 1
            if text.startswith("I could not find"):
                fb_ok += 1
            else:
                failures.append((q.question, "unanswerable question got a non-fallback answer", text))
        else:
            n_ans += 1
            d, f = q.expected_doc in sources, q.expected_fact in text
            doc_ok += d; fact_ok += f
            if not (d and f):
                failures.append((q.question, f"doc_in_top3={d} fact_in_answer={f}", text))
    res = dict(answerable=n_ans, doc_in_top3=doc_ok, fact_in_answer=fact_ok, unanswerable=n_unans, fallback=fb_ok)
    return res, failures
