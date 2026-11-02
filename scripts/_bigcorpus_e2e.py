"""End-to-end: with real retrieval, does reranking change the answer?

Two arms, same reader and prompt, same 200 questions:
    A  BM25 top-10 from the 4930-paragraph corpus
    B  BM25 top-50 -> cross-encoder rerank -> top-10
Everything else identical. This is the experiment that was meaningless in the degenerate
10-doc setting, where coverage was 100% by construction.
"""
import importlib.util
import json
import string
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "r3" / "bigcorpus"
MODEL = "qwen2.5:3b"
INSTR = "Answer the question with a short phrase only. No explanation."
NUMW = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
DROP = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "by", "with"}


def norm(s):
    s = (s or "").lower().strip().translate(str.maketrans("", "", string.punctuation))
    return " ".join(NUMW.get(t, t) for t in s.split() if t not in DROP)


def em(p, g):
    p = norm(p)
    return int(bool(p) and any(p == norm(x) for x in g))


def call(prompt):
    body = dict(model=MODEL, prompt=prompt, stream=False,
                options=dict(temperature=0, num_predict=24, seed=0))
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return (json.loads(urllib.request.urlopen(req, timeout=180).read()).get("response") or "").strip()


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "hotpot"
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    corpus_q = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    K, POOL = 10, 50

    run_r3 = load_mod("run_r3_mod", ROOT / "scripts" / "run_r3.py")
    splits = run_r3.load_splits()
    all_qids = [str(q) for q in splits[ds]["held_out_qids"]]
    exs = run_r3.load_examples_by_qids(ds, all_qids[:corpus_q])
    corpus, seen = [], set()
    for ex in exs:
        for t, d in zip(ex["titles"], ex["docs"]):
            key = (t, d[:80])
            if key not in seen:
                seen.add(key)
                corpus.append(d)
    test = exs[:n_q]
    print(f"corpus {len(corpus)} paragraphs, {len(test)} questions", flush=True)

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([d.lower().split() for d in corpus])
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"e2e_{ds}_{n_q}.jsonl"
    done = set()
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["qid"])
    t0 = time.perf_counter()
    with f.open("a", encoding="utf-8") as fh:
        for ex in test:
            qid = str(ex.get("qid"))
            if qid in done:
                continue
            q = ex["question"]
            scores = np.asarray(bm25.get_scores(q.lower().split()))
            pool_idx = list(np.argsort(-scores)[:POOL])
            ce_s = ce.predict([(q, corpus[i][:1200]) for i in pool_idx])
            ce_order = [pool_idx[i] for i in np.argsort(-np.asarray(ce_s, float))]

            res = {}
            for arm, idx in [("bm25", pool_idx[:K]), ("ce", ce_order[:K])]:
                ctx = "\n".join(f"[{j+1}] {corpus[i].strip()}" for j, i in enumerate(idx))
                prompt = f"{INSTR}\n\nContext:\n{ctx}\n\nQuestion: {q}\nShort answer:"
                ans = call(prompt)
                res[arm] = dict(answer=ans, em=em(ans, ex["answers"]))
            fh.write(json.dumps(dict(qid=qid, golds=list(ex["answers"]), arms=res),
                                ensure_ascii=False) + "\n")
            fh.flush()
            if (len(done) + 1) % 20 == 0:
                print(f"[e2e] {len(done)+1} {time.perf_counter()-t0:.0f}s", flush=True)
            done.add(qid)
    print("[e2e] done", flush=True)


if __name__ == "__main__":
    main()
