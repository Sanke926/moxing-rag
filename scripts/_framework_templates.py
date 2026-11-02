"""Published framework templates, verbatim, as a natural experiment.

Three widely used open-source RAG frameworks ship a default question-answering prompt.
Two of them contain no abstention instruction; one does. We run all three exactly as
published, on the same data and reader, in two conditions (RAG with ten documents, and
closed book with none).

Provenance
  LC  langchain.chains.retrieval_qa.prompt.PROMPT
      (docs/langchain/.../retrieval_qa/prompt.py, v0.1-era, unchanged in kind since)
  LI  llama_index.core.prompts.default_prompts.DEFAULT_TEXT_QA_PROMPT_TMPL
  FR  FlashRAG examples/methods/... base_prompt (answer_prompt)
  LC_minus   LC with the abstention sentence deleted, everything else byte-identical
  LC_neutral LC with that sentence replaced by a neutral sentence of the same length

Conditions: rag (top-10 BM25 documents), closed (empty context).
"""
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "r3" / "frameworks"
MODEL = "qwen2.5:3b"

# --- verbatim published templates -------------------------------------------
LC = ("Use the following pieces of context to answer the question at the end. "
      "If you don't know the answer, just say that you don't know, don't try to make "
      "up an answer.\n{context}\nQuestion: {question}\nHelpful Answer:")
LI = ("Context information is below.\n---------------------\n{context}\n"
      "---------------------\nGiven the context information and not prior knowledge, "
      "answer the query.\nQuery: {question}\nAnswer: ")
FR = ("Answer the question based on the given document."
      "Only give me the answer and do not output any other words.\n"
      "Context:\n{context}\nQuestion: {question}\nAnswer:")

CLAUSE = ("If you don't know the answer, just say that you don't know, don't try to make "
          "up an answer.")
NEUTRAL = ("Read the passages below carefully and then answer the question at the end as "
           "accurately as you can.")

LC_MINUS = LC.replace(CLAUSE + "\n", "").replace(CLAUSE, "")
LC_NEUTRAL = LC.replace(CLAUSE, NEUTRAL)

TEMPLATES = {
    "LC": LC,
    "LI": LI,
    "FR": FR,
    "LC_minus": LC_MINUS,
    "LC_neutral": LC_NEUTRAL,
    "LC_cleared": LC.replace(CLAUSE, ""),
}

PROVENANCE = {
    "LC": "langchain chains/retrieval_qa/prompt.py PROMPT (verbatim)",
    "LI": "llama_index core/prompts/default_prompts DEFAULT_TEXT_QA_PROMPT_TMPL (verbatim)",
    "FR": "FlashRAG base_prompt answer_prompt (verbatim)",
    "LC_minus": "LC with the abstention sentence deleted",
    "LC_neutral": "LC with that sentence replaced by an equal-length neutral sentence",
    "LC_cleared": "LC with the abstention sentence replaced by the empty string",
}


def build(tmpl, question, docs):
    ctx = "" if not docs else "\n".join(f"[{i+1}] {d.strip()}" for i, d in enumerate(docs))
    return tmpl.replace("{context}", ctx).replace("{question}", question)


def gen(prompt, model, num_predict=32):
    body = dict(model=model, prompt=prompt, stream=False,
                options=dict(temperature=0.0, num_predict=num_predict, seed=0))
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return (json.loads(urllib.request.urlopen(req, timeout=180).read()).get("response") or "").strip()


def load_mod(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from src.bm25_index import BM25Index, tokenize  # noqa: E402
import numpy as np  # noqa: E402
from src.metrics import max_em_f1_acc  # noqa: E402


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "hotpot"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    model = sys.argv[3] if len(sys.argv) > 3 else MODEL
    tmpl_keys = sys.argv[4].split(",") if len(sys.argv) > 4 else list(TEMPLATES)
    conds = sys.argv[5].split(",") if len(sys.argv) > 5 else ["rag", "closed"]

    slug = model.replace(":", "").replace(".", "")
    run_r3 = load_mod("run_r3_mod", ROOT / "scripts" / "run_r3.py")
    splits = run_r3.load_splits()
    qids = [str(q) for q in splits[ds]["held_out_qids"]][:n]
    examples = run_r3.load_examples_by_qids(ds, qids)

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"fw_{ds}_{slug}_n{n}.jsonl"
    done = set()
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["qid"], r["template"], r["condition"]))
    print(f"[fw] {ds} {model} n={len(examples)} templates={tmpl_keys} conds={conds} "
          f"skip={len(done)}", flush=True)

    t0 = time.perf_counter()
    emitted = 0
    with f.open("a", encoding="utf-8") as fh:
        for ex in examples:
            qid = str(ex.get("qid"))
            bm = BM25Index(ex["docs"])
            sc = np.asarray(bm._bm25.get_scores(tokenize(ex["question"])), dtype=np.float64)
            order = [int(i) for i in np.argsort(-sc)][: min(10, len(ex["docs"]))]
            docs = [ex["docs"][j] for j in order]
            for tk in tmpl_keys:
                for cond in conds:
                    if (qid, tk, cond) in done:
                        continue
                    use_docs = docs if cond == "rag" else []
                    prompt = build(TEMPLATES[tk], ex["question"], use_docs)
                    ans = gen(prompt, model)
                    em, f1, acc, _ = max_em_f1_acc(ans or "", ex["answers"])
                    low = (ans or "").lower()
                    contains = int(any(a.lower() in low for a in ex["answers"] if a))
                    fh.write(json.dumps(dict(
                        qid=qid, template=tk, condition=cond, model=model,
                        dataset=ds, answer=ans, em=em, f1=f1, contains=contains,
                        prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()[:16],
                        provenance=PROVENANCE[tk]), ensure_ascii=False) + "\n")
                    fh.flush()
                    emitted += 1
                    if emitted % 200 == 0:
                        el = time.perf_counter() - t0
                        print(f"[fw] {emitted} emitted {el:.0f}s ({el/emitted:.3f} s/call)",
                              flush=True)
    print("[fw] done", flush=True)


if __name__ == "__main__":
    main()

