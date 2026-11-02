"""LLM-as-a-judge check for Study A.

Containment can be criticised for rewarding long responses. Here a separate model
(qwen2.5:7b, i.e. a different size from the 3B reader) judges whether each stored response
answers the question correctly, independent of its format. Run over the same 200 HotpotQA
questions for the three templates.
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "results" / "r3" / "frameworks"
OUT = FW / "judge_hotpot_200.jsonl"
JUDGE = "qwen2.5:7b"

PROMPT = """You are grading answers to a question. You are given the question, the gold
answer, and a response produced by a system. Decide whether the response conveys the gold
answer, regardless of whether it is written as a short phrase or as a long explanation, and
regardless of extra correct detail.

Question: {q}
Gold answer: {g}
Response: {r}

If the response conveys the gold answer, output YES. If it does not, or the response refuses
to answer, output NO. Output only YES or NO."""


def gen(prompt):
    body = dict(model=JUDGE, prompt=prompt, stream=False,
                options=dict(temperature=0.0, num_predict=4, seed=0))
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return (json.loads(urllib.request.urlopen(req, timeout=180).read()).get("response") or "").strip()


import importlib.util
spec = importlib.util.spec_from_file_location("r3mod", ROOT / "scripts" / "run_r3.py")
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)
qids = [str(q) for q in r3.load_splits()["hotpot"]["held_out_qids"]][:200]
ex = {str(e.get("qid")): e for e in r3.load_examples_by_qids("hotpot", qids)}

rows = []
for f in sorted(FW.glob("fw_hotpot_*.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

todo = [r for r in rows
        if r["condition"] == "rag" and r["template"] in ("FR", "LI", "LC")
        and r["qid"] in ex and r["model"] == "qwen2.5:3b"]

done = set()
if OUT.exists():
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            done.add((d["qid"], d["template"]))

print(f"[judge] {len(todo)} rows, {len(done)} already done", flush=True)
t0 = time.perf_counter()
n = 0
with OUT.open("a", encoding="utf-8") as fh:
    for r in todo:
        key = (r["qid"], r["template"])
        if key in done:
            continue
        e = ex[r["qid"]]
        gold = e["answers"][0] if e["answers"] else ""
        p = PROMPT.format(q=e["question"], g=gold, r=(r["answer"] or "")[:1500])
        out = gen(p).upper()
        verdict = 1 if re.search(r"\bYES\b", out) else 0
        fh.write(json.dumps(dict(qid=r["qid"], template=r["template"], model=r["model"],
                                 verdict=verdict, raw=out[:20]), ensure_ascii=False) + "\n")
        fh.flush()
        n += 1
        if n % 100 == 0:
            el = time.perf_counter() - t0
            print(f"[judge] {n} done {el:.0f}s", flush=True)
print("[judge] done", flush=True)
