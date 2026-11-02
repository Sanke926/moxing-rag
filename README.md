# Your RAG Number Is a Property of the Template You Copied

Code and data for the paper *"Your RAG Number Is a Property of the Template You Copied: An
Audit of the Default Question-Answering Prompts Shipped by Three RAG Frameworks"*.

## What this is

Three widely used RAG frameworks ship three different default question-answering prompts. We
run all three **verbatim** on the same reader, the same documents and the same questions, and
measure how far apart the reported numbers land. They are up to **46 exact-match points** apart
(HotpotQA: 46.4 vs 0.0), while a format-insensitive measure — does the gold answer appear
anywhere in the response — puts them within five points.

We also isolate one sentence inside LangChain's default (`If you don't know the answer, just say
that you don't know...`) with a byte-level ablation. With an empty context it drives abstention
to **79–95%**, against **0–2%** once deleted, without changing the format-insensitive measure.

## Contents

```
scripts/    runner, analysis, verification and figure scripts
results/    per-question predictions (fw_*.jsonl), 21,000 reader calls
figures/    figures used in the paper
paper/      manuscript source (.tex) and PDF
```

## The three templates (verbatim, with provenance)

| id | source | abstention clause | output cue |
|---|---|---|---|
| LC | `langchain.chains.retrieval_qa.prompt.PROMPT` | **yes** | `Helpful Answer:` |
| LI | `llama_index.core.prompts.DEFAULT_TEXT_QA_PROMPT_TMPL` | no | `Answer: ` |
| FR | FlashRAG `base_prompt` (answer prompt) | no | `Answer:` |

## Reproduction

Requires Python 3.10+, a local Ollama server with the four readers
(`llama3.2:1b`, `qwen2.5:3b`, `qwen2.5:7b`, `llama3:8b`), and the three QA datasets in the
project cache. Set `PYTHONIOENCODING=utf-8`.

```powershell
# 1. run the three templates, both conditions, on all datasets
python scripts/_framework_templates.py hotpot 500 qwen2.5:3b LC,LC_minus,LC_neutral,LC_cleared rag,closed
python scripts/_framework_templates.py hotpot 500 qwen2.5:3b LI,FR rag
python scripts/_framework_templates.py 2wiki  500 qwen2.5:3b LC,LC_minus,LC_neutral,LC_cleared rag,closed
python scripts/_framework_templates.py 2wiki  500 qwen2.5:3b LI,FR rag
python scripts/_framework_templates.py musique 500 qwen2.5:3b LC,LC_minus,LC_neutral,LC_cleared rag,closed
python scripts/_framework_templates.py musique 500 qwen2.5:3b LI,FR rag

# 2. reader generality
python scripts/_framework_templates.py hotpot 200 qwen2.5:7b LC,LC_minus,rag,closed
python scripts/_framework_templates.py hotpot 200 llama3:8b  FR,LI,LC rag,closed
python scripts/_framework_templates.py hotpot 200 llama3.2:1b LC,LI,FR rag,closed

# 3. tables and figures
python scripts/_framework_analyze.py
python scripts/_framework_stats.py
python scripts/_truncation_check.py
python scripts/_answer_length.py
python scripts/_llama_sig.py
python scripts/plot_templates.py
python scripts/plot_graphical_abstract.py

# 4. verify every number quoted in the paper (40 checks)
python scripts/_template_paper_verify2.py
```

The verification script recomputes all 40 values quoted in the manuscript from the raw
predictions and compares them with the text; a clean run reports `0 mismatches out of 40`.

## Key results

| dataset | FlashRAG EM | LlamaIndex EM | LangChain EM |
|---|---:|---:|---:|
| HotpotQA | **46.4** | 0.0 | 0.0 |
| 2WikiMultiHopQA | **37.8** | 0.0 | 0.0 |
| MuSiQue | **12.2** | 0.0 | 0.0 |

Containment for the same cells: 50.6 / 47.8 / 52.2 (HotpotQA); the paired FlashRAG-vs-LangChain
difference is −1.6, exact McNemar *p* = 0.54.

Empty-context abstention under LangChain's template: 91.2 / 79.0 / 95.0 (%), against
1.0 / 0.0 / 2.0 (%) once the sentence is deleted.

Anchor: FlashRAG reports HotpotQA F1 35.3 (RAG) and 28.4 (no retrieval) with
Llama3-8B-instruct. We obtain 24.9 in the closed-book condition (within 3.5) and 67.6 with
retrieval; the latter is not comparable because our pool contains the gold paragraph by
construction.

## Citing

```bibtex
@article{liu2026template,
  title  = {Your RAG Number Is a Property of the Template You Copied:
            An Audit of the Default Question-Answering Prompts Shipped by Three RAG Frameworks},
  author = {Liu, Fangqi and Liu, Yuhe},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

MIT for the code; the datasets keep their original licenses
(HotpotQA, 2WikiMultiHopQA, MuSiQue).
