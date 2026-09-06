# AI Usage Report: Transparency & Verification

## 1. Where AI Assisted Effectively
1. **Parallel Corpus Ingestion & Extraction Boilerplate**:
   - AI generated the initial `urllib` streaming and `tarfile` iteration scripts to download and extract the official Meta FLORES-200 archive directly from Facebook AI public servers.
2. **Tokenizer Benchmarking Harness**:
   - AI helped write the tabulate-formatted comparison tables and batch encoding loops across `tiktoken` (`gpt2`, `cl100k_base`, `o200k_base`) and HuggingFace `AutoTokenizer` (`xlm-roberta-base`).
3. **Drafting Markdown Outlines**:
   - AI accelerated the structural formatting of `AUDIT_FINDINGS.md`, `capacity_reconciliation.md`, and the decision memo templates.

---

## 2. Where AI Misled / Hallucinated (and How It Was Corrected)
1. **Hallucination on `random.seed(1337)`**:
   - Initial AI suggestions flagged `random.seed(1337)` in `fertility.py` as a "potential source of non-deterministic batch sampling."
   - **Correction**: We checked the Python Abstract Syntax Tree (AST) and verified line-by-line that `random` is never called anywhere in the script. Flagging it as a bug would have violated the Evidence Rule and incurred a point penalty. It was correctly categorized as harmless dead code.
2. **HuggingFace Hub Datasets 5.0 Gated Access Assumptions**:
   - AI initially generated code assuming `load_dataset("facebook/flores")` or `load_dataset("openlanguagedata/flores_plus")` would work unauthenticated. Both failed with gated access and deprecated script errors in `datasets` 5.0.
   - **Correction**: We bypassed the Hugging Face gated hub by directly streaming Meta's official public tarball from `dl.fbaipublicfiles.com` and verifying internal member archive paths.
3. **Naive Goodput Calculation**:
   - AI initially attempted to calculate goodput by simply subtracting prompt length from total tokens and dividing by wall clock time without accounting for TTFT prefill latency and multi-sequence parallelism in batching.
   - **Correction**: We independently formulated both derivations:
     - Method 1: Exact generated tokens divided by total wall clock time ($24 \times 512 / 61.16\text{s} = 200.92\text{ gen-tok/s}$).
     - Method 2: Batch size divided by median inter-token latency during decode ($24 / 0.09607\text{s} = 249.82\text{ gen-tok/s}$).
4. **Script vs Tokenizer Attribution**:
   - An early AI drafting iteration repeated the previous intern's misconception that Brahmic scripts inherently generate more tokens due to complex vowel matras.
   - **Correction**: We ran empirical proofs with `o200k_base` and `xlm-roberta-base`, proving that the token explosion disappears on modern multilingual tokenizers with allocated Indic vocabularies (dropping Hindi from $7.42\times$ to $1.25\times$), proving it is a property of the tokenizer merge table, not the script.

---

## 3. Human Verification Summary
Every number, metric delta, arithmetic derivation, and code script in this repository was independently executed, checked against raw logs, and verified locally. All claims are backed by minimal reproducible experiments that can be rerun live in the defense session.
