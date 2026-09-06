# AI Team Intern Assignment — The Audit

This repository contains the complete audit, empirical benchmarks, mathematical derivations, and decision memos for the **AI Team Intern Assignment: The Audit**.

---

## 📁 Repository Structure

```
.
├── README.md                       # Main project overview and defense guide
├── NOTEBOOK.md                     # Chronological lab notebook with hypotheses & dead ends
├── AI_USAGE.md                     # Honest audit of AI assistance & critical human checks
├── run_all.sh                      # 1-command runner for the entire benchmark & audit suite
│
├── partA/                          # Part A: Tokenizer Audit & Metrics
│   ├── build_corpus.py             # Script downloading & extracting FLORES-200 parallel sentences
│   ├── corpus/                     # 1,012 parallel sentences across 8 languages
│   ├── CORPUS_NOTES.md             # Corpus documentation, domain, normalization & boundary caveats
│   ├── audit_experiments.py        # Minimal reproduction experiments for fertility.py bugs
│   ├── AUDIT_FINDINGS.md           # Exhaustive evidence-based audit with measured deltas
│   ├── corrected_fertility.py      # Multi-tokenizer (4 tokenizers) & multi-denominator benchmark
│   ├── audit_experiments_output.txt# Raw output logs of audit experiments
│   ├── corrected_fertility_output.txt # Raw output logs of corrected benchmark
│   └── memo.md                     # ≤1 page executive leadership memo on tokenizer economics
│
├── partB/                          # Part B: Capacity Reconciliation
│   ├── verify_capacity.py          # Arithmetic & empirical verification script
│   ├── verify_capacity_output.txt  # Raw output logs of capacity math
│   └── capacity_reconciliation.md  # Detailed answers to B1–B4 with mathematical proofs
│
├── partC/                          # Part C: Decision Memo
│   └── memo.md                     # ≤1 page decision memo for casual Indic responses
│
├── bench/                          # Serving setup spec and raw load-test log
│   ├── model_spec.md
│   └── bench_log.csv
│
├── corpus_sample/                  # Original 10-sentence sample corpora
└── REPORT_v0.md                    # The original draft report being audited
```

---

## 🚀 Quickstart & Reproduction

To reproduce all experiments, benchmarks, and capacity calculations from source:

```bash
# Set up environment & run the entire suite
./run_all.sh
```

Or run individual components:

```bash
# Part A: Audit Experiments (Evidence Rule)
python3 partA/audit_experiments.py

# Part A: Corrected Multilingual Tokenizer Benchmark
python3 partA/corrected_fertility.py

# Part B: Capacity Reconciliation & Goodput Arithmetic
python3 partB/verify_capacity.py
```

---

## 🔍 Key Audit Discoveries & Headline Corrections

### Part A: Tokenizer Economics & The "Property of the Script" Fallacy
- **`REPORT_v0` Claim:** *"Hindi fertility is 5.89× worse than English... this is an inherent property of the script."*
- **The Reality:** The previous intern benchmarked an English-only tokenizer (`gpt2`) with zero Devanagari subwords, causing complete byte-level token fallback.
- **Evidence-Based Correction on FLORES-200 (1,012 parallel sentences):**
  - On **`gpt2`**: Hindi is $7.42\times$, Kannada is $13.58\times$, Tamil is $15.54\times$ English token count.
  - On **`xlm-roberta-base` (Indic-Aware)**: Hindi is **$1.25\times$**, Kannada is **$1.35\times$**, Tamil is **$1.35\times$**, Malayalam is **$1.38\times$** English token count!
- **Denominator Insight:** Agglutinative Dravidian languages express a 5-word English phrase in 1–2 compound words. Measuring `tokens/word` artificially inflates perceived cost by $+36\%\text{–}+50\%$. The only valid routing and capacity metric is **Tokens per Parallel Semantic Sentence** (holding user intent constant).

### Part B: Serving Capacity & Goodput Derivations
- **Exact KV Cache per Token:** $2 \times 28 \text{ layers} \times 8 \text{ KV heads} \times 128 \text{ head\_dim} \times 2 \text{ bytes} = \mathbf{114,688\text{ bytes}} = \mathbf{112\text{ KiB}}$.
- **Max 4096-Token Sequences on 24GB L4:** $\lfloor 12.77\text{ GiB} / 0.4375\text{ GiB} \rfloor = \mathbf{29\text{ sequences}}$ (empirically $25\text{–}26$ accounting for allocator metadata).
- **Long-Context Throughput Anomaly:** Saturated at batch 24 (`kv_cache_util` = 0.93). At batch 32 & 48, memory exhaustion triggers **preemption thrashing** ($7$ and $23$ preempted requests), destroying generation throughput.
- **Goodput vs Harness Throughput:** `reported_tok_s` was inflated because $87.5\%$ of all processed tokens were prompt prefill matrix multiplications.
  - At batch 24: Honest Goodput is **$200.9\text{ gen-tok/s}$** (wall-clock) and **$249.8\text{ gen-tok/s}$** (ITL median), not $1,607.4\text{ tok/s}$.
  - At batch 48: Throughput collapsed to **$162.3\text{ gen-tok/s}$**, refuting the intern's projected $3,200\text{ tok/s}$.

### Part C: Decision Strategy for Casual Indic Output
- **Selected Path:** **Path (c) — System Prompt Engineering with Dynamic Few-Shot Exemplars & Prefix Caching**.
- **Constraint Fit:** With only 1 reviewer for 30 hours total (covering Hindi + Kannada only) and $0 external API budget:
  - SFT risks catastrophic degradation in the 4 unmonitored languages (Tamil, Telugu, Bengali, Marathi).
  - 1B Rewriter doubles inference latency and degrades on complex Dravidian morphology.
  - Prompt engineering utilizes 600 reviewer evaluations (300 HIN / 300 KAN), preserves base reasoning, and introduces 0ms decode overhead via KV prefix caching.
- **Success Metric:** $\ge 75\%$ pairwise blind preference win rate over textbook baseline.
- **Kill Criterion:** If win rate is $<60\%$ by Day 8 (End of Week 1), pivot to LoRA adapter training.

---

## 📊 Interactive Model
- **Desmos Capacity & Goodput Model**: Interactive graphs of KV-cache footprint and preemption curves are available at [Desmos Graphing Calculator](https://www.desmos.com/calculator).
