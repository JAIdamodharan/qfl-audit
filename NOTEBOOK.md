# Chronological Lab Notebook: The Audit

---

## Log Entry 01 — Workspace Triage & Baseline Assessment
- **Hypothesis:** `REPORT_v0.md` claims that Hindi has a $\sim 6\times$ serving cost penalty due to an "inherent property of the script." Let's inspect `fertility.py` and the sample corpora.
- **Action:** Read `REPORT_v0.md`, `fertility.py`, and `corpus_sample/`.
- **Observations:**
  - `corpus_sample/` only has 10 sentences for English and Hindi.
  - Line 7 of `eng_sample.txt` (`"Please keep the books  in the cupboard."`) has a double space.
  - Line 10 of `hin_sample.txt` (`"किताबें  अलमारी में रखी हैं।"`) has a double space.
  - `fertility.py` uses `line.split(" ")` rather than `line.split()`.

---

## Log Entry 02 — Building the Evaluation Corpus & Encountering Dead Ends
- **Hypothesis:** We need a standardized multilingual parallel eval corpus across English, Hindi, and Dravidian languages (Kannada, Tamil, Telugu, Malayalam). FLORES-200 is the gold standard.
- **Experiment 1 (Attempt via HuggingFace Datasets API):**
  - Wrote script to load `facebook/flores` and `muennighoff/flores200` using `datasets.load_dataset`.
  - **Result / Failure:** Failed with `RuntimeError: Dataset scripts are no longer supported, but found flores200.py` and HuggingFace Hub 401 Unauthorized / Gated Repo errors on `openlanguagedata/flores_plus` and `bri25yu/flores200_val_test`.
- **Revision / Solution:**
  - Located Meta AI's official open dataset tarball at `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`.
  - Discovered member paths were prefixed with `./flores200_dataset/devtest/{code}.devtest`.
  - Extracted 8 languages: `eng_Latn`, `hin_Deva`, `kan_Knda`, `tam_Taml`, `tel_Telu`, `mal_Mlym`, `ben_Beng`, `mar_Deva`.
  - Verified exact parallel alignment across all 8 files ($N = 1,012$ sentences per language) with strict NFC Unicode normalization. Saved to `partA/corpus/`.

---

## Log Entry 03 — Auditing `fertility.py` under the Evidence Rule
- **Flaw 1 Hypothesis:** `line.split(" ")` creates empty strings on double spaces, deflating fertility.
  - **Experiment:** Compared `split(" ")` vs `split()` on sample corpora.
  - **Result:**
    - English word count drops from 79 to 78; fertility increases from $1.2652$ to $1.2831$ ($+1.41\%$).
    - Hindi word count drops from 62 to 61; fertility increases from $7.4485$ to $7.5985$ ($+2.01\%$).
    - Distortion proven.
- **Flaw 2 Hypothesis:** Statistical aggregation macro-averaging (`sum(fertility)/N`) skews results toward short sentences.
  - **Experiment:** Compared mean of ratios vs ratio of totals ($\sum \text{tokens} / \sum \text{words}$).
  - **Result:** On FLORES English ($N=1012$), macro-average is $1.2874$ vs micro-average $1.2782$ ($-0.71\%$). On sample Hindi, distortion is $-0.97\%$.
- **Flaw 3 Hypothesis:** `line.lower()` alters tokenization.
  - **Experiment:** Ran lowercased vs raw casing across 1012 FLORES sentences.
  - **Result:** English token count without `lower()` is $27,044$ vs with `lower()` $27,994$ ($-3.39\%$ difference). On Hindi/Kannada, delta is $<0.01\%$ because Brahmic scripts have no case distinctions.
- **Dead End / Harmless Construct Check:**
  - Checked `random.seed(1337)`: Verified AST — zero calls to `random` functions exist in `fertility.py`. Harmless dead code.
  - Checked `unicodedata.normalize("NFC")`: Proved it is essential to prevent decomposed combining marks (NFD) from artificially inflating token and character counts.
- **Flaw 4 Hypothesis (The Big Myth):** "Hindi has high fertility because of the script."
  - **Experiment:** Evaluated `gpt2` (50k vocab) vs `cl100k_base` (100k vocab) vs `o200k_base` (200k vocab) vs `xlm-roberta-base` (250k vocab) on 1012 parallel sentences.
  - **Result:**
    - `gpt2`: Hindi $= 200,688\text{ tokens}$ ($7.42\times$ English).
    - `cl100k_base` (GPT-4): Hindi $= 129,573\text{ tokens}$ ($4.77\times$ English).
    - `o200k_base` (GPT-4o): Hindi $= 42,247\text{ tokens}$ ($1.57\times$ English).
    - `xlm-roberta-base`: Hindi $= 38,221\text{ tokens}$ (**$1.25\times$ English**), Kannada $= 41,459\text{ tokens}$ (**$1.35\times$ English**).
  - **Conclusion:** Disproved the intern's claim. High fertility in GPT-2 was caused by the absence of Devanagari merges in GPT-2's vocabulary, not by the script.

---

## Log Entry 04 — Part B: Serving Capacity Reconciliation & Goodput Derivation
- **Arithmetic Check:**
  - Model: 28 layers, 8 KV heads, head_dim 128, fp16 (2 bytes).
  - KV bytes/token $= 2 \times 28 \times 8 \times 128 \times 2 = 114,688\text{ bytes} = 112\text{ KiB}$.
  - GPU: 24GB L4 with $0.92$ utilization limit $= 22.08\text{ GiB}$.
  - Subtracting weights ($8.4\text{ GB} = 7.82\text{ GiB}$) and overhead ($1.6\text{ GB} = 1.49\text{ GiB}$) leaves $\sim 12.77\text{ GiB} = 13.7\text{ GB}$ for the KV cache pool.
  - Sequence KV cache (4096 tokens) $= 4096 \times 114,688 = 448\text{ MiB} = 0.4375\text{ GiB}$.
  - Theoretical limit $= 12.77 / 0.4375 \approx 29.18 \implies \sim 25\text{–}29\text{ sequences}$.
- **Empirical Confirmation:**
  - In `bench_log.csv`, batch 24 (4096 tokens) hits $0.93$ KV utilization ($24 / 0.93 = 25.8$ max capacity).
  - At batch 32, `preempted_seqs` $= 7$ (meaning $32 - 7 = 25$ fit in memory).
  - At batch 48, `preempted_seqs` $= 23$ (meaning $48 - 23 = 25$ fit in memory).
- **Goodput Derivation (Batch 24, Long Prompt):**
  - Harness reported $1607.4\text{ tok/s}$.
  - Method 1 (Wall clock): $24 \text{ seqs} \times 512 \text{ gen tokens} / 61.16\text{s} = \mathbf{200.92\text{ gen-tok/s}}$.
  - Method 2 (ITL): $24 / (96.07\text{ms} / 1000) = \mathbf{249.82\text{ gen-tok/s}}$.
  - Proved that $87.5\%$ of `reported_tok_s` was prefill matrix multiplications, exposing the intern's projection of $3200\text{ tok/s}$ at batch 48 as completely false.

---

## Log Entry 05 — Part C: Decision Modeling & Reviewer Bottleneck Analysis
- **Problem:** Make responses casual across 6 Indic languages under extreme constraints: 1 reviewer (Hindi + Kannada only, 10h/w, 3w = 30h), 1x A100 (2w), $0 external API budget.
- **Trade-off Modeling:**
  - SFT requires 30k–60k pairs; without external APIs, generating synthetic data on our 4B base model creates low-quality colloquial outputs and causes catastrophic forgetting in the 4 unmonitored languages (Tamil, Telugu, Bengali, Marathi).
  - 1B Rewriter doubles inference latency and degrades on morphologically complex Dravidian syntax.
  - Prompt Engineering with dynamic few-shot exemplars and KV prefix caching uses 600 reviewer evaluations (fitting within 25 hours), adds 0ms decode overhead, and preserves base model reasoning.
- **Final Recommendation:** Formulated Path (c) with an explicit 75% blind win-rate success metric and Day 8 kill criterion.
