# Audit of `fertility.py` and `REPORT_v0.md` Metrics

Every claim below adheres strictly to the **Evidence Rule**: each claimed bug, conceptual flaw, or benign artifact is isolated, quantitatively benchmarked, and accompanied by the exact delta and mechanical explanation.

---

## 1. Code Bug: Whitespace Splitting Bug (`line.split(" ")` vs `line.split()`)

### Claim
`fertility.py` uses `line.split(" ")` instead of `line.split()`. When a sentence contains consecutive whitespace characters (such as double spaces or tabs), `split(" ")` produces empty strings `""` in the word list, artificially inflating the word count denominator and deflating the reported fertility.

### Evidence & Measurement
- **Isolation Test**: Run `fertility.py` with `line.split(" ")` vs `line.split()` on `corpus_sample/`:
  - In `eng_sample.txt` (line 7: `"Please keep the books  in the cupboard."`), raw word count drops from 79 to 78.
    - Fertility (split `" "`): **1.2652 tok/word**
    - Fertility (split `None`): **1.2831 tok/word**
    - **Delta**: $+0.0179$ tok/word ($+1.41\%$ fertility increase upon fix).
  - In `hin_sample.txt` (line 10: `"किताबें  अलमारी में रखी हैं।"`), raw word count drops from 62 to 61.
    - Fertility (split `" "`): **7.4485 tok/word**
    - Fertility (split `None`): **7.5985 tok/word**
    - **Delta**: $+0.1500$ tok/word ($+2.01\%$ fertility increase upon fix).
- **Direction & Magnitude**: Distorts fertility downward by ~1.4% to 2.0% in the sample corpus.

---

## 2. Statistical Aggregation Flaw: Arithmetic Mean of Ratios vs Ratio of Totals

### Claim
`fertility.py` computes the unweighted arithmetic mean of per-line fertility ratios ($\frac{1}{N} \sum \frac{\text{tokens}_i}{\text{words}_i}$) rather than the corpus micro-average ($\frac{\sum \text{tokens}}{\sum \text{words}}$).

### Evidence & Measurement
- **Isolation Test**:
  - Sample English (10 sentences):
    - Mean of Ratios: **1.2831**
    - Ratio of Totals: **1.2692**
    - **Delta**: $-0.0138$ ($-1.08\%$)
  - Sample Hindi (10 sentences):
    - Mean of Ratios: **7.5985**
    - Ratio of Totals: **7.5246**
    - **Delta**: $-0.0739$ ($-0.97\%$)
  - FLORES English (1,012 sentences):
    - Mean of Ratios: **1.2874**
    - Ratio of Totals: **1.2782**
    - **Delta**: $-0.0092$ ($-0.71\%$)
- **Mechanism**: Short sentences (e.g. 2–3 words) with unusual tokenizations exert equal weight to long 40-word sentences in an arithmetic average, biasing the statistic toward short-sentence tokenization noise.

---

## 3. Conceptual Metric Flaw 1: Inappropriate Cross-Lingual Denominator (`tok/word`)

### Claim
`tokens / whitespace_word` is a fundamentally flawed cross-linguistic metric. It assumes a "word" carries equal semantic information across typologically diverse languages. In agglutinative languages (e.g. Kannada, Tamil, Malayalam), grammatical particles, case markers, prepositions, and verb conjugations are fused into a single whitespace token.

### Evidence & Measurement
- On FLORES-200 parallel data (1,012 identical semantic sentences):
  - English requires **21,950 words** (21.7 words/sentence).
  - Kannada requires only **16,100 words** (15.9 words/sentence — 27% fewer words!).
  - Malayalam requires only **14,930 words** (14.8 words/sentence — 32% fewer words!).
- When evaluating with `xlm-roberta-base`:
  - English: **1.40 tok/word**
  - Kannada: **2.58 tok/word** (looks $1.84\times$ worse per word!)
  - **However**, total tokens for the same meaning are 30,661 (ENG) vs 41,459 (KAN) = **only $1.35\times$ total tokens!**
- **Direction & Magnitude**: `tok/word` artificially inflates the perceived cost of agglutinative Indic languages by **+36% to +50%** relative to the true semantic information transferred.

---

## 4. Conceptual Metric Flaw 2: False Equivalence of `tok/char`

### Claim
`REPORT_v0` claims `tok/char` ($1.579$ vs $0.226 = 7.0\times$) "confirms" the per-word number. This is invalid: Python `len(line)` measures Unicode codepoints (UTF-16 code units), which do not represent visual or semantic units in Brahmic scripts.

### Evidence & Measurement
- In Devanagari, a single visual akshara/syllable (e.g. *"स्था"* or *"किं"*) consists of 2–4 Unicode codepoints (base consonant + virama + conjunct consonant + matra + anusvara).
- In English, 1 codepoint = 1 letter = 1 ASCII byte.
- In Hindi, 1 visual akshara = ~2.5–3.0 codepoints = 3–9 UTF-8 bytes.
- When measuring GPT-2:
  - English: 0.205 tok/char (GPT-2 compresses ~4.9 characters per token).
  - Hindi: 1.530 tok/char (GPT-2 lacks Devanagari merges, falling back to byte tokens: 3 bytes/char = ~1.5 tokens/char).
- **Mechanism**: The $7.0\times$ ratio does not confirm word fertility; it simply measures GPT-2's complete lack of Hindi byte merges.

---

## 5. Casing Distortion: `line.lower()` on Mixed Multilingual Text

### Claim
`line.lower()` in `fertility.py` distorts English tokenization by removing title/capitalization tokens, while doing nothing for native Brahmic scripts.

### Evidence & Measurement
- On FLORES-200 English:
  - Tokens with `lower()`: 27,994
  - Tokens without `lower()` (raw): 27,044
  - **Delta**: $-950$ tokens ($-3.39\%$ distortion on English).
- On FLORES-200 Hindi:
  - Tokens with `lower()`: 200,696 vs without `lower()`: 200,688 (Delta: 8 tokens, $<0.01\%$, due solely to embedded Latin acronyms).

---

## 6. Harmless / Benign Elements

### 1. `random.seed(1337)`
- **Observation**: `random.seed(1337)` is executed at script import.
- **Audit**: Zero random operations are called in `fertility.py`.
- **Verdict**: Harmless dead code. Flagging this as affecting metrics is incorrect.

### 2. `unicodedata.normalize("NFC", line)`
- **Observation**: Lines are normalized to Unicode NFC.
- **Audit**: In Indic scripts, text can be encoded in NFD (decomposed combining marks) or NFC (composed). If text were decomposed, character counts would inflate and subword tokenizers might fragment further. NFC standardizes text to canonical composition.
- **Verdict**: Correct and standard practice.

---

## 7. Refutation of REPORT_v0 Core Conclusion: "Property of the Script"

### Claim in REPORT_v0
*"Root cause: Hindi simply has more Unicode characters per word, so any tokenizer will struggle. This is a property of the script, not the tokenizer."*

### Measured Refutation
On the 1,012 parallel sentence benchmark:
- **`gpt2` (50k vocab)**: Hindi = 200,688 tokens (**$7.42\times$ English**)
- **`cl100k_base` (100k vocab)**: Hindi = 129,573 tokens (**$4.77\times$ English**)
- **`o200k_base` (200k vocab)**: Hindi = 42,247 tokens (**$1.57\times$ English**)
- **`xlm-roberta-base` (250k vocab)**: Hindi = 38,221 tokens (**$1.25\times$ English**)

### Conclusion
The $7.4\times$ inflation was entirely an artifact of using the English-only `gpt2` tokenizer. On modern multilingual tokenizers with allocated Indic vocabularies, Hindi is only **$1.25\times$** English, and Dravidian languages are only **$1.35\times$** English. The intern's recommendation to budget $6\times$ serving cost was catastrophic.
