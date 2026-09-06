# Decision Memo: Casual Indic Response Strategy

**To:** Product Leadership & AI Engineering  
**From:** AI Engineering Audit Team  
**Date:** September 6, 2026  
**Subject:** Technical Recommendation for Casual & Conversational Indic Response Generation  
**Selected Strategy:** **Path (c) — System Prompt Engineering with Dynamic Few-Shot Exemplars & Prefix Caching**  

---

### 1. Explicit Assumptions
1. **Reviewer Capacity & Language Scope**: 1 native reviewer (Hindi + Kannada only) for 10 h/week across 3 weeks = **30 total human hours** ($\le 720$ total evaluated response pairs). Tamil, Telugu, Bengali, and Marathi have **zero dedicated external reviewer hours** and must rely on automated linguistic heuristic checks and internal bilingual sanity checks.
2. **Compute & Budget Bounds**: 1× A100-80GB GPU available for 2 weeks; **$0 external API budget** (prohibiting distillation or synthetic generation from external proprietary LLMs like GPT-4o/Claude).
3. **Serving Latency SLA**: End-to-end P95 latency must remain $<2.5\text{s}$; conversational adjustments cannot double inference latency.

---

### 2. Back-of-Envelope Arithmetic & Trade-Off Analysis

| Metric / Dimension | (a) SFT on Synthetic Pairs | (b) ≤1B Rewriter Model | **(c) Prompt Engineering (Selected)** |
|---|---|---|---|
| **Data Requirements** | 5,000–10,000 pairs / language ($30\text{k}$–$60\text{k}$ total) | 10,000+ parallel rewrite pairs | **15–20 gold exemplars / language** (100 total) |
| **Reviewer Load** | Needs 2,000+ audits (Exceeds 30h by $3\times$) | Needs 1,500+ audits (Exceeds 30h by $2\times$) | **~600 audits** (Fits within 25h: 300 HIN + 300 KAN) |
| **GPU Compute Used** | ~180 GPU-hrs training + 100h eval | ~220 GPU-hrs training + serving | **~20 GPU-hrs** batch evaluation & validation |
| **Serving Cost & Latency**| $1.0\times$ latency; model-drift risk | **$+40\text{–}60\text{ms}$ / token** ($2\times$ model forward passes) | **$+0\text{ms}$ decode overhead** (KV prefix caching enabled) |
| **Unmonitored Lang Risk**| **HIGH**: Catastrophic forgetting / syntax degradation | **HIGH**: Hallucination & grammatical breakdown | **LOW**: Base model reasoning preserved; prompt scoped |

#### Reviewer Throughput Arithmetic
$$\text{Reviewer Speed} = 2.5 \text{ min / pair} \implies 24 \text{ pairs / hour}$$
$$\text{Total Reviewer Capacity} = 30 \text{ hours} \times 24 \text{ pairs/hr} = \mathbf{720 \text{ evaluated pairs max}}$$
*Path (c) utilizes 600 reviewer evaluations (300 in Hindi, 300 in Kannada) for rigorous iterative prompt validation, leaving 5 reviewer hours buffer.*

---

### 3. Success Metric & Numeric Threshold
- **Primary Metric**: **Casual Naturalness Win Rate (Pairwise Blind A/B vs Baseline)** judged by human evaluation (for HIN/KAN) and LLM-as-a-judge / lexical register scoring (for all 6 languages).
- **Numeric Threshold**: **$\ge 75\%$ Preference Win Rate** for casual prompt over textbook baseline on a 200-sample test set in Hindi and Kannada, with **$\le 1.0\%$ safety/factuality regression**.
- **Lexical Colloquial Index (LCI)**: $\ge 40\%$ reduction in formal textbook markers (e.g. Sanskritized formal pronouns/verbs like *"कृपया अवगत कराएं"*) in favor of natural conversational markers (e.g. *"बताइए"* / *"ಹೇಳಿ"*).

---

### 4. Kill Criterion & Timeline
- **Kill Criterion**: If by **Day 8 (End of Week 1)**, iterative system prompt tuning fails to achieve **$\ge 60\%$ casual preference win rate** on the reviewer test set in Hindi/Kannada, OR causes a **$>2.0\%$ drop in factual accuracy/safety**, we immediately abort Path (c) and pivot GPU resources to LoRA adapter fine-tuning on human-curated seed pairs.
- **Decision Checkpoint**: **September 15, 2026 (Day 8)**.

---

### 5. Day 1 Experiment Plan
1. **Hour 0–2**: Deploy the baseline FLM-4B model on the A100 GPU with vLLM and enable automatic prefix caching (`enable_prefix_caching = True`).
2. **Hour 2–4**: Construct 5 system prompt variations defining persona, tone, register guidelines, and pronoun conventions (*tum/aap*, colloquial affirmative tags) with 3 few-shot conversational exemplars per language.
3. **Hour 4–6**: Generate 50 test responses across Hindi and Kannada across diverse conversational domains (troubleshooting, small talk, advice).
4. **Hour 6–8**: Execute the first 2-hour blind evaluation session with the native reviewer to establish Day 1 baseline win rate.
