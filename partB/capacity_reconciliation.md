# Part B — Serving Capacity & Performance Reconciliation

This document provides the mathematical derivations, empirical validation against `bench/bench_log.csv`, root-cause diagnosis of serving anomalies, and corrected capacity planning rules for the FLM-4B-Instruct model on NVIDIA L4 hardware.

---

## B1. Theoretical KV-Cache Arithmetic & Concurrency Derivation

### (a) Exact KV-Cache Bytes Per Token
For an autoregressive Transformer utilizing Grouped Query Attention (GQA), each token stores a Key vector and a Value vector across every layer.

**Parameters from `bench/model_spec.md`:**
- Number of layers ($L$): $28$
- Number of KV heads ($H_{kv}$): $8$
- Head dimension ($d_k$): $128$
- Key precision: $\text{fp16}$ ($2\text{ bytes}$)
- Value precision: $\text{fp16}$ ($2\text{ bytes}$)

$$\text{KV Bytes per Token} = 2 \times L \times H_{kv} \times d_k \times \text{bytes\_per\_element}$$
$$\text{KV Bytes per Token} = 2 \times 28 \times 8 \times 128 \times 2 = \mathbf{114,688 \text{ bytes}}$$

$$\mathbf{114,688 \text{ bytes} = 112.0 \text{ KiB} = 114.688 \text{ KB}}$$

---

### (b) Maximum Concurrent 4096-Token Sequences
We determine the memory remaining on the GPU dedicated to the dynamic KV cache pool after loading model weights and allocating runtime buffers.

**1. Allocatable GPU VRAM:**
- Total VRAM: $1\times \text{NVIDIA L4} = 24\text{ GB} = 24 \times 1024^3 = 25,769,803,776\text{ bytes}$ ($24.0\text{ GiB}$)
- Serving utilization limit (`gpu_memory_utilization = 0.92`):
  $$M_{\text{allocatable}} = 0.92 \times 25,769,803,776 = 23,708,219,474\text{ bytes} \approx \mathbf{22.08 \text{ GiB}}$$

**2. Fixed Memory Allocations:**
- **Model Weights (FLM-4B-Instruct, FP16):**
  $$M_{\text{weights}} = 4.2 \times 10^9 \text{ params} \times 2\text{ bytes} = 8,400,000,000\text{ bytes} \approx \mathbf{7.823 \text{ GiB}}$$
- **Non-KV Runtime Overhead (Activations, CUDA Graphs, PagedAttention metadata):**
  $$M_{\text{overhead}} \approx 1.6 \text{ GB} = 1,600,000,000\text{ bytes} \approx \mathbf{1.490 \text{ GiB}}$$

**3. Memory Available for KV Cache Pool:**
$$M_{\text{KV\_pool}} = M_{\text{allocatable}} - M_{\text{weights}} - M_{\text{overhead}}$$
$$M_{\text{KV\_pool}} = 23,708,219,474 - 8,400,000,000 - 1,600,000,000 = \mathbf{13,708,219,474 \text{ bytes}} \approx \mathbf{12.767 \text{ GiB}}$$

**4. KV Cache Requirement per 4096-Token Sequence:**
$$\text{Memory per Sequence} = 4096 \text{ tokens} \times 114,688 \text{ bytes/token} = 469,762,048 \text{ bytes} = \mathbf{448.0 \text{ MiB} = 0.4375 \text{ GiB}}$$

**5. Maximum Concurrent Sequences:**
$$\text{Max Sequences} = \left\lfloor \frac{13,708,219,474\text{ bytes}}{469,762,048\text{ bytes}} \right\rfloor = \left\lfloor 29.18 \right\rfloor = \mathbf{29 \text{ sequences}}$$

*(Note: In real serving engines like vLLM with 16-token block granularity and memory fragmentation buffers, usable capacity is ~25–26 sequences).*

---

### Empirical Verification Against `bench/bench_log.csv`
Looking at the long-context sweep (`prompt_len = 3584` + `gen_len = 512` = $4096\text{ total tokens}$):

| Batch Size | Prompt + Gen | Peak `kv_cache_util` | Preempted Sequences | Observed Behavior |
|---|---|---|---|---|
| **24** | 4096 | **0.93 (93%)** | **0** | Fits comfortably in VRAM ($24 / 0.93 = 25.8$ theoretical capacity) |
| **32** | 4096 | **0.97 (97%)** | **7** | VRAM saturated; $32 - 7 = \mathbf{25}$ active sequences accommodated |
| **48** | 4096 | **0.97 (97%)** | **23** | VRAM saturated; $48 - 23 = \mathbf{25}$ active sequences accommodated |

**Conclusion:** The arithmetic prediction of $\sim 25\text{–}29$ sequences matches the empirical log.

---

## B2. Long-Context Throughput Anomaly & Root Cause Diagnosis

### The Anomaly
In `bench_log.csv`, as batch size increases from 4 to 24, reported throughput scales monotonically from $565.4\text{ tok/s}$ up to a peak of $1607.4\text{ tok/s}$.
However, past batch 24, throughput **collapses**:
- **Batch 24**: $1607.4\text{ tok/s}$ (Wall clock: $61.16\text{s}$, Preemptions: $0$, KV Util: $0.93$)
- **Batch 32**: **$1384.0\text{ tok/s}$ ($-13.9\%$ drop)** (Wall clock jumps to $94.71\text{s}$, Preemptions: **$7$**, KV Util: $0.97$)
- **Batch 48**: **$1298.5\text{ tok/s}$ ($-19.2\%$ drop)** (Wall clock jumps to $151.41\text{s}$, Preemptions: **$23$**, KV Util: $0.97$)

### Mechanism
The degradation is caused by **KV-Cache Exhaustion and Scheduler Preemption Thrashing**.
1. When batch size exceeds GPU capacity ($>25$ sequences of 4096 tokens), all sequences enter decode simultaneously.
2. As sequences generate new tokens, memory demands exceed available physical blocks. `kv_cache_util` hits the safety ceiling ($0.97$).
3. The vLLM scheduler is forced to preempt active requests ($7$ sequences in batch 32, $23$ sequences in batch 48).
4. Preempted requests are evicted (swapped to host CPU memory or aborted) and later resumed by **recomputing their entire 3,584-token prefill from scratch**.
5. This recomputation thrashing inflates wall-clock time from $61\text{s}$ to $151\text{s}$, spikes p95 latency to $105.4\text{s}$, and destroys generation efficiency.

### Proposed Configuration / Deployment Change
**Option 1 (Immediate Config Change): Set Concurrency Limit `max_num_seqs = 24`**
- *Mechanism*: Prevents the scheduler from admitting more than 24 concurrent 4096-token requests into the active KV pool, holding excess requests in the pending queue.
- *Quantitative Prediction*: When 48 requests arrive, they are processed cleanly as 2 consecutive batches of 24 with **0 preemptions**. Total wall-clock time drops from $151.41\text{s}$ to $2 \times 61.16\text{s} = \mathbf{122.32\text{s}}$ (a **$19.2\%$ reduction in total time**), restoring peak throughput to **$1607\text{ tok/s}$** ($+23.8\%$ throughput increase).

**Option 2 (Architectural Enhancement): Enable FP8 KV Cache (`kv_cache_dtype="fp8"`)**
- *Mechanism*: Halves KV cache footprint from 112 KiB to 56 KiB per token.
- *Quantitative Prediction*: Increases GPU capacity to $\sim 51$ concurrent 4096-token sequences. Batch 48 executes in a single pass with 0 preemptions (`kv_cache_util` $\approx 0.94$), completing in $\sim 82\text{s}$ at **$>2,350\text{ tok/s}$**.

---

## B3. Exposing the REPORT_v0 Misreading & Deriving Honest Goodput

### The Misreading
`REPORT_v0` made two disastrous claims:
1. *"Longer prompts clearly give better GPU utilization (1311 vs 883 tok/s at batch 16)."*
2. *"Assume ~1600 tok/s per L4 and scale linearly with batch size, so batch 48 should give ~3200 tok/s."*

**The Root Error:** The intern misread `reported_tok_s` as **generation throughput (goodput)**.
`reported_tok_s` is total harness throughput:
$$\text{reported\_tok\_s} = \frac{\text{Prompt Tokens} + \text{Generated Tokens}}{\text{Wall Clock Time}}$$

In the long-prompt benchmark ($3584\text{ prompt} + 512\text{ gen} = 4096\text{ tokens}$), **$87.5\%$ of all tokens are prompt prefill tokens**.
- **Prefill** is compute-bound (matrix multiplications in parallel across 3584 tokens), running at thousands of tokens/second.
- **Decode (generation)** is memory-bandwidth bound (reading model weights once per token for every sequence).

Long prompts inflated `reported_tok_s` solely because 87.5% of the work was prefill matrix multiplications, not because the GPU was serving user tokens faster!

---

### Two Independent Derivations of Honest Goodput (Batch 24, Long Prompt)

#### Derivation 1: Total Generated Output Tokens / Wall Clock Time
$$\text{Total Output Tokens} = \text{Batch Size} \times \text{gen\_len} = 24 \times 512 = 12,288 \text{ tokens}$$
$$\text{Wall Clock Time} = 61.16 \text{ seconds}$$
$$\mathbf{\text{Honest Goodput}_{\text{total}} = \frac{12,288 \text{ gen tokens}}{61.16 \text{ s}} = 200.92 \text{ gen-tok/s}}$$

*(If isolating purely the generation phase by subtracting TTFT latency $0.5005\text{s}$: $\text{Goodput}_{\text{decode}} = \frac{12,288}{60.66\text{s}} = \mathbf{202.57\text{ gen-tok/s}}$).*

#### Derivation 2: From Median Inter-Token Latency (`itl_ms_p50`)
The median inter-token latency during decode is $\text{itl\_ms\_p50} = 96.07\text{ ms} = 0.09607\text{ s}$ per decode step.
In each step, 24 tokens are emitted (one per active sequence):
$$\mathbf{\text{Decode Rate} = \frac{\text{Batch Size}}{\text{ITL (seconds)}} = \frac{24 \text{ tokens}}{0.09607 \text{ s}} = 249.82 \text{ gen-tok/s}}$$

---

### What REPORT_v0 Should Have Stated
| Metric | Intern Claim (`REPORT_v0`) | Honest Reality | Correction Factor |
|---|---|---|---|
| **Batch 24 Throughput** | $1607.4\text{ tok/s}$ | **$200.9\text{–}249.8\text{ gen-tok/s}$** | Intern overstated by **$6.4\times$ – $8.0\times$** |
| **Batch 48 Throughput** | $\sim 3200\text{ tok/s}$ (projected) | **$162.3\text{ gen-tok/s}$** (actual) | Intern overstated by **$19.7\times$** |
| **Long Prompt Impact** | "Gives better throughput" | "Consumes 87.5% capacity on prefill; triggers preemption collapse" | Fatal capacity flaw |

---

## B4. Production Monitoring Counter to Detect Preemption Thrashing

**Counter / Metric:**
`vllm:num_preemptions_total` (Prometheus Counter) or `vllm:gpu_cache_usage_factor` (Gauge).

**Expected Values & Monitoring Rule:**
- **Under Normal Healthy Load ($\text{Batch} \le 24$):** Value must equal **`0`** (`gpu_cache_usage_factor` $<0.95$).
- **Under Thrashing / Preemption Overload ($\text{Batch} \ge 32$):** Value will show **$\ge 7$** (jumping to **$\ge 23$** at batch 48), accompanied by `gpu_cache_usage_factor` pegged at $\ge 0.97$.
- **Alert Rule:** Trigger a P1 alert when `rate(vllm:num_preemptions_total[1m]) > 0` to automatically throttle ingress traffic and adjust `max_num_seqs`.
