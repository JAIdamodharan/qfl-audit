#!/usr/bin/env python3
"""
verify_capacity.py -- Verification script for Part B capacity reconciliation arithmetic.
"""

import pandas as pd
import numpy as np

def verify():
    # 1. Model Spec Constants
    layers = 28
    d_model = 3072
    q_heads = 24
    kv_heads = 8
    head_dim = 128
    prec_bytes = 2 # fp16
    
    # KV Cache bytes per token
    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * prec_bytes
    print(f"1. KV Cache Bytes per token: {kv_bytes_per_token:,} bytes ({kv_bytes_per_token / 1024:.1f} KiB)")
    
    # GPU Memory
    gpu_total_bytes = 24 * (1024**3) # 24 GiB = 25,769,803,776 bytes
    gpu_target_util = 0.92
    gpu_allocatable = gpu_total_bytes * gpu_target_util
    
    model_weights_bytes = 4.2 * (10**9) * 2 # 8.4 GB = 8,400,000,000 bytes
    overhead_bytes = 1.6 * (10**9) # 1.6 GB = 1,600,000,000 bytes
    
    kv_pool_bytes = gpu_allocatable - model_weights_bytes - overhead_bytes
    kv_seq_4096_bytes = 4096 * kv_bytes_per_token
    
    max_concurrent_seqs = kv_pool_bytes / kv_seq_4096_bytes
    print(f"2. Max Concurrent 4096-token Sequences Theoretical: {max_concurrent_seqs:.2f} (approx {int(max_concurrent_seqs)} seqs)")
    
    # 2. Check against bench_log.csv
    df = pd.read_csv("bench/bench_log.csv")
    print("\n3. Bench Log Analysis for Prompt 3584 + Gen 512 (Total 4096):")
    long_df = df[df["prompt_len"] == 3584].copy()
    
    long_df["total_tokens"] = long_df["batch_size"] * (long_df["prompt_len"] + long_df["gen_len"])
    long_df["gen_tokens"] = long_df["batch_size"] * long_df["gen_len"]
    long_df["honest_goodput_wallclock"] = long_df["gen_tokens"] / long_df["wall_clock_s"]
    long_df["honest_goodput_itl"] = (long_df["batch_size"] / (long_df["itl_ms_p50"] / 1000.0))
    long_df["computed_reported_tok_s"] = long_df["total_tokens"] / long_df["wall_clock_s"]
    
    for _, row in long_df.iterrows():
        print(f"Batch {int(row['batch_size']):2d} | Wall: {row['wall_clock_s']:6.2f}s | "
              f"Reported: {row['reported_tok_s']:6.1f} tok/s | "
              f"Goodput (Wall): {row['honest_goodput_wallclock']:5.1f} gen-tok/s | "
              f"Goodput (ITL): {row['honest_goodput_itl']:5.1f} gen-tok/s | "
              f"Preempted: {int(row['preempted_seqs']):2d} | KV Util: {row['kv_cache_util']:.2f}")

if __name__ == "__main__":
    verify()
