#!/usr/bin/env python3
"""
corrected_fertility.py -- Corrected, multi-tokenizer, multi-denominator multilingual benchmark.

Evaluates:
  - Languages: English (eng), Hindi (hin), Kannada (kan), Tamil (tam), Telugu (tel), Malayalam (mal), Bengali (ben), Marathi (mar)
  - Tokenizers:
      1. gpt2 (OpenAI legacy, 50k vocab)
      2. cl100k_base (GPT-4/ChatGPT, 100k vocab)
      3. o200k_base (GPT-4o, 200k vocab)
      4. xlm-roberta-base (Multilingual encoder, 250k vocab)
  - Denominators:
      1. Per Parallel Sentence (Semantic content holding constant) + Normalized to English (Cost multiplier)
      2. Per UTF-8 Byte (Raw information compression efficiency)
      3. Per Grapheme Cluster / Akshara (Orthographic unit via regex \X)
      4. Per Whitespace Word (Lexical unit with proper split())
      5. Per Unicode Codepoint (Character count)

Outputs clean markdown tables and structured summaries.
"""

import os
import sys
import unicodedata
import regex as re
import tiktoken
from transformers import AutoTokenizer
from tabulate import tabulate

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
LANGUAGES = ["eng", "hin", "kan", "tam", "tel", "mal", "ben", "mar"]
LANG_NAMES = {
    "eng": "English",
    "hin": "Hindi",
    "kan": "Kannada",
    "tam": "Tamil",
    "tel": "Telugu",
    "mal": "Malayalam",
    "ben": "Bengali",
    "mar": "Marathi",
}

def load_corpus():
    data = {}
    for lang in LANGUAGES:
        path = os.path.join(CORPUS_DIR, f"{lang}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing corpus file: {path}. Run build_corpus.py first.")
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                l = raw.strip()
                if l:
                    lines.append(unicodedata.normalize("NFC", l))
        data[lang] = lines
    return data

def count_grapheme_clusters(text):
    # \X matches any extended grapheme cluster
    return len(re.findall(r"\X", text))

def get_tokenizers():
    xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return {
        "gpt2": lambda s: tiktoken.get_encoding("gpt2").encode(s),
        "cl100k_base (GPT-4)": lambda s: tiktoken.get_encoding("cl100k_base").encode(s),
        "o200k_base (GPT-4o)": lambda s: tiktoken.get_encoding("o200k_base").encode(s),
        "xlm-roberta-base": lambda s: xlm_tok.encode(s, add_special_tokens=False),
    }

def benchmark():
    corpus = load_corpus()
    num_sentences = len(corpus["eng"])
    tokenizers = get_tokenizers()
    
    print(f"Benchmark Dataset: FLORES-200 Devtest ({num_sentences} parallel sentences per language)")
    print("=" * 80)
    
    # Calculate corpus baselines (words, bytes, graphemes, characters)
    stats = {}
    for lang in LANGUAGES:
        lines = corpus[lang]
        total_words = sum(len(l.split()) for l in lines)
        total_bytes = sum(len(l.encode("utf-8")) for l in lines)
        total_graphemes = sum(count_grapheme_clusters(l) for l in lines)
        total_chars = sum(len(l) for l in lines)
        stats[lang] = {
            "sentences": num_sentences,
            "words": total_words,
            "bytes": total_bytes,
            "graphemes": total_graphemes,
            "chars": total_chars,
        }
        
    # Print Corpus Statistics Table
    corp_table = []
    for lang in LANGUAGES:
        s = stats[lang]
        corp_table.append([
            f"{lang} ({LANG_NAMES[lang]})",
            s["sentences"],
            s["words"],
            f"{s['words']/s['sentences']:.1f}",
            s["chars"],
            s["graphemes"],
            s["bytes"],
            f"{s['bytes']/s['chars']:.2f}",
        ])
    print("\n### 1. Corpus Statistics (FLORES-200 Parallel Split)")
    print(tabulate(corp_table, headers=["Language", "Sentences", "Total Words", "Words/Sent", "Codepoints (Chars)", "Graphemes", "UTF-8 Bytes", "Bytes/Char"], tablefmt="github"))
    
    # Run Tokenizer Benchmarks across all denominators
    all_results = {}
    for tok_name, enc_fn in tokenizers.items():
        tok_stats = {}
        for lang in LANGUAGES:
            lines = corpus[lang]
            total_tokens = sum(len(enc_fn(l)) for l in lines)
            s = stats[lang]
            tok_stats[lang] = {
                "total_tokens": total_tokens,
                "tok_per_sent": total_tokens / s["sentences"],
                "tok_per_word": total_tokens / s["words"],
                "tok_per_byte": total_tokens / s["bytes"],
                "tok_per_grapheme": total_tokens / s["graphemes"],
                "tok_per_char": total_tokens / s["chars"],
            }
        all_results[tok_name] = tok_stats
        
    # Print results for each tokenizer
    for tok_name, tok_stats in all_results.items():
        print(f"\n### Tokenizer: `{tok_name}`")
        eng_tokens = tok_stats["eng"]["total_tokens"]
        eng_tok_sent = tok_stats["eng"]["tok_per_sent"]
        
        table_rows = []
        for lang in LANGUAGES:
            ts = tok_stats[lang]
            cost_mult = ts["total_tokens"] / eng_tokens
            table_rows.append([
                f"{lang} ({LANG_NAMES[lang]})",
                f"{ts['total_tokens']:,}",
                f"{ts['tok_per_sent']:.2f}",
                f"{cost_mult:.2f}x",
                f"{ts['tok_per_word']:.2f}",
                f"{ts['tok_per_grapheme']:.2f}",
                f"{ts['tok_per_byte']:.3f}",
                f"{ts['tok_per_char']:.3f}",
            ])
        print(tabulate(table_rows, headers=[
            "Language", "Total Tokens", "Tok / Sentence", "Cost Multiplier (vs ENG)", "Tok / Word", "Tok / Grapheme", "Tok / Byte", "Tok / Char"
        ], tablefmt="github"))

    # Print Headline Comparison of Cost Multipliers (Tokens per Semantic Sentence vs English)
    print("\n### 2. Cross-Tokenizer Semantic Cost Multiplier vs English (Holding Meaning Constant)")
    comp_headers = ["Language"] + list(tokenizers.keys())
    comp_rows = []
    for lang in LANGUAGES:
        row = [f"{lang} ({LANG_NAMES[lang]})"]
        for tok_name in tokenizers.keys():
            eng_tok = all_results[tok_name]["eng"]["total_tokens"]
            lang_tok = all_results[tok_name][lang]["total_tokens"]
            mult = lang_tok / eng_tok
            row.append(f"{mult:.2f}x")
        comp_rows.append(row)
    print(tabulate(comp_rows, headers=comp_headers, tablefmt="github"))

if __name__ == "__main__":
    benchmark()
