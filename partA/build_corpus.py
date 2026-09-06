#!/usr/bin/env python3
"""
build_corpus.py -- Download and extract standardized FLORES-200 multilingual parallel eval corpus.

Languages:
  - eng: English (eng_Latn)
  - hin: Hindi (hin_Deva)
  - kan: Kannada (kan_Knda) [Dravidian]
  - tam: Tamil (tam_Taml) [Dravidian]
  - tel: Telugu (tel_Telu) [Dravidian]
  - mal: Malayalam (mal_Mlym) [Dravidian]
  - ben: Bengali (ben_Beng) [Indic / Part C context]
  - mar: Marathi (mar_Deva) [Indic / Part C context]
"""

import os
import sys
import tarfile
import urllib.request
import unicodedata

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

LANG_MAP = {
    "eng": "eng_Latn",
    "hin": "hin_Deva",
    "kan": "kan_Knda",
    "tam": "tam_Taml",
    "tel": "tel_Telu",
    "mal": "mal_Mlym",
    "ben": "ben_Beng",
    "mar": "mar_Deva",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "corpus")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading FLORES-200 dataset from {FLORES_URL}...")
    
    target_members = {
        f"./flores200_dataset/devtest/{code}.devtest": lang
        for lang, code in LANG_MAP.items()
    }
    # Also handle without leading ./
    target_members_alt = {
        f"flores200_dataset/devtest/{code}.devtest": lang
        for lang, code in LANG_MAP.items()
    }

    req = urllib.request.Request(FLORES_URL, headers={"User-Agent": "Mozilla/5.0"})
    extracted = {}
    
    with urllib.request.urlopen(req) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                lang = target_members.get(member.name) or target_members_alt.get(member.name)
                if lang:
                    f = tar.extractfile(member)
                    if f is not None:
                        lines = [
                            unicodedata.normalize("NFC", line.decode("utf-8").strip())
                            for line in f
                            if line.decode("utf-8").strip()
                        ]
                        extracted[lang] = lines
                        print(f"  -> Extracted {len(lines)} sentences for {lang} ({LANG_MAP[lang]})")

    # Assert parallel alignment
    assert len(extracted) == len(LANG_MAP), f"Missing languages! Extracted: {list(extracted.keys())}"
    counts = {l: len(s) for l, s in extracted.items()}
    assert len(set(counts.values())) == 1, f"Sentence counts differ: {counts}"
    n_sentences = list(counts.values())[0]
    print(f"\nAll {len(extracted)} languages verified strictly parallel with {n_sentences} sentences each.")

    # Save full devtest files
    for lang, lines in extracted.items():
        out_path = os.path.join(OUTPUT_DIR, f"{lang}.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            for line in lines:
                out_f.write(line + "\n")
        print(f"Saved: {out_path} ({len(lines)} lines)")

    # Save a 100-sentence smoke-test subset
    subset_dir = os.path.join(OUTPUT_DIR, "subset_100")
    os.makedirs(subset_dir, exist_ok=True)
    for lang, lines in extracted.items():
        out_path = os.path.join(subset_dir, f"{lang}.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            for line in lines[:100]:
                out_f.write(line + "\n")
    print(f"\nSaved 100-sentence subsets in {subset_dir}")

if __name__ == "__main__":
    main()
