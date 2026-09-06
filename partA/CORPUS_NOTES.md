# Multilingual Evaluation Corpus Documentation

## 1. Overview & Dataset Selection
To replace the 10-sentence smoke-test sample from `REPORT_v0`, we assembled a standardized, high-quality parallel evaluation benchmark using the **FLORES-200** `devtest` split.

FLORES-200 is an open, professionally translated parallel benchmark created by Meta AI for multi-way evaluation across 200+ languages. Every sentence in the dataset is strictly aligned across all target languages, ensuring that the semantic information content is held strictly constant across languages.

## 2. Language Coverage & Statistics

We evaluate **8 languages**, exceeding the minimum requirement (English + Hindi + at least two Dravidian languages) and providing full coverage for the Indic languages evaluated in Part C:

| Language Code | Language | Family / Branch | Script (ISO 15924) | Parallel Sentences | Total Words | Words / Sentence | UTF-8 Bytes | Bytes / Char |
|---|---|---|---|---|---|---|---|---|
| `eng` | English | Indo-European (Germanic) | Latin (`Latn`) | 1,012 | 21,950 | 21.7 | 131,894 | 1.00 |
| `hin` | Hindi | Indo-European (Indo-Aryan) | Devanagari (`Deva`) | 1,012 | 25,643 | 25.3 | 337,439 | 2.57 |
| `kan` | Kannada | Dravidian (Southern) | Kannada (`Knda`) | 1,012 | 16,100 | 15.9 | 375,341 | 2.72 |
| `tam` | Tamil | Dravidian (Southern) | Tamil (`Taml`) | 1,012 | 16,775 | 16.6 | 421,635 | 2.74 |
| `tel` | Telugu | Dravidian (South-Central) | Telugu (`Telu`) | 1,012 | 16,938 | 16.7 | 353,661 | 2.67 |
| `mal` | Malayalam | Dravidian (Southern) | Malayalam (`Mlym`) | 1,012 | 14,930 | 14.8 | 411,741 | 2.76 |
| `ben` | Bengali | Indo-European (Indo-Aryan) | Bengali (`Beng`) | 1,012 | 19,506 | 19.3 | 348,729 | 2.67 |
| `mar` | Marathi | Indo-European (Indo-Aryan) | Devanagari (`Deva`) | 1,012 | 19,046 | 18.8 | 355,511 | 2.67 |

## 3. Domain & Preprocessing Pipeline
- **Domain**: Broad general domain sampled from multi-topic sources (Wiki articles, travel guides, news, government publications, narratives).
- **Preprocessing Pipeline**:
  1. Strict sentence alignment verification across all 8 languages ($N = 1,012$ parallel pairs).
  2. Unicode Normalization: Standardized to **NFC** (`unicodedata.normalize("NFC")`) to ensure combining characters (matras, nuktas, halant/virama) are canonicalized.
  3. Casing preservation: Preserved original casing to reflect realistic production input streams.

## 4. Critical Corpus Caveats: What This Corpus Cannot Tell Us

While FLORES-200 provides an immaculate semantic baseline holding meaning invariant, production teams must understand its fundamental boundaries:

1. **Formal/Standard vs Colloquial Register**: FLORES-200 consists of standard, grammatical, textbook-style prose. Real-world user traffic in India is heavily conversational, informal, and contains colloquial contractions, slang, and dialectal variations that may exhibit different subword fragmentation.
2. **Code-Switching & Romanized Transliteration (Hinglish/Tanglish/Kanglish)**: A massive portion of consumer AI traffic in India is typed using Latin script (e.g. *"Kya haal hai bhai?"* or *"Naan nalla irukken"*). FLORES-200 is exclusively in native Brahmic scripts and Latin English; it cannot quantify tokenization fertility on code-switched or romanized Indic text (which frequently fragments severely due to non-standardized phonetic spelling).
3. **Noisy Text & Formatting Artifacts**: FLORES sentences are clean and curated. They do not test robustness against OCR errors, speech-to-text transcript noise, emoji spam, or unnormalized multi-codepoint sequences.
4. **Sentence Length Distribution**: FLORES sentences average 15–25 words (~120–150 characters). It does not test long-document or short micro-utterance (1–3 word prompt) edge cases.
