#!/usr/bin/env bash
set -e

echo "================================================================="
echo "  AI Team Intern Assignment — The Audit Execution Suite"
echo "================================================================="
echo ""

PYTHON="./.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Virtual environment not found. Using system python3..."
    PYTHON="python3"
fi

echo "1. Checking / Extracting Parallel Multilingual Corpus..."
$PYTHON partA/build_corpus.py

echo ""
echo "2. Running Part A Tokenizer Audit Experiments (Evidence Rule)..."
$PYTHON partA/audit_experiments.py | tee partA/audit_experiments_output.txt

echo ""
echo "3. Running Part A Corrected Multi-Tokenizer & Multi-Denominator Benchmark..."
$PYTHON partA/corrected_fertility.py | tee partA/corrected_fertility_output.txt

echo ""
echo "4. Running Part B Capacity & Goodput Mathematical Verification..."
$PYTHON partB/verify_capacity.py | tee partB/verify_capacity_output.txt

echo ""
echo "================================================================="
echo "  All audit benchmarks and verification suites completed successfully!"
echo "================================================================="
