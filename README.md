# Evaluating Hybrid RAG for Arabic Regulatory Question Answering

A comparative evaluation of **Hybrid Retrieval-Augmented Generation (RAG)** versus an identical **No-RAG baseline** for Arabic financial regulatory question answering using Saudi Central Bank (SAMA) documents.

## Research Objective

This project investigates whether document-grounded retrieval reduces hallucinations and improves factual accuracy in Arabic regulatory question answering.

The proposed system combines:

- Sentence-BERT dense retrieval
- BM25 lexical retrieval
- Reciprocal Rank Fusion (RRF)
- Aya 8B multilingual language model

## Experimental Results

| Benchmark | No-RAG | Hybrid RAG |
|-----------|--------|------------|
| 50 Questions | 2.0% | **54.0%** |
| 30 Specificity | 13.3% | **43.3%** |

Additional evaluation includes:

- Retrieval Recall@5
- Question-level comparison
- Retrieval vs Generation error analysis

## Repository Structure

```text
src/        Core implementation
paper/      Publication manuscript
docs/       Technical report
figures/    Paper figures
data/       Evaluation benchmarks
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/evaluating-hybrid-rag-arabic-regulatory-qa.git
cd evaluating-hybrid-rag-arabic-regulatory-qa
pip install -r requirements.txt
```

## Usage

Build the retrieval index:

```bash
python -m src.indexer
```

Run Hybrid RAG evaluation:

```bash
python -m src.evaluation.run_rag
```

Run No-RAG baseline:

```bash
python -m src.evaluation.run_no_rag
```

## Dataset

The experiments use a 289-page Arabic regulatory corpus published by the Saudi Central Bank (SAMA).

> **Note:** The original SAMA PDF is not included due to redistribution considerations.

## Citation

If you use this repository, please cite the accompanying research paper included in the `paper/` directory.

## Author

**Muhammad Junaid Khan**

Department of Computer Science and Data Science

LORDS Institute of Engineering and Technology

India

Email: Belikejunaid007@gmail.com
