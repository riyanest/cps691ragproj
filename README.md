# Robust Agentic RAG Benchmark: HotpotQA & Noise Resilience

Comparative study evaluating Retrieval-Augmented Generation (RAG) systems on multi-hop question answering under noise and distractor conditions (HotpotQA and Cuconasu et al., 2024).

## Systems Compared
1. **Standard RAG (Naive RAG)**: Direct top-k retrieval followed by answer synthesis.
2. **Canonical LangGraph Agentic RAG**: Baseline agent with binary document grading, global query rewriting, and retrieval retry loops.
3. **Extended Agentic RAG (Proposed)**:
   - **Passage-level scoring with threshold filtering ($\tau = 0.6$)**: Checks each retrieved passage separately to purge distractors and spurious noise.
   - **Sufficiency verification gate**: Evaluates whether surviving evidence is complete before answering; rewrites query only targeting the missing multi-hop gap.
   - **Noise resilience**: Evaluated against natural distractors and random noise injection.

## Project Structure
```
├── data/
│   ├── hotpot_eval_subset.json    # 50 balanced HotpotQA validation questions (bridge & comparison)
│   └── random_noise_corpus.json   # 200 random Wikipedia passages for Cuconasu noise injection
├── src/
│   ├── data_loader.py             # HotpotQA dataset preparation and noise corpus creation
│   ├── vector_store.py            # ChromaDB management with Ollama nomic-embed-text
│   ├── eval_metrics.py            # Token F1, Exact Match, Containment, Passage Precision/Recall
│   ├── standard_rag.py            # Baseline 1: Standard naive RAG
│   ├── langgraph_baseline_rag.py  # Baseline 2: Canonical LangGraph Agentic RAG
│   ├── extended_agentic_rag.py    # Proposed: Passage-scored Sufficiency-gated RAG
│   ├── experiments.py             # Benchmark runner across noise conditions
│   └── analysis.py                # Statistical analysis and CSV results summarizer
├── results/                       # Output CSV benchmark logs and evaluation summaries
├── requirements.txt               # Python package dependencies
└── README.md
```

## Setup & Quickstart

### 1. Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Local Ollama Models
Ensure Ollama is running and pull the models:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 3. Run Experiments
```bash
python -m src.experiments
```

## References
- Yan et al. (2024), *Corrective Retrieval Augmented Generation (CRAG)*: https://arxiv.org/abs/2401.15884
- Asai et al. (2024), *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*: https://arxiv.org/abs/2310.11511
- Cuconasu et al. (2024), *The Power of Noise: On the Role of Noise in Retrieval-Augmented Generation*: https://doi.org/10.1145/3626772.3657834
- Yang et al. (2018), *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*: https://arxiv.org/abs/1809.09600
- LangGraph Agentic RAG: https://docs.langchain.com/oss/python/langgraph/agentic-rag
