"""
Smoke test comparing 1 sample across all three systems.
"""

import json
from src.vector_store import create_vector_store_from_passages
from src.standard_rag import run_standard_rag
from src.langgraph_baseline_rag import run_langgraph_baseline
from src.extended_agentic_rag import run_extended_agentic_rag

with open("data/hotpot_eval_subset.json", "r") as f:
    samples = json.load(f)

sample = samples[0]
print(f"Testing Question: {sample['question']}")
print(f"Ground Truth: {sample['answer']}")
print(f"Gold Titles: {sample['gold_titles']}")

# Index passages
vs = create_vector_store_from_passages(sample["all_context_passages"])

print("\n--- 1. Testing Standard RAG ---")
res_std = run_standard_rag(
    question=sample["question"],
    ground_truth=sample["answer"],
    gold_titles=sample["gold_titles"],
    vectorstore=vs
)
print(f"Prediction: {res_std['prediction']}")
print(f"F1: {res_std['f1']}, EM: {res_std['exact_match']}, Prec: {res_std['passage_precision']}")
print(f"Latency: {res_std['total_latency_seconds']}s, Calls: {res_std['llm_calls']}, Tokens: {res_std['total_tokens']}")

print("\n--- 2. Testing LangGraph Baseline ---")
res_base = run_langgraph_baseline(
    question=sample["question"],
    ground_truth=sample["answer"],
    gold_titles=sample["gold_titles"],
    vectorstore=vs,
    max_retries=1
)
print(f"Prediction: {res_base['prediction']}")
print(f"F1: {res_base['f1']}, EM: {res_base['exact_match']}, Prec: {res_base['passage_precision']}")
print(f"Latency: {res_base['total_latency_seconds']}s, Calls: {res_base['llm_calls']}, Tokens: {res_base['total_tokens']}")

print("\n--- 3. Testing Extended Agentic RAG ---")
res_ext = run_extended_agentic_rag(
    question=sample["question"],
    ground_truth=sample["answer"],
    gold_titles=sample["gold_titles"],
    vectorstore=vs,
    threshold=0.6,
    max_retries=1
)
print(f"Prediction: {res_ext['prediction']}")
print(f"F1: {res_ext['f1']}, EM: {res_ext['exact_match']}, Prec: {res_ext['passage_precision']}")
print(f"Latency: {res_ext['total_latency_seconds']}s, Calls: {res_ext['llm_calls']}, Tokens: {res_ext['total_tokens']}")
print(f"Kept titles: {res_ext['kept_titles']}")
