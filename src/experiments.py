"""
Benchmark experiment runner comparing:
1. Standard RAG (Naive)
2. LangGraph Baseline Agentic RAG
3. Extended Agentic RAG (Ours)

Evaluated under two noise conditions (Cuconasu et al., 2024):
- Condition A: "distractor" (2 gold + 8 natural HotpotQA distractors)
- Condition B: "random_noise" (2 gold + 8 distractors + 5 injected random passages)

Supports automatic resuming: skips already evaluated (question_id, condition, system) triplets.
Saves detailed per-sample outputs incrementally to CSV in results/.
"""

import argparse
import json
import os
import random
import time
from typing import Dict, List, Any, Set, Tuple
import pandas as pd
from tqdm import tqdm

from src.vector_store import create_vector_store_from_passages
from src.standard_rag import run_standard_rag
from src.langgraph_baseline_rag import run_langgraph_baseline
from src.extended_agentic_rag import run_extended_agentic_rag
from src.analysis import analyze_results


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "benchmark_results.csv")
SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, "summary_metrics.csv")


def load_dataset_and_noise():
    subset_path = os.path.join(DATA_DIR, "hotpot_eval_subset.json")
    noise_path = os.path.join(DATA_DIR, "random_noise_corpus.json")

    with open(subset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    with open(noise_path, "r", encoding="utf-8") as f:
        noise_pool = json.load(f)

    return samples, noise_pool


def inject_random_noise(
    context_passages: List[Dict],
    noise_pool: List[Dict],
    current_q_id: str,
    num_random: int = 5,
    rng: random.Random = None
) -> List[Dict]:
    """
    Injects completely random passages from other articles (Cuconasu et al., 2024).
    """
    if rng is None:
        rng = random.Random(42)

    candidates = [p for p in noise_pool if p.get("source_id") != current_q_id]
    selected_noise = rng.sample(candidates, min(num_random, len(candidates)))

    noisy_passages = list(context_passages)
    for p in selected_noise:
        noisy_passages.append({
            "title": p["title"],
            "text": p["text"],
            "is_gold": False,
            "is_random_noise": True
        })

    rng.shuffle(noisy_passages)
    return noisy_passages


def load_existing_records() -> Tuple[List[Dict[str, Any]], Set[Tuple[str, str, str]]]:
    """Loads existing records from CSV to avoid redundant re-runs."""
    if os.path.exists(RESULTS_CSV_PATH):
        try:
            df = pd.read_csv(RESULTS_CSV_PATH)
            existing_records = df.to_dict(orient="records")
            existing_keys = {
                (str(r["question_id"]), str(r["condition"]), str(r["system"]))
                for r in existing_records
            }
            return existing_records, existing_keys
        except Exception:
            return [], set()
    return [], set()


def save_checkpoint(records: List[Dict[str, Any]]):
    """Flushes records to CSV immediately."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_CSV_PATH, index=False)

    summary_cols = [
        "exact_match", "f1", "containment",
        "passage_precision", "passage_recall",
        "total_latency_seconds", "llm_calls",
        "prompt_tokens", "completion_tokens", "total_tokens"
    ]
    df_summary = df.groupby(["condition", "system"])[summary_cols].mean().reset_index()
    df_summary.to_csv(SUMMARY_CSV_PATH, index=False)


def run_benchmark(
    num_samples: int = 11,
    conditions: List[str] = None,
    threshold: float = 0.6,
    max_retries: int = 1
):
    """
    Runs the full comparative benchmark across systems and noise conditions.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if conditions is None:
        conditions = ["distractor", "random_noise"]

    samples, noise_pool = load_dataset_and_noise()
    eval_samples = samples[:num_samples]
    rng = random.Random(1337)

    all_records, existing_keys = load_existing_records()
    print(f"\n==================================================================")
    print(f"Starting Benchmark: {len(eval_samples)} samples across {len(conditions)} conditions")
    print(f"Loaded {len(existing_keys)} existing evaluations from checkpoint.")
    print(f"Conditions: {conditions}")
    print(f"Scoring Threshold: {threshold} | Max Retries: {max_retries}")
    print(f"==================================================================\n")

    for c_idx, condition in enumerate(conditions):
        print(f"\n>>> Running Condition [{c_idx+1}/{len(conditions)}]: {condition.upper()}")

        for s_idx, sample in enumerate(eval_samples):
            q_id = sample["id"]
            question = sample["question"]
            ground_truth = sample["answer"]
            q_type = sample["type"]
            gold_titles = sample["gold_titles"]

            # Check if all 3 systems already evaluated for this question & condition
            needed_systems = [
                s for s in ["Standard RAG", "LangGraph Baseline", "Extended Agentic RAG"]
                if (str(q_id), str(condition), s) not in existing_keys
            ]

            if not needed_systems:
                continue

            print(f"[{condition.upper()} {s_idx+1}/{len(eval_samples)}] Q: {question[:65]}... (GT: {ground_truth})")
            print(f"  Evaluating systems: {needed_systems}")

            # Prepare passage corpus based on condition
            if condition == "distractor":
                passages = sample["all_context_passages"]
            elif condition == "random_noise":
                passages = inject_random_noise(
                    sample["all_context_passages"],
                    noise_pool,
                    current_q_id=q_id,
                    num_random=5,
                    rng=rng
                )
            else:
                passages = sample["all_context_passages"]

            # Build in-memory vector store
            vs = create_vector_store_from_passages(passages)

            # System 1: Standard Naive RAG
            if "Standard RAG" in needed_systems:
                try:
                    res_std = run_standard_rag(
                        question=question,
                        ground_truth=ground_truth,
                        gold_titles=gold_titles,
                        vectorstore=vs
                    )
                    res_std.update({
                        "question_id": q_id,
                        "type": q_type,
                        "condition": condition
                    })
                    all_records.append(res_std)
                    existing_keys.add((str(q_id), str(condition), "Standard RAG"))
                except Exception as e:
                    print(f"  [Error Standard RAG]: {e}")

            # System 2: LangGraph Baseline
            if "LangGraph Baseline" in needed_systems:
                try:
                    res_base = run_langgraph_baseline(
                        question=question,
                        ground_truth=ground_truth,
                        gold_titles=gold_titles,
                        vectorstore=vs,
                        max_retries=max_retries
                    )
                    res_base.update({
                        "question_id": q_id,
                        "type": q_type,
                        "condition": condition
                    })
                    all_records.append(res_base)
                    existing_keys.add((str(q_id), str(condition), "LangGraph Baseline"))
                except Exception as e:
                    print(f"  [Error LangGraph Baseline]: {e}")

            # System 3: Extended Agentic RAG
            if "Extended Agentic RAG" in needed_systems:
                try:
                    res_ext = run_extended_agentic_rag(
                        question=question,
                        ground_truth=ground_truth,
                        gold_titles=gold_titles,
                        vectorstore=vs,
                        threshold=threshold,
                        max_retries=max_retries
                    )
                    res_ext.update({
                        "question_id": q_id,
                        "type": q_type,
                        "condition": condition
                    })
                    all_records.append(res_ext)
                    existing_keys.add((str(q_id), str(condition), "Extended Agentic RAG"))
                except Exception as e:
                    print(f"  [Error Extended Agentic RAG]: {e}")

            # Checkpoint to disk on every sample
            save_checkpoint(all_records)

    print("\nBenchmark completed successfully! Generating analysis report...\n")
    analyze_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG benchmark on HotpotQA")
    parser.add_argument("--samples", type=int, default=11, help="Number of HotpotQA questions to evaluate")
    parser.add_argument("--threshold", type=float, default=0.6, help="Passage relevance threshold")
    parser.add_argument("--retries", type=int, default=1, help="Max retry loops")
    args = parser.parse_args()

    run_benchmark(num_samples=args.samples, threshold=args.threshold, max_retries=args.retries)
