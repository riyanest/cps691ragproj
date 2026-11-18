"""
Statistical analysis and reporting script for RAG benchmark results.
Reads results/benchmark_results.csv and results/summary_metrics.csv to produce detailed breakdowns.
"""

import os
import pandas as pd
import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def analyze_results():
    results_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    print("\n" + "=" * 95)
    print("DETAILED EXPERIMENTAL ANALYSIS & BENCHMARK REPORT")
    print("=" * 95)

    print(f"Total Evaluations: {len(df)} runs across {df['question_id'].nunique()} unique HotpotQA questions.")
    print(f"Conditions: {df['condition'].unique().tolist()}")
    print(f"Systems: {df['system'].unique().tolist()}")

    # 1. Overall Performance by System
    print("\n" + "-" * 95)
    print("1. OVERALL SYSTEM COMPARISON (Averaged across all noise conditions)")
    print("-" * 95)
    overall = df.groupby("system").agg(
        EM=("exact_match", lambda x: f"{x.mean()*100:.2f}%"),
        F1=("f1", lambda x: f"{x.mean()*100:.2f}%"),
        Containment=("containment", lambda x: f"{x.mean()*100:.2f}%"),
        Passage_Precision=("passage_precision", lambda x: f"{x.mean()*100:.2f}%"),
        Passage_Recall=("passage_recall", lambda x: f"{x.mean()*100:.2f}%"),
        Avg_Latency=("total_latency_seconds", lambda x: f"{x.mean():.2f}s"),
        Avg_Tokens=("total_tokens", lambda x: f"{x.mean():.0f}"),
        Avg_Calls=("llm_calls", lambda x: f"{x.mean():.1f}")
    ).reset_index()
    print(overall.to_string(index=False))

    # 2. Performance by Noise Condition (Cuconasu et al. Analysis)
    print("\n" + "-" * 95)
    print("2. NOISE IMPACT COMPARISON (Cuconasu et al., 2024: Distractor vs Injected Random Noise)")
    print("-" * 95)
    noise_comp = df.groupby(["condition", "system"]).agg(
        EM=("exact_match", lambda x: f"{x.mean()*100:.2f}%"),
        F1=("f1", lambda x: f"{x.mean()*100:.2f}%"),
        Passage_Precision=("passage_precision", lambda x: f"{x.mean()*100:.2f}%"),
        Avg_Latency=("total_latency_seconds", lambda x: f"{x.mean():.2f}s"),
        Avg_Tokens=("total_tokens", lambda x: f"{x.mean():.0f}")
    ).reset_index()
    print(noise_comp.to_string(index=False))

    # 3. Breakdown by Question Type (Bridge vs Comparison)
    print("\n" + "-" * 95)
    print("3. BREAKDOWN BY QUESTION REASONING TYPE (Bridge vs Comparison Multi-Hop)")
    print("-" * 95)
    type_comp = df.groupby(["type", "system"]).agg(
        EM=("exact_match", lambda x: f"{x.mean()*100:.2f}%"),
        F1=("f1", lambda x: f"{x.mean()*100:.2f}%"),
        Passage_Precision=("passage_precision", lambda x: f"{x.mean()*100:.2f}%")
    ).reset_index()
    print(type_comp.to_string(index=False))

    # 4. Passage Filtering Efficacy
    print("\n" + "-" * 95)
    print("4. PASSAGE FILTERING EFFICACY (Elimination of Irrelevant / Distractor Passages)")
    print("-" * 95)
    filter_stats = df.groupby("system").agg(
        Avg_Passages_Kept=("num_kept_passages", "mean"),
        Passage_Precision=("passage_precision", lambda x: f"{x.mean()*100:.2f}%"),
        Passage_Recall=("passage_recall", lambda x: f"{x.mean()*100:.2f}%")
    ).reset_index()
    print(filter_stats.to_string(index=False))
    print("=" * 95 + "\n")


if __name__ == "__main__":
    analyze_results()
