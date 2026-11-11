"""
Data loader and preprocessing for HotpotQA benchmark.
Extracts gold passages, related distractors, and creates a random noise corpus (Cuconasu et al., 2024).
"""

import json
import os
import random
from typing import Dict, List, Tuple
from datasets import load_dataset


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SUBSET_PATH = os.path.join(DATA_DIR, "hotpot_eval_subset.json")
NOISE_POOL_PATH = os.path.join(DATA_DIR, "random_noise_corpus.json")


def prepare_hotpot_subset(num_samples: int = 50, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Downloads validation split of HotpotQA distractor setting and extracts:
    - Balanced evaluation subset (bridge and comparison questions)
    - Pool of random passages for Cuconasu noise injection
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(SUBSET_PATH) and os.path.exists(NOISE_POOL_PATH):
        print(f"Loading existing cached subset from {SUBSET_PATH}...")
        with open(SUBSET_PATH, "r", encoding="utf-8") as f:
            eval_samples = json.load(f)
        with open(NOISE_POOL_PATH, "r", encoding="utf-8") as f:
            noise_pool = json.load(f)
        return eval_samples, noise_pool

    print("Downloading HotpotQA validation dataset...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)

    rng = random.Random(seed)
    bridge_samples = []
    comparison_samples = []
    noise_pool = []

    target_per_type = num_samples // 2

    for item in ds:
        q_id = item["id"]
        question = item["question"]
        answer = item["answer"]
        q_type = item["type"]
        level = item["level"]
        supporting_facts = item["supporting_facts"]
        context = item["context"]

        gold_titles = list(set(supporting_facts["title"]))

        # Build list of passages
        passages = []
        gold_passages = []
        distractor_passages = []

        for title, sents in zip(context["title"], context["sentences"]):
            text = " ".join(sents).strip()
            is_gold = title in gold_titles
            p_obj = {
                "title": title,
                "text": text,
                "is_gold": is_gold
            }
            passages.append(p_obj)
            if is_gold:
                gold_passages.append(p_obj)
            else:
                distractor_passages.append(p_obj)

        # Collect extra distractor passages into the global noise pool
        if len(noise_pool) < 200 and distractor_passages:
            noise_pool.extend([
                {"title": p["title"], "text": p["text"], "source_id": q_id}
                for p in distractor_passages[:2]
            ])

        # Filter out questions that don't have both gold passages in context
        if len(gold_passages) < 2:
            continue

        sample_record = {
            "id": q_id,
            "question": question,
            "answer": answer,
            "type": q_type,
            "level": level,
            "gold_titles": gold_titles,
            "gold_passages": gold_passages,
            "distractor_passages": distractor_passages,
            "all_context_passages": passages
        }

        if q_type == "bridge" and len(bridge_samples) < target_per_type:
            bridge_samples.append(sample_record)
        elif q_type == "comparison" and len(comparison_samples) < target_per_type:
            comparison_samples.append(sample_record)

        if len(bridge_samples) >= target_per_type and len(comparison_samples) >= target_per_type:
            break

    eval_samples = bridge_samples + comparison_samples
    rng.shuffle(eval_samples)

    print(f"Prepared {len(eval_samples)} evaluation samples ({len(bridge_samples)} bridge, {len(comparison_samples)} comparison).")
    print(f"Prepared {len(noise_pool)} random noise passages in noise corpus.")

    with open(SUBSET_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_samples, f, indent=2)

    with open(NOISE_POOL_PATH, "w", encoding="utf-8") as f:
        json.dump(noise_pool, f, indent=2)

    return eval_samples, noise_pool


if __name__ == "__main__":
    prepare_hotpot_subset(num_samples=50)
