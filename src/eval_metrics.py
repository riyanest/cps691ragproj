"""
Evaluation metrics for HotpotQA RAG systems:
- HotpotQA official Token F1 and Exact Match (EM)
- Normalized string containment
- Passage-level Precision and Recall against gold supporting facts
"""

import re
import string
from collections import Counter
from typing import List, Set, Tuple


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Exact match of normalized strings."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score according to official HotpotQA evaluation."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    pred_tokens = normalized_prediction.split()
    gt_tokens = normalized_ground_truth.split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def containment_score(prediction: str, ground_truth: str) -> float:
    """
    Checks if normalized ground truth is contained within the prediction,
    useful when generative LLM provides an explanation alongside the short answer.
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    if not norm_gt:
        return 0.0
    if norm_gt in norm_pred or norm_pred in norm_gt:
        return 1.0
    return 0.0


def passage_level_metrics(kept_titles: List[str], gold_titles: List[str]) -> Tuple[float, float]:
    """
    Computes passage-level Precision and Recall:
    - Precision: |Kept ∩ Gold| / |Kept|
    - Recall: |Kept ∩ Gold| / |Gold|
    """
    kept_set: Set[str] = set(kept_titles)
    gold_set: Set[str] = set(gold_titles)

    if not kept_set:
        return 0.0, 0.0

    overlap = kept_set.intersection(gold_set)
    precision = len(overlap) / len(kept_set) if kept_set else 0.0
    recall = len(overlap) / len(gold_set) if gold_set else 0.0

    return precision, recall
