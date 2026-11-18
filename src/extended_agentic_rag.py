"""
Extended Agentic RAG implementation with:
1. Passage-level scoring with threshold filtering (s >= tau).
2. Evidence sufficiency verification gate (routes directly to generation if evidence is complete;
   rewrites query only targeting the missing multi-hop gap if insufficient).
3. Robustness against distractor and random noise passages (Cuconasu et al., 2024).
"""

import re
import time
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from src.llm_client import get_llm, invoke_with_metrics, UsageMetrics
from src.eval_metrics import exact_match_score, f1_score, containment_score, passage_level_metrics


PASSAGE_SCORER_PROMPT = PromptTemplate.from_template(
    """You are an expert evaluator assessing whether a passage provides useful evidence for answering a question.
Keep in mind that complex questions require multi-hop clues from different passages.
A passage is USEFUL if it provides facts, attributes (such as nationality, birth, role, dates), or details about ANY person, entity, or relationship involved in the question.

Passage:
[{title}]: {text}

Question: {question}

Scoring guide:
- 0.8 to 1.0: Directly provides key attributes or facts about a person or entity mentioned in the question (e.g., their nationality, profession, identity).
- 0.5 to 0.7: Relevant supporting context about an entity mentioned in the question.
- 0.2 to 0.4: Distractor (e.g., an unrelated movie or place sharing a similar name).
- 0.0 to 0.1: Completely irrelevant or random noise.

Score (number from 0.0 to 1.0):"""
)


SUFFICIENCY_CHECKER_PROMPT = PromptTemplate.from_template(
    """You are a multi-hop evidence verifier.
Assess whether the provided evidence passages contain SUFFICIENT facts to fully answer the question.

Question: {question}

Evidence Passages:
{evidence}

Is the evidence sufficient to answer the question?
Format your response exactly as:
Status: [SUFFICIENT or INSUFFICIENT]
Missing: [If INSUFFICIENT, specify the exact missing entity or fact; if SUFFICIENT, write None]"""
)


TARGETED_REWRITE_PROMPT = PromptTemplate.from_template(
    """You are an expert at formulating targeted search queries for multi-hop question answering.
We are answering: "{question}"
We currently have partial evidence, but are missing this specific piece:
Missing: {missing}

Formulate a concise, focused search query to find this missing information.
Targeted Query (provide only the query, no quotes or explanation):"""
)


ANSWER_GENERATION_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant answering questions based on the provided verified evidence passages.

Evidence:
{context}

Question: {question}

Instructions:
1. For yes/no questions, answer strictly "yes" or "no".
2. For other questions, provide ONLY the concise name, entity, location, or date.
Keep your response as short as possible without conversational filler.

Answer:"""
)


class ExtendedGraphState(TypedDict):
    question: str
    current_query: str
    all_retrieved_docs: List[Document]
    filtered_docs: List[Document]
    passage_scores: List[float]
    is_sufficient: bool
    missing_gap: str
    loop_step: int
    max_retries: int
    threshold: float
    generation: str
    vectorstore: Any
    llm: Any
    metrics: Any


def parse_score(score_str: str) -> float:
    """Extracts a float score between 0.0 and 1.0 from LLM output."""
    # Look for explicit score mention first
    score_match = re.search(r"score\s*:\s*(\d+(\.\d+)?)", score_str, re.IGNORECASE)
    if score_match:
        val = float(score_match.group(1))
    else:
        match = re.search(r"(\d+(\.\d+)?)", score_str)
        if match:
            val = float(match.group(1))
        else:
            return 0.5

    # Normalize if on 1-10 or 1-100 scale
    if val > 1.0 and val <= 10.0:
        val = val / 10.0
    elif val > 10.0 and val <= 100.0:
        val = val / 100.0
    return min(max(val, 0.0), 1.0)


def retrieve_passages_node(state: ExtendedGraphState) -> Dict[str, Any]:
    """Retrieves top-k candidate passages from vectorstore."""
    vectorstore: Chroma = state["vectorstore"]
    query = state["current_query"]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    new_docs = retriever.invoke(query)

    existing_titles = {d.metadata.get("title") for d in state["all_retrieved_docs"]}
    combined = list(state["all_retrieved_docs"])
    for d in new_docs:
        if d.metadata.get("title") not in existing_titles:
            combined.append(d)
            existing_titles.add(d.metadata.get("title"))

    return {
        "all_retrieved_docs": combined
    }


def score_and_filter_node(state: ExtendedGraphState) -> Dict[str, Any]:
    """
    Extension 1: Individually scores each passage and filters with threshold tau.
    Purges unhelpful distractors and random noise passages.
    """
    llm = state["llm"]
    metrics: UsageMetrics = state["metrics"]
    question = state["question"]
    docs = state["all_retrieved_docs"]
    tau = state["threshold"]

    filtered = []
    scores = []

    for doc in docs:
        title = doc.metadata.get("title", "")
        text = doc.metadata.get("text", doc.page_content)
        prompt_text = PASSAGE_SCORER_PROMPT.format(
            title=title,
            text=text,
            question=question
        )
        resp = invoke_with_metrics(llm, prompt_text, metrics)
        score = parse_score(resp)
        scores.append(score)

        if score >= tau:
            doc.metadata["utility_score"] = score
            filtered.append(doc)

    # Fallback: if strict threshold removed all docs, keep the highest scoring doc
    if not filtered and docs:
        best_idx = int(scores.index(max(scores))) if scores else 0
        filtered.append(docs[best_idx])

    return {
        "filtered_docs": filtered,
        "passage_scores": scores
    }


def check_sufficiency_node(state: ExtendedGraphState) -> Dict[str, Any]:
    """
    Extension 2: Sufficiency Verification Gate.
    Verifies if the surviving filtered evidence has enough facts to answer the question.
    """
    llm = state["llm"]
    metrics: UsageMetrics = state["metrics"]
    question = state["question"]
    filtered_docs = state["filtered_docs"]

    evidence_text = "\n\n".join([
        f"[{d.metadata.get('title', '')}]: {d.metadata.get('text', d.page_content)}"
        for d in filtered_docs
    ])

    prompt_text = SUFFICIENCY_CHECKER_PROMPT.format(
        question=question,
        evidence=evidence_text if evidence_text else "No evidence available."
    )
    resp = invoke_with_metrics(llm, prompt_text, metrics)

    is_sufficient = "status: sufficient" in resp.lower() or ("sufficient" in resp.lower() and "insufficient" not in resp.lower())
    missing_gap = ""
    if "missing:" in resp.lower():
        missing_gap = resp.lower().split("missing:", 1)[1].strip()

    return {
        "is_sufficient": is_sufficient,
        "missing_gap": missing_gap
    }


def rewrite_gap_query_node(state: ExtendedGraphState) -> Dict[str, Any]:
    """
    Formulates a targeted rewrite query focusing on the missing multi-hop gap.
    """
    llm = state["llm"]
    metrics: UsageMetrics = state["metrics"]
    question = state["question"]
    missing = state["missing_gap"]
    loop_step = state["loop_step"] + 1

    prompt_text = TARGETED_REWRITE_PROMPT.format(
        question=question,
        missing=missing if missing else "additional supporting facts"
    )
    targeted_query = invoke_with_metrics(llm, prompt_text, metrics).strip().strip('"\'')

    return {
        "current_query": targeted_query,
        "loop_step": loop_step
    }


def generate_answer_node(state: ExtendedGraphState) -> Dict[str, Any]:
    """Synthesizes final answer from verified, filtered passages."""
    llm = state["llm"]
    metrics: UsageMetrics = state["metrics"]
    question = state["question"]
    docs = state["filtered_docs"]

    context_str = "\n\n".join([
        f"[{d.metadata.get('title', '')}]: {d.metadata.get('text', d.page_content)}"
        for d in docs
    ])

    prompt_text = ANSWER_GENERATION_PROMPT.format(
        context=context_str,
        question=question
    )
    answer = invoke_with_metrics(llm, prompt_text, metrics).strip()
    return {"generation": answer}


def sufficiency_route_decision(state: ExtendedGraphState) -> str:
    """
    Routes directly to generate if evidence is sufficient.
    Only routes to rewrite and retry if evidence is insufficient and retries remain.
    """
    if state["is_sufficient"]:
        return "generate_answer"
    if state["loop_step"] < state["max_retries"]:
        return "rewrite_gap_query"
    return "generate_answer"


def build_extended_graph():
    """Builds and compiles the Extended Agentic RAG workflow."""
    workflow = StateGraph(ExtendedGraphState)

    workflow.add_node("retrieve_passages", retrieve_passages_node)
    workflow.add_node("score_and_filter", score_and_filter_node)
    workflow.add_node("check_sufficiency", check_sufficiency_node)
    workflow.add_node("rewrite_gap_query", rewrite_gap_query_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("retrieve_passages")
    workflow.add_edge("retrieve_passages", "score_and_filter")
    workflow.add_edge("score_and_filter", "check_sufficiency")
    workflow.add_conditional_edges(
        "check_sufficiency",
        sufficiency_route_decision,
        {
            "generate_answer": "generate_answer",
            "rewrite_gap_query": "rewrite_gap_query"
        }
    )
    workflow.add_edge("rewrite_gap_query", "retrieve_passages")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


_EXTENDED_GRAPH = None


def run_extended_agentic_rag(
    question: str,
    ground_truth: str,
    gold_titles: List[str],
    vectorstore: Chroma,
    threshold: float = 0.6,
    max_retries: int = 1,
    llm: Any = None
) -> Dict[str, Any]:
    """
    Executes Extended Agentic RAG on a given question.
    """
    global _EXTENDED_GRAPH
    if _EXTENDED_GRAPH is None:
        _EXTENDED_GRAPH = build_extended_graph()

    if llm is None:
        llm = get_llm()

    metrics = UsageMetrics()
    t_start = time.time()

    initial_state: ExtendedGraphState = {
        "question": question,
        "current_query": question,
        "all_retrieved_docs": [],
        "filtered_docs": [],
        "passage_scores": [],
        "is_sufficient": False,
        "missing_gap": "",
        "loop_step": 0,
        "max_retries": max_retries,
        "threshold": threshold,
        "generation": "",
        "vectorstore": vectorstore,
        "llm": llm,
        "metrics": metrics
    }

    final_state = _EXTENDED_GRAPH.invoke(initial_state)
    total_latency = time.time() - t_start

    prediction = final_state["generation"]
    kept_docs = final_state["filtered_docs"]
    kept_titles = [doc.metadata.get("title", "") for doc in kept_docs]

    em = exact_match_score(prediction, ground_truth)
    f1 = f1_score(prediction, ground_truth)
    cont = containment_score(prediction, ground_truth)
    pass_prec, pass_rec = passage_level_metrics(kept_titles, gold_titles)

    usage = metrics.to_dict()

    return {
        "system": "Extended Agentic RAG",
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "exact_match": em,
        "f1": round(f1, 4),
        "containment": cont,
        "passage_precision": round(pass_prec, 4),
        "passage_recall": round(pass_rec, 4),
        "kept_titles": kept_titles,
        "num_kept_passages": len(kept_titles),
        "retries_performed": final_state["loop_step"],
        "is_sufficient": final_state["is_sufficient"],
        "total_latency_seconds": round(total_latency, 3),
        "llm_calls": usage["llm_calls"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "llm_latency_seconds": usage["llm_latency_seconds"]
    }
