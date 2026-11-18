"""
Standard RAG Baseline (Retrieve -> Generate).
Retrieves top-k passages from Chroma and generates an answer in a single pass without verification.
"""

import time
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from src.llm_client import get_llm, invoke_with_metrics, UsageMetrics
from src.eval_metrics import exact_match_score, f1_score, containment_score, passage_level_metrics


STANDARD_RAG_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant answering questions based on the provided context passages.

Context:
{context}

Question: {question}

Instructions:
1. For yes/no questions, answer strictly "yes" or "no".
2. For other questions, provide ONLY the concise name, entity, location, or date.
Keep your response as short as possible without conversational filler.

Answer:"""
)


def run_standard_rag(
    question: str,
    ground_truth: str,
    gold_titles: List[str],
    vectorstore: Chroma,
    k: int = 4,
    llm: Any = None
) -> Dict[str, Any]:
    """
    Executes standard RAG on a given question and vectorstore.
    """
    if llm is None:
        llm = get_llm()

    metrics = UsageMetrics()
    t_start = time.time()

    # Step 1: Retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)

    retrieved_titles = [doc.metadata.get("title", "") for doc in retrieved_docs]
    context_str = "\n\n".join([
        f"[{doc.metadata.get('title', '')}]: {doc.metadata.get('text', doc.page_content)}"
        for doc in retrieved_docs
    ])

    # Step 2: Generation
    prompt_text = STANDARD_RAG_PROMPT.format(context=context_str, question=question)
    prediction = invoke_with_metrics(llm, prompt_text, metrics).strip()

    total_latency = time.time() - t_start

    # Metrics
    em = exact_match_score(prediction, ground_truth)
    f1 = f1_score(prediction, ground_truth)
    cont = containment_score(prediction, ground_truth)
    pass_prec, pass_rec = passage_level_metrics(retrieved_titles, gold_titles)

    usage = metrics.to_dict()

    return {
        "system": "Standard RAG",
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "exact_match": em,
        "f1": round(f1, 4),
        "containment": cont,
        "passage_precision": round(pass_prec, 4),
        "passage_recall": round(pass_rec, 4),
        "kept_titles": retrieved_titles,
        "num_kept_passages": len(retrieved_titles),
        "total_latency_seconds": round(total_latency, 3),
        "llm_calls": usage["llm_calls"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "llm_latency_seconds": usage["llm_latency_seconds"]
    }
