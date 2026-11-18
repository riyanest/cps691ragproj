"""
Baseline LangGraph Agentic RAG implementation.
Faithfully reproduces the canonical LangGraph Agentic RAG / CRAG workflow:
1. retrieve: Retrieves candidate documents from Chroma.
2. grade_documents: Evaluates retrieved text with a binary yes/no relevance decision.
3. conditional route: If relevant -> generate; If irrelevant and retries available -> rewrite_question and retry.
4. rewrite_question: Global query rewrite for retrieval.
5. generate: Answer synthesis.
"""

import time
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from src.llm_client import get_llm, invoke_with_metrics, UsageMetrics
from src.eval_metrics import exact_match_score, f1_score, containment_score, passage_level_metrics


# Binary Relevance Grader Prompt
BINARY_GRADER_PROMPT = PromptTemplate.from_template(
    """You are a grader assessing whether a retrieved document is relevant to a question.
Document:
{document}

Question: {question}

Assess if the document contains keywords, entities, or semantic meaning related to the question.
Answer with ONLY "yes" if relevant, or "no" if irrelevant.

Decision (yes/no):"""
)

# Query Rewriter Prompt
REWRITE_PROMPT = PromptTemplate.from_template(
    """You are an expert at optimizing search queries for vector retrieval.
The previous search query did not retrieve sufficient relevant documents.
Look at the initial question and formulate an improved, clearer web/vector search query.

Initial Question: {question}
Current Query: {current_query}

Improved Query (provide only the new query string, no explanation):"""
)

# Answer Generation Prompt
GENERATION_PROMPT = PromptTemplate.from_template(
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


class BaselineGraphState(TypedDict):
    question: str
    current_query: str
    documents: List[Document]
    loop_step: int
    max_retries: int
    is_relevant: bool
    generation: str
    vectorstore: Any
    llm: Any
    metrics: Any


def retrieve_node(state: BaselineGraphState) -> Dict[str, Any]:
    """Retrieve top-k documents based on current query."""
    vectorstore: Chroma = state["vectorstore"]
    query = state["current_query"]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    return {"documents": docs}


def grade_documents_node(state: BaselineGraphState) -> Dict[str, Any]:
    """
    Grades retrieved documents with binary yes/no relevance decisions.
    Following LangGraph tutorial: keeps docs graded 'yes'.
    If at least one doc is relevant, overall is_relevant is True; else False.
    """
    llm = state["llm"]
    metrics: UsageMetrics = state["metrics"]
    question = state["question"]
    docs = state["documents"]

    filtered_docs = []
    any_relevant = False

    for doc in docs:
        prompt_text = BINARY_GRADER_PROMPT.format(
            document=f"[{doc.metadata.get('title', '')}]: {doc.page_content}",
            question=question
        )
        res = invoke_with_metrics(llm, prompt_text, metrics).lower().strip()
        if "yes" in res:
            filtered_docs.append(doc)
            any_relevant = True

    # If none were graded relevant, keep original docs so generation can still attempt
    if not filtered_docs:
        filtered_docs = docs
        any_relevant = False

    return {
        "documents": filtered_docs,
        "is_relevant": any_relevant
    }


def rewrite_question_node(state: BaselineGraphState) -> Dict[str, Any]:
    """Rewrites the query to retry retrieval."""
    llm = state["llm"]
    metrics: UsageMetrics = state["metrics"]
    question = state["question"]
    current_query = state["current_query"]
    loop_step = state["loop_step"] + 1

    prompt_text = REWRITE_PROMPT.format(
        question=question,
        current_query=current_query
    )
    new_query = invoke_with_metrics(llm, prompt_text, metrics).strip()
    new_query = new_query.strip('"\'')

    return {
        "current_query": new_query,
        "loop_step": loop_step
    }


def generate_node(state: BaselineGraphState) -> Dict[str, Any]:
    """Generates the final answer from filtered documents."""
    llm = state["llm"]
    metrics: UsageMetrics = state["metrics"]
    question = state["question"]
    docs = state["documents"]

    context_str = "\n\n".join([
        f"[{doc.metadata.get('title', '')}]: {doc.metadata.get('text', doc.page_content)}"
        for doc in docs
    ])

    prompt_text = GENERATION_PROMPT.format(context=context_str, question=question)
    answer = invoke_with_metrics(llm, prompt_text, metrics).strip()
    return {"generation": answer}


def route_decision(state: BaselineGraphState) -> str:
    """Conditional edge deciding whether to generate or rewrite."""
    if state["is_relevant"]:
        return "generate"
    if state["loop_step"] < state["max_retries"]:
        return "rewrite_question"
    return "generate"


def build_baseline_graph():
    """Builds and compiles the canonical LangGraph Agentic RAG workflow."""
    workflow = StateGraph(BaselineGraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_question", rewrite_question_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_decision,
        {
            "generate": "generate",
            "rewrite_question": "rewrite_question"
        }
    )
    workflow.add_edge("rewrite_question", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


_BASELINE_GRAPH = None


def run_langgraph_baseline(
    question: str,
    ground_truth: str,
    gold_titles: List[str],
    vectorstore: Chroma,
    max_retries: int = 1,
    llm: Any = None
) -> Dict[str, Any]:
    """
    Executes the canonical LangGraph baseline agentic RAG on a question.
    """
    global _BASELINE_GRAPH
    if _BASELINE_GRAPH is None:
        _BASELINE_GRAPH = build_baseline_graph()

    if llm is None:
        llm = get_llm()

    metrics = UsageMetrics()
    t_start = time.time()

    initial_state: BaselineGraphState = {
        "question": question,
        "current_query": question,
        "documents": [],
        "loop_step": 0,
        "max_retries": max_retries,
        "is_relevant": False,
        "generation": "",
        "vectorstore": vectorstore,
        "llm": llm,
        "metrics": metrics
    }

    final_state = _BASELINE_GRAPH.invoke(initial_state)
    total_latency = time.time() - t_start

    prediction = final_state["generation"]
    kept_docs = final_state["documents"]
    kept_titles = [doc.metadata.get("title", "") for doc in kept_docs]

    em = exact_match_score(prediction, ground_truth)
    f1 = f1_score(prediction, ground_truth)
    cont = containment_score(prediction, ground_truth)
    pass_prec, pass_rec = passage_level_metrics(kept_titles, gold_titles)

    usage = metrics.to_dict()

    return {
        "system": "LangGraph Baseline",
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
        "total_latency_seconds": round(total_latency, 3),
        "llm_calls": usage["llm_calls"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "llm_latency_seconds": usage["llm_latency_seconds"]
    }
