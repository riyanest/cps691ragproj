"""
LLM client and token/cost tracker for Ollama.
Wraps model invocations to measure latency, call counts, and token usage.
"""

import time
from typing import Any, Dict, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage


class UsageMetrics:
    def __init__(self):
        self.llm_calls: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.latency_seconds: float = 0.0

    def record_call(self, prompt_tokens: int, completion_tokens: int, duration: float):
        self.llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.latency_seconds += duration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_latency_seconds": round(self.latency_seconds, 3)
        }


def get_llm(model: str = "llama3.2:3b", temperature: float = 0.0) -> ChatOllama:
    """Returns a ChatOllama instance."""
    return ChatOllama(model=model, temperature=temperature)


def invoke_with_metrics(
    llm: ChatOllama,
    input_messages: Any,
    metrics: UsageMetrics
) -> str:
    """Invokes LLM, records metrics, and returns text content."""
    t0 = time.time()
    response = llm.invoke(input_messages)
    elapsed = time.time() - t0

    metadata = getattr(response, "response_metadata", {}) or {}
    p_tokens = metadata.get("prompt_eval_count", 0)
    c_tokens = metadata.get("eval_count", 0)

    # Fallback heuristic if not provided
    if p_tokens == 0:
        p_tokens = len(str(input_messages)) // 4
    if c_tokens == 0:
        c_tokens = len(str(response.content)) // 4

    metrics.record_call(p_tokens, c_tokens, elapsed)
    return str(response.content).strip()
