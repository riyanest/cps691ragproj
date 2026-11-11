"""
Vector store management using Chroma and OllamaEmbeddings.
Constructs document indices for HotpotQA questions under various noise conditions.
"""

from typing import List, Dict, Optional
import uuid
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


_GLOBAL_EMBEDDINGS: Optional[OllamaEmbeddings] = None


def get_embeddings() -> OllamaEmbeddings:
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is None:
        _GLOBAL_EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")
    return _GLOBAL_EMBEDDINGS


def create_vector_store_from_passages(
    passages: List[Dict],
    collection_name_prefix: str = "eval"
) -> Chroma:
    """
    Creates an in-memory Chroma vector store populated with the provided passages.
    Passages format: [{'title': str, 'text': str, 'is_gold': bool, ...}]
    """
    embeddings = get_embeddings()
    col_name = f"{collection_name_prefix}_{uuid.uuid4().hex[:8]}"

    docs = [
        Document(
            page_content=f"{p['title']}: {p['text']}",
            metadata={
                "title": p["title"],
                "text": p["text"],
                "is_gold": p.get("is_gold", False),
                "is_random_noise": p.get("is_random_noise", False)
            }
        )
        for p in passages
    ]

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=col_name
    )
    return vectorstore
