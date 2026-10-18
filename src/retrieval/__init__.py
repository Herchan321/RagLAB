"""Public retrieval API for the RAG project.

This package is the import target used by the application code. The project also
contains a root-level `src/retrieval.py` module with the baseline vector search,
so the package re-exports the relevant symbols from that implementation while
adding the hybrid/BM25 retrieval utilities.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .bm25_search import search_bm25
from .hybrid_search import search_hybrid


def _load_legacy_retrieval_module():
    """Load the legacy retrieval module from the sibling file path."""
    module_path = Path(__file__).resolve().parents[1] / "retrieval.py"
    spec = importlib.util.spec_from_file_location("legacy_retrieval_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load retrieval module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_retrieval_module()

INDEX_PATH = _legacy.INDEX_PATH
MODEL_NAME = _legacy.MODEL_NAME
load_chunks = _legacy.load_chunks
build_index = _legacy.build_index
search = _legacy.search
expand_context = _legacy.expand_context

__all__ = [
    "INDEX_PATH",
    "MODEL_NAME",
    "load_chunks",
    "build_index",
    "search",
    "expand_context",
    "search_bm25",
    "search_hybrid",
]
