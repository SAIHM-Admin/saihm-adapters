"""SAIHM memory for Python — one owned, client-side-encrypted store for your AI agents.

A single distribution behind every supported framework. The core client needs only ``mcp``;
each framework adapter is an optional extra, imported lazily, so you install only what you use::

    pip install saihm-adapters                    # core client (any Python app)
    pip install "saihm-adapters[langchain]"       # + LangChain chat history
    pip install "saihm-adapters[langgraph]"       # + LangGraph BaseStore
    pip install "saihm-adapters[llamaindex]"      # + LlamaIndex memory
    pip install "saihm-adapters[rag]"             # + LlamaIndex retriever for RAG
    pip install "saihm-adapters[crewai]"          # + CrewAI storage backend
    pip install "saihm-adapters[autogen]"         # + AutoGen memory

Every cell is sealed by a bundled Node sidecar — Python holds no key — portable across models
and frameworks, and provably erasable (GDPR Art. 17)::

    from saihm_adapters import SaihmMemoryClient              # core, any app
    from saihm_adapters import SaihmChatMessageHistory        # LangChain    [langchain]
    from saihm_adapters import SaihmStore                     # LangGraph    [langgraph]
    from saihm_adapters import SaihmRetriever                 # RAG          [rag]
    from saihm_adapters import SaihmStorageBackend            # CrewAI       [crewai]
    from saihm_adapters.autogen_memory import SaihmMemory     # AutoGen      [autogen]
    from saihm_adapters.llamaindex_memory import SaihmMemory  # LlamaIndex   [llamaindex]

``SaihmMemory`` is provided by two different frameworks (AutoGen and LlamaIndex) as distinct
classes, so it is imported from its framework submodule rather than the top level — the two can
never collide on one name.
"""
from __future__ import annotations

import importlib

from .client import Memory, SaihmMemoryClient, SaihmTimeout

__all__ = [
    "SaihmMemoryClient",
    "Memory",
    "SaihmTimeout",
    "SaihmChatMessageHistory",
    "SaihmStore",
    "SaihmRetriever",
    "SaihmStorageBackend",
]

# Unambiguous adapter classes exposed at the top level: name -> (submodule, extra).
_LAZY = {
    "SaihmChatMessageHistory": ("langchain_memory", "langchain"),
    "SaihmStore": ("langgraph_store", "langgraph"),
    "SaihmRetriever": ("rag_memory", "rag"),
    "SaihmStorageBackend": ("crewai_storage", "crewai"),
}

# Names carried by more than one framework submodule as different classes. Importing them from
# the top level would be a silent collision, so it is refused with a directive to the submodule.
_AMBIGUOUS = {
    "SaihmMemory": (
        ("saihm_adapters.autogen_memory", "autogen"),
        ("saihm_adapters.llamaindex_memory", "llamaindex"),
    ),
}


def __getattr__(name):  # PEP 562 — lazy adapter imports
    if name in _LAZY:
        module, extra = _LAZY[name]
        try:
            mod = importlib.import_module(f"{__name__}.{module}")
        except ModuleNotFoundError as exc:
            # The adapter module ships in every install; a ModuleNotFoundError here means the
            # framework dependency is absent. Name the extra rather than leak a bare import error.
            raise ImportError(
                f"{name} needs the optional '{extra}' extra — install it with:  "
                f'pip install "saihm-adapters[{extra}]"  (missing dependency: {exc.name})'
            ) from exc
        return getattr(mod, name)
    if name in _AMBIGUOUS:
        hint = " or ".join(
            f"`from {mod} import {name}`  ({extra})" for mod, extra in _AMBIGUOUS[name]
        )
        raise AttributeError(
            f"{name} is provided by more than one framework as different classes; import it "
            f"from its framework submodule instead of the top level: {hint}."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys()) + __all__))
