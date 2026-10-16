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

Every cell is sealed by a bundled Node sidecar (with the key in a key file, Python never reads
it), portable across models and frameworks, and provably erasable (GDPR Art. 17)::

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
import importlib.util

from .client import Memory, SaihmMemoryClient, SaihmNoIdentityError, SaihmStartupError, SaihmTimeout

# Unambiguous adapter classes exposed at the top level: name -> (submodule, extra, framework).
_LAZY = {
    "SaihmChatMessageHistory": ("langchain_memory", "langchain", "langchain_core"),
    "SaihmStore": ("langgraph_store", "langgraph", "langgraph"),
    "SaihmRetriever": ("rag_memory", "rag", "llama_index.core"),
    "SaihmStorageBackend": ("crewai_storage", "crewai", "crewai"),
}


def _installed(module: str) -> bool:
    """Whether a framework can be imported, looked up without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # a dotted name whose parent package is missing
        return False


# An adapter is listed only when its framework is installed, so help(saihm_adapters) and
# ``from saihm_adapters import *`` work with any set of extras. Asking for a missing one by name
# still raises an ImportError that names the extra to install.
__all__ = [
    "SaihmMemoryClient",
    "Memory",
    "SaihmTimeout",
    "SaihmStartupError",
    "SaihmNoIdentityError",
] + [name for name, (_, _, framework) in _LAZY.items() if _installed(framework)]

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
        module, extra, _ = _LAZY[name]
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
