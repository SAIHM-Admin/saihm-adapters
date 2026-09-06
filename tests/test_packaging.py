"""Packaging-invariant tests — need only the core client, no framework installed.

These lock in the single-distribution layout that replaced the four separate packages: one
top-level ``saihm_adapters`` carrying every adapter as a submodule, the ambiguous ``SaihmMemory``
name refused at the top level (it is a different class under AutoGen and under LlamaIndex), and
the Node sidecar shipped as package data.
"""
from __future__ import annotations

import importlib.resources
import importlib.util

import pytest

import saihm_adapters

ADAPTER_SUBMODULES = [
    "saihm_adapters.autogen_memory",
    "saihm_adapters.langchain_memory",
    "saihm_adapters.langgraph_store",
    "saihm_adapters.llamaindex_memory",
    "saihm_adapters.rag_memory",
    "saihm_adapters.crewai_storage",
]

SIDECAR_DATA = ["server.mjs", "sandbox.mjs", "package.json", "package-lock.json"]


def test_core_is_importable_without_any_framework():
    assert saihm_adapters.SaihmMemoryClient is not None
    assert saihm_adapters.SaihmTimeout is not None


def test_all_lists_core_and_unambiguous_adapters_only():
    assert "SaihmMemoryClient" in saihm_adapters.__all__
    for name in ("SaihmChatMessageHistory", "SaihmStore", "SaihmRetriever", "SaihmStorageBackend"):
        assert name in saihm_adapters.__all__
    # The ambiguous name is deliberately NOT exported at the top level.
    assert "SaihmMemory" not in saihm_adapters.__all__


def test_ambiguous_name_refused_with_directive():
    with pytest.raises(AttributeError) as exc:
        saihm_adapters.SaihmMemory  # noqa: B018
    msg = str(exc.value)
    assert "autogen_memory" in msg and "llamaindex_memory" in msg


def test_unknown_attribute_still_raises_plain_attributeerror():
    with pytest.raises(AttributeError):
        saihm_adapters.NoSuchThing  # noqa: B018


def test_every_adapter_ships_in_the_one_distribution():
    # find_spec locates the module file without executing it, so no framework dependency is
    # imported — this proves all five adapters ship in this single distribution.
    for mod in ADAPTER_SUBMODULES:
        assert importlib.util.find_spec(mod) is not None, mod


def test_sidecar_bundled_as_package_data():
    root = importlib.resources.files("saihm_adapters") / "_sidecar"
    for name in SIDECAR_DATA:
        assert (root / name).is_file(), name


def _installed(mod: str) -> bool:
    # find_spec raises (not returns None) when a dotted name's parent is absent, so guard it.
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize(
    "attr,extra,dep",
    [
        ("SaihmChatMessageHistory", "langchain", "langchain_core"),
        ("SaihmStore", "langgraph", "langgraph"),
        ("SaihmRetriever", "rag", "llama_index"),
        ("SaihmStorageBackend", "crewai", "crewai"),
    ],
)
def test_missing_extra_gives_directive_importerror(attr, extra, dep):
    # Only meaningful when the framework is absent; in the full [dev] env it is installed.
    if _installed(dep):
        pytest.skip(f"{dep} is installed; missing-extra path not exercised in this env")
    with pytest.raises(ImportError) as exc:
        getattr(saihm_adapters, attr)
    assert extra in str(exc.value)
