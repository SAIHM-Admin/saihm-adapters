"""Packaging-invariant tests — need only the core client, no framework installed.

These lock in the single-distribution layout that replaced the four separate packages: one
top-level ``saihm_adapters`` carrying every adapter as a submodule, the ambiguous ``SaihmMemory``
name refused at the top level (it is a different class under AutoGen and under LlamaIndex), and
the Node sidecar shipped as package data.
"""
from __future__ import annotations

import importlib.metadata
import importlib.resources
import importlib.util
import json
import os
import pydoc
import subprocess
import sys

import pytest
from packaging.version import Version

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
    # An adapter is listed exactly when its framework is installed at a version its extra allows.
    for name, expected in _expected_listing().items():
        assert (name in saihm_adapters.__all__) == expected, name
        assert (name in dir(saihm_adapters)) == expected, name
    # The ambiguous name is deliberately NOT exported at the top level.
    assert "SaihmMemory" not in saihm_adapters.__all__


def test_help_renders_with_the_installed_extras():
    # pydoc walks __all__; a listed adapter whose framework is missing used to replace the whole
    # page with an ImportError.
    text = pydoc.render_doc(saihm_adapters, renderer=pydoc.plaintext)
    assert "SaihmMemoryClient" in text and "SaihmTimeout" in text


def test_star_import_works_with_the_installed_extras():
    names = {}
    exec("from saihm_adapters import *", names)
    assert "SaihmMemoryClient" in names
    for name, expected in _expected_listing().items():
        assert (name in names) == expected, name


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


# The floor of each adapter's extra in pyproject.toml, written out here so the test does not
# trust the table it checks.
_FLOORS = {
    "SaihmChatMessageHistory": ("langchain-core", "0.3"),
    "SaihmStore": ("langgraph", "1.2"),
    "SaihmRetriever": ("llama-index-core", "0.14"),
    "SaihmStorageBackend": ("crewai", "1.15"),
}


def _expected_listing() -> dict:
    """Which adapters should be listed: the framework distribution is installed at or above its
    extra's floor."""
    found = {}
    for name, (dist, floor) in _FLOORS.items():
        try:
            found[name] = Version(importlib.metadata.version(dist)) >= Version(floor)
        except importlib.metadata.PackageNotFoundError:
            found[name] = False
    return found


_PROBE = r"""
import inspect, json, pydoc, sys
import saihm_adapters
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("crewai", "langgraph", "langchain_core", "llama_index", "autogen_core"))
pydoc.render_doc(saihm_adapters, renderer=pydoc.plaintext)
names = {}
exec("from saihm_adapters import *", names)
inspect.getmembers(saihm_adapters)
print(json.dumps({"loaded": loaded, "listed": [n for n in saihm_adapters._LAZY if n in saihm_adapters.__all__],
                  "star": [n for n in saihm_adapters._LAZY if n in names]}))
"""


def _probe(cwd, extra_path=None):
    # A fresh interpreter, so modules this test session already imported do not count.
    env = dict(os.environ)
    if extra_path:
        env["PYTHONPATH"] = os.pathsep.join([str(extra_path)] + [p for p in [env.get("PYTHONPATH")] if p])
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _stand_in(root, package, init="", dist=None, version=None):
    # A package on sys.path, optionally with installed metadata naming a distribution.
    (root / package).mkdir(parents=True, exist_ok=True)
    (root / package / "__init__.py").write_text(init, encoding="utf-8")
    if dist:
        info = root / f"{dist.replace('-', '_')}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n", encoding="utf-8")


def test_import_runs_no_framework_code(tmp_path):
    # Listing the adapters reads metadata only: no framework module is imported with the core.
    result = _probe(tmp_path)
    assert result["loaded"] == []
    assert result["listed"] == result["star"] == [n for n, ok in _expected_listing().items() if ok]


def test_framework_below_its_extra_floor_is_not_listed(tmp_path):
    # CrewAI 0.x has no crewai.memory.types; listing it would break help() and the star import.
    _stand_in(tmp_path / "site", "crewai", dist="crewai", version="0.203.2")
    result = _probe(tmp_path, tmp_path / "site")
    assert "SaihmStorageBackend" not in result["listed"] and "SaihmStorageBackend" not in result["star"]
    assert result["loaded"] == []


def test_folder_named_like_a_framework_is_not_an_install(tmp_path):
    # An empty crewai/ folder beside the script is importable as a namespace package, but no
    # CrewAI is installed by it.
    (tmp_path / "crewai").mkdir()
    result = _probe(tmp_path)
    assert ("SaihmStorageBackend" in result["listed"]) == _expected_listing()["SaihmStorageBackend"]
    assert result["listed"] == result["star"]


@pytest.mark.parametrize(
    "package,dist,version",
    [("llama_index", None, None), ("llama_index", "llama-index", "0.9.48"), ("crewai", "crewai", "1.15.0")],
)
def test_a_framework_that_fails_on_import_cannot_break_the_core_import(tmp_path, package, dist, version):
    # A framework package whose own import raises (a broken or legacy install) is never run by
    # `import saihm_adapters`, whatever it raises.
    site = tmp_path / "site"
    _stand_in(site, package, init='raise RuntimeError("stand-in framework ran")\n', dist=dist, version=version)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(site)] + [p for p in [env.get("PYTHONPATH")] if p])
    out = subprocess.run(
        [sys.executable, "-c", "import sys, saihm_adapters; print(sorted(m for m in sys.modules if m.split('.')[0] == %r))" % package],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]"


def test_unreadable_framework_metadata_cannot_break_the_core_import(tmp_path):
    # A broken install whose METADATA is not UTF-8 makes importlib.metadata raise
    # UnicodeDecodeError. The core import still works, and the adapter is not listed.
    info = tmp_path / "site" / "crewai-1.15.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_bytes(b"Metadata-Version: 2.1\nName: crewai\nVersion: 1.15.0\nSummary: \xff\xfe\n")
    result = _probe(tmp_path, tmp_path / "site")
    assert "SaihmStorageBackend" not in result["listed"]
    assert result["loaded"] == []


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
    assert attr not in saihm_adapters.__all__ and attr not in dir(saihm_adapters)
    with pytest.raises(ImportError) as exc:
        getattr(saihm_adapters, attr)
    assert extra in str(exc.value)
