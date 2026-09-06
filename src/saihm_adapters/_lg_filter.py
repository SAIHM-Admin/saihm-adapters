"""JSONB-style filter + namespace matching, bound to LangGraph's own helpers when available.

``SaihmStore`` must apply ``filter=`` and ``list_namespaces(prefix=/suffix=)`` exactly as
LangGraph's reference ``InMemoryStore`` does, so a graph behaves identically whichever store is
plugged in. The cheapest way to guarantee that is to call LangGraph's own implementation.

Those helpers are private (``_compare_values`` / ``_does_match`` / ``_apply_operator``) and live in
``langgraph.store.memory``, which is shipped by the ``langgraph-checkpoint`` distribution — a
transitive dependency this package does not pin. ``langgraph`` admits
``langgraph-checkpoint>=4.1.0,<5.0.0``, so any 4.x release could rename or drop an underscore
symbol without a semver signal.

So: import them when they are there, and fall back to a byte-faithful vendored copy when they are
not. ``USING_VENDORED`` records which path was taken; the test suite forces the fallback branch and
asserts the vendored copies agree with upstream across a filter/namespace matrix, so the fallback is
proven equivalent rather than merely present.

The vendored functions below are copied from LangGraph (``langgraph/store/memory/__init__.py``,
langgraph-checkpoint 4.2.0), MIT licensed, (c) LangChain, Inc. Kept verbatim on purpose: any edit
would be a silent divergence from the parity this module exists to preserve.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotation-only: `from __future__ import annotations` keeps this out of runtime
    from langgraph.store.base import MatchCondition

__all__ = ["USING_VENDORED", "compare_values", "does_match"]


# --- vendored copies (used only if the upstream import below fails) ---------------------------

def _vendored_apply_operator(value: Any, operator: str, op_value: Any) -> bool:
    """Apply a comparison operator, matching PostgreSQL's JSONB behavior."""
    if operator == "$eq":
        return value == op_value
    elif operator == "$gt":
        return float(value) > float(op_value)
    elif operator == "$gte":
        return float(value) >= float(op_value)
    elif operator == "$lt":
        return float(value) < float(op_value)
    elif operator == "$lte":
        return float(value) <= float(op_value)
    elif operator == "$ne":
        return value != op_value
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def _vendored_compare_values(item_value: Any, filter_value: Any) -> bool:
    """Compare values in a JSONB-like way, handling nested objects."""
    if isinstance(filter_value, dict):
        if any(k.startswith("$") for k in filter_value):
            return all(
                _vendored_apply_operator(item_value, op_key, op_value)
                for op_key, op_value in filter_value.items()
            )
        if not isinstance(item_value, dict):
            return False
        return all(
            _vendored_compare_values(item_value.get(k), v)
            for k, v in filter_value.items()
        )
    elif isinstance(filter_value, (list, tuple)):
        return (
            isinstance(item_value, (list, tuple))
            and len(item_value) == len(filter_value)
            and all(
                _vendored_compare_values(iv, fv)
                for iv, fv in zip(item_value, filter_value, strict=False)
            )
        )
    else:
        return item_value == filter_value


def _vendored_does_match(match_condition: MatchCondition, key: tuple[str, ...]) -> bool:
    """Whether a namespace key matches a match condition."""
    match_type = match_condition.match_type
    path = match_condition.path

    if len(key) < len(path):
        return False

    if match_type == "prefix":
        for k_elem, p_elem in zip(key, path, strict=False):
            if p_elem == "*":
                continue  # Wildcard matches any element
            if k_elem != p_elem:
                return False
        return True
    elif match_type == "suffix":
        for k_elem, p_elem in zip(reversed(key), reversed(path), strict=False):
            if p_elem == "*":
                continue  # Wildcard matches any element
            if k_elem != p_elem:
                return False
        return True
    else:
        raise ValueError(f"Unsupported match type: {match_type}")


# --- bind: upstream when present, vendored otherwise ------------------------------------------

try:
    from langgraph.store.memory import (  # type: ignore[attr-defined]
        _compare_values as compare_values,
        _does_match as does_match,
    )

    USING_VENDORED = False
except ImportError:  # pragma: no cover - covered by tests that force this branch
    compare_values = _vendored_compare_values
    does_match = _vendored_does_match
    USING_VENDORED = True

    # Engaging the fallback silently would be the wrong trade. The copies match
    # langgraph-checkpoint 4.2.0 exactly, so a REMOVED symbol is handled correctly — but a symbol
    # that was CHANGED rather than removed would leave SaihmStore on 4.2.0 semantics while
    # InMemoryStore moved on, and that divergence should be visible, not swallowed.
    warnings.warn(
        "saihm_adapters: LangGraph's private filter helpers (_compare_values/_does_match) could not "
        "be imported from langgraph.store.memory, so SaihmStore is using vendored copies taken "
        "from langgraph-checkpoint 4.2.0. Filter and namespace behaviour is unchanged for that "
        "version, but if LangGraph altered these semantics rather than merely relocating them, "
        "SaihmStore may now differ from InMemoryStore. Please report this so the adapter can be "
        "updated: https://saihm.coti.global",
        RuntimeWarning,
        stacklevel=2,
    )
