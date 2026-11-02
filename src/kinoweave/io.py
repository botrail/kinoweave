"""Strict JSON documents, with explicit formats, schema versions, and digests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TypeVar, overload

from ._contracts import ContractError, value_as
from .manipulation import ManipulationProblem, ManipulationResult, TaskDomain
from .models import PlanningProblem
from .results import ExecutionReport, PlanResult

Document = (
    PlanningProblem
    | PlanResult
    | ExecutionReport
    | ManipulationProblem
    | ManipulationResult
    | TaskDomain
)
T = TypeVar(
    "T",
    PlanningProblem,
    PlanResult,
    ExecutionReport,
    ManipulationProblem,
    ManipulationResult,
    TaskDomain,
)
_FORMATS = {
    "kinoweave.problem": PlanningProblem,
    "kinoweave.result": PlanResult,
    "kinoweave.execution": ExecutionReport,
    "kinoweave.manipulation.problem": ManipulationProblem,
    "kinoweave.manipulation.result": ManipulationResult,
    "kinoweave.manipulation.tasks": TaskDomain,
}


def _plain(value):
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def dumps(document: Document) -> str:
    """Encode a validated document as deterministic UTF-8-compatible JSON."""
    if type(document) not in _FORMATS.values():
        raise TypeError("expected a supported KinoWeave document")
    return json.dumps(
        _plain(document), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    )


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"json.{key}", "duplicate object key")
        result[key] = value
    return result


def _constant(value: str):
    raise ContractError("json", f"non-finite JSON number {value!r} is not allowed")


@overload
def loads(text: str, *, expected: type[T]) -> T: ...


@overload
def loads(text: str, *, expected: None = None) -> Document: ...


def loads(text: str, *, expected: type[T] | None = None) -> T | Document:
    """Decode known formats only; reject unknown/missing versions and fields."""
    try:
        data = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except json.JSONDecodeError as error:
        raise ContractError("json", f"invalid JSON at line {error.lineno}: {error.msg}") from error
    if not isinstance(data, dict):
        raise ContractError("document", "expected an object")
    for key in ("format", "schema_version", "conventions"):
        if key not in data:
            raise ContractError(key, "missing required envelope field")
    name = data["format"]
    if not isinstance(name, str) or name not in _FORMATS:
        raise ContractError("format", f"unknown document format {name!r}")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ContractError("schema_version", "only schema version 1 is supported")
    cls = _FORMATS[name]
    if expected is not None and cls is not expected:
        raise ContractError("format", f"expected {expected.__name__}, got {name!r}")
    return value_as(data, cls, name, decode=True)


def dump(document: Document, path: str | Path) -> None:
    """Write JSON after encoding succeeds; do not create parent directories."""
    encoded = dumps(document)
    Path(path).write_text(encoded + "\n", encoding="utf-8")


@overload
def load(path: str | Path, *, expected: type[T]) -> T: ...


@overload
def load(path: str | Path, *, expected: None = None) -> Document: ...


def load(path: str | Path, *, expected: type[T] | None = None) -> T | Document:
    text = Path(path).read_text(encoding="utf-8")
    if expected is None:
        return loads(text)
    return loads(text, expected=expected)


def document_digest(document: Document) -> str:
    """SHA-256 of the canonical semantic fields, independent of JSON whitespace."""
    if type(document) not in _FORMATS.values():
        raise TypeError("expected a supported KinoWeave document")
    canonical = json.dumps(
        _plain(document),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
