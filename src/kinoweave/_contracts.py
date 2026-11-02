"""Strict, dependency-free validation shared by the public data contracts."""

from __future__ import annotations

import math
import types
from dataclasses import MISSING, fields, is_dataclass
from functools import cache
from typing import Literal, Union, get_args, get_origin, get_type_hints


class ContractError(ValueError):
    """An invalid contract value, with its field path preserved."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


@cache
def hints(cls: type) -> dict:
    return get_type_hints(cls)


def value_as(value: object, annotation: object, path: str, *, decode: bool = False):
    """Check types without string/boolean numeric coercion; freeze input sequences."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        if not any(type(value) is type(option) and value == option for option in args):
            raise ContractError(path, f"expected one of {args!r}, got {value!r}")
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        records = [arg for arg in args if isinstance(arg, type) and is_dataclass(arg)]
        if decode and isinstance(value, dict) and len(records) == 1:
            return value_as(value, records[0], path, decode=True)
        if decode and isinstance(value, dict) and len(records) > 1:
            # Shape unions use an explicit discriminator, including when fields overlap.
            kind = value.get("kind")
            matching = [cls for cls in records if kind in get_args(hints(cls).get("kind"))]
            if len(matching) != 1:
                raise ContractError(f"{path}.kind", f"unknown or missing shape kind {kind!r}")
            return value_as(value, matching[0], path, decode=True)
        for option in args:
            try:
                return value_as(value, option, path, decode=decode)
            except ContractError:
                pass
        raise ContractError(path, f"value does not match {annotation!r}")
    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            raise ContractError(path, "expected an array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                value_as(item, args[0], f"{path}[{i}]", decode=decode)
                for i, item in enumerate(value)
            )
        if len(value) != len(args):
            raise ContractError(path, f"expected {len(args)} entries, got {len(value)}")
        return tuple(
            value_as(item, item_type, f"{path}[{i}]", decode=decode)
            for i, (item, item_type) in enumerate(zip(value, args, strict=True))
        )
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(path, "expected a finite number")
        try:
            number = float(value)
        except OverflowError as error:
            raise ContractError(path, "number is outside the finite float range") from error
        if not math.isfinite(number):
            raise ContractError(path, "expected a finite number")
        return number
    if annotation is int:
        if type(value) is not int:
            raise ContractError(path, "expected an integer")
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise ContractError(path, "expected a boolean")
        return value
    if annotation is str:
        if not isinstance(value, str) or not value.strip():
            raise ContractError(path, "expected a non-empty string")
        return value
    if annotation is type(None):
        if value is not None:
            raise ContractError(path, "expected null")
        return None
    if isinstance(annotation, type) and is_dataclass(annotation):
        if decode:
            if not isinstance(value, dict):
                raise ContractError(path, "expected an object")
            allowed = {field.name for field in fields(annotation)}
            unknown = value.keys() - allowed
            if unknown:
                key = sorted(unknown)[0]
                raise ContractError(f"{path}.{key}", "unknown field")
            values = {}
            for field in fields(annotation):
                if field.name in value:
                    values[field.name] = value_as(
                        value[field.name],
                        hints(annotation)[field.name],
                        f"{path}.{field.name}",
                        decode=True,
                    )
                elif field.default is MISSING and field.default_factory is MISSING:
                    raise ContractError(f"{path}.{field.name}", "missing required field")
            try:
                return annotation(**values)
            except ContractError as error:
                prefix = annotation.__name__
                suffix = error.path.removeprefix(prefix)
                raise ContractError(path + suffix, error.message) from error
        if type(value) is not annotation:
            raise ContractError(path, f"expected {annotation.__name__}")
        return value
    raise TypeError(f"unsupported contract annotation: {annotation!r}")


class Contract:
    """Dataclass base; subclasses add domain checks after structural validation."""

    __slots__ = ()

    def __post_init__(self) -> None:
        for field in fields(self):
            checked = value_as(
                getattr(self, field.name),
                hints(type(self))[field.name],
                f"{type(self).__name__}.{field.name}",
            )
            object.__setattr__(self, field.name, checked)
        self._validate()

    def _validate(self) -> None:
        pass

    def _require(self, condition: bool, field: str, message: str) -> None:
        if not condition:
            raise ContractError(f"{type(self).__name__}.{field}", message)

    def _unique(self, values: tuple, field: str) -> None:
        seen = set()
        for value in values:
            self._require(value not in seen, field, f"duplicate name/value {value!r}")
            seen.add(value)
