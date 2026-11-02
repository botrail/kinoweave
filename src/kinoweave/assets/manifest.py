"""Hash-checked, portable link visuals; no geometry SDK is required to copy them."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from .._contracts import Contract, ContractError, value_as
from ..io import _constant, _pairs, document_digest
from ..models import Conventions, PlanningProblem, RobotModel


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_digest(model: RobotModel) -> str:
    return sha256(
        json.dumps(asdict(model), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelFingerprint(Contract):
    name: str
    sha256: str

    def _validate(self):
        self._require(
            re.fullmatch("[0-9a-f]{64}", self.sha256) is not None, "sha256", "invalid digest"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceFile(Contract):
    name: str
    sha256: str

    def _validate(self):
        self._require(
            re.fullmatch("[0-9a-f]{64}", self.sha256) is not None, "sha256", "invalid digest"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LinkVisual(Contract):
    model: str
    link: str
    file: str
    sha256: str

    def _validate(self):
        path = PurePosixPath(self.file)
        self._require(
            not path.is_absolute() and ".." not in path.parts and "\\" not in self.file,
            "file",
            "expected a contained relative path",
        )
        self._require(
            re.fullmatch("[0-9a-f]{64}", self.sha256) is not None, "sha256", "invalid digest"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AssetManifest(Contract):
    problem_digest: str
    models: tuple[ModelFingerprint, ...]
    visuals: tuple[LinkVisual, ...] = ()
    sources: tuple[SourceFile, ...] = ()
    notices: tuple[SourceFile, ...] = ()
    conventions: Conventions = Conventions()
    format: Literal["kinoweave.assets"] = "kinoweave.assets"
    schema_version: Literal[1] = 1

    def _validate(self):
        self._require(
            re.fullmatch("[0-9a-f]{64}", self.problem_digest) is not None,
            "problem_digest",
            "invalid digest",
        )
        self._unique(tuple(m.name for m in self.models), "models")
        self._unique(tuple((v.model, v.link) for v in self.visuals), "visuals")
        self._unique(tuple(n.name for n in self.notices), "notices")

    def check_problem(self, problem: PlanningProblem):
        self._require(
            self.problem_digest == document_digest(problem),
            "problem_digest",
            "assets belong to a different problem",
        )
        self._require(
            {m.name: m.sha256 for m in self.models}
            == {m.name: model_digest(m) for m in problem.models},
            "models",
            "native model fingerprints do not match",
        )
        known = {(m.name, link) for m in problem.models for link in m.links}
        self._require(
            all((v.model, v.link) in known for v in self.visuals), "visuals", "unknown model/link"
        )


def read_manifest(path: str | Path, problem: PlanningProblem) -> AssetManifest:
    try:
        data = json.loads(
            Path(path).read_text(), object_pairs_hook=_pairs, parse_constant=_constant
        )
    except json.JSONDecodeError as error:
        raise ContractError("assets", "invalid JSON") from error
    if not isinstance(data, dict) or not {"format", "schema_version", "conventions"} <= data.keys():
        raise ContractError("assets", "missing manifest envelope")
    manifest = value_as(data, AssetManifest, "assets", decode=True)
    manifest.check_problem(problem)
    return manifest


def visual_bytes(path: str | Path, manifest: AssetManifest) -> dict[tuple[str, str], bytes]:
    root = Path(path).resolve().parent
    result = {}
    for visual in manifest.visuals:
        file = (root / visual.file).resolve()
        if not file.is_relative_to(root):
            raise ContractError("assets.file", "asset escapes the bundle")
        data = file.read_bytes()
        if sha256(data) != visual.sha256 or not data.startswith(b"glTF"):
            raise ContractError(f"assets.{visual.model}.{visual.link}", "GLB content/hash mismatch")
        result[(visual.model, visual.link)] = data
    return result


def notice_bytes(path: str | Path, manifest: AssetManifest):
    root = Path(path).resolve().parent
    result = []
    for notice in manifest.notices:
        file = (root / notice.name).resolve()
        if not file.is_relative_to(root):
            raise ContractError("assets.notices", "notice escapes the bundle")
        data = file.read_bytes()
        if sha256(data) != notice.sha256:
            raise ContractError("assets.notices", "notice hash mismatch")
        result.append((notice.name, data))
    return result


def load_visuals(path: str | Path, problem: PlanningProblem):
    manifest = read_manifest(path, problem)
    notice_bytes(path, manifest)
    return visual_bytes(path, manifest)


def write_manifest(
    directory: str | Path, problem: PlanningProblem, visuals, sources=(), notices=()
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for (model, link), data in sorted(visuals.items()):
        digest = sha256(data)
        relative = f"assets/{digest}.glb"
        file = directory / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
        records.append(LinkVisual(model=model, link=link, file=relative, sha256=digest))
    notice_records = {}
    for _, data in notices:
        digest = sha256(data)
        relative = f"notices/{digest}.txt"
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_bytes(data)
        notice_records[relative] = SourceFile(name=relative, sha256=digest)
    manifest = AssetManifest(
        problem_digest=document_digest(problem),
        models=tuple(ModelFingerprint(name=m.name, sha256=model_digest(m)) for m in problem.models),
        visuals=tuple(records),
        sources=tuple(sources),
        notices=tuple(notice_records.values()),
    )
    (directory / "assets.manifest.json").write_text(
        json.dumps(asdict(manifest), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def copy_assets(
    source: str | Path, original: PlanningProblem, target: str | Path, effective: PlanningProblem
):
    """Copy checked visuals and explicitly rebind a backend/configuration override."""
    manifest = read_manifest(source, original)
    if original.models != effective.models:
        raise ContractError("assets.models", "cannot rebind changed native models")
    return write_manifest(
        target,
        effective,
        visual_bytes(source, manifest),
        manifest.sources,
        notice_bytes(source, manifest),
    )
