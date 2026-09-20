"""Small deterministic I/O helpers used by every BeamRisk stage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .errors import BeamRiskError


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256(project_root: Path) -> str:
    """Fingerprint all executable BeamRisk code and the frozen main config."""

    beamrisk_root = project_root.resolve() / "beamrisk_sft"
    paths = sorted(
        {
            *beamrisk_root.glob("configs/*.yaml"),
            *beamrisk_root.glob("run_*.sh"),
            *beamrisk_root.glob("scripts/*.py"),
            *beamrisk_root.glob("src/**/*.py"),
        },
        key=lambda path: str(path.relative_to(beamrisk_root)),
    )
    if not paths:
        raise BeamRiskError(f"找不到 BeamRisk 实现文件：{beamrisk_root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = str(path.relative_to(beamrisk_root)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, *, name: str = "JSON") -> dict[str, Any]:
    if not path.is_file():
        raise BeamRiskError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BeamRiskError(f"{name} 读取失败：{path}") from error
    if not isinstance(value, dict):
        raise BeamRiskError(f"{name} 必须是 JSON object：{path}")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise BeamRiskError(f"JSONL 不存在：{path}")
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise BeamRiskError(
                    f"JSONL 第 {line_number} 行解析失败：{path}"
                ) from error
            if not isinstance(value, dict):
                raise BeamRiskError(
                    f"JSONL 第 {line_number} 行必须是 object：{path}"
                )
            yield line_number, value


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    digest = hashlib.sha256()
    rows = 0
    try:
        with temporary.open("wb") as stream:
            for value in values:
                payload = (
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                stream.write(payload)
                digest.update(payload)
                rows += 1
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return rows, digest.hexdigest()
