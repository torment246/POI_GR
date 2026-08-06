"""Load and validate method-level pipeline contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "poi-gr-method-v1"
REQUIRED_STAGES = ("data", "identifier", "train", "evaluate")
VALID_GROUPS = ("baseline", "current", "innovation")
VALID_METHOD_STATUSES = ("implemented", "partial", "planned")
VALID_STAGE_STATUSES = ("ready", "partial", "missing")
GROUP_DIRECTORIES = {
    "baseline": "baselines",
    "current": "current",
    "innovation": "innovations",
}


class MethodCatalogError(ValueError):
    """Raised when a method contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class CommandSpec:
    """One existing repository command that contributes to a method stage."""

    entrypoint: Path
    config: Path | None
    purpose: str


@dataclass(frozen=True)
class StageSpec:
    """Implementation readiness and reusable commands for one pipeline stage."""

    name: str
    status: str
    commands: tuple[CommandSpec, ...]
    notes: str


@dataclass(frozen=True)
class MethodSpec:
    """Validated method-level experiment contract."""

    method_id: str
    group: str
    display_name: str
    status: str
    summary: str
    paper: str | None
    task: Mapping[str, Any]
    identifier: Mapping[str, Any]
    stages: tuple[StageSpec, ...]
    config_path: Path

    def stage(self, name: str) -> StageSpec:
        """Return one required pipeline stage by name."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MethodCatalogError(f"{name} 必须是映射")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MethodCatalogError(f"{name} 必须是非空字符串")
    return value.strip()


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, name)


def _resolve_repo_path(value: Any, project_root: Path, name: str) -> Path:
    raw_path = Path(_require_string(value, name))
    path = raw_path if raw_path.is_absolute() else project_root / raw_path
    resolved = path.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise MethodCatalogError(f"{name} 必须位于仓库内：{resolved}") from error
    if not resolved.is_file():
        raise MethodCatalogError(f"{name} 不存在：{resolved}")
    return resolved


def _load_command(
    value: Any,
    *,
    project_root: Path,
    stage_name: str,
    index: int,
) -> CommandSpec:
    name = f"pipeline.{stage_name}.commands[{index}]"
    payload = _require_mapping(value, name)
    allowed_keys = {"entrypoint", "config", "purpose"}
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise MethodCatalogError(
            f"{name} 包含未知字段：" + ", ".join(unknown_keys)
        )
    entrypoint = _resolve_repo_path(
        payload.get("entrypoint"),
        project_root,
        f"{name}.entrypoint",
    )
    config_value = payload.get("config")
    config = (
        None
        if config_value is None
        else _resolve_repo_path(config_value, project_root, f"{name}.config")
    )
    return CommandSpec(
        entrypoint=entrypoint,
        config=config,
        purpose=_require_string(payload.get("purpose"), f"{name}.purpose"),
    )


def _load_stage(
    name: str,
    value: Any,
    *,
    project_root: Path,
) -> StageSpec:
    payload = _require_mapping(value, f"pipeline.{name}")
    allowed_keys = {"status", "commands", "notes"}
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise MethodCatalogError(
            f"pipeline.{name} 包含未知字段：" + ", ".join(unknown_keys)
        )
    status = _require_string(payload.get("status"), f"pipeline.{name}.status")
    if status not in VALID_STAGE_STATUSES:
        raise MethodCatalogError(
            f"pipeline.{name}.status 必须是 {VALID_STAGE_STATUSES} 之一"
        )
    raw_commands = payload.get("commands", [])
    if not isinstance(raw_commands, list):
        raise MethodCatalogError(f"pipeline.{name}.commands 必须是列表")
    commands = tuple(
        _load_command(
            command,
            project_root=project_root,
            stage_name=name,
            index=index,
        )
        for index, command in enumerate(raw_commands)
    )
    if status == "ready" and not commands:
        raise MethodCatalogError(f"ready 阶段 pipeline.{name} 必须至少有一个命令")
    if status == "missing" and commands:
        raise MethodCatalogError(f"missing 阶段 pipeline.{name} 不应声明可运行命令")
    return StageSpec(
        name=name,
        status=status,
        commands=commands,
        notes=_require_string(payload.get("notes"), f"pipeline.{name}.notes"),
    )


def load_method_config(config_path: Path, project_root: Path) -> MethodSpec:
    """Load one YAML method contract and validate repository references."""
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise MethodCatalogError(f"无法读取方法配置 {config_path}：{error}") from error
    root = _require_mapping(payload, str(config_path))
    if root.get("schema_version") != SCHEMA_VERSION:
        raise MethodCatalogError(
            f"{config_path} schema_version 必须是 {SCHEMA_VERSION}"
        )

    method = _require_mapping(root.get("method"), "method")
    method_id = _require_string(method.get("id"), "method.id")
    group = _require_string(method.get("group"), "method.group")
    if group not in VALID_GROUPS:
        raise MethodCatalogError(f"method.group 必须是 {VALID_GROUPS} 之一")
    expected_parent = GROUP_DIRECTORIES[group]
    if config_path.parent.name != expected_parent:
        raise MethodCatalogError(
            f"{method_id} 的 group={group}，配置应位于 {expected_parent}/"
        )
    status = _require_string(method.get("status"), "method.status")
    if status not in VALID_METHOD_STATUSES:
        raise MethodCatalogError(
            f"method.status 必须是 {VALID_METHOD_STATUSES} 之一"
        )

    task = _require_mapping(root.get("task"), "task")
    _require_string(task.get("type"), "task.type")
    target_fields = task.get("target_fields")
    history_fields = task.get("history_fields")
    if not isinstance(target_fields, list) or not target_fields:
        raise MethodCatalogError("task.target_fields 必须是非空列表")
    if not all(isinstance(value, str) and value.strip() for value in target_fields):
        raise MethodCatalogError("task.target_fields 只能包含非空字符串")
    if not isinstance(history_fields, list):
        raise MethodCatalogError("task.history_fields 必须是列表")
    if not all(isinstance(value, str) and value.strip() for value in history_fields):
        raise MethodCatalogError("task.history_fields 只能包含非空字符串")
    max_history_events = task.get("max_history_events")
    if max_history_events is not None and (
        not isinstance(max_history_events, int) or max_history_events <= 0
    ):
        raise MethodCatalogError("task.max_history_events 必须是正整数或 null")

    identifier = _require_mapping(root.get("identifier"), "identifier")
    _require_string(identifier.get("type"), "identifier.type")
    token_order = identifier.get("token_order")
    if not isinstance(token_order, list) or not token_order:
        raise MethodCatalogError("identifier.token_order 必须是非空列表")

    pipeline = _require_mapping(root.get("pipeline"), "pipeline")
    missing_stages = sorted(set(REQUIRED_STAGES) - set(pipeline))
    extra_stages = sorted(set(pipeline) - set(REQUIRED_STAGES))
    if missing_stages or extra_stages:
        details = []
        if missing_stages:
            details.append("缺少 " + ", ".join(missing_stages))
        if extra_stages:
            details.append("未知 " + ", ".join(extra_stages))
        raise MethodCatalogError("pipeline 阶段不完整：" + "；".join(details))
    stages = tuple(
        _load_stage(name, pipeline[name], project_root=project_root)
        for name in REQUIRED_STAGES
    )
    if status == "implemented" and any(stage.status != "ready" for stage in stages):
        raise MethodCatalogError("implemented 方法的四个阶段都必须是 ready")

    return MethodSpec(
        method_id=method_id,
        group=group,
        display_name=_require_string(method.get("display_name"), "method.display_name"),
        status=status,
        summary=_require_string(method.get("summary"), "method.summary"),
        paper=_optional_string(method.get("paper"), "method.paper"),
        task=task,
        identifier=identifier,
        stages=stages,
        config_path=config_path,
    )


def load_method_catalog(config_root: Path, project_root: Path) -> dict[str, MethodSpec]:
    """Discover every method YAML below the configured method root."""
    config_root = config_root.resolve()
    if not config_root.is_dir():
        raise MethodCatalogError(f"方法配置目录不存在：{config_root}")
    config_paths = tuple(sorted(config_root.glob("*/*.yaml")))
    if not config_paths:
        raise MethodCatalogError(f"方法配置目录为空：{config_root}")
    catalog: dict[str, MethodSpec] = {}
    for config_path in config_paths:
        spec = load_method_config(config_path, project_root)
        if spec.method_id in catalog:
            previous = catalog[spec.method_id].config_path
            raise MethodCatalogError(
                f"方法 ID 重复：{spec.method_id}，来自 {previous} 和 {config_path}"
            )
        catalog[spec.method_id] = spec
    return dict(sorted(catalog.items()))


def method_spec_payload(spec: MethodSpec, project_root: Path) -> dict[str, Any]:
    """Return a stable JSON-serializable representation for CLI output."""
    project_root = project_root.resolve()

    def display(path: Path) -> str:
        return str(path.resolve().relative_to(project_root))

    return {
        "id": spec.method_id,
        "group": spec.group,
        "display_name": spec.display_name,
        "status": spec.status,
        "summary": spec.summary,
        "paper": spec.paper,
        "task": dict(spec.task),
        "identifier": dict(spec.identifier),
        "config": display(spec.config_path),
        "pipeline": {
            stage.name: {
                "status": stage.status,
                "notes": stage.notes,
                "commands": [
                    {
                        "entrypoint": display(command.entrypoint),
                        "config": (
                            None if command.config is None else display(command.config)
                        ),
                        "purpose": command.purpose,
                    }
                    for command in stage.commands
                ],
            }
            for stage in spec.stages
        },
    }
