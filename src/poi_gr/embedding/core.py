"""Build POI text embeddings while preserving row-to-ID alignment."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence

import numpy as np
import yaml
from tqdm import tqdm


class ConfigError(ValueError):
    pass


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DataConfig:
    input_dir: Path
    file_pattern: str
    success_marker: str | None
    id_field: str
    text_field: str
    expected_rows: int | None
    check_unique_ids: bool


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    device: str
    batch_size: int
    encode_buffer_size: int
    max_seq_length: int
    torch_dtype: str
    attention: str | None
    padding_side: str
    normalize_embeddings: bool
    truncate_dim: int | None
    prompt_name: str | None


@dataclass(frozen=True)
class OutputConfig:
    dir: Path
    embedding_dtype: str
    checkpoint_interval_batches: int
    resume: bool


@dataclass(frozen=True)
class EmbeddingJobConfig:
    job_name: str
    data: DataConfig
    model: ModelConfig
    output: OutputConfig


@dataclass(frozen=True)
class PreparedInput:
    files: tuple[Path, ...]
    total_rows: int
    fingerprint: str
    sources: tuple[dict[str, Any], ...]


class TextEncoder(Protocol):
    def get_sentence_embedding_dimension(self) -> int | None:
        ...

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> np.ndarray:
        ...


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} 必须是 YAML mapping")
    return value


def _resolve_path(value: Any, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} 必须是非空路径")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} 必须是正整数或 null")
    return value


def load_job_config(config_path: Path, project_root: Path) -> EmbeddingJobConfig:
    """Load a job config and resolve its paths against the project root."""

    with config_path.open("r", encoding="utf-8") as handle:
        raw = _require_mapping(yaml.safe_load(handle), "配置根节点")

    data_raw = _require_mapping(raw.get("data"), "data")
    model_raw = _require_mapping(raw.get("model"), "model")
    output_raw = _require_mapping(raw.get("output"), "output")

    job_name = raw.get("job_name")
    if not isinstance(job_name, str) or not job_name.strip():
        raise ConfigError("job_name 必须是非空字符串")

    data = DataConfig(
        input_dir=_resolve_path(data_raw.get("input_dir"), project_root, "data.input_dir"),
        file_pattern=str(data_raw.get("file_pattern", "part-*.json")),
        success_marker=data_raw.get("success_marker"),
        id_field=str(data_raw.get("id_field", "poi_id")),
        text_field=str(data_raw.get("text_field", "text")),
        expected_rows=_optional_positive_int(data_raw.get("expected_rows"), "data.expected_rows"),
        check_unique_ids=bool(data_raw.get("check_unique_ids", True)),
    )
    batch_size = _optional_positive_int(
        model_raw.get("batch_size"), "model.batch_size"
    ) or 32
    model = ModelConfig(
        path=_resolve_path(model_raw.get("path"), project_root, "model.path"),
        device=str(model_raw.get("device", "auto")),
        batch_size=batch_size,
        encode_buffer_size=_optional_positive_int(
            model_raw.get("encode_buffer_size"), "model.encode_buffer_size"
        )
        or batch_size,
        max_seq_length=_optional_positive_int(
            model_raw.get("max_seq_length"), "model.max_seq_length"
        )
        or 512,
        torch_dtype=str(model_raw.get("torch_dtype", "bfloat16")),
        attention=model_raw.get("attention"),
        padding_side=str(model_raw.get("padding_side", "left")),
        normalize_embeddings=bool(model_raw.get("normalize_embeddings", True)),
        truncate_dim=_optional_positive_int(model_raw.get("truncate_dim"), "model.truncate_dim"),
        prompt_name=model_raw.get("prompt_name"),
    )
    output = OutputConfig(
        dir=_resolve_path(output_raw.get("dir"), project_root, "output.dir"),
        embedding_dtype=str(output_raw.get("embedding_dtype", "float16")),
        checkpoint_interval_batches=_optional_positive_int(
            output_raw.get("checkpoint_interval_batches"),
            "output.checkpoint_interval_batches",
        )
        or 20,
        resume=bool(output_raw.get("resume", True)),
    )
    _validate_config(data, model, output)
    return EmbeddingJobConfig(job_name=job_name, data=data, model=model, output=output)


def _validate_config(data: DataConfig, model: ModelConfig, output: OutputConfig) -> None:
    if not data.file_pattern:
        raise ConfigError("data.file_pattern 不能为空")
    if not data.id_field or not data.text_field:
        raise ConfigError("data.id_field 和 data.text_field 不能为空")
    if model.padding_side not in {"left", "right"}:
        raise ConfigError("model.padding_side 只能是 left 或 right")
    if model.batch_size <= 0:
        raise ConfigError("model.batch_size 必须是正整数")
    if model.encode_buffer_size <= 0:
        raise ConfigError("model.encode_buffer_size 必须是正整数")
    if model.encode_buffer_size < model.batch_size:
        raise ConfigError("model.encode_buffer_size 不能小于 model.batch_size")
    if model.torch_dtype not in {"float16", "bfloat16", "float32"}:
        raise ConfigError("model.torch_dtype 只能是 float16、bfloat16 或 float32")
    if output.embedding_dtype not in {"float16", "float32"}:
        raise ConfigError("output.embedding_dtype 只能是 float16 或 float32")


def apply_overrides(
    config: EmbeddingJobConfig,
    *,
    model_path: Path | None = None,
    output_dir: Path | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    encode_buffer_size: int | None = None,
    resume: bool | None = None,
) -> EmbeddingJobConfig:
    """Apply supported command-line overrides to a job config."""

    model = replace(
        config.model,
        path=model_path or config.model.path,
        device=device or config.model.device,
        batch_size=config.model.batch_size if batch_size is None else batch_size,
        encode_buffer_size=(
            config.model.encode_buffer_size
            if encode_buffer_size is None
            else encode_buffer_size
        ),
    )
    output = replace(
        config.output,
        dir=output_dir or config.output.dir,
        resume=config.output.resume if resume is None else resume,
    )
    _validate_config(config.data, model, output)
    return replace(config, model=model, output=output)


def discover_input_files(config: DataConfig) -> tuple[Path, ...]:
    """Return input shards in deterministic row order."""

    if not config.input_dir.is_dir():
        raise DataValidationError(f"输入目录不存在：{config.input_dir}")
    if config.success_marker and not (config.input_dir / config.success_marker).is_file():
        raise DataValidationError(f"缺少完成标记：{config.success_marker}")
    files = tuple(sorted(config.input_dir.glob(config.file_pattern)))
    if not files:
        raise DataValidationError(f"没有匹配到输入文件：{config.file_pattern}")
    return files


def _read_record(
    raw_line: bytes,
    *,
    path: Path,
    line_number: int,
    id_field: str,
    text_field: str,
) -> tuple[str, str]:
    if not raw_line.strip():
        raise DataValidationError(f"{path.name}:{line_number} 是空行")
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"{path.name}:{line_number} JSON 解析失败") from exc
    if not isinstance(record, dict):
        raise DataValidationError(f"{path.name}:{line_number} 不是 JSON object")

    poi_id = record.get(id_field)
    text = record.get(text_field)
    if not isinstance(poi_id, str) or not poi_id.strip():
        raise DataValidationError(f"{path.name}:{line_number} 的 {id_field} 为空或不是字符串")
    if not isinstance(text, str) or not text.strip():
        raise DataValidationError(f"{path.name}:{line_number} 的 {text_field} 为空或不是字符串")
    return poi_id, text


def scan_and_prepare_input(
    files: Sequence[Path],
    config: DataConfig,
    ids_path: Path,
    *,
    max_rows: int | None = None,
) -> PreparedInput:
    """Validate input rows and write the aligned POI ID mapping."""

    if max_rows is not None and max_rows <= 0:
        raise ConfigError("max_rows 必须是正整数")

    ids_path.parent.mkdir(parents=True, exist_ok=True)
    temp_ids_path = ids_path.with_name(f".{ids_path.name}.tmp")
    seen_ids: set[str] | None = set() if config.check_unique_ids else None
    total_rows = 0
    sources: list[dict[str, Any]] = []

    try:
        with temp_ids_path.open("w", encoding="utf-8") as ids_handle:
            for path in files:
                file_hash = hashlib.sha256()
                file_rows = 0
                file_bytes = 0
                with path.open("rb") as input_handle:
                    for line_number, raw_line in enumerate(input_handle, start=1):
                        if max_rows is not None and total_rows >= max_rows:
                            break
                        poi_id, _ = _read_record(
                            raw_line,
                            path=path,
                            line_number=line_number,
                            id_field=config.id_field,
                            text_field=config.text_field,
                        )
                        if seen_ids is not None:
                            if poi_id in seen_ids:
                                raise DataValidationError(
                                    f"{path.name}:{line_number} 出现重复 {config.id_field}"
                                )
                            seen_ids.add(poi_id)
                        ids_handle.write(json.dumps(poi_id, ensure_ascii=False) + "\n")
                        file_hash.update(raw_line)
                        file_rows += 1
                        file_bytes += len(raw_line)
                        total_rows += 1

                if file_rows:
                    sources.append(
                        {
                            "name": path.name,
                            "rows_scanned": file_rows,
                            "bytes_scanned": file_bytes,
                            "sha256_scanned": file_hash.hexdigest(),
                        }
                    )
                if max_rows is not None and total_rows >= max_rows:
                    break

        if max_rows is not None and total_rows < max_rows:
            raise DataValidationError(
                f"请求 {max_rows} 行，但输入只有 {total_rows} 行"
            )
        if max_rows is None and config.expected_rows is not None:
            if total_rows != config.expected_rows:
                raise DataValidationError(
                    f"实际行数 {total_rows} != expected_rows {config.expected_rows}"
                )
        os.replace(temp_ids_path, ids_path)
    except BaseException:
        temp_ids_path.unlink(missing_ok=True)
        raise

    fingerprint_payload = {
        "id_field": config.id_field,
        "text_field": config.text_field,
        "total_rows": total_rows,
        "sources": sources,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return PreparedInput(
        files=tuple(files),
        total_rows=total_rows,
        fingerprint=fingerprint,
        sources=tuple(sources),
    )


def iter_texts(
    prepared: PreparedInput,
    config: DataConfig,
    *,
    start_row: int = 0,
) -> Iterator[str]:
    """Yield texts in the scanned order, optionally skipping completed rows."""

    current_row = 0
    for source in prepared.sources:
        path = next(path for path in prepared.files if path.name == source["name"])
        rows_to_read = int(source["rows_scanned"])
        with path.open("rb") as input_handle:
            for line_number, raw_line in enumerate(
                itertools.islice(input_handle, rows_to_read), start=1
            ):
                _, text = _read_record(
                    raw_line,
                    path=path,
                    line_number=line_number,
                    id_field=config.id_field,
                    text_field=config.text_field,
                )
                if current_row >= start_row:
                    yield text
                current_row += 1


def _chunked(values: Iterable[str], chunk_size: int) -> Iterator[list[str]]:
    iterator = iter(values)
    while chunk := list(itertools.islice(iterator, chunk_size)):
        yield chunk


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _load_encoder(config: ModelConfig) -> tuple[TextEncoder, str, float]:
    import torch
    from sentence_transformers import SentenceTransformer

    if not config.path.is_dir():
        raise ConfigError(f"模型目录不存在：{config.path}")
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前进程无法访问 GPU")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype_map[config.torch_dtype]}
    if config.attention:
        model_kwargs["attn_implementation"] = config.attention

    started = time.perf_counter()
    model = SentenceTransformer(
        str(config.path),
        device=device,
        local_files_only=True,
        trust_remote_code=False,
        truncate_dim=config.truncate_dim,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": config.padding_side},
    )
    model.max_seq_length = config.max_seq_length
    return model, device, time.perf_counter() - started


def _job_signature(
    config: EmbeddingJobConfig,
    prepared: PreparedInput,
    embedding_dim: int,
) -> str:
    payload = {
        "input_fingerprint": prepared.fingerprint,
        "total_rows": prepared.total_rows,
        "model": {
            **asdict(config.model),
            "path": str(config.model.path),
        },
        "output_dtype": config.output.embedding_dtype,
        "embedding_dim": embedding_dim,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _create_npy_file(path: Path, dtype: np.dtype[Any], shape: tuple[int, int]) -> int:
    header = {
        "descr": np.lib.format.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": shape,
    }
    with path.open("wb") as handle:
        np.lib.format.write_array_header_2_0(handle, header)
        return handle.tell()


def _read_npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype[Any], bool, int]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise RuntimeError(f"不支持的 NPY 版本：{version}")
        return shape, dtype, fortran_order, handle.tell()


def encode_texts_to_npy(
    encoder: TextEncoder,
    texts: Iterable[str],
    *,
    embeddings_path: Path,
    progress_path: Path,
    total_rows: int,
    embedding_dim: int,
    start_row: int,
    model_config: ModelConfig,
    output_config: OutputConfig,
    signature: str,
    show_progress: bool = True,
    progress_desc: str = "POI embedding",
) -> dict[str, Any]:
    """Encode buffered text batches into a resumable NPY file."""

    output_dtype = np.dtype(output_config.embedding_dtype)
    expected_shape = (total_rows, embedding_dim)
    if start_row == 0:
        data_offset = _create_npy_file(
            embeddings_path,
            output_dtype,
            expected_shape,
        )
    else:
        if not embeddings_path.is_file():
            raise RuntimeError("存在断点状态，但 embeddings.npy 不存在")
        shape, dtype, fortran_order, data_offset = _read_npy_header(embeddings_path)
        if shape != expected_shape:
            raise RuntimeError(
                f"已有向量 shape {shape} 与预期 {expected_shape} 不一致"
            )
        if dtype != output_dtype:
            raise RuntimeError(
                f"已有向量 dtype {dtype} 与预期 {output_dtype} 不一致"
            )
        if fortran_order:
            raise RuntimeError("不支持 Fortran-order NPY 断点文件")

    next_row = start_row
    row_bytes = embedding_dim * output_dtype.itemsize
    resume_offset = data_offset + start_row * row_bytes
    current_size = embeddings_path.stat().st_size
    if current_size < resume_offset:
        raise RuntimeError("embeddings.npy 比断点记录的已完成行更短")

    embeddings_handle = embeddings_path.open("r+b", buffering=8 * 1024 * 1024)
    embeddings_handle.truncate(resume_offset)
    embeddings_handle.seek(resume_offset)
    batches_since_checkpoint = 0
    encode_started = time.perf_counter()
    progress = tqdm(
        total=total_rows,
        initial=start_row,
        unit="row",
        desc=progress_desc,
        disable=not show_progress,
    )
    try:
        for text_buffer in _chunked(texts, model_config.encode_buffer_size):
            vectors = np.asarray(
                encoder.encode(
                    text_buffer,
                    batch_size=model_config.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=model_config.normalize_embeddings,
                    prompt_name=model_config.prompt_name,
                )
            )
            expected_vector_shape = (len(text_buffer), embedding_dim)
            if vectors.shape != expected_vector_shape:
                raise RuntimeError(
                    f"模型输出 shape {vectors.shape} 与预期 {expected_vector_shape} 不一致"
                )
            batch_end = next_row + len(text_buffer)
            if batch_end > total_rows:
                raise RuntimeError("编码行数超过扫描得到的总行数")
            embeddings_handle.write(
                vectors.astype(output_dtype, copy=False).tobytes(order="C")
            )
            next_row = batch_end
            batches_since_checkpoint += (
                len(text_buffer) + model_config.batch_size - 1
            ) // model_config.batch_size
            progress.update(len(text_buffer))

            if batches_since_checkpoint >= output_config.checkpoint_interval_batches:
                embeddings_handle.flush()
                os.fsync(embeddings_handle.fileno())
                _write_json_atomic(
                    progress_path,
                    {
                        "signature": signature,
                        "status": "running",
                        "next_row": next_row,
                        "total_rows": total_rows,
                        "updated_at": _utc_now(),
                    },
                )
                batches_since_checkpoint = 0

        if next_row != total_rows:
            raise RuntimeError(f"只编码了 {next_row} 行，预期 {total_rows} 行")
        embeddings_handle.flush()
        os.fsync(embeddings_handle.fileno())
        elapsed = time.perf_counter() - encode_started
        _write_json_atomic(
            progress_path,
            {
                "signature": signature,
                "status": "completed",
                "next_row": next_row,
                "total_rows": total_rows,
                "updated_at": _utc_now(),
            },
        )
        return {
            "rows_encoded_this_run": next_row - start_row,
            "encoding_seconds": elapsed,
            "rows_per_second": (next_row - start_row) / elapsed if elapsed else None,
        }
    finally:
        progress.close()
        embeddings_handle.close()


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def run_embedding_job(
    config: EmbeddingJobConfig,
    *,
    project_root: Path,
    max_rows: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run one embedding job and persist its reproducibility metadata."""

    output_dir = config.output.dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "poi_ids.jsonl"
    embeddings_path = output_dir / "embeddings.npy"
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    has_progress = progress_path.is_file()
    if not has_progress and (embeddings_path.exists() or manifest_path.exists()):
        raise RuntimeError("输出目录已有产物但缺少 progress.json，请更换输出目录")

    scan_started = time.perf_counter()
    files = discover_input_files(config.data)
    # Keep the existing ID mapping untouched until resume compatibility is verified.
    scan_ids_path = (
        output_dir / ".poi_ids.resume.jsonl" if has_progress else ids_path
    )
    prepared = scan_and_prepare_input(
        files,
        config.data,
        scan_ids_path,
        max_rows=max_rows,
    )
    scan_seconds = time.perf_counter() - scan_started

    encoder, device, model_load_seconds = _load_encoder(config.model)
    embedding_dim = encoder.get_sentence_embedding_dimension()
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise RuntimeError("模型未返回有效的 Embedding 维度")
    signature = _job_signature(config, prepared, embedding_dim)

    start_row = 0
    if has_progress:
        with progress_path.open("r", encoding="utf-8") as handle:
            progress_state = json.load(handle)
        if progress_state.get("signature") != signature:
            scan_ids_path.unlink(missing_ok=True)
            raise RuntimeError("输出目录中的断点与当前数据或配置不一致，请更换输出目录")
        if not config.output.resume:
            scan_ids_path.unlink(missing_ok=True)
            raise RuntimeError("输出目录存在断点，但 output.resume=false")
        start_row = int(progress_state.get("next_row", 0))
        if progress_state.get("status") == "completed" and start_row == prepared.total_rows:
            scan_ids_path.unlink(missing_ok=True)
            raise RuntimeError("该输出目录中的任务已经完成")
        os.replace(scan_ids_path, ids_path)

    _write_json_atomic(
        progress_path,
        {
            "signature": signature,
            "status": "running",
            "next_row": start_row,
            "total_rows": prepared.total_rows,
            "updated_at": _utc_now(),
        },
    )

    manifest: dict[str, Any] = {
        "job_name": config.job_name,
        "status": "running",
        "started_at": _utc_now(),
        "input": {
            "dir": _display_path(config.data.input_dir, project_root),
            "file_pattern": config.data.file_pattern,
            "id_field": config.data.id_field,
            "text_field": config.data.text_field,
            "total_rows": prepared.total_rows,
            "fingerprint": prepared.fingerprint,
            "sources": prepared.sources,
        },
        "model": {
            "path": _display_path(config.model.path, project_root),
            "device": device,
            "batch_size": config.model.batch_size,
            "encode_buffer_size": config.model.encode_buffer_size,
            "max_seq_length": config.model.max_seq_length,
            "torch_dtype": config.model.torch_dtype,
            "attention": config.model.attention,
            "padding_side": config.model.padding_side,
            "normalize_embeddings": config.model.normalize_embeddings,
            "truncate_dim": config.model.truncate_dim,
            "prompt_name": config.model.prompt_name,
            "embedding_dim": embedding_dim,
        },
        "output": {
            "dir": _display_path(output_dir, project_root),
            "embeddings": embeddings_path.name,
            "poi_ids": ids_path.name,
            "shape": [prepared.total_rows, embedding_dim],
            "dtype": config.output.embedding_dtype,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "sentence_transformers": _package_version("sentence-transformers"),
            "scan_seconds": scan_seconds,
            "model_load_seconds": model_load_seconds,
        },
        "signature": signature,
    }
    _write_json_atomic(manifest_path, manifest)

    cuda_device = None
    if device.startswith("cuda"):
        import torch

        cuda_device = torch.device(device)
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        manifest["runtime"]["gpu"] = torch.cuda.get_device_name(cuda_device)
        _write_json_atomic(manifest_path, manifest)

    try:
        encode_metrics = encode_texts_to_npy(
            encoder,
            iter_texts(prepared, config.data, start_row=start_row),
            embeddings_path=embeddings_path,
            progress_path=progress_path,
            total_rows=prepared.total_rows,
            embedding_dim=embedding_dim,
            start_row=start_row,
            model_config=config.model,
            output_config=config.output,
            signature=signature,
            show_progress=show_progress,
        )
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            allocated = int(torch.cuda.max_memory_allocated(cuda_device))
            reserved = int(torch.cuda.max_memory_reserved(cuda_device))
            encode_metrics.update(
                {
                    "cuda_peak_memory_allocated_bytes": allocated,
                    "cuda_peak_memory_reserved_bytes": reserved,
                    "cuda_peak_memory_allocated_gib": allocated / 1024**3,
                    "cuda_peak_memory_reserved_gib": reserved / 1024**3,
                }
            )
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _write_json_atomic(manifest_path, manifest)
        raise

    manifest.update(
        {
            "status": "completed",
            "finished_at": _utc_now(),
            "metrics": encode_metrics,
        }
    )
    _write_json_atomic(manifest_path, manifest)
    return manifest
