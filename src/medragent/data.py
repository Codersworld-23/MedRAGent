"""Validated loaders for the MedRAGent benchmark and evidence corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


class DatasetValidationError(ValueError):
    """Raised when a dataset record does not satisfy the project schema."""


@dataclass(frozen=True)
class MedQARecord:
    """A normalized MedQA-US multiple-choice record."""

    id: str
    question: str
    options: dict[str, str]
    answer_index: int | None
    split: str
    source_file: str
    metadata: dict[str, Any]

    @property
    def answer_letter(self) -> str | None:
        """Return the answer as an A-E letter when a label is available."""
        if self.answer_index is None:
            return None
        letters = tuple(self.options)
        if 0 <= self.answer_index < len(letters):
            return letters[self.answer_index]
        return None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable normalized representation."""
        return {
            "id": self.id,
            "question": self.question,
            "options": dict(self.options),
            "answer_index": self.answer_index,
            "answer_letter": self.answer_letter,
            "split": self.split,
            "source_file": self.source_file,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TextChunk:
    """A normalized retrievable textbook chunk."""

    chunk_id: str
    title: str
    text: str
    source_file: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable normalized representation."""
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "text": self.text,
            "source_file": self.source_file,
            "metadata": dict(self.metadata),
        }


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read non-empty JSONL records with line-numbered validation errors."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetValidationError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise DatasetValidationError(
                        f"{path}:{line_number}: expected a JSON object"
                    )
                yield line_number, record
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset file not found: {path}") from exc


def _required_text(record: Mapping[str, Any], field: str, path: Path, line: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{path}:{line}: '{field}' must be non-empty text")
    return value.strip()


def _normalize_options(
    value: Any, path: Path, line: int
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise DatasetValidationError(f"{path}:{line}: 'options' must be a non-empty object")
    options: dict[str, str] = {}
    for key, option in value.items():
        if not isinstance(key, str) or not key.strip():
            raise DatasetValidationError(f"{path}:{line}: option labels must be text")
        if not isinstance(option, str) or not option.strip():
            raise DatasetValidationError(
                f"{path}:{line}: option '{key}' must contain non-empty text"
            )
        options[key.strip().upper()] = option.strip()
    return options


def _normalize_answer_index(value: Any, options: Mapping[str, str], path: Path, line: int) -> int | None:
    if value is None:
        return None
    labels = tuple(options)
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in options:
            return labels.index(normalized)
        if normalized.isdigit():
            value = int(normalized)
        else:
            raise DatasetValidationError(
                f"{path}:{line}: answer_idx '{value}' is not a valid option label"
            )
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetValidationError(
            f"{path}:{line}: 'answer_idx' must be an option letter or integer"
        )
    if not 0 <= value < len(labels):
        raise DatasetValidationError(
            f"{path}:{line}: answer_idx {value} is outside {len(labels)} options"
        )
    return value


def load_medqa_jsonl(path: str | Path, split: str | None = None) -> list[MedQARecord]:
    """Load and validate MedQA-US JSONL records in deterministic file order."""
    dataset_path = Path(path)
    resolved_split = split or dataset_path.stem
    records: list[MedQARecord] = []
    seen_ids: set[str] = set()

    for line_number, raw in _read_jsonl(dataset_path):
        raw_id = raw.get("id")
        record_id = (
            raw_id.strip()
            if isinstance(raw_id, str) and raw_id.strip()
            else f"{resolved_split}-{line_number:06d}"
        )
        if record_id in seen_ids:
            raise DatasetValidationError(f"{dataset_path}:{line_number}: duplicate id '{record_id}'")
        seen_ids.add(record_id)
        question = _required_text(raw, "question", dataset_path, line_number)
        options = _normalize_options(raw.get("options"), dataset_path, line_number)
        answer_index = _normalize_answer_index(
            raw.get("answer_idx"), options, dataset_path, line_number
        )
        metadata = {
            key: value
            for key, value in raw.items()
            if key not in {"id", "question", "options", "answer_idx"}
        }
        metadata["source_line"] = line_number
        records.append(
            MedQARecord(
                id=record_id,
                question=question,
                options=options,
                answer_index=answer_index,
                split=resolved_split,
                source_file=str(dataset_path),
                metadata=metadata,
            )
        )
    return records


def load_text_chunks_jsonl(path: str | Path) -> list[TextChunk]:
    """Load and validate MedRAG textbook chunks in deterministic file order."""
    dataset_path = Path(path)
    records: list[TextChunk] = []
    seen_ids: set[str] = set()

    for line_number, raw in _read_jsonl(dataset_path):
        chunk_id = _required_text(raw, "id", dataset_path, line_number)
        if chunk_id in seen_ids:
            raise DatasetValidationError(f"{dataset_path}:{line_number}: duplicate id '{chunk_id}'")
        seen_ids.add(chunk_id)
        title = _required_text(raw, "title", dataset_path, line_number)
        content = raw.get("content")
        contents = raw.get("contents")
        text = content if isinstance(content, str) and content.strip() else contents
        if not isinstance(text, str) or not text.strip():
            raise DatasetValidationError(
                f"{dataset_path}:{line_number}: 'content' or 'contents' must be non-empty text"
            )
        metadata = {
            key: value
            for key, value in raw.items()
            if key not in {"id", "title", "content", "contents"}
        }
        records.append(
            TextChunk(
                chunk_id=chunk_id,
                title=title,
                text=text.strip(),
                source_file=str(dataset_path),
                metadata=metadata,
            )
        )
    return records
