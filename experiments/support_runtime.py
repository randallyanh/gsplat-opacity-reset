"""Runtime logging and exceptions for reproducible gsplat probes."""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


class GsplatOpacityResetError(Exception):
    """Base error with structured context and a stable process exit code."""

    exit_code = 1

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_record(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "context": _jsonable(self.context),
            "exit_code": self.exit_code,
        }


class ConfigurationError(GsplatOpacityResetError):
    exit_code = 2


class EnvironmentCheckError(GsplatOpacityResetError):
    exit_code = 3


class DependencyInstallError(GsplatOpacityResetError):
    exit_code = 4


class KaggleInputError(GsplatOpacityResetError):
    exit_code = 5


class CudaUnavailableError(EnvironmentCheckError):
    pass


class ArtifactWriteError(GsplatOpacityResetError):
    exit_code = 6


class CommandExecutionError(GsplatOpacityResetError):
    exit_code = 7

    def __init__(self, message: str, command: list[str], returncode: int) -> None:
        super().__init__(message, command=command, returncode=returncode)
        self.command = command
        self.returncode = returncode


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


class RunLogger:
    """Small structured logger that is safe for Kaggle stdout and JSONL files."""

    def __init__(
        self,
        name: str,
        log_path: str | Path | None = None,
        *,
        stream: TextIO | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.stream = stream if stream is not None else sys.stdout
        self.context = context or {}
        self.started = time.monotonic()
        self._file: TextIO | None = None
        self.log_path = Path(log_path) if log_path else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def event(
        self,
        event: str,
        message: str | None = None,
        *,
        level: str = "INFO",
        **fields: Any,
    ) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.monotonic() - self.started, 3),
            "logger": self.name,
            "level": level,
            "event": event,
            "message": message or event,
            **self.context,
            **fields,
        }
        record = _jsonable(record)
        console = self._format_console(record)
        print(console, file=self.stream, flush=True)
        if self._file is not None:
            self._file.write(json.dumps(record, sort_keys=True) + "\n")
            self._file.flush()
        return record

    def info(self, event: str, message: str | None = None, **fields: Any) -> None:
        self.event(event, message, level="INFO", **fields)

    def warning(self, event: str, message: str | None = None, **fields: Any) -> None:
        self.event(event, message, level="WARNING", **fields)

    def error(self, event: str, message: str | None = None, **fields: Any) -> None:
        self.event(event, message, level="ERROR", **fields)

    def section(self, title: str, **fields: Any) -> None:
        self.event("section", title, level="INFO", **fields)

    def progress(self, name: str, *, step: int, total: int, **fields: Any) -> None:
        pct = 100.0 * step / total if total else 0.0
        self.event(
            "progress",
            f"{name} {step}/{total}",
            level="INFO",
            progress=name,
            step=step,
            total=total,
            pct=round(pct, 2),
            **fields,
        )

    def exception(self, event: str, exc: BaseException, **fields: Any) -> None:
        if isinstance(exc, GsplatOpacityResetError):
            error_record = exc.to_record()
        else:
            error_record = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "context": {},
                "exit_code": 1,
            }
        self.event(
            event,
            error_record["message"],
            level="ERROR",
            error=error_record,
            traceback="".join(traceback.format_exception(exc)),
            **fields,
        )

    def _format_console(self, record: dict[str, Any]) -> str:
        extras = []
        for key, value in record.items():
            if key in {"ts", "elapsed_s", "logger", "level", "event", "message", "traceback"}:
                continue
            if key in self.context:
                continue
            extras.append(f"{key}={value!r}")
        suffix = " " + " ".join(extras) if extras else ""
        return (
            f"[{record['ts']}] {record['level']} {record['event']}: "
            f"{record['message']}{suffix}"
        )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ArtifactWriteError(
            "Failed to write JSON artifact",
            path=output_path,
            original_error=repr(exc),
        ) from exc
