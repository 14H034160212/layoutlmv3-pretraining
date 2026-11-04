"""Lightweight experiment tracking: one JSON line per run in ``experiments/registry.jsonl``.

    with RunTracker("pretrain", args, output_dir=a.output_dir, tags=a.tags) as run:
        run.config.update(model=..., corpus=...)
        ...
        run.metrics.update(best=...)

On exit the record (run id, UTC start/end, status, git commit + dirty flag + diff hash, command line,
parsed args, resolved config, environment, outputs, metrics) is appended to the registry in a single
locked write, and a copy is written to ``<output_dir>/run_record.json``. If the block raises, the record
is still written with ``status = "failed"`` and the error, and the exception propagates.

The registry location is ``$LAYOUTLMV3_REGISTRY`` if set, else ``<repo>/experiments/registry.jsonl``.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_registry() -> Path:
    return Path(os.environ.get("LAYOUTLMV3_REGISTRY", REPO_ROOT / "experiments" / "registry.jsonl"))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _git(*cmd, cwd=REPO_ROOT) -> str | None:
    try:
        return subprocess.run(["git", *cmd], cwd=cwd, capture_output=True, text=True, check=True,
                              timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def git_info(cwd: Path = REPO_ROOT) -> dict:
    """HEAD commit, dirty flag and, if dirty, a hash of the tracked diff plus the untracked file names."""
    head = _git("rev-parse", "HEAD", cwd=cwd)
    if head is None:
        return {"commit": None, "dirty": None, "diff_sha256": None}
    status = _git("status", "--porcelain", cwd=cwd) or ""
    info = {"commit": head.strip(), "dirty": bool(status.strip()), "diff_sha256": None, "untracked": []}
    if info["dirty"]:
        diff = _git("diff", "HEAD", "--binary", cwd=cwd) or ""
        info["diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
        info["untracked"] = [line[3:] for line in status.splitlines() if line.startswith("??")]
    return info


def environment() -> dict:
    env = {"python": platform.python_version(), "hostname": socket.gethostname(),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch
        env.update(torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                   if torch.cuda.is_available() else [])
    except ImportError:
        pass
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    return env


def to_jsonable(x):
    """Best-effort conversion (Paths, numpy scalars, tensors, sets) so a record never fails to serialise."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "item") and callable(x.item):
        try:
            return x.item()
        except (ValueError, RuntimeError):
            pass
    if hasattr(x, "tolist"):
        return x.tolist()
    return str(x)


def append_record(record: dict, registry: Path | str | None = None) -> Path:
    """Append one JSON line under an exclusive lock (one write call, so concurrent runs never interleave)."""
    path = Path(registry or default_registry())
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(to_jsonable(record), sort_keys=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line)
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return path


def write_json_atomic(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=1))
    os.replace(tmp, path)


def read_registry(registry: Path | str | None = None) -> list[dict]:
    path = Path(registry or default_registry())
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class RunTracker:
    """Context manager that records one run. ``config``, ``metrics``, ``outputs`` and ``tags`` are mutable."""

    def __init__(self, kind: str, args=None, output_dir: str | Path | None = None, tags=(),
                 registry: str | Path | None = None, argv: list[str] | None = None):
        self.kind, self.registry = kind, registry
        self.args = vars(args) if args is not None and hasattr(args, "__dict__") else dict(args or {})
        self.output_dir = Path(output_dir) if output_dir else None
        self.tags = list(tags or [])
        self.argv = list(sys.argv if argv is None else argv)
        self.config: dict = {}
        self.metrics: dict = {}
        self.outputs: dict = {}
        self.run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{kind}-{uuid.uuid4().hex[:6]}"
        self.start = None
        self.record: dict | None = None

    def __enter__(self):
        self.start = utc_now()
        self._git, self._env = git_info(), environment()
        return self

    def build_record(self, status: str, error: str | None = None) -> dict:
        outputs = dict(self.outputs)
        if self.output_dir is not None:
            outputs.setdefault("output_dir", str(self.output_dir))
        return {"run_id": self.run_id, "kind": self.kind, "status": status, "backfilled": False,
                "tags": self.tags, "start_utc": self.start, "end_utc": utc_now(), "git": self._git,
                "command": " ".join([Path(sys.executable).name, *self.argv]), "argv": self.argv,
                "cwd": os.getcwd(), "args": self.args, "config": self.config, "environment": self._env,
                "outputs": outputs, "metrics": self.metrics, "error": error}

    def __exit__(self, exc_type, exc, tb):
        failed = exc_type is not None
        error = f"{exc_type.__name__}: {exc}" if failed else None
        self.record = self.build_record("failed" if failed else "completed", error)
        append_record(self.record, self.registry)
        if self.output_dir is not None:
            try:
                write_json_atomic(self.output_dir / "run_record.json", self.record)
            except OSError:
                pass
        return False                                     # never swallow the exception
