"""Runtime and source provenance capture."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from .artifacts import sha256_file


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256_tree(path: Path) -> str:
    root = Path(path).resolve()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"cannot hash empty directory {root}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def runtime_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda_available"] = torch.cuda.is_available()
        record["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
            record["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        record["torch"] = None
        record["cuda_available"] = False
    return record

