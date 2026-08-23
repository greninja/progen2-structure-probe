"""Remote-only technical smoke checks for model APIs and resource use."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from .artifacts import write_json_atomic


SMOKE_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def synthetic_sequence(length: int) -> str:
    if length < 16:
        raise ValueError("smoke sequence length must be at least 16")
    return (SMOKE_ALPHABET * ((length + len(SMOKE_ALPHABET) - 1) // len(SMOKE_ALPHABET)))[:length]


def _measure_cuda(torch: Any, operation: Callable[[], Any]) -> tuple[Any, float, dict[str, int]]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = operation()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    memory = {
        "allocated_after_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
    }
    return result, elapsed, memory


def run_model_smoke(
    progen_model: object,
    esm_model: object,
    output_dir: Path,
    sequence_length: int = 128,
) -> dict[str, Any]:
    """Validate extraction shapes, causal masking, determinism, and peak VRAM."""

    sequence = synthetic_sequence(sequence_length)
    progen_first, progen_seconds, progen_memory = _measure_cuda(
        progen_model.torch, lambda: progen_model.extract(sequence)
    )
    progen_second, repeat_seconds, repeat_memory = _measure_cuda(
        progen_model.torch, lambda: progen_model.extract(sequence)
    )
    if progen_first.attentions.shape != (27, 16, sequence_length, sequence_length):
        raise ValueError(f"unexpected ProGen2 attention shape {progen_first.attentions.shape}")
    if progen_first.hidden_states.shape != (28, sequence_length, 1536):
        raise ValueError(f"unexpected ProGen2 hidden-state shape {progen_first.hidden_states.shape}")
    if not np.isfinite(progen_first.attentions).all() or not np.isfinite(progen_first.hidden_states).all():
        raise ValueError("ProGen2 extraction contains non-finite values")
    if not np.allclose(np.triu(progen_first.attentions, k=1), 0.0, atol=1e-6):
        raise ValueError("ProGen2 attention is not causally masked")
    progen_attention_delta = float(
        np.max(np.abs(progen_first.attentions - progen_second.attentions))
    )
    progen_hidden_delta = float(
        np.max(np.abs(progen_first.hidden_states - progen_second.hidden_states))
    )
    if progen_attention_delta > 1e-6 or progen_hidden_delta > 1e-6:
        raise ValueError("repeated ProGen2 extraction is not deterministic within tolerance")

    esm_first, esm_seconds, esm_memory = _measure_cuda(
        esm_model.torch, lambda: esm_model.extract_attention(sequence)
    )
    esm_second, esm_repeat_seconds, esm_repeat_memory = _measure_cuda(
        esm_model.torch, lambda: esm_model.extract_attention(sequence)
    )
    if esm_first.ndim != 4 or esm_first.shape[0] != 12 or esm_first.shape[-2:] != (
        sequence_length,
        sequence_length,
    ):
        raise ValueError(f"unexpected ESM-2 attention shape {esm_first.shape}")
    if not np.isfinite(esm_first).all():
        raise ValueError("ESM-2 extraction contains non-finite values")
    esm_attention_delta = float(np.max(np.abs(esm_first - esm_second)))
    if esm_attention_delta > 1e-6:
        raise ValueError("repeated ESM-2 extraction is not deterministic within tolerance")

    result = {
        "status": "passed",
        "synthetic_sequence_length": sequence_length,
        "synthetic_sequence": sequence,
        "progen2": {
            "attention_shape": list(progen_first.attentions.shape),
            "hidden_state_shape": list(progen_first.hidden_states.shape),
            "first_seconds": progen_seconds,
            "repeat_seconds": repeat_seconds,
            "attention_max_abs_repeat_delta": progen_attention_delta,
            "hidden_max_abs_repeat_delta": progen_hidden_delta,
            "first_memory": progen_memory,
            "repeat_memory": repeat_memory,
            "returned_attention_bytes": int(progen_first.attentions.nbytes),
            "returned_hidden_state_bytes": int(progen_first.hidden_states.nbytes),
        },
        "esm2": {
            "attention_shape": list(esm_first.shape),
            "first_seconds": esm_seconds,
            "repeat_seconds": esm_repeat_seconds,
            "attention_max_abs_repeat_delta": esm_attention_delta,
            "first_memory": esm_memory,
            "repeat_memory": esm_repeat_memory,
            "returned_attention_bytes": int(esm_first.nbytes),
        },
    }
    write_json_atomic(Path(output_dir) / "smoke.json", result)
    return result

