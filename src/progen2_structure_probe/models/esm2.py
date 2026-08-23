"""Raw-attention adapter for the official archived ESM implementation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from ..copy_bias import validate_protein_sequence


class OfficialESM2:
    def __init__(self, upstream_repo: Path, device: str = "cuda") -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("install the pinned RunPod model environment first") from error
        sys.path.insert(0, str(Path(upstream_repo).resolve()))
        import esm

        self.torch = torch
        self.device = torch.device(device)
        self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.model = self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    def extract_attention(self, sequence: str) -> np.ndarray:
        seq = validate_protein_sequence(sequence)
        _, _, tokens = self.batch_converter([("protein", seq)])
        tokens = tokens.to(self.device)
        with self.torch.inference_mode():
            output = self.model(tokens, repr_layers=[], need_head_weights=True, return_contacts=False)
        # Official ESM result shape: [batch, layers, heads, tokens, tokens].
        attention = output["attentions"][0, :, :, 1 : len(seq) + 1, 1 : len(seq) + 1]
        return attention.detach().cpu().float().numpy()

