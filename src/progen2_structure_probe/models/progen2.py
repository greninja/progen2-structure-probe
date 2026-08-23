"""Adapter for Salesforce's official ProGen2 implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from ..copy_bias import repeat_prompt, summarize_duplicate_log_probs, validate_protein_sequence


@dataclass(frozen=True)
class ProGenExtraction:
    token_ids: np.ndarray
    attentions: np.ndarray  # [layers, heads, residues, residues]
    hidden_states: np.ndarray  # [embedding+layers, residues, hidden]


class OfficialProGen2:
    def __init__(
        self,
        upstream_repo: Path,
        checkpoint: Path,
        device: str = "cuda",
        fp16: bool = True,
    ) -> None:
        try:
            import torch
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError("install the pinned RunPod model environment first") from error

        progen2_root = Path(upstream_repo).resolve() / "progen2"
        if not (progen2_root / "models/progen/modeling_progen.py").exists():
            raise ValueError(f"not an official ProGen2 checkout: {progen2_root}")
        sys.path.insert(0, str(progen2_root))
        from models.progen.modeling_progen import ProGenForCausalLM

        self.torch = torch
        self.device = torch.device(device)
        tokenizer_path = progen2_root / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        dtype = torch.float16 if fp16 else torch.float32
        self.model = ProGenForCausalLM.from_pretrained(
            str(Path(checkpoint).resolve()), torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

    def _encode(self, context: str) -> Any:
        ids = self.tokenizer.encode(context).ids
        return self.torch.tensor(ids, dtype=self.torch.long, device=self.device)[None, :]

    def extract(self, sequence: str) -> ProGenExtraction:
        seq = validate_protein_sequence(sequence)
        input_ids = self._encode("1" + seq + "2")
        if input_ids.shape[1] != len(seq) + 2:
            raise ValueError("tokenizer did not produce one token per residue plus terminals")
        with self.torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        attention = self.torch.stack(output.attentions, dim=0)[:, 0, :, 1:-1, 1:-1]
        hidden = self.torch.stack(output.hidden_states, dim=0)[:, 0, 1:-1, :]
        return ProGenExtraction(
            token_ids=input_ids[0].detach().cpu().numpy(),
            attentions=attention.detach().cpu().float().numpy(),
            hidden_states=hidden.detach().cpu().float().numpy(),
        )

    def duplicate_perplexity(self, sequence: str) -> tuple[float, float, np.ndarray]:
        seq = validate_protein_sequence(sequence)
        input_ids = self._encode("1" + seq + seq + "2")
        with self.torch.inference_mode():
            logits = self.model(input_ids=input_ids, use_cache=False, return_dict=True).logits
            next_log_probs = self.torch.log_softmax(logits[:, :-1, :], dim=-1)
            targets = input_ids[:, 1:]
            selected = next_log_probs.gather(-1, targets[..., None]).squeeze(-1)[0]
        residue_log_probs = selected[: 2 * len(seq)].detach().cpu().numpy()
        original, repeated = summarize_duplicate_log_probs(residue_log_probs, len(seq))
        return original, repeated, residue_log_probs

    def generate_repeat(
        self,
        sequence: str,
        new_residues: int = 500,
        strategy: str = "greedy",
        top_p: float = 0.95,
        temperature: float = 0.8,
        seed: int = 20260822,
    ) -> tuple[str, int]:
        context, prefix_length = repeat_prompt(sequence)
        input_ids = self._encode(context)
        self.torch.manual_seed(seed)
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(seed)
        generation = {
            "input_ids": input_ids,
            "max_length": input_ids.shape[1] + new_residues,
            "pad_token_id": self.tokenizer.encode("<|pad|>").ids[0],
        }
        if strategy == "greedy":
            generation["do_sample"] = False
        elif strategy == "nucleus":
            generation.update(do_sample=True, top_p=top_p, temperature=temperature)
        else:
            raise ValueError("strategy must be 'greedy' or 'nucleus'")
        with self.torch.inference_mode():
            output_ids = self.model.generate(**generation)
        generated_ids = output_ids[0, input_ids.shape[1] :].detach().cpu().tolist()
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        residues = "".join(character for character in decoded if character in "ACDEFGHIKLMNPQRSTVWY")
        return residues, prefix_length
