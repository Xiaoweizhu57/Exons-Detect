"""Core detector implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import weighted_score
from .utils import assert_tokenizer_consistency

torch.set_grad_enabled(False)


def _resolve_device(index: int) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if torch.cuda.device_count() > index:
        return f"cuda:{index}"
    return "cuda:0"


@dataclass(slots=True)
class DetectorConfig:
    observer_name_or_path: str = "tiiuae/falcon-7b"
    performer_name_or_path: str = "tiiuae/falcon-7b-instruct"
    use_bfloat16: bool = False
    max_token_observed: int = 1024
    tau: float = 0.15
    alpha: float = 10.0
    hidden_num: int = 32
    trust_remote_code: bool = False
    hf_token: str | None = None


class ExonsDetect:
    """Weighted dual-model text detector."""

    def __init__(self, **kwargs: object) -> None:
        self.config = DetectorConfig(
            hf_token=os.environ.get("HF_TOKEN"),
            **kwargs,
        )
        assert_tokenizer_consistency(
            self.config.observer_name_or_path,
            self.config.performer_name_or_path,
        )

        self.observer_device = _resolve_device(0)
        self.performer_device = _resolve_device(1)
        self.observer_model = self._load_model(
            self.config.observer_name_or_path,
            self.observer_device,
        )
        self.performer_model = self._load_model(
            self.config.performer_name_or_path,
            self.performer_device,
        )

        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.observer_name_or_path,
            token=self.config.hf_token,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self, model_name_or_path: str, device: str) -> AutoModelForCausalLM:
        dtype = torch.bfloat16 if self.config.use_bfloat16 else torch.float32
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=self.config.hf_token,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=dtype,
            device_map={"": device},
        )

    def _tokenize(self, texts: list[str]) -> transformers.BatchEncoding:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_token_observed,
            return_token_type_ids=False,
        )

    @torch.inference_mode()
    def _forward_with_hidden(
        self, encodings: transformers.BatchEncoding
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        encodings_observer = {key: value.to(self.observer_device) for key, value in encodings.items()}
        encodings_performer = {key: value.to(self.performer_device) for key, value in encodings.items()}

        observer_output = self.observer_model(
            **encodings_observer,
            output_hidden_states=True,
        )
        performer_output = self.performer_model(
            **encodings_performer,
            output_hidden_states=True,
        )

        observer_logits = observer_output.logits
        performer_logits = performer_output.logits.to(observer_logits.device)
        observer_hidden_states = tuple(observer_output.hidden_states)
        performer_hidden_states = tuple(
            hidden.to(observer_logits.device) for hidden in performer_output.hidden_states
        )

        if self.observer_device.startswith("cuda"):
            torch.cuda.synchronize()

        return (
            observer_logits,
            performer_logits,
            observer_hidden_states,
            performer_hidden_states,
        )

    def score_batch(self, texts: Iterable[str]) -> list[float]:
        """Score a batch of texts and return one float per input text."""
        batch = list(texts)
        if not batch:
            return []

        encodings = self._tokenize(batch)
        observer_logits, performer_logits, observer_hidden_states, performer_hidden_states = (
            self._forward_with_hidden(encodings)
        )

        input_ids = encodings["input_ids"].to(observer_logits.device)
        attention_mask = encodings["attention_mask"].to(observer_logits.device)
        scores = weighted_score(
            input_ids=input_ids,
            attention_mask=attention_mask,
            performer_logits=performer_logits,
            observer_logits=observer_logits,
            pad_token_id=self.tokenizer.pad_token_id,
            observer_hidden_states=observer_hidden_states,
            performer_hidden_states=performer_hidden_states,
            tau=self.config.tau,
            alpha=self.config.alpha,
            hidden_num=self.config.hidden_num,
        )
        return scores.detach().cpu().tolist()

    def score_text(self, text: str) -> float:
        """Score a single text."""
        return self.score_batch([text])[0]

    def compute_w_score(self, input_text: str | list[str]) -> float | list[float]:
        """Backward-compatible wrapper for the original API."""
        if isinstance(input_text, str):
            return self.score_text(input_text)
        return self.score_batch(input_text)

    def cleanup(self) -> None:
        """Release models and clear CUDA memory if available."""
        if getattr(self, "observer_model", None) is not None:
            del self.observer_model
            self.observer_model = None
        if getattr(self, "performer_model", None) is not None:
            del self.performer_model
            self.performer_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


Exons_Detect = ExonsDetect
