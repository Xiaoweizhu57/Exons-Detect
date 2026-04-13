"""Command-line interface for Exons-Detect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .detector import ExonsDetect


def _load_texts(input_path: Path, text_key: str | None) -> list[str]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        if not all(isinstance(item, str) for item in payload):
            raise ValueError("When the input JSON is a list, every element must be a string.")
        return payload

    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be either a dictionary or a list of strings.")

    if text_key is None:
        for candidate in ("text", "human_text", "machine_text"):
            if candidate in payload:
                text_key = candidate
                break

    if text_key is None or text_key not in payload:
        raise ValueError(
            "Could not find a usable text field. Pass --text-key explicitly or use one of: "
            "text, human_text, machine_text."
        )

    texts = payload[text_key]
    if not isinstance(texts, list) or not all(isinstance(item, str) for item in texts):
        raise ValueError(f"Field '{text_key}' must be a list of strings.")
    return texts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score text with Exons-Detect.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", required=True, help="Path to the output JSON file.")
    parser.add_argument("--text-key", default=None, help="JSON key containing a list of texts.")
    parser.add_argument("--observer", required=True, help="Observer model name or local path.")
    parser.add_argument("--performer", required=True, help="Performer model name or local path.")
    parser.add_argument("--max-token-observed", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--hidden-num", type=int, default=32)
    parser.add_argument("--use-bfloat16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    texts = _load_texts(input_path, args.text_key)

    detector = ExonsDetect(
        observer_name_or_path=args.observer,
        performer_name_or_path=args.performer,
        use_bfloat16=args.use_bfloat16,
        max_token_observed=args.max_token_observed,
        tau=args.tau,
        alpha=args.alpha,
        hidden_num=args.hidden_num,
        trust_remote_code=args.trust_remote_code,
    )

    predictions = []
    for text in tqdm(texts, desc="Scoring"):
        predictions.append(detector.score_text(text))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"text": texts, "predictions": predictions}, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
