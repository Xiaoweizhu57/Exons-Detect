"""Public package interface for Exons-Detect."""

from __future__ import annotations

__all__ = ["ExonsDetect", "Exons_Detect"]


def __getattr__(name: str):
    if name in {"ExonsDetect", "Exons_Detect"}:
        from .detector import ExonsDetect, Exons_Detect

        return {"ExonsDetect": ExonsDetect, "Exons_Detect": Exons_Detect}[name]
    raise AttributeError(f"module 'exons_detect' has no attribute '{name}'")
