from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping


SIGNAL_KEYS = ("ast", "config", "graph", "lexical")
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

DEFAULT_CALIBRATION: Dict[str, Dict[str, float]] = {
    "default": {
        "weight.ast": 3.0,
        "weight.config": 3.0,
        "weight.graph": 2.0,
        "weight.lexical": 1.0,
        "high_min": 5.0,
        "medium_min": 2.0,
    },
    "entrypoint": {
        "high_min": 4.0,
        "medium_min": 2.0,
    },
    "route": {
        "high_min": 4.0,
        "medium_min": 2.0,
    },
}


@dataclass(frozen=True)
class EvidenceScore:
    domain: str
    score: float
    confidence: str
    signals: Dict[str, int]


def _calibration_path() -> Path:
    return Path(__file__).resolve().parents[1] / "references" / "evidence-calibration.tsv"


@lru_cache(maxsize=1)
def load_calibration() -> Dict[str, Dict[str, float]]:
    merged: Dict[str, Dict[str, float]] = {
        domain: values.copy() for domain, values in DEFAULT_CALIBRATION.items()
    }
    path = _calibration_path()
    if not path.exists():
        return merged
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return merged
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [token.strip() for token in line.split("\t")]
        if len(parts) != 3:
            continue
        domain, key, value_raw = parts
        try:
            value = float(value_raw)
        except ValueError:
            continue
        bucket = merged.setdefault(domain, {})
        bucket[key] = value
    return merged


def _domain_config(domain: str) -> Dict[str, float]:
    calibration = load_calibration()
    base = calibration.get("default", {}).copy()
    base.update(calibration.get(domain, {}))
    return base


def _normalize_signals(signals: Mapping[str, int]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for key in SIGNAL_KEYS:
        raw = signals.get(key, 0)
        normalized[key] = 1 if raw and int(raw) > 0 else 0
    return normalized


def score_candidate(
    domain: str,
    *,
    ast: int = 0,
    config: int = 0,
    graph: int = 0,
    lexical: int = 0,
) -> EvidenceScore:
    signals = _normalize_signals(
        {
            "ast": ast,
            "config": config,
            "graph": graph,
            "lexical": lexical,
        }
    )
    cfg = _domain_config(domain)
    score = 0.0
    for key in SIGNAL_KEYS:
        weight = float(cfg.get(f"weight.{key}", 0.0))
        score += weight * signals[key]
    high_min = float(cfg.get("high_min", 5.0))
    medium_min = float(cfg.get("medium_min", 2.0))
    confidence = "low"
    if score >= high_min:
        confidence = "high"
    elif score >= medium_min:
        confidence = "medium"
    return EvidenceScore(domain=domain, score=score, confidence=confidence, signals=signals)


def confidence_at_least(confidence: str, minimum: str) -> bool:
    return CONFIDENCE_RANK.get(confidence, -1) >= CONFIDENCE_RANK.get(minimum, -1)
