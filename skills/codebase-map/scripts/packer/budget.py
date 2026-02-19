from __future__ import annotations

import re
from typing import Dict

_TOKENIZER = None
_TOKENIZER_READY = False
_USE_PRECISE_TOKENS = False
_TOKEN_SPLIT_RE = re.compile(r"[A-Za-z0-9_]+|[^\s]")

SECTION_BUDGETS: Dict[str, float] = {
    "REPO": 0.03,
    "SUMMARY": 0.08,
    "ARCHITECTURE": 0.05,
    "LAYERS": 0.05,
    "PURPOSE": 0.06,
    "FILES": 0.35,
    "EDGES": 0.15,
    "ROUTES": 0.12,
    "FLOWS": 0.12,
    "ENTITIES": 0.08,
    "DOCS": 0.05,
    "DOCS_QUALITY": 0.04,
    "DIAGRAM": 0.07,
    "DEAD_CODE": 0.04,
    "STATIC_TRACES": 0.06,
    "TRACE_GRAPH": 0.06,
    "ENTRY_DETAILS": 0.08,
    "INVARIANTS": 0.04,
    "INVARIANTS_HC": 0.04,
    "ARCH_STYLE": 0.04,
    "PATTERNS": 0.05,
    "CALL_CHAINS": 0.05,
    "CAPABILITIES": 0.04,
    "CAPABILITY_GAPS": 0.03,
    "TEST_CONFIDENCE": 0.04,
    "CHANGE_RISK": 0.04,
    "TEST_MAPPING": 0.05,
    "API_CONTRACTS": 0.06,
    "TYPE_HIERARCHY": 0.05,
    "CYCLES": 0.04,
    "TRACEABILITY": 0.04,
    "TOP_HUBS": 0.04,
    "COMPONENTS": 0.06,
    "COMPONENT_EDGES": 0.05,
    "ENTRYPOINTS": 0.06,
    "ENTRYPOINT_GROUPS": 0.04,
    "MIDDLEWARE": 0.04,
    "PUBLIC_API": 0.04,
    "DIR": 0.04,
    "CODEQL": 0.05,
    "QUALITY": 0.05,
    "CONFIDENCE_GATE": 0.03,
    "PACKAGES": 0.04,
    "WARNINGS": 0.02,
    "LIMITS": 0.02,
    "LEGEND": 0.02,
}


def configure_tokenizer(precise: bool) -> None:
    global _TOKENIZER, _TOKENIZER_READY, _USE_PRECISE_TOKENS
    _USE_PRECISE_TOKENS = bool(precise)
    if not _USE_PRECISE_TOKENS:
        return
    if _TOKENIZER_READY:
        return
    try:
        import tiktoken  # type: ignore

        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        _TOKENIZER_READY = True
    except Exception:
        _TOKENIZER = None
        _TOKENIZER_READY = False
        _USE_PRECISE_TOKENS = False


def estimate_tokens(text: str) -> int:
    if _USE_PRECISE_TOKENS and _TOKENIZER is not None:
        try:
            return max(1, len(_TOKENIZER.encode(text)))
        except Exception:
            pass
    tokens = _TOKEN_SPLIT_RE.findall(text)
    return max(1, len(tokens))
