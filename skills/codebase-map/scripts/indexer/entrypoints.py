from __future__ import annotations

from collections import defaultdict
import json
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

from evidence_model import confidence_at_least, score_candidate

from .constants import ENTRYPOINT_NAME_ROLES
from .entities import is_noise_path

NON_RUNTIME_PATH_TOKENS = (
    "/example/",
    "/examples/",
    "/sample/",
    "/samples/",
    "/fixture/",
    "/fixtures/",
    "/mock/",
    "/mocks/",
    "/bench/",
    "/benches/",
)

ENTRYPOINT_CONTENT_EXTENSIONS = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
)

ENTRYPOINT_SECOND_PASS_PATH_HINTS = (
    "scripts/",
    "script/",
    "cmd/",
    "bin/",
    "tools/",
    "worker/",
    "workers/",
    "jobs/",
    "job/",
    "cron/",
    "dag/",
    "dags/",
)

ENTRYPOINT_SECOND_PASS_NAME_HINTS = (
    "main",
    "cli",
    "server",
    "worker",
    "queue",
    "job",
    "dag",
    "monitor",
    "daemon",
    "scheduler",
    "runner",
    "ingest",
)

ENTRYPOINT_SECOND_PASS_LIMIT = 240


def summarize_file(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    for line in lines[:40]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            return stripped.lstrip("/").strip()[:160]
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:160]
        if stripped.startswith("/*"):
            stripped = stripped.lstrip("/*").strip()
            return stripped[:160]
        if stripped.startswith('"""') or stripped.startswith("'''"):
            stripped = stripped.strip("\"'")
            return stripped[:160]
        break
    return ""


def is_non_runtime_path(path: str) -> bool:
    lower = f"/{path.lower().lstrip('/')}/"
    return any(token in lower for token in NON_RUNTIME_PATH_TOKENS)


def entrypoint_role_for_path(path: str) -> Optional[str]:
    path_lower = path.lower()
    name = Path(path_lower).name
    if name in ENTRYPOINT_NAME_ROLES:
        return ENTRYPOINT_NAME_ROLES[name]
    if "/bin/" in path_lower and name.endswith((".ts", ".tsx", ".py", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs")):
        return "cli"
    if name in {"main.go", "main.rs"}:
        if "/cmd/" in path_lower or path_lower.startswith("cmd/") or path_lower in {"main.go", "main.rs", "src/main.go", "src/main.rs"}:
            return "cli"
        return None
    if path_lower.endswith(
        (
            "/src/index.tsx",
            "/src/index.ts",
            "/src/main.tsx",
            "/src/main.ts",
            "/src/index.jsx",
            "/src/index.js",
        )
    ):
        return "ui"
    if path_lower.endswith(("/src/app.tsx", "/src/app.ts", "/src/app.jsx", "/src/app.js")):
        return "ui"
    if path_lower.endswith(("/src/app.go", "/src/app.rs", "/src/server.go", "/src/server.rs")):
        return "http"
    return None


def entrypoint_role_for_content(path: str, content: str) -> Optional[str]:
    lower = content.lower()
    if any(token in lower for token in ("fastapi(", "flask(", "@app.route", "uvicorn.run", "gunicorn")):
        return "http"
    if any(token in lower for token in ("django", "asgi", "wsgi")):
        return "http"
    if any(token in lower for token in ("argparse", "click", "typer", "__main__")):
        return "cli"
    if any(
        token in lower
        for token in (
            "celery",
            "dramatiq",
            "rq",
            "apscheduler",
            "airflow",
            "dag(",
            " dag =",
            "schedule_interval",
        )
    ):
        return "worker"
    if path.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")):
        if any(token in lower for token in ("express(", "fastify(", "createserver(", "koa(")):
            return "http"
        if any(token in lower for token in ("commander", "yargs")):
            return "cli"
    if path.endswith(".rs"):
        if any(token in lower for token in ("axum::router", "actix_web", "rocket::build", "warp::", "router::new")):
            return "http"
        if "fn main(" in lower:
            return "cli"
    if path.endswith(".go"):
        if any(token in lower for token in ("func main()", "http.listenandserve", "gin.default()", "fiber.new(", "echo.new()", "chi.newrouter()")):
            if any(token in lower for token in ("listenandserve", "gin.default", "fiber.new", "echo.new", "newrouter")):
                return "http"
            return "cli"
    return None


def resolve_entrypoint_candidate(candidate: str, files_set: Set[str]) -> Optional[str]:
    if not candidate:
        return None
    candidate = candidate.strip().strip('"').strip("'")
    candidate = candidate.lstrip("./")
    if candidate in files_set:
        return candidate
    if candidate.endswith(".js"):
        ts_candidate = candidate[:-3] + ".ts"
        tsx_candidate = candidate[:-3] + ".tsx"
        if ts_candidate in files_set:
            return ts_candidate
        if tsx_candidate in files_set:
            return tsx_candidate
    for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py"):
        path = candidate + ext
        if path in files_set:
            return path
    for ext in (".go", ".rs"):
        path = candidate + ext
        if path in files_set:
            return path
    for ext in (
        "index.ts",
        "index.tsx",
        "index.js",
        "index.jsx",
        "index.mjs",
        "index.cjs",
        "index.py",
        "main.go",
        "main.rs",
    ):
        path = f"{candidate}/{ext}"
        if path in files_set:
            return path
    return None


def entrypoints_from_package_json(repo: Path, files_set: Set[str]) -> Dict[str, str]:
    entrypoints: Dict[str, str] = {}
    package_json = repo / "package.json"
    if not package_json.exists():
        return entrypoints
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return entrypoints

    main = data.get("main")
    if isinstance(main, str):
        resolved = resolve_entrypoint_candidate(main, files_set)
        if resolved:
            entrypoints[resolved] = "library"

    bin_val = data.get("bin")
    if isinstance(bin_val, str):
        resolved = resolve_entrypoint_candidate(bin_val, files_set)
        if resolved:
            entrypoints[resolved] = "cli"
    elif isinstance(bin_val, dict):
        for value in bin_val.values():
            if not isinstance(value, str):
                continue
            resolved = resolve_entrypoint_candidate(value, files_set)
            if resolved:
                entrypoints[resolved] = "cli"

    scripts = data.get("scripts", {})
    if isinstance(scripts, dict):
        for name, cmd in scripts.items():
            if not isinstance(name, str) or not isinstance(cmd, str):
                continue
            lowered = name.lower()
            if not any(token in lowered for token in ("start", "dev", "serve", "preview", "worker", "queue", "job", "cli")):
                continue
            match = re.search(r"([\\w./-]+\\.(?:ts|tsx|js|jsx|mjs|cjs|py|go|rs))", cmd)
            if not match:
                continue
            resolved = resolve_entrypoint_candidate(match.group(1), files_set)
            if not resolved:
                continue
            role = "http"
            if any(token in lowered for token in ("worker", "queue", "job")):
                role = "worker"
            elif "cli" in lowered:
                role = "cli"
            entrypoints[resolved] = role

    return entrypoints


def entrypoints_from_pyproject(repo: Path, files_set: Set[str]) -> Dict[str, str]:
    entrypoints: Dict[str, str] = {}
    pyproject = repo / "pyproject.toml"
    if not pyproject.exists():
        return entrypoints
    try:
        lines = pyproject.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return entrypoints
    in_scripts = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[]").strip()
            in_scripts = section in {"project.scripts", "tool.poetry.scripts"}
            continue
        if not in_scripts or "=" not in stripped or ":" not in stripped:
            continue
        _, value = stripped.split("=", 1)
        value = value.strip().strip('"').strip("'")
        module = value.split(":", 1)[0].strip()
        if not module:
            continue
        candidate = module.replace(".", "/") + ".py"
        resolved = resolve_entrypoint_candidate(candidate, files_set)
        if resolved:
            entrypoints[resolved] = "cli"
    return entrypoints


def entrypoints_from_dockerfile(repo: Path, files_set: Set[str]) -> Dict[str, str]:
    entrypoints: Dict[str, str] = {}
    for name in ("Dockerfile", "dockerfile"):
        dockerfile = repo / name
        if not dockerfile.exists():
            continue
        try:
            content = dockerfile.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or not (stripped.startswith("ENTRYPOINT") or stripped.startswith("CMD")):
                continue
            match = re.search(r"([\\w./-]+\\.(?:py|js|ts|tsx|jsx|mjs|cjs|go|rs))", stripped)
            if not match:
                continue
            resolved = resolve_entrypoint_candidate(match.group(1), files_set)
            if resolved:
                entrypoints[resolved] = "http"
    return entrypoints


def detect_entrypoints(repo: Path, files: Sequence[str], tests: Set[str]) -> Dict[str, str]:
    files_set = set(files)
    role_priority = {
        "http": 5,
        "api": 5,
        "worker": 4,
        "job": 4,
        "cli": 3,
        "ui": 3,
        "library": 1,
        "entrypoint": 1,
    }
    signal_priority = {
        "config": 4,
        "ast": 3,
        "graph": 2,
        "lexical": 1,
    }
    entrypoints: Dict[str, str] = {}
    role_signal_level: Dict[str, int] = {}
    evidence_signals = defaultdict(
        lambda: {"ast": 0, "config": 0, "graph": 0, "lexical": 0}
    )

    def add_entrypoint(path: str, role: str, signal: str) -> None:
        if signal in evidence_signals[path]:
            evidence_signals[path][signal] = 1
        current = entrypoints.get(path)
        incoming_level = signal_priority.get(signal, 0)
        if current is None:
            entrypoints[path] = role
            role_signal_level[path] = incoming_level
            return
        current_level = role_signal_level.get(path, 0)
        if incoming_level > current_level:
            entrypoints[path] = role
            role_signal_level[path] = incoming_level
            return
        if incoming_level < current_level:
            return
        current_pri = role_priority.get(current, 1)
        next_pri = role_priority.get(role, 1)
        if current == "library" or next_pri > current_pri:
            entrypoints[path] = role
            role_signal_level[path] = incoming_level

    def path_has_second_pass_hint(path: str) -> bool:
        lower = path.lower().lstrip("./")
        return any(
            lower.startswith(hint) or f"/{hint}" in lower
            for hint in ENTRYPOINT_SECOND_PASS_PATH_HINTS
        )

    for path in files:
        if path in tests or is_noise_path(path) or is_non_runtime_path(path):
            continue
        role = entrypoint_role_for_path(path)
        if role:
            add_entrypoint(path, role, "lexical")

    def merge_entrypoints(new_entries: Dict[str, str]) -> None:
        for path, role in new_entries.items():
            if path in tests or is_noise_path(path) or is_non_runtime_path(path):
                continue
            add_entrypoint(path, role, "config")

    merge_entrypoints(entrypoints_from_package_json(repo, files_set))
    merge_entrypoints(entrypoints_from_pyproject(repo, files_set))
    merge_entrypoints(entrypoints_from_dockerfile(repo, files_set))

    def is_supported_content_file(path: str) -> bool:
        return path.endswith(ENTRYPOINT_CONTENT_EXTENSIONS)

    def read_entrypoint_content(path: str) -> str:
        try:
            return (repo / path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""

    def refine_entrypoint_role(path: str, role: Optional[str]) -> Optional[str]:
        if not is_supported_content_file(path):
            return None
        content = read_entrypoint_content(path)
        if not content:
            return None
        refined = entrypoint_role_for_content(path, content[:20000])
        if not refined:
            return None
        if role is None or role in {"library", "entrypoint"} or refined != role:
            return refined
        return role

    for path, role in list(entrypoints.items()):
        if path in tests or is_noise_path(path) or is_non_runtime_path(path):
            entrypoints.pop(path, None)
            continue
        if not is_supported_content_file(path):
            continue
        refined = refine_entrypoint_role(path, role)
        if refined:
            add_entrypoint(path, refined, "ast")

    def second_pass_candidate(path: str) -> bool:
        if (
            path in entrypoints
            or path in tests
            or is_noise_path(path)
            or is_non_runtime_path(path)
            or not is_supported_content_file(path)
        ):
            return False
        if path_has_second_pass_hint(path):
            return True
        lower = path.lower().lstrip("./")
        name = Path(lower).name
        return any(token in name for token in ENTRYPOINT_SECOND_PASS_NAME_HINTS)

    def second_pass_rank(path: str) -> Tuple[int, int, str]:
        lower = path.lower().lstrip("./")
        path_hint = path_has_second_pass_hint(path)
        name = Path(lower).name
        name_hits = sum(1 for token in ENTRYPOINT_SECOND_PASS_NAME_HINTS if token in name)
        return (0 if path_hint else 1, -name_hits, lower)

    second_pass_paths = sorted(
        [path for path in files if second_pass_candidate(path)],
        key=second_pass_rank,
    )[:ENTRYPOINT_SECOND_PASS_LIMIT]
    for path in second_pass_paths:
        refined = refine_entrypoint_role(path, None)
        if refined:
            lexical = 1 if path_has_second_pass_hint(path) else 0
            confidence = score_candidate(
                "entrypoint",
                ast=1,
                lexical=lexical,
            )
            if confidence_at_least(confidence.confidence, "medium"):
                add_entrypoint(path, refined, "ast")
                if lexical:
                    add_entrypoint(path, refined, "lexical")
            continue
        lexical = 1 if path_has_second_pass_hint(path) else 0
        lower = path.lower().lstrip("./")
        graph = 1 if "/scripts/" in f"/{lower}/" or "/bin/" in f"/{lower}/" else 0
        confidence = score_candidate("entrypoint", lexical=lexical, graph=graph)
        if not confidence_at_least(confidence.confidence, "medium"):
            continue
        fallback_role = entrypoint_role_for_path(path) or "cli"
        add_entrypoint(path, fallback_role, "lexical")
        if graph:
            add_entrypoint(path, fallback_role, "graph")
    return entrypoints
