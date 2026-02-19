from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from utils import ToolState, tool_version

from .constants import (
    CONFIG_FILES,
    MONOREPO_MARKERS,
    PACKAGE_MANIFESTS,
    REPOMAP_CONFIG_FILES,
)
from .discovery import code_files_for_languages, code_files_size_mb

if TYPE_CHECKING:
    from . import IndexOptions


def load_repo_config(repo: Path, warnings: List[str]) -> Tuple[Dict[str, object], Optional[str]]:
    for filename in REPOMAP_CONFIG_FILES:
        path = repo / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Failed to parse {filename}: {exc}")
            return {}, filename
        if not isinstance(payload, dict):
            warnings.append(f"Invalid {filename}: expected a JSON object")
            return {}, filename
        routes = payload.get("routes") if isinstance(payload.get("routes"), dict) else {}
        entrypoints = payload.get("entrypoints") if isinstance(payload.get("entrypoints"), dict) else {}
        entry_details = entrypoints.get("details") if isinstance(entrypoints.get("details"), dict) else {}
        routes_mode = payload.get("routes_mode")
        if not isinstance(routes_mode, str):
            routes_mode = None
        nested_mode = routes.get("mode") if isinstance(routes.get("mode"), str) else None
        routes_mode = (nested_mode or routes_mode or "").strip() or None

        def as_int(value: Any) -> Optional[int]:
            return value if isinstance(value, int) else None

        config = {
            "docs_globs": normalize_globs(payload.get("docs_globs")),
            "docs_exclude_globs": normalize_globs(payload.get("docs_exclude_globs")),
            "tests_globs": normalize_globs(payload.get("tests_globs")),
            "tests_exclude_globs": normalize_globs(payload.get("tests_exclude_globs")),
            "configs_globs": normalize_globs(payload.get("configs_globs")),
            "configs_exclude_globs": normalize_globs(payload.get("configs_exclude_globs")),
            "routes_globs": normalize_globs(payload.get("routes_globs"))
            + normalize_globs(routes.get("globs"))
            + normalize_globs(routes.get("paths")),
            "routes_names": normalize_globs(payload.get("routes_names"))
            + normalize_globs(routes.get("names")),
            "routes_exclude_globs": normalize_globs(payload.get("routes_exclude_globs"))
            + normalize_globs(routes.get("exclude_globs")),
            "routes_mode": routes_mode or "auto",
            "entrypoints": {
                "frameworks": normalize_str_list(entrypoints.get("frameworks")),
                "max_items": as_int(entrypoints.get("max_items")),
                "max_per_group": as_int(entrypoints.get("max_per_group")),
                "include_middleware": entrypoints.get("include_middleware") if "include_middleware" in entrypoints else None,
                "details_limit": as_int(entrypoints.get("details_limit") or entry_details.get("limit")),
                "details_per_kind": as_int(entrypoints.get("details_per_kind") or entry_details.get("per_kind")),
            },
        }
        return config, filename
    return {}, None


def normalize_globs(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def normalize_str_list(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def detect_framework(repo: Path, config: Dict[str, object]) -> Optional[str]:
    """Detect framework from package.json (Node.js) or pyproject.toml (Python)."""
    package_json = repo / "package.json"
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
            deps = {
                **data.get("dependencies", {}),
                **data.get("devDependencies", {}),
                **data.get("peerDependencies", {}),
            }
            if "next" in deps:
                return "nextjs"
            if "express" in deps:
                return "express"
            if "fastify" in deps:
                return "fastify"
            if "hapi" in deps:
                return "hapi"
        except (OSError, json.JSONDecodeError):
            pass

    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib

            with pyproject.open("rb") as handle:
                data = tomllib.load(handle)

            def _normalize_dep_names(value: Any) -> Set[str]:
                names: Set[str] = set()

                def _push(raw: str) -> None:
                    token = raw.strip()
                    if not token:
                        return
                    # Handles common PEP 508 forms like:
                    # "fastapi>=0.110; python_version>='3.10'"
                    token = token.split(";", 1)[0]
                    token = token.split("[", 1)[0]
                    for delim in ("==", ">=", "<=", "~=", "!=", ">", "<", "="):
                        if delim in token:
                            token = token.split(delim, 1)[0]
                            break
                    token = token.strip().lower()
                    if token:
                        names.add(token)

                if isinstance(value, dict):
                    for key in value.keys():
                        if isinstance(key, str):
                            _push(key)
                    return names
                if isinstance(value, str):
                    _push(value)
                    return names
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            _push(item)
                        elif isinstance(item, dict):
                            for key in item.keys():
                                if isinstance(key, str):
                                    _push(key)
                return names

            deps: Set[str] = set()
            project = data.get("project", {})
            if isinstance(project, dict):
                deps.update(_normalize_dep_names(project.get("dependencies", [])))
            tool = data.get("tool", {})
            if isinstance(tool, dict):
                poetry = tool.get("poetry", {})
                if isinstance(poetry, dict):
                    deps.update(_normalize_dep_names(poetry.get("dependencies", {})))

            if any(dep.startswith("fastapi") for dep in deps):
                return "fastapi"
            if any(dep.startswith("django") for dep in deps):
                return "django"
            if any(dep.startswith("flask") for dep in deps):
                return "flask"
            if "aiohttp" in deps:
                return "aiohttp"
        except (OSError, ImportError, tomllib.TOMLDecodeError):
            try:
                content = pyproject.read_text(encoding="utf-8")
                if "fastapi" in content.lower():
                    return "fastapi"
                if "django" in content.lower():
                    return "django"
                if "flask" in content.lower():
                    return "flask"
            except OSError:
                pass

    cargo_toml = repo / "Cargo.toml"
    if cargo_toml.exists():
        try:
            content = cargo_toml.read_text(encoding="utf-8", errors="ignore").lower()
            if any(token in content for token in ("axum", "actix-web", "rocket", "warp", "poem")):
                return "rust"
        except OSError:
            pass

    go_mod = repo / "go.mod"
    if go_mod.exists():
        try:
            content = go_mod.read_text(encoding="utf-8", errors="ignore").lower()
            if any(
                token in content
                for token in (
                    "github.com/gin-gonic/gin",
                    "github.com/gofiber/fiber",
                    "github.com/labstack/echo",
                    "github.com/go-chi/chi",
                    "gorilla/mux",
                )
            ):
                return "go"
        except OSError:
            pass

    return None


def strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments from JSON-like text."""
    out: List[str] = []
    in_str = False
    escape = False
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            idx += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            idx += 1
            continue
        if ch == "/" and idx + 1 < len(text):
            nxt = text[idx + 1]
            if nxt == "/":
                idx = text.find("\n", idx + 2)
                if idx == -1:
                    break
                continue
            if nxt == "*":
                end = text.find("*/", idx + 2)
                if end == -1:
                    break
                idx = end + 2
                continue
        out.append(ch)
        idx += 1
    return "".join(out)


def load_tsconfig_paths(
    repo: Path,
    warnings: List[str],
    *,
    files: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Load baseUrl/paths from tsconfig or jsconfig for alias resolution."""

    def parse_tsconfig(path: Path) -> Tuple[Optional[str], Dict[str, List[str]]]:
        try:
            raw = path.read_text(encoding="utf-8")
            payload = json.loads(strip_json_comments(raw))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Failed to parse {path.name}: {exc}")
            return None, {}
        if not isinstance(payload, dict):
            warnings.append(f"Invalid {path.name}: expected a JSON object")
            return None, {}
        compiler = payload.get("compilerOptions", {})
        if not isinstance(compiler, dict):
            compiler = {}
        base_url = compiler.get("baseUrl", ".")
        if not isinstance(base_url, str) or not base_url.strip():
            base_url = "."
        paths = compiler.get("paths", {})
        normalized: Dict[str, List[str]] = {}
        if isinstance(paths, dict):
            for key, value in paths.items():
                if not isinstance(key, str):
                    continue
                if isinstance(value, str):
                    normalized[key] = [value]
                elif isinstance(value, list):
                    normalized[key] = [item for item in value if isinstance(item, str)]
        return base_url, normalized

    for name in ("tsconfig.json", "jsconfig.json"):
        path = repo / name
        if not path.exists():
            continue
        base_url, normalized = parse_tsconfig(path)
        if normalized:
            return {
                "baseUrl": base_url or ".",
                "paths": normalized,
                "source": name,
            }

    if files:
        repo_root = repo.resolve()
        repo_root_str = repo_root.as_posix()
        merged: Dict[str, List[str]] = {}
        config_paths = [
            Path(repo_root, path)
            for path in files
            if path.endswith(("tsconfig.json", "jsconfig.json"))
        ]
        for config_path in config_paths:
            base_url, normalized = parse_tsconfig(config_path)
            if not normalized:
                continue
            base_url = base_url or "."
            config_dir = config_path.parent
            for key, targets in normalized.items():
                for target in targets:
                    full_path = (config_dir / base_url / target).as_posix()
                    if full_path.startswith(repo_root_str + "/"):
                        full_path = full_path[len(repo_root_str) + 1 :]
                    merged.setdefault(key, []).append(full_path)
        if merged:
            return {
                "baseUrl": ".",
                "paths": {k: sorted(set(v)) for k, v in merged.items()},
                "source": "tsconfig.json (multi)",
            }

    return {}


def tsconfig_fingerprint(config: Dict[str, object]) -> str:
    if not config:
        return ""
    payload = json.dumps(config, ensure_ascii=True, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def collect_package_roots(files: Iterable[str]) -> List[Dict[str, str]]:
    packages: Dict[str, Dict[str, str]] = {}
    for path in files:
        name = Path(path).name
        manifest_type = PACKAGE_MANIFESTS.get(name)
        if not manifest_type:
            continue
        root = Path(path).parent.as_posix() or "."
        if root not in packages:
            packages[root] = {"path": root, "manifest": name, "type": manifest_type}
    return sorted(packages.values(), key=lambda item: (item["path"] != ".", item["path"]))


def detect_monorepo(repo: Path, files: Iterable[str]) -> Dict[str, object]:
    file_set = set(files)
    markers = [marker for marker in MONOREPO_MARKERS if marker in file_set]
    packages = collect_package_roots(files)
    subpackages = [pkg for pkg in packages if pkg["path"] not in {".", ""}]
    if markers and subpackages:
        return {
            "type": MONOREPO_MARKERS.get(markers[0], "workspace"),
            "markers": markers,
            "packages": packages,
        }
    if len(subpackages) >= 2:
        return {
            "type": "multi-package",
            "markers": [],
            "packages": packages,
        }
    return {}


def config_fingerprint(config: Dict[str, object]) -> str:
    if not config:
        return ""
    payload = json.dumps(config, ensure_ascii=True, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def index_signature(
    options: "IndexOptions",
    config: Dict[str, object],
    tool_versions: Dict[str, str],
    *,
    tsconfig_fp: str = "",
) -> Dict[str, Any]:
    languages = sorted(options.languages) if options.languages else ["auto"]
    return {
        "version": 1,
        "options": {
            "languages": languages,
            "ast": options.ast,
            "codeql": options.codeql,
            "ast_mode": options.ast_mode,
            "codeql_mode": options.codeql_mode,
            "ast_auto_max_files": options.ast_auto_max_files,
            "codeql_auto_max_files": options.codeql_auto_max_files,
            "codeql_auto_max_mb": options.codeql_auto_max_mb,
            "include_internal": options.include_internal,
        },
        "config_fingerprint": config_fingerprint(config),
        "tsconfig_fingerprint": tsconfig_fp,
        "tool_versions": tool_versions,
    }


def probe_tool_versions(
    repo: Path,
    options: "IndexOptions",
    *,
    warnings: List[str],
    tools: ToolState,
) -> Dict[str, str]:
    tool_versions: Dict[str, str] = {}
    version = tool_version(["fd", "--version"], cwd=repo, warnings=warnings, tools=tools)
    if version:
        tool_versions["fd"] = version
    version = tool_version(["rg", "--version"], cwd=repo, warnings=warnings, tools=tools)
    if version:
        tool_versions["rg"] = version
    if options.ast:
        version = tool_version(["ast-grep", "--version"], cwd=repo, warnings=warnings, tools=tools)
        if version:
            tool_versions["ast-grep"] = version
    if options.codeql:
        version = tool_version(["codeql", "version"], cwd=repo, warnings=warnings, tools=tools)
        if version:
            tool_versions["codeql"] = version
    return tool_versions


def resolve_auto_flags(
    repo: Path,
    options: "IndexOptions",
    files: List[str],
    languages: Set[str],
    *,
    warnings: List[str],
    tools: ToolState,
) -> Dict[str, object]:
    code_files = code_files_for_languages(files, languages)
    code_file_count = len(code_files)
    code_mb = 0.0
    if options.codeql_mode == "auto":
        code_mb = code_files_size_mb(repo, code_files)

    ast_enabled = options.ast
    if options.ast_mode == "auto":
        if code_file_count > options.ast_auto_max_files:
            ast_enabled = False
            warnings.append(
                f"AST auto-disabled: code files={code_file_count} > limit={options.ast_auto_max_files}"
            )
        else:
            version = tool_version(
                ["ast-grep", "--version"], cwd=repo, warnings=warnings, tools=tools
            )
            if not version:
                ast_enabled = False
                warnings.append("AST auto-disabled: ast-grep not available")
            else:
                ast_enabled = True

    codeql_enabled = options.codeql
    if options.codeql_mode == "auto":
        if code_file_count > options.codeql_auto_max_files:
            codeql_enabled = False
            warnings.append(
                f"CodeQL auto-disabled: code files={code_file_count} > limit={options.codeql_auto_max_files}"
            )
        elif options.codeql_auto_max_mb > 0 and code_mb > options.codeql_auto_max_mb:
            codeql_enabled = False
            warnings.append(
                f"CodeQL auto-disabled: code size={code_mb:.1f}MB > limit={options.codeql_auto_max_mb}MB"
            )
        else:
            version = tool_version(
                ["codeql", "version"], cwd=repo, warnings=warnings, tools=tools
            )
            if not version:
                codeql_enabled = False
                warnings.append("CodeQL auto-disabled: codeql not available")
            else:
                codeql_enabled = True

    options.ast = bool(ast_enabled)
    options.codeql = bool(codeql_enabled)

    return {
        "code_files": code_files,
        "code_file_count": code_file_count,
        "code_mb": code_mb,
        "ast_enabled": options.ast,
        "codeql_enabled": options.codeql,
    }


def load_package_jsons(repo: Path, files: Iterable[str], warnings: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for path in files:
        if Path(path).name != "package.json":
            continue
        full_path = repo / path
        try:
            data = json.loads(full_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            warnings.append(f"Failed to read {path}")
            continue
        scripts = data.get("scripts") if isinstance(data, dict) else None
        script_keys = []
        if isinstance(scripts, dict):
            script_keys = sorted(str(k) for k in scripts.keys())
        rows.append(
            [
                path,
                str(data.get("name", "-")) if isinstance(data, dict) else "-",
                ",".join(script_keys),
                str(data.get("main", "-")) if isinstance(data, dict) else "-",
                str(data.get("type", "-")) if isinstance(data, dict) else "-",
            ]
        )
    return rows
