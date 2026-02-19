from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

INTERNAL_SOURCE_PREFIXES = (
    "src/",
    "app/",
    "lib/",
    "internal/",
    "cmd/",
    "crates/",
    "packages/",
    "server/",
    "backend/",
    "frontend/",
    "client/",
    "web/",
)


def match_globs(path: str, globs: Sequence[str]) -> bool:
    if not globs:
        return False
    lower_path = path.lower()
    lower_name = Path(path).name.lower()
    for pattern in globs:
        lowered = pattern.lower()
        if any(token in lowered for token in ("*", "?", "[")):
            if fnmatch.fnmatch(lower_path, lowered) or fnmatch.fnmatch(lower_name, lowered):
                return True
            continue
        if lowered in lower_path or lowered == lower_name:
            return True
    return False


def resolve_alias_candidates(
    module: str,
    *,
    base_url: str,
    paths: Dict[str, List[str]],
) -> List[str]:
    candidates: List[str] = []
    base = Path(base_url) if base_url else Path(".")

    def qualify(value: str) -> str:
        if value.startswith(("/", "./", "../")):
            return value
        if base_url and value.startswith(f"{base_url.rstrip('/')}/"):
            return value
        return str(base / value)

    for pattern, targets in paths.items():
        if "*" in pattern:
            prefix, suffix = pattern.split("*", 1)
            if module.startswith(prefix) and module.endswith(suffix):
                token = module[len(prefix) : len(module) - len(suffix)]
                for target in targets:
                    if "*" in target:
                        candidates.append(qualify(target.replace("*", token)))
                    else:
                        candidates.append(qualify(target))
        else:
            if module == pattern:
                candidates.extend(qualify(target) for target in targets)
    if base_url:
        candidates.append(str(base / module))
    return candidates


def resolve_module_candidates(
    candidates: Sequence[str], files_set: Set[str], *, exts: Sequence[str]
) -> Optional[str]:
    for raw in candidates:
        target = os.path.normpath(raw).replace("\\", "/").lstrip("./")
        if Path(target).suffix:
            if target in files_set:
                return target
            continue
        for ext in exts:
            path = f"{target}{ext}"
            if path in files_set:
                return path
        for ext in exts:
            path = f"{target}/index{ext}"
            if path in files_set:
                return path
    return None


def resolve_ts_module(
    module: str,
    importer: str,
    files_set: Set[str],
    *,
    alias_config: Optional[Dict[str, object]] = None,
) -> Optional[str]:
    if not module.startswith("."):
        if alias_config:
            base_url = str(alias_config.get("baseUrl", "."))
            paths = alias_config.get("paths", {})
            if isinstance(paths, dict):
                candidates = resolve_alias_candidates(
                    module, base_url=base_url, paths=paths  # type: ignore[arg-type]
                )
                resolved = resolve_module_candidates(
                    candidates,
                    files_set,
                    exts=(".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"),
                )
                if resolved:
                    return resolved
        return None
    base = Path(importer).parent
    target = os.path.normpath((base / module).as_posix()).replace("\\", "/")
    if Path(target).suffix:
        return target if target in files_set else None
    candidates = [
        f"{target}.ts",
        f"{target}.tsx",
        f"{target}/index.ts",
        f"{target}/index.tsx",
    ]
    for candidate in candidates:
        if candidate in files_set:
            return candidate
    return None


def resolve_js_module(
    module: str,
    importer: str,
    files_set: Set[str],
    *,
    alias_config: Optional[Dict[str, object]] = None,
) -> Optional[str]:
    if not module.startswith("."):
        if alias_config:
            base_url = str(alias_config.get("baseUrl", "."))
            paths = alias_config.get("paths", {})
            if isinstance(paths, dict):
                candidates = resolve_alias_candidates(
                    module, base_url=base_url, paths=paths  # type: ignore[arg-type]
                )
                resolved = resolve_module_candidates(
                    candidates,
                    files_set,
                    exts=(".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"),
                )
                if resolved:
                    return resolved
        return None
    base = Path(importer).parent
    target = os.path.normpath((base / module).as_posix()).replace("\\", "/")
    if Path(target).suffix:
        return target if target in files_set else None
    candidates = [
        f"{target}.js",
        f"{target}.jsx",
        f"{target}.mjs",
        f"{target}.cjs",
        f"{target}/index.js",
        f"{target}/index.jsx",
    ]
    for candidate in candidates:
        if candidate in files_set:
            return candidate
    return None


def resolve_py_module(module: str, importer: str, files_set: Set[str]) -> Optional[str]:
    if module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        remainder = module.lstrip(".")
        base = Path(importer).parent
        for _ in range(max(dots - 1, 0)):
            base = base.parent
        module_path = remainder.replace(".", "/")
        target = os.path.normpath((base / module_path).as_posix()).replace("\\", "/")
    else:
        target = os.path.normpath(module.replace(".", "/")).replace("\\", "/")
    candidates = [f"{target}.py", f"{target}/__init__.py"]
    for candidate in candidates:
        if candidate in files_set:
            return candidate
    if not module.startswith("."):
        base = Path(importer).parent
        target = os.path.normpath((base / module.replace(".", "/")).as_posix()).replace("\\", "/")
        candidates = [f"{target}.py", f"{target}/__init__.py"]
        for candidate in candidates:
            if candidate in files_set:
                return candidate
    return None


def resolve_go_module(module: str, importer: str, files_set: Set[str]) -> Optional[str]:
    target = ""
    if module.startswith("."):
        base = Path(importer).parent
        target = os.path.normpath((base / module).as_posix()).replace("\\", "/")
        if target.endswith(".go") and target in files_set:
            return target
        if target in files_set:
            return target
        if f"{target}.go" in files_set:
            return f"{target}.go"
        package_files = sorted(
            path
            for path in files_set
            if path.startswith(f"{target}/") and path.endswith(".go") and not path.endswith("_test.go")
        )
        return package_files[0] if package_files else None

    normalized = module.strip().lstrip("/")
    if not normalized:
        return None
    path_candidates = [normalized]
    parts = [part for part in normalized.split("/") if part]
    for idx in range(1, len(parts)):
        path_candidates.append("/".join(parts[idx:]))
    for candidate in path_candidates:
        if candidate.endswith(".go") and candidate in files_set:
            return candidate
        if f"{candidate}.go" in files_set:
            return f"{candidate}.go"
        package_files = sorted(
            path
            for path in files_set
            if path.startswith(f"{candidate}/") and path.endswith(".go") and not path.endswith("_test.go")
        )
        if package_files:
            return package_files[0]
    return None


def resolve_rust_module(module: str, importer: str, files_set: Set[str]) -> Optional[str]:
    normalized = module.strip().strip(":")
    if not normalized:
        return None
    normalized = normalized.replace("::", "/")

    candidate_bases: List[str] = []
    if normalized.startswith("."):
        base = Path(importer).parent
        candidate_bases.append(
            os.path.normpath((base / normalized).as_posix()).replace("\\", "/")
        )
    else:
        candidate_bases.append(normalized)
        if "/src/" in importer:
            crate_root = importer.split("/src/", 1)[0] + "/src"
            candidate_bases.append(
                os.path.normpath((Path(crate_root) / normalized).as_posix()).replace("\\", "/")
            )

    for base in candidate_bases:
        if base.endswith(".rs") and base in files_set:
            return base
        for candidate in (f"{base}.rs", f"{base}/mod.rs"):
            if candidate in files_set:
                return candidate
    return None


def normalize_module_hint(module: str) -> str:
    normalized = (module or "").strip().strip("'\"")
    if not normalized:
        return ""
    if normalized.startswith("crate::"):
        normalized = normalized[len("crate::") :]
    elif normalized.startswith("self::"):
        normalized = "./" + normalized[len("self::") :]
    elif normalized.startswith("super::"):
        normalized = "../" + normalized[len("super::") :]
    normalized = normalized.replace("::", "/").replace("\\", "/")
    normalized = normalized.replace("//", "/")
    return normalized


def looks_like_internal_module(module: str, repo_roots: Optional[Set[str]] = None) -> bool:
    normalized = normalize_module_hint(module)
    if not normalized:
        return False
    if normalized.startswith((".", "/", "@/", "~/")):
        return True
    if normalized.startswith(INTERNAL_SOURCE_PREFIXES):
        return True
    if "/" in normalized and repo_roots:
        head = normalized.split("/", 1)[0].lstrip("./")
        if head in repo_roots:
            return True
    return False


def resolve_internal_fallback_module(
    module: str,
    importer: str,
    files_set: Set[str],
    *,
    repo_roots: Optional[Set[str]] = None,
) -> Optional[str]:
    normalized = normalize_module_hint(module)
    if not looks_like_internal_module(normalized, repo_roots=repo_roots):
        return None

    importer_dir = Path(importer).parent
    candidate_roots: List[str] = []
    if normalized.startswith("@/") or normalized.startswith("~/"):
        normalized = f"src/{normalized[2:]}"
    if normalized.startswith(("./", "../")):
        candidate_roots.append(
            os.path.normpath((importer_dir / normalized).as_posix()).replace("\\", "/")
        )
    else:
        candidate_roots.append(normalized.lstrip("/"))

    roots: List[str] = []
    seen_roots: Set[str] = set()
    for root in candidate_roots:
        cleaned = os.path.normpath(root).replace("\\", "/").lstrip("./")
        if not cleaned or cleaned in seen_roots:
            continue
        seen_roots.add(cleaned)
        roots.append(cleaned)

    for root in roots:
        if root in files_set:
            return root
        if Path(root).suffix:
            continue
        for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".go", ".rs"):
            candidate = f"{root}{ext}"
            if candidate in files_set:
                return candidate
        for candidate in (
            f"{root}/index.ts",
            f"{root}/index.tsx",
            f"{root}/index.js",
            f"{root}/index.jsx",
            f"{root}/index.mjs",
            f"{root}/index.cjs",
            f"{root}/__init__.py",
            f"{root}/mod.rs",
        ):
            if candidate in files_set:
                return candidate
        go_package_files = sorted(
            path
            for path in files_set
            if path.startswith(f"{root}/") and path.endswith(".go") and not path.endswith("_test.go")
        )
        if go_package_files:
            return go_package_files[0]
    return None
