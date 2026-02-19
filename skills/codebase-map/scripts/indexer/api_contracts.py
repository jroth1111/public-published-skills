from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


def extract_api_contracts(
    repo: Path,
    files: Sequence[str],
    *,
    max_entries: int = 80,
    file_hashes: Optional[Dict[str, str]] = None,
    cache: Optional[Dict[str, object]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    api_entries: List[Dict[str, object]] = []
    cache_files = {}
    cache_version = 2
    if isinstance(cache, dict):
        if cache.get("version") == cache_version:
            cache_files = cache.get("files", {}) if isinstance(cache.get("files"), dict) else {}
    new_cache: Dict[str, object] = {"version": cache_version, "files": {}}
    candidates = []
    file_set = set(files)
    for path in files:
        lower = path.lower()
        name = Path(path).name.lower()
        if not path.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs", ".proto")):
            continue
        if path.endswith(".proto"):
            candidates.append(path)
        elif any(token in lower for token in ("/api/", "/routes/", "/routers/", "/controller", "/controllers/", "/handlers/")):
            candidates.append(path)
        elif "route" in name or "router" in name:
            candidates.append(path)
        elif name in {
            "app.ts",
            "app.tsx",
            "app.js",
            "app.jsx",
            "index.ts",
            "index.tsx",
            "index.js",
            "index.jsx",
        }:
            candidates.append(path)
    seen: Set[Tuple[str, str, str]] = set()
    for path in candidates:
        cached = cache_files.get(path) if isinstance(cache_files, dict) else None
        current_hash = file_hashes.get(path) if file_hashes else None
        if (
            isinstance(cached, dict)
            and current_hash
            and cached.get("hash") == current_hash
            and isinstance(cached.get("api_contracts"), list)
        ):
            cached_entries = cached.get("api_contracts", [])
            new_cache["files"][path] = cached
            for entry in cached_entries:
                if isinstance(entry, dict):
                    api_entries.append(entry)
                    if len(api_entries) >= max_entries:
                        return api_entries, new_cache
            continue

        file_path = repo / path
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(content) > 200_000:
            continue
        lines = content.splitlines()
        ext = Path(path).suffix.lower()
        file_entries: List[Dict[str, object]] = []
        if ext == ".proto":
            current_service = ""
            for idx, line in enumerate(lines, start=1):
                service_match = re.match(r"\s*service\s+(\w+)", line)
                if service_match:
                    current_service = service_match.group(1)
                    continue
                rpc_match = re.match(r"\s*rpc\s+(\w+)\s*\(", line)
                if rpc_match:
                    rpc_name = rpc_match.group(1)
                    route = f"{current_service}.{rpc_name}" if current_service else rpc_name
                    entry = {
                        "method": "grpc",
                        "route": route,
                        "path": path,
                        "line": idx,
                    }
                    file_entries.append(entry)
                    api_entries.append(entry)
                    if len(api_entries) >= max_entries:
                        new_cache["files"][path] = {"hash": current_hash or "", "api_contracts": file_entries}
                        return api_entries, new_cache
        elif ext == ".py":
            for idx, line in enumerate(lines, start=1):
                match = re.search(
                    r"@(?:app|router|api_router|bp|blueprint)\.([a-zA-Z_]+)\(\s*[\"']([^\"']+)[\"']",
                    line,
                )
                if not match:
                    continue
                method = match.group(1).lower()
                if method == "websocket":
                    method = "ws"
                route_path = match.group(2)
                response = ""
                auth = ""
                status = ""
                if "response_model" in line:
                    resp_match = re.search(r"response_model\s*=\s*([A-Za-z0-9_.]+)", line)
                    if resp_match:
                        response = resp_match.group(1)
                if "status_code" in line:
                    status_match = re.search(r"status_code\s*=\s*([0-9]{3})", line)
                    if status_match:
                        status = status_match.group(1)
                if "Depends(" in line or "require" in line.lower():
                    auth = "likely"

                handler = ""
                params: List[str] = []
                response_hint = ""
                for look_ahead in range(idx, min(idx + 5, len(lines))):
                    def_match = re.match(r"\s*def\s+(\w+)\(([^)]*)\)\s*(?:->\s*([^\s:]+))?:", lines[look_ahead])
                    if def_match:
                        handler = def_match.group(1)
                        param_blob = def_match.group(2)
                        response_hint = def_match.group(3) or ""
                        for part in param_blob.split(","):
                            part = part.strip()
                            if not part:
                                continue
                            name_type = part.split(":", 1)
                            if len(name_type) == 2:
                                param_name = name_type[0].strip()
                                param_type = name_type[1].split("=", 1)[0].strip()
                                params.append(f"{param_name}:{param_type}")
                            else:
                                params.append(part.split("=", 1)[0].strip())
                        break
                if response_hint and not response:
                    response = response_hint
                if any("Depends(" in p for p in params):
                    auth = "likely"

                key = (path, method, route_path)
                if key in seen:
                    continue
                seen.add(key)
                entry = {
                    "method": method,
                    "route": route_path,
                    "path": path,
                    "line": idx,
                }
                if handler:
                    entry["handler"] = handler
                if params:
                    entry["params"] = params
                if response:
                    entry["response"] = response
                if auth:
                    entry["auth"] = auth
                if status:
                    entry["status"] = status
                file_entries.append(entry)
                api_entries.append(entry)
                if len(api_entries) >= max_entries:
                    new_cache["files"][path] = {"hash": current_hash or "", "api_contracts": file_entries}
                    return api_entries, new_cache
        else:
            if ext == ".go":
                for idx, line in enumerate(lines, start=1):
                    match = re.search(
                        r"\b(?:router|r|mux|engine|e)\.(GET|POST|PUT|DELETE|PATCH)\(\s*[\"']([^\"']+)[\"']",
                        line,
                    )
                    if not match:
                        match = re.search(
                            r"\b(?:http|router|r|mux)\.(HandleFunc|Handle|Path|PathPrefix)\(\s*[\"']([^\"']+)[\"']",
                            line,
                        )
                        if not match:
                            continue
                        method = "handle"
                        route_path = match.group(2)
                    else:
                        method = match.group(1).lower()
                        route_path = match.group(2)
                    key = (path, method, route_path)
                    if key in seen:
                        continue
                    seen.add(key)
                    entry = {
                        "method": method,
                        "route": route_path,
                        "path": path,
                        "line": idx,
                    }
                    file_entries.append(entry)
                    api_entries.append(entry)
                    if len(api_entries) >= max_entries:
                        new_cache["files"][path] = {"hash": current_hash or "", "api_contracts": file_entries}
                        return api_entries, new_cache
            elif ext == ".rs":
                for idx, line in enumerate(lines, start=1):
                    match = re.search(
                        r"#\[(get|post|put|patch|delete)\s*\(\s*[\"']([^\"']+)[\"']",
                        line,
                    )
                    if match:
                        method = match.group(1).lower()
                        route_path = match.group(2)
                    else:
                        match = re.search(
                            r"\.route\(\s*[\"']([^\"']+)[\"']\s*,\s*(get|post|put|patch|delete)\s*\(",
                            line,
                        )
                        if not match:
                            continue
                        method = match.group(2).lower()
                        route_path = match.group(1)
                    key = (path, method, route_path)
                    if key in seen:
                        continue
                    seen.add(key)
                    entry = {
                        "method": method,
                        "route": route_path,
                        "path": path,
                        "line": idx,
                    }
                    file_entries.append(entry)
                    api_entries.append(entry)
                    if len(api_entries) >= max_entries:
                        new_cache["files"][path] = {"hash": current_hash or "", "api_contracts": file_entries}
                        return api_entries, new_cache
            else:
                for idx, line in enumerate(lines, start=1):
                    match = re.search(r"(app|router)\.(get|post|put|delete|patch|ws|websocket)\(\s*[\"']([^\"']+)[\"']", line)
                    if match:
                        method = match.group(2).lower()
                        if method == "websocket":
                            method = "ws"
                        route_path = match.group(3)
                        key = (path, method, route_path)
                        if key in seen:
                            continue
                        seen.add(key)
                        entry = {
                            "method": method,
                            "route": route_path,
                            "path": path,
                            "line": idx,
                        }
                        file_entries.append(entry)
                        api_entries.append(entry)
                        if len(api_entries) >= max_entries:
                            new_cache["files"][path] = {"hash": current_hash or "", "api_contracts": file_entries}
                            return api_entries, new_cache
                    if "websocket" in line and ("WebSocket" in line or "websocket" in line):
                        entry = {
                            "method": "ws",
                            "route": "",
                            "path": path,
                            "line": idx,
                        }
                        file_entries.append(entry)
                        api_entries.append(entry)
                        if len(api_entries) >= max_entries:
                            new_cache["files"][path] = {"hash": current_hash or "", "api_contracts": file_entries}
                            return api_entries, new_cache
                    if ("grpc" in line or "proto" in line) and ".proto" in line:
                        proto_match = re.search(r"([A-Za-z0-9_./-]+\.proto)", line)
                        route = proto_match.group(1) if proto_match else ""
                        if route:
                            entry = {
                                "method": "grpc",
                                "route": route,
                                "path": path,
                                "line": idx,
                            }
                            file_entries.append(entry)
                            api_entries.append(entry)
                            if len(api_entries) >= max_entries:
                                new_cache["files"][path] = {"hash": current_hash or "", "api_contracts": file_entries}
                                return api_entries, new_cache
        if ext == ".py":
            for idx, line in enumerate(lines, start=1):
                if "websocket" in line and ("WebSocket" in line or "websocket" in line):
                    entry = {
                        "method": "ws",
                        "route": "",
                        "path": path,
                        "line": idx,
                    }
                    file_entries.append(entry)
                    api_entries.append(entry)
                    if len(api_entries) >= max_entries:
                        new_cache["files"][path] = {"hash": current_hash or "", "api_contracts": file_entries}
                        return api_entries, new_cache
                if "grpc" in line and ".proto" in line:
                    proto_match = re.search(r"([A-Za-z0-9_./-]+\.proto)", line)
                    route = proto_match.group(1) if proto_match else ""
                    if route:
                        entry = {
                            "method": "grpc",
                            "route": route,
                            "path": path,
                            "line": idx,
                        }
                        file_entries.append(entry)
                        api_entries.append(entry)
                        if len(api_entries) >= max_entries:
                            new_cache["files"][path] = {"hash": current_hash or "", "api_contracts": file_entries}
                            return api_entries, new_cache
        new_cache["files"][path] = {"hash": current_hash or "", "api_contracts": file_entries}

    if isinstance(cache_files, dict):
        for path, cached in cache_files.items():
            if path not in new_cache["files"] and path in file_set:
                new_cache["files"][path] = cached
    return api_entries, new_cache
