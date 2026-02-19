from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def rule_to_yaml(rule_id: str, language: str, pattern: str) -> str:
    escaped = pattern.replace("'", "''")
    if "\n" in pattern:
        lines = [
            f"id: {rule_id}",
            f"language: {language}",
            "rule:",
            "  pattern: |",
        ]
        for line in pattern.splitlines():
            lines.append(f"    {line}")
        return "\n".join(lines)
    return "\n".join(
        [
            f"id: {rule_id}",
            f"language: {language}",
            "rule:",
            f"  pattern: '{escaped}'",
        ]
    )


def rules_to_yaml(rules: List[Tuple[str, str, str]]) -> str:
    parts = [rule_to_yaml(rule_id, language, pattern) for rule_id, language, pattern in rules]
    return "\n---\n".join(parts)


def ast_rules_for_lang(
    lang: str, include_internal: bool
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    rules: List[Tuple[str, str, str]] = []
    meta: Dict[str, Dict[str, Any]] = {}

    def add_rule(rule_id: str, language: str, pattern: str, info: Dict[str, Any]) -> None:
        rules.append((rule_id, language, pattern))
        meta[rule_id] = info

    if lang in {"ts", "tsx"}:
        language = lang
        import_patterns = [
            'import { $$$ } from "$MOD"',
            'import type { $$$ } from "$MOD"',
            'import $NAME from "$MOD"',
            'import * as $NAME from "$MOD"',
            'import "$MOD"',
        ]
        reexport_patterns = [
            'export { $$$ } from "$MOD"',
            'export type { $$$ } from "$MOD"',
            'export * from "$MOD"',
            'export * as $NAME from "$MOD"',
            'export { default as $NAME } from "$MOD"',
        ]
        for idx, pattern in enumerate(import_patterns, start=1):
            rule_id = f"{lang}-import-{idx}"
            add_rule(rule_id, language, pattern, {"category": "import"})
        for idx, pattern in enumerate(reexport_patterns, start=1):
            rule_id = f"{lang}-reexport-{idx}"
            add_rule(rule_id, language, pattern, {"category": "reexport"})

        export_patterns: List[Tuple[str, str, bool, str | None]] = [
            ("export async function $NAME($$$PARAMS): $RET { $$$ }", "function", True, None),
            ("export async function $NAME($$$PARAMS) { $$$ }", "function", True, None),
            ("export function $NAME($$$PARAMS): $RET { $$$ }", "function", True, None),
            ("export function $NAME($$$PARAMS) { $$$ }", "function", True, None),
            ("export default function $NAME($$$PARAMS): $RET { $$$ }", "function", True, None),
            ("export default function $NAME($$$PARAMS) { $$$ }", "function", True, None),
            ("export default function ($$$PARAMS): $RET { $$$ }", "function", True, "default"),
            ("export default function ($$$PARAMS) { $$$ }", "function", True, "default"),
            ("export class $NAME extends $BASE { $$$ }", "class", True, None),
            ("export class $NAME { $$$ }", "class", True, None),
            ("export default class $NAME extends $BASE { $$$ }", "class", True, None),
            ("export default class $NAME { $$$ }", "class", True, None),
            ("export default class { $$$ }", "class", True, "default"),
            ("export interface $NAME extends $BASE { $$$ }", "interface", True, None),
            ("export interface $NAME { $$$ }", "interface", True, None),
            ("export type $NAME = $TYPE", "type", True, None),
            ("export enum $NAME { $$$ }", "enum", True, None),
            ("export const $NAME = async ($$$PARAMS): $RET => $$$", "const", True, None),
            ("export const $NAME = async ($$$PARAMS) => $$$", "const", True, None),
            ("export const $NAME = ($$$PARAMS): $RET => $$$", "const", True, None),
            ("export const $NAME = ($$$PARAMS) => $$$", "const", True, None),
            ("export const $NAME = $VALUE", "const", True, None),
        ]
        internal_patterns: List[Tuple[str, str, bool, str | None]] = [
            ("async function $NAME($$$PARAMS): $RET { $$$ }", "function", False, None),
            ("async function $NAME($$$PARAMS) { $$$ }", "function", False, None),
            ("function $NAME($$$PARAMS): $RET { $$$ }", "function", False, None),
            ("function $NAME($$$PARAMS) { $$$ }", "function", False, None),
            ("class $NAME extends $BASE { $$$ }", "class", False, None),
            ("class $NAME { $$$ }", "class", False, None),
            ("interface $NAME extends $BASE { $$$ }", "interface", False, None),
            ("interface $NAME { $$$ }", "interface", False, None),
            ("type $NAME = $TYPE", "type", False, None),
            ("enum $NAME { $$$ }", "enum", False, None),
            ("const $NAME = async ($$$PARAMS): $RET => $$$", "const", False, None),
            ("const $NAME = async ($$$PARAMS) => $$$", "const", False, None),
            ("const $NAME = ($$$PARAMS): $RET => $$$", "const", False, None),
            ("const $NAME = ($$$PARAMS) => $$$", "const", False, None),
            ("const $NAME = $VALUE", "const", False, None),
        ]
        sym_patterns = export_patterns + (internal_patterns if include_internal else [])
        for idx, (pattern, kind, exported, default_name) in enumerate(sym_patterns, start=1):
            rule_id = f"{lang}-sym-{idx}"
            add_rule(
                rule_id,
                language,
                pattern,
                {
                    "category": "symbol",
                    "kind": kind,
                    "exported": exported,
                    "default_name": default_name,
                },
            )
    elif lang in {"javascript", "jsx"}:
        language = lang
        import_patterns = [
            'import { $$$ } from "$MOD"',
            'import $NAME from "$MOD"',
            'import * as $NAME from "$MOD"',
            'import "$MOD"',
        ]
        reexport_patterns = [
            'export { $$$ } from "$MOD"',
            'export * from "$MOD"',
            'export * as $NAME from "$MOD"',
            'export { default as $NAME } from "$MOD"',
        ]
        for idx, pattern in enumerate(import_patterns, start=1):
            rule_id = f"{lang}-import-{idx}"
            add_rule(rule_id, language, pattern, {"category": "import"})
        for idx, pattern in enumerate(reexport_patterns, start=1):
            rule_id = f"{lang}-reexport-{idx}"
            add_rule(rule_id, language, pattern, {"category": "reexport"})

        export_patterns = [
            ("export async function $NAME($$$PARAMS) { $$$ }", "function", True, None),
            ("export function $NAME($$$PARAMS) { $$$ }", "function", True, None),
            ("export default function $NAME($$$PARAMS) { $$$ }", "function", True, None),
            ("export default function ($$$PARAMS) { $$$ }", "function", True, "default"),
            ("export class $NAME extends $BASE { $$$ }", "class", True, None),
            ("export class $NAME { $$$ }", "class", True, None),
            ("export default class $NAME extends $BASE { $$$ }", "class", True, None),
            ("export default class $NAME { $$$ }", "class", True, None),
            ("export default class { $$$ }", "class", True, "default"),
            ("export const $NAME = async ($$$PARAMS) => $$$", "const", True, None),
            ("export const $NAME = ($$$PARAMS) => $$$", "const", True, None),
            ("export const $NAME = $VALUE", "const", True, None),
        ]
        internal_patterns = [
            ("async function $NAME($$$PARAMS) { $$$ }", "function", False, None),
            ("function $NAME($$$PARAMS) { $$$ }", "function", False, None),
            ("class $NAME extends $BASE { $$$ }", "class", False, None),
            ("class $NAME { $$$ }", "class", False, None),
            ("const $NAME = async ($$$PARAMS) => $$$", "const", False, None),
            ("const $NAME = ($$$PARAMS) => $$$", "const", False, None),
            ("const $NAME = $VALUE", "const", False, None),
        ]
        sym_patterns = export_patterns + (internal_patterns if include_internal else [])
        for idx, (pattern, kind, exported, default_name) in enumerate(sym_patterns, start=1):
            rule_id = f"{lang}-sym-{idx}"
            add_rule(
                rule_id,
                language,
                pattern,
                {
                    "category": "symbol",
                    "kind": kind,
                    "exported": exported,
                    "default_name": default_name,
                },
            )
    elif lang == "python":
        language = "python"
        import_patterns = [
            "import $MOD",
            "from $MOD import $NAME",
        ]
        for idx, pattern in enumerate(import_patterns, start=1):
            rule_id = f"{lang}-import-{idx}"
            add_rule(rule_id, language, pattern, {"category": "import"})
        sym_patterns = [
            ("async def $NAME($$$PARAMS) -> $RET:\n    $$$", "function", True, None),
            ("async def $NAME($$$PARAMS):\n    $$$", "function", True, None),
            ("def $NAME($$$PARAMS) -> $RET:\n    $$$", "function", True, None),
            ("def $NAME($$$PARAMS):\n    $$$", "function", True, None),
            ("class $NAME($$$BASES):\n    $$$", "class", True, None),
            ("class $NAME:\n    $$$", "class", True, None),
        ]
        for idx, (pattern, kind, exported, default_name) in enumerate(sym_patterns, start=1):
            rule_id = f"{lang}-sym-{idx}"
            add_rule(
                rule_id,
                language,
                pattern,
                {
                    "category": "symbol",
                    "kind": kind,
                    "exported": exported,
                    "default_name": default_name,
                },
            )
    elif lang == "rust":
        language = "rust"
        import_patterns = [
            "use $MOD;",
            "use $MOD::$NAME;",
            "use $MOD::{ $$$NAMES };",
        ]
        for idx, pattern in enumerate(import_patterns, start=1):
            rule_id = f"{lang}-import-{idx}"
            add_rule(rule_id, language, pattern, {"category": "import"})

        export_patterns = [
            ("pub fn $NAME($$$PARAMS) -> $RET { $$$ }", "function", True, None),
            ("pub fn $NAME($$$PARAMS) { $$$ }", "function", True, None),
            ("pub struct $NAME { $$$ }", "struct", True, None),
            ("pub enum $NAME { $$$ }", "enum", True, None),
            ("pub trait $NAME { $$$ }", "trait", True, None),
        ]
        internal_patterns = [
            ("fn $NAME($$$PARAMS) -> $RET { $$$ }", "function", False, None),
            ("fn $NAME($$$PARAMS) { $$$ }", "function", False, None),
            ("struct $NAME { $$$ }", "struct", False, None),
            ("enum $NAME { $$$ }", "enum", False, None),
            ("trait $NAME { $$$ }", "trait", False, None),
        ]
        sym_patterns = export_patterns + (internal_patterns if include_internal else [])
        for idx, (pattern, kind, exported, default_name) in enumerate(sym_patterns, start=1):
            rule_id = f"{lang}-sym-{idx}"
            add_rule(
                rule_id,
                language,
                pattern,
                {
                    "category": "symbol",
                    "kind": kind,
                    "exported": exported,
                    "default_name": default_name,
                },
            )

    return rules_to_yaml(rules), meta


def normalize_match_path(repo: Path, path: str) -> str:
    if not path:
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(repo.resolve()).as_posix()
        except (ValueError, OSError):
            return candidate.as_posix()
    return candidate.as_posix()
