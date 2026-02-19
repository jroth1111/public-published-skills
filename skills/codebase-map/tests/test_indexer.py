import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os
import tempfile
import argparse

# Add scripts/ to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))

from indexer import (
    EXCLUDE_DIRS,
    FD_EXCLUDES,
    RG_EXCLUDES,
    detect_languages,
    unsupported_language_counts,
    detect_monorepo,
    detect_framework,
    match_globs,
    filter_docs,
    filter_tests,
    filter_configs,
    resolve_ts_module,
    resolve_py_module,
    IndexOptions,
    extract_entities,
    attach_symbol_docs,
    build_test_mapping,
    extract_api_contracts,
    build_entity_graph,
    is_noise_path,
    list_repo_files_fallback,
    detect_entrypoints,
    ast_rules_for_lang,
)
from indexer.ast_edges import build_ast_primary_edges
from cli.profiles import apply_profile
from ir import file_id

class TestIndexer(unittest.TestCase):
    def test_generated_state_dirs_are_default_excludes(self):
        self.assertIn(".code-state", EXCLUDE_DIRS)
        self.assertIn(".code-state", FD_EXCLUDES)
        self.assertIn("!**/.code-state/**", RG_EXCLUDES)
        self.assertIn(".beads", EXCLUDE_DIRS)
        self.assertIn(".beads", FD_EXCLUDES)
        self.assertIn("!**/.beads/**", RG_EXCLUDES)
        self.assertIn(".tanstack", EXCLUDE_DIRS)
        self.assertIn(".output", EXCLUDE_DIRS)
        self.assertIn(".nitro", EXCLUDE_DIRS)
        self.assertIn(".vinxi", EXCLUDE_DIRS)
        self.assertIn("out", EXCLUDE_DIRS)
        self.assertIn(".vercel", EXCLUDE_DIRS)
        self.assertIn("!**/next-env.d.ts", RG_EXCLUDES)
        self.assertIn("!**/*.tsbuildinfo", RG_EXCLUDES)
        self.assertIn("!**/.yarn/install-state.gz", RG_EXCLUDES)
        self.assertIn("!**/*.py[cod]", RG_EXCLUDES)
        self.assertIn("!**/target/**", RG_EXCLUDES)
        self.assertIn("!**/.vinxi/**", RG_EXCLUDES)

    def test_is_noise_path_detects_state_dirs(self):
        self.assertTrue(is_noise_path("codebase/.code-state/cache/summary.txt"))
        self.assertTrue(is_noise_path("codebase/.beads/issues.jsonl"))
        self.assertTrue(is_noise_path("apps/web/.next/server/chunks/1.js"))
        self.assertTrue(is_noise_path("apps/web/.output/server/index.mjs"))
        self.assertTrue(is_noise_path("apps/web/.vinxi/server/chunks/1.js"))
        self.assertTrue(is_noise_path("services/api/.pytest_cache/v/cache/nodeids"))
        self.assertFalse(is_noise_path("src/services/user_service.py"))

    def test_repo_file_fallback_skips_generated_noise_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "src").mkdir(parents=True, exist_ok=True)
            (repo / "src/app.ts").write_text("export const ok = 1\n", encoding="utf-8")
            (repo / "next-env.d.ts").write_text("declare namespace Next {}\n", encoding="utf-8")
            (repo / ".yarn").mkdir(parents=True, exist_ok=True)
            (repo / ".yarn/install-state.gz").write_text("binary", encoding="utf-8")
            (repo / "cache.tsbuildinfo").write_text("{}", encoding="utf-8")
            files = list_repo_files_fallback(repo)
            self.assertIn("src/app.ts", files)
            self.assertNotIn("next-env.d.ts", files)
            self.assertNotIn(".yarn/install-state.gz", files)
            self.assertNotIn("cache.tsbuildinfo", files)

    def test_detect_languages(self):
        self.assertEqual(detect_languages([], {"ts"}), {"ts"})
        self.assertEqual(detect_languages(["a.ts"], set()), {"ts"})
        self.assertEqual(detect_languages(["a.py", "b.tsx"], set()), {"py", "tsx"})
        self.assertEqual(detect_languages(["a.js"], set()), {"js"})

    def test_match_globs(self):
        self.assertTrue(match_globs("src/test.ts", ["**/*.ts"]))
        self.assertTrue(match_globs("src/test.ts", ["test.ts"]))
        self.assertFalse(match_globs("src/test.js", ["**/*.ts"]))
        self.assertTrue(match_globs("vendor/lib.js", ["vendor/**"]))

    def test_filter_docs(self):
        files = ["README.md", "src/main.ts", "docs/api.md", "CONTRIBUTING", "references/guide.md", "SKILL.md"]
        docs = filter_docs(files)
        self.assertIn("README.md", docs)
        self.assertIn("docs/api.md", docs)
        self.assertIn("references/guide.md", docs)
        self.assertIn("SKILL.md", docs)
        self.assertNotIn("src/main.ts", docs)

    def test_filter_tests(self):
        files = [
            "src/main.ts",
            "src/main.test.ts",
            "tests/integration.py",
            "spec/test.ts",
            "skill.spec.json",
        ]
        tests = filter_tests(files)
        self.assertIn("src/main.test.ts", tests)
        self.assertIn("tests/integration.py", tests)
        self.assertNotIn("skill.spec.json", tests)
        self.assertNotIn("src/main.ts", tests)

    def test_filter_configs(self):
        files = [
            "package.json",
            "src/index.ts",
            "tsconfig.json",
            "pyproject.toml",
            "skill.spec.json",
            "repomap.json",
            ".repomap.json",
        ]
        configs = filter_configs(files)
        self.assertIn("package.json", configs)
        self.assertIn("tsconfig.json", configs)
        self.assertIn("pyproject.toml", configs)
        self.assertIn("skill.spec.json", configs)
        self.assertIn("repomap.json", configs)
        self.assertIn(".repomap.json", configs)
        self.assertNotIn("src/index.ts", configs)

    def test_detect_entrypoints_ignores_non_runtime_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "examples").mkdir(parents=True, exist_ok=True)
            (repo / "src").mkdir(parents=True, exist_ok=True)
            (repo / "examples/app.py").write_text(
                "from fastapi import FastAPI\napp = FastAPI()\n",
                encoding="utf-8",
            )
            (repo / "src/main.py").write_text(
                "def main():\n    return 0\n",
                encoding="utf-8",
            )
            files = ["examples/app.py", "src/main.py"]
            entrypoints = detect_entrypoints(repo, files, tests=set())
            self.assertNotIn("examples/app.py", entrypoints)
            self.assertIn("src/main.py", entrypoints)

    def test_detect_entrypoints_refines_role_by_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "src").mkdir(parents=True, exist_ok=True)
            (repo / "src/server.py").write_text(
                "import argparse\n"
                "def main():\n"
                "    parser = argparse.ArgumentParser()\n"
                "    parser.parse_args()\n",
                encoding="utf-8",
            )
            entrypoints = detect_entrypoints(repo, ["src/server.py"], tests=set())
            self.assertEqual(entrypoints.get("src/server.py"), "cli")

    def test_detect_entrypoints_second_pass_for_scripts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "scripts").mkdir(parents=True, exist_ok=True)
            (repo / "scripts/pipeline_monitor.py").write_text(
                "from airflow import DAG\n"
                "with DAG('daily-ingest', schedule_interval='@daily') as dag:\n"
                "    pass\n",
                encoding="utf-8",
            )
            entrypoints = detect_entrypoints(repo, ["scripts/pipeline_monitor.py"], tests=set())
            self.assertEqual(entrypoints.get("scripts/pipeline_monitor.py"), "worker")

    def test_unsupported_language_counts(self):
        files = ["src/app.ts", "src/app.js", "lib/main.rs", "src/main.java", "README.md", "scripts/tool.py"]
        counts = unsupported_language_counts(files)
        self.assertEqual(counts.get("java"), 1)
        self.assertNotIn("rust", counts)
        self.assertNotIn("python", counts)
        self.assertNotIn("javascript", counts)

    def test_detect_monorepo(self):
        files = ["pnpm-workspace.yaml", "packages/a/package.json", "packages/b/package.json"]
        monorepo = detect_monorepo(Path("."), files)
        self.assertEqual(monorepo.get("type"), "pnpm")
        package_paths = [pkg.get("path") for pkg in monorepo.get("packages", [])]
        self.assertIn("packages/a", package_paths)
        self.assertIn("packages/b", package_paths)

    def test_detect_framework_project_dependencies_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "pyproject.toml").write_text(
                "[project]\n"
                "name = \"demo\"\n"
                "version = \"0.1.0\"\n"
                "dependencies = [\n"
                "  \"fastapi>=0.100.0\",\n"
                "  \"uvicorn[standard]>=0.30.0\",\n"
                "]\n",
                encoding="utf-8",
            )
            self.assertEqual(detect_framework(repo, {}), "fastapi")

    def test_detect_framework_poetry_dependencies_dict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "pyproject.toml").write_text(
                "[tool.poetry]\n"
                "name = \"demo\"\n"
                "version = \"0.1.0\"\n"
                "[tool.poetry.dependencies]\n"
                "python = \">=3.11\"\n"
                "django = \"^5.0\"\n",
                encoding="utf-8",
            )
            self.assertEqual(detect_framework(repo, {}), "django")

    def test_ast_rules_include_rust(self):
        rules_yaml, meta = ast_rules_for_lang("rust", include_internal=True)
        self.assertIn("language: rust", rules_yaml)
        categories = {info.get("category") for info in meta.values()}
        self.assertIn("import", categories)
        symbol_kinds = {info.get("kind") for info in meta.values() if info.get("category") == "symbol"}
        self.assertIn("function", symbol_kinds)
        self.assertIn("struct", symbol_kinds)

    def test_resolve_ts_module(self):
        files = {"src/utils.ts", "src/components/Button.tsx", "src/index.ts"}
        
        # Relative import
        self.assertEqual(resolve_ts_module("./utils", "src/main.ts", files), "src/utils.ts")
        # Relative import with extension
        self.assertEqual(resolve_ts_module("./utils.ts", "src/main.ts", files), "src/utils.ts")
        # Parent directory
        self.assertEqual(resolve_ts_module("../index", "src/components/Button.tsx", files), "src/index.ts")
        # Missing file
        self.assertIsNone(resolve_ts_module("./missing", "src/main.ts", files))
        # Non-relative import (ignored)
        self.assertIsNone(resolve_ts_module("react", "src/main.ts", files))

    def test_resolve_ts_module_alias(self):
        files = {"src/utils.ts", "src/index.ts"}
        alias_config = {"baseUrl": "src", "paths": {"@/*": ["src/*"]}}

        # Alias import
        self.assertEqual(
            resolve_ts_module("@/utils", "src/main.ts", files, alias_config=alias_config),
            "src/utils.ts",
        )
        # baseUrl import
        self.assertEqual(
            resolve_ts_module("utils", "src/main.ts", files, alias_config=alias_config),
            "src/utils.ts",
        )

    def test_resolve_py_module(self):
        files = {"app/utils.py", "app/models/__init__.py", "app/core/config.py"}

        # Absolute-ish import (within repo)
        self.assertEqual(resolve_py_module("app.utils", "app/main.py", files), "app/utils.py")
        self.assertEqual(resolve_py_module("app.models", "app/main.py", files), "app/models/__init__.py")

        # Fallback to importer directory for absolute imports
        self.assertEqual(resolve_py_module("utils", "app/main.py", files), "app/utils.py")
        
        # Relative import (.)
        self.assertEqual(resolve_py_module(".config", "app/core/main.py", files), "app/core/config.py")
        
        # Missing
        self.assertIsNone(resolve_py_module("app.missing", "app/main.py", files))

    def test_entity_graph_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            model_path = repo / "models/user.py"
            service_path = repo / "services/user_service.py"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            service_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text(
                "from pydantic import BaseModel\n"
                "class User(BaseModel):\n"
                "    id: str\n",
                encoding="utf-8",
            )
            service_path.write_text(
                "from models.user import User\n"
                "def get_user() -> User:\n"
                "    return User(id='1')\n",
                encoding="utf-8",
            )
            files = ["models/user.py", "services/user_service.py"]
            entities = extract_entities(repo, files)
            file_edges = [
                {
                    "from": file_id("services/user_service.py"),
                    "to": file_id("models/user.py"),
                }
            ]
            nodes, edges = build_entity_graph(entities, file_edges=file_edges)
            self.assertTrue(any(node.get("name") == "User" for node in nodes.values()))
            self.assertTrue(
                any(
                    edge.get("kind") == "defines"
                    and edge.get("from") == file_id("models/user.py")
                    for edge in edges
                )
            )
            self.assertTrue(
                any(
                    edge.get("kind") == "uses"
                    and edge.get("from") == file_id("services/user_service.py")
                    for edge in edges
                )
            )

    def test_extract_entities_drizzle_typeorm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = "\n".join(
                [
                    "import { pgTable, serial, text } from 'drizzle-orm/pg-core'",
                    "export const users = pgTable('users', {",
                    "  id: serial('id').primaryKey(),",
                    "  name: text('name'),",
                    "})",
                    "",
                    "@Entity()",
                    "export class Account {",
                    "  @PrimaryGeneratedColumn()",
                    "  id!: number",
                    "  @Column()",
                    "  email!: string",
                    "}",
                ]
            )
            rel_path = "src/models.ts"
            (root / "src").mkdir(parents=True, exist_ok=True)
            (root / rel_path).write_text(content, encoding="utf-8")
            entities = extract_entities(root, [rel_path], max_entities=10, max_fields=4)
            by_name = {entity["name"]: entity for entity in entities}
            self.assertEqual(by_name.get("users", {}).get("kind"), "drizzle")
            self.assertEqual(by_name.get("Account", {}).get("kind"), "typeorm")

    def test_extract_entities_graphql(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = "\n".join(
                [
                    "type User {",
                    "  id: ID!",
                    "  email: String",
                    "}",
                    "",
                    "type Post {",
                    "  id: ID!",
                    "  title: String",
                    "}",
                ]
            )
            rel_path = "schema.graphql"
            (root / rel_path).write_text(content, encoding="utf-8")
            entities = extract_entities(root, [rel_path], max_entities=10, max_fields=4)
            by_name = {entity["name"]: entity for entity in entities}
            self.assertEqual(by_name.get("User", {}).get("kind"), "graphql")
            self.assertEqual(by_name.get("Post", {}).get("kind"), "graphql")

    def test_profile_defaults(self):
        args = argparse.Namespace()
        apply_profile(args, "balanced")
        self.assertTrue(getattr(args, "static_traces", False))
        self.assertTrue(getattr(args, "api_contracts", False))
        self.assertTrue(getattr(args, "include_routes", False))
        args = argparse.Namespace()
        apply_profile(args, "deep")
        self.assertTrue(getattr(args, "semantic_cluster", False))
        self.assertTrue(getattr(args, "doc_quality", False))
        self.assertTrue(getattr(args, "call_chains", False))
        self.assertEqual(getattr(args, "codeql", None), "auto")
        args = argparse.Namespace()
        apply_profile(args, "fast")
        self.assertFalse(getattr(args, "static_traces", True))

    def test_build_ast_primary_edges_for_py_ts_rs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "src").mkdir(parents=True, exist_ok=True)
            (repo / "src/main.py").write_text(
                "def helper(x):\n"
                "    return x + 1\n"
                "\n"
                "def run(v):\n"
                "    out = helper(v)\n"
                "    return out\n",
                encoding="utf-8",
            )
            (repo / "src/main.ts").write_text(
                "function helper(v: number) {\n"
                "  return v + 1;\n"
                "}\n"
                "function run(v: number) {\n"
                "  const out = helper(v);\n"
                "  return out;\n"
                "}\n",
                encoding="utf-8",
            )
            (repo / "src/main.rs").write_text(
                "fn helper(v: i32) -> i32 {\n"
                "    v + 1\n"
                "}\n"
                "fn run(v: i32) -> i32 {\n"
                "    let out = helper(v);\n"
                "    out\n"
                "}\n",
                encoding="utf-8",
            )
            files = ["src/main.py", "src/main.ts", "src/main.rs"]
            info = build_ast_primary_edges(repo, files, {"py", "ts", "rs"})
            counts = info.get("counts", {})
            self.assertGreater(int(counts.get("py_call_edges_ast", 0)), 0)
            self.assertGreater(int(counts.get("ts_call_edges_ast", 0)), 0)
            self.assertGreater(int(counts.get("rs_call_edges_ast", 0)), 0)
            self.assertIn(info.get("confidence"), {"high", "medium", "low"})

    def test_build_ast_primary_edges_low_confidence_detection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "src").mkdir(parents=True, exist_ok=True)
            (repo / "src/a.ts").write_text("export {};\n", encoding="utf-8")
            (repo / "src/b.ts").write_text("export {};\n", encoding="utf-8")
            info = build_ast_primary_edges(repo, ["src/a.ts", "src/b.ts"], {"ts"})
            self.assertTrue(bool(info.get("low_confidence")))
            self.assertIn("ts", info.get("low_confidence_languages", []))

    def test_attach_symbol_docs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            js_content = "\n".join(
                [
                    "/**",
                    " * Adds two numbers.",
                    " * @param {number} a first number",
                    " * @param {number} b second number",
                    " * @returns {number} sum",
                    " */",
                    "export function add(a, b) {",
                    "  return a + b",
                    "}",
                ]
            )
            py_content = "\n".join(
                [
                    "def greet(name):",
                    '    """Greet a user.',
                    "",
                    "    Args:",
                    "        name (str): user name.",
                    "",
                    "    Returns:",
                    "        str: friendly greeting.",
                    '    """',
                    "    return f\"Hi {name}\"",
                ]
            )
            (root / "src").mkdir(parents=True, exist_ok=True)
            (root / "src/add.js").write_text(js_content, encoding="utf-8")
            (root / "src/utils.py").write_text(py_content, encoding="utf-8")
            symbol_nodes = {
                "sym:add": {
                    "id": "sym:add",
                    "name": "add",
                    "defined_in": {"path": "src/add.js", "line": 7},
                    "doc_1l": "",
                },
                "sym:greet": {
                    "id": "sym:greet",
                    "name": "greet",
                    "defined_in": {"path": "src/utils.py", "line": 1},
                    "doc_1l": "",
                },
            }
            attach_symbol_docs(root, symbol_nodes)
            self.assertEqual(symbol_nodes["sym:add"].get("doc_1l"), "Adds two numbers.")
            self.assertEqual(symbol_nodes["sym:greet"].get("doc_1l"), "Greet a user.")
            js_doc = symbol_nodes["sym:add"].get("documentation") or {}
            self.assertEqual(js_doc.get("returns_type"), "number")
            self.assertEqual(js_doc.get("params", [])[0].get("type"), "number")
            py_doc = symbol_nodes["sym:greet"].get("documentation") or {}
            self.assertEqual(py_doc.get("returns_type"), "str")
            self.assertEqual(py_doc.get("params", [])[0].get("type"), "str")

    def test_build_test_mapping(self):
        ir = {
            "files": {
                "file:src/app.ts": {"path": "src/app.ts", "role": "library"},
                "file:src/app.test.ts": {"path": "src/app.test.ts", "role": "test"},
            },
            "edges": {
                "file_dep": [
                    {"from": "file:src/app.test.ts", "to": "file:src/app.ts"},
                ]
            },
        }
        mapping, summary = build_test_mapping(ir)
        self.assertEqual(len(mapping), 1)
        self.assertEqual(mapping[0]["test"], "src/app.test.ts")
        self.assertIn("src/app.ts", mapping[0]["targets"])
        self.assertIn("orphan_tests", summary)

    def test_build_test_mapping_symbols(self):
        ir = {
            "files": {
                "file:src/app.ts": {"path": "src/app.ts", "role": "library"},
                "file:tests/app.test.ts": {"path": "tests/app.test.ts", "role": "test"},
            },
            "symbols": {
                "sym:tests/app.test.ts#L1:testIt": {
                    "id": "sym:tests/app.test.ts#L1:testIt",
                    "name": "testIt",
                    "defined_in": {"path": "tests/app.test.ts", "line": 1},
                },
                "sym:src/app.ts#L1:run": {
                    "id": "sym:src/app.ts#L1:run",
                    "name": "run",
                    "defined_in": {"path": "src/app.ts", "line": 1},
                },
            },
            "edges": {
                "file_dep": [],
                "symbol_ref": [
                    {"from": "sym:tests/app.test.ts#L1:testIt", "to": "sym:src/app.ts#L1:run"}
                ],
            },
        }
        mapping, summary = build_test_mapping(ir)
        self.assertEqual(len(mapping), 1)
        self.assertIn("run", mapping[0].get("symbols", []))
        self.assertIn("tested_by", ir["symbols"]["sym:src/app.ts#L1:run"])
        self.assertIn("untested_entrypoints", summary)

    def test_extract_api_contracts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            py_content = "\n".join(
                [
                    "from fastapi import APIRouter",
                    "router = APIRouter()",
                    "@router.get('/items', response_model=Item, status_code=200)",
                    "def list_items(limit: int) -> Item:",
                    "    return Item()",
                ]
            )
            js_content = "\n".join(
                [
                    "const router = require('express').Router()",
                    "router.post('/login', loginHandler)",
                ]
            )
            (root / "api").mkdir(parents=True, exist_ok=True)
            (root / "api/routes.py").write_text(py_content, encoding="utf-8")
            (root / "api/router.js").write_text(js_content, encoding="utf-8")
            files = ["api/routes.py", "api/router.js"]
            file_hashes = {path: "x" for path in files}
            contracts, cache = extract_api_contracts(root, files, file_hashes=file_hashes, cache={})
            self.assertTrue(any(entry.get("route") == "/items" for entry in contracts))
            self.assertTrue(any(entry.get("route") == "/login" for entry in contracts))
            self.assertIn("files", cache)

if __name__ == "__main__":
    unittest.main()
