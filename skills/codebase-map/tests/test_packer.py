import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))

from packer import (
    entrypoint_detail_lines,
    entrypoint_inventory_lines,
    expand_focus_neighbors,
    flow_lines,
    entity_section_lines,
    digest,
    architecture_overview_line,
    architecture_section_lines,
    type_hierarchy_lines,
)
from packer.sections.routes import (
    nextjs_route_path,
    route_candidates,
    route_entries_for_file,
)
from ir import new_ir, file_id


def write_file(root: Path, rel_path: str, content: str) -> None:
    full_path = root / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")


class TestPacker(unittest.TestCase):
    def _basic_ir(self, repo: Path, file_paths: List[str]) -> dict:
        ir = new_ir(repo, ["ts"], {path: "x" for path in file_paths})
        for path in file_paths:
            fid = file_id(path)
            ir["files"][fid] = {
                "id": fid,
                "path": path,
                "language": "ts",
                "imports": [],
                "exports": [],
                "defines": [],
                "score": 0.0,
            }
        return ir

    def test_section_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            paths = [
                "src/entry.ts",
                "src/routes/user.ts",
                "src/lib/util.ts",
            ]
            ir = self._basic_ir(repo, paths)
            ir["files"][file_id("src/entry.ts")]["role"] = "entrypoint"
            ir["edges"]["file_dep"] = [
                {"from": file_id("src/entry.ts"), "to": file_id("src/lib/util.ts")}
            ]
            result = digest(
                ir,
                2000,
                include_routes=True,
                routes_mode="heuristic",
                static_traces=True,
                compress_paths=False,
            )
            text = result.get("digest", "")
            self.assertIn("[SUMMARY]", text)
            self.assertIn("[FILES]", text)
            self.assertIn("[ROUTES]", text)
            summary_idx = text.find("[SUMMARY]")
            files_idx = text.find("[FILES]")
            routes_idx = text.find("[ROUTES]")
            self.assertTrue(0 <= summary_idx < files_idx < routes_idx)

    def test_section_budgets_skip_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = self._basic_ir(repo, ["src/a.ts", "src/b.ts"])
            result = digest(
                ir,
                1000,
                section_budgets={"FILES": 0.001},
                include_routes=False,
            )
            text = result.get("digest", "")
            self.assertIn("[SUMMARY]", text)
            self.assertNotIn("F1 ", text)

    def test_section_budgets_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = self._basic_ir(repo, ["src/a.ts", "src/b.ts"])
            result = digest(
                ir,
                1000,
                section_budgets={"files": 0.001},
                include_routes=False,
            )
            text = result.get("digest", "")
            self.assertIn("[SUMMARY]", text)
            self.assertNotIn("F1 ", text)

    def test_section_budgets_skip_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = self._basic_ir(repo, ["src/routes/user.ts"])
            baseline = digest(
                ir,
                1000,
                include_routes=True,
                routes_mode="heuristic",
                compress_paths=False,
            )
            self.assertIn("[ROUTES]", baseline.get("digest", ""))
            result = digest(
                ir,
                1000,
                include_routes=True,
                routes_mode="heuristic",
                section_budgets={"routes": 0.001},
                compress_paths=False,
            )
            self.assertNotIn("[ROUTES]", result.get("digest", ""))

    def test_route_candidates_include_http_entrypoints(self) -> None:
        files = {
            "file:http": {
                "id": "file:http",
                "path": "crates/remote/src/app.rs",
                "score": 0.1,
                "entrypoint_role": "http",
            },
            "file:lib": {
                "id": "file:lib",
                "path": "crates/remote/src/lib.rs",
                "score": 0.9,
            },
        }
        ranked = route_candidates(files, routes_mode="heuristic")
        self.assertIn("file:http", ranked)

    def test_route_entries_use_entrypoint_role_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_file(root, "crates/remote/src/app.rs", "fn main() {}\n")
            entries = route_entries_for_file(
                path="crates/remote/src/app.rs",
                prefix="",
                routes_mode="heuristic",
                repo_root=root,
                entrypoint_role="http",
            )
            self.assertTrue(entries)
            self.assertEqual(entries[0].get("kind"), "api")
            self.assertTrue(str(entries[0].get("label", "")).startswith("file:"))

    def test_routes_keep_same_label_from_multiple_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            write_file(
                repo,
                "service_a/routes.py",
                "@app.get('/health')\n"
                "def health_a():\n"
                "    return {'ok': True}\n",
            )
            write_file(
                repo,
                "service_b/routes.py",
                "@app.get('/health')\n"
                "def health_b():\n"
                "    return {'ok': True}\n",
            )
            ir = new_ir(repo, ["py"], {"service_a/routes.py": "x", "service_b/routes.py": "y"})
            for path in ["service_a/routes.py", "service_b/routes.py"]:
                fid = file_id(path)
                ir["files"][fid] = {
                    "id": fid,
                    "path": path,
                    "language": "py",
                    "imports": [],
                    "exports": [],
                    "defines": [],
                    "score": 0.0,
                }
            result = digest(
                ir,
                3000,
                include_routes=True,
                routes_mode="heuristic",
                routes_limit=8,
                compress_paths=False,
            )
            text = result.get("digest", "")
            self.assertGreaterEqual(text.count("get:/health"), 2)

    def test_nextjs_route_path_handles_monorepo_prefix(self) -> None:
        path = "apps/web/src/app/api/auth/status/route.ts"
        self.assertEqual(nextjs_route_path(path, "apps/"), "/api/auth/status")

    def test_section_budgets_skip_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = self._basic_ir(repo, ["src/a.ts"])
            fid = file_id("src/a.ts")
            sid = "sym:a"
            ir["symbols"][sid] = {
                "id": sid,
                "name": "add",
                "kind": "function",
                "signature": "add(a, b)",
                "doc_1l": "Adds two numbers.",
                "defined_in": {"path": "src/a.ts", "line": 1},
                "visibility": "public",
                "score": 1.0,
            }
            ir["files"][fid]["exports"] = [sid]
            ir["files"][fid]["defines"] = [sid]
            baseline = digest(ir, 1200, include_routes=False, compress_paths=False)
            self.assertIn("[DOCS]", baseline.get("digest", ""))
            capped = digest(
                ir,
                1200,
                include_routes=False,
                compress_paths=False,
                section_budgets={"docs": 0.001},
            )
            self.assertNotIn("[DOCS]", capped.get("digest", ""))

    def test_section_budgets_skip_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = self._basic_ir(repo, ["src/a.ts"])
            ir["meta"]["quality"] = {"code_files_total": 1, "ast_enabled": True}
            baseline = digest(ir, 1200, include_routes=False, compress_paths=False)
            self.assertIn("[QUALITY]", baseline.get("digest", ""))
            capped = digest(
                ir,
                1200,
                include_routes=False,
                compress_paths=False,
                section_budgets={"quality": 0.001},
            )
            self.assertNotIn("[QUALITY]", capped.get("digest", ""))

    def test_section_budgets_skip_docs_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = self._basic_ir(repo, ["src/a.ts"])
            fid = file_id("src/a.ts")
            sid = "sym:a"
            ir["symbols"][sid] = {
                "id": sid,
                "name": "add",
                "kind": "function",
                "signature": "add(a: string) -> string",
                "doc_1l": "",
                "defined_in": {"path": "src/a.ts", "line": 1},
                "visibility": "public",
                "score": 1.0,
            }
            ir["files"][fid]["exports"] = [sid]
            ir["files"][fid]["defines"] = [sid]
            baseline = digest(
                ir,
                1200,
                include_routes=False,
                compress_paths=False,
                doc_quality=True,
            )
            self.assertIn("[DOCS_QUALITY]", baseline.get("digest", ""))
            capped = digest(
                ir,
                1200,
                include_routes=False,
                compress_paths=False,
                doc_quality=True,
                section_budgets={"docs_quality": 0.001},
            )
            self.assertNotIn("[DOCS_QUALITY]", capped.get("digest", ""))

    def test_section_budgets_skip_api_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = self._basic_ir(repo, ["src/api.ts"])
            ir["meta"]["api_contracts"] = [
                {
                    "method": "get",
                    "route": "/api/test",
                    "path": "src/api.ts",
                    "line": 1,
                }
            ]
            baseline = digest(ir, 1200, include_routes=False, compress_paths=False)
            self.assertIn("[API_CONTRACTS]", baseline.get("digest", ""))
            capped = digest(
                ir,
                1200,
                include_routes=False,
                compress_paths=False,
                section_budgets={"api_contracts": 0.001},
            )
            self.assertNotIn("[API_CONTRACTS]", capped.get("digest", ""))

    def test_section_budgets_skip_trace_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            paths = ["src/entry.ts", "src/service.ts"]
            ir = self._basic_ir(repo, paths)
            ir["files"][file_id("src/entry.ts")]["role"] = "entrypoint"
            ir["edges"]["file_dep"] = [
                {"from": file_id("src/entry.ts"), "to": file_id("src/service.ts")}
            ]
            baseline = digest(
                ir,
                1200,
                static_traces=True,
                include_routes=False,
                compress_paths=False,
                trace_depth=2,
                trace_max=4,
            )
            self.assertIn("[TRACE_GRAPH]", baseline.get("digest", ""))
            capped = digest(
                ir,
                1200,
                static_traces=True,
                include_routes=False,
                compress_paths=False,
                trace_depth=2,
                trace_max=4,
                section_budgets={"trace_graph": 0.001},
            )
            self.assertNotIn("[TRACE_GRAPH]", capped.get("digest", ""))

    def test_bucket_quotas_keep_entrypoints_and_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            paths = [
                "src/entry.ts",
                "src/routes/user.ts",
                "src/high1.ts",
                "src/high2.ts",
                "src/other.ts",
            ]
            ir = self._basic_ir(repo, paths)
            ir["files"][file_id("src/entry.ts")]["role"] = "entrypoint"
            ir["files"][file_id("src/high1.ts")]["score"] = 100.0
            ir["files"][file_id("src/high2.ts")]["score"] = 99.0
            result = digest(
                ir,
                2000,
                include_routes=True,
                routes_mode="heuristic",
                max_files=2,
                compress_paths=False,
            )
            included = set(result.get("included", {}).get("files", []))
            self.assertIn(file_id("src/entry.ts"), included)
            self.assertIn(file_id("src/routes/user.ts"), included)

    def test_budget_governor_drops_diagram_before_flows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            paths = [f"src/mod{i}.ts" for i in range(1, 9)]
            paths[0] = "src/entry.ts"
            ir = self._basic_ir(repo, paths)
            ir["files"][file_id("src/entry.ts")]["role"] = "entrypoint"
            ir["edges"]["file_dep"] = [
                {"from": file_id(paths[i]), "to": file_id(paths[i + 1])}
                for i in range(len(paths) - 1)
            ]
            result = digest(
                ir,
                40,
                include_routes=False,
                compress_paths=False,
                diagram_mode="compact",
            )
            text = result.get("digest", "")
            self.assertIn("[FLOWS]", text)
            self.assertNotIn("[DIAGRAM]", text)

    def test_static_trace_drops_flows_when_tight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            paths = [
                "src/very/long/path/ui/entrypoint_component.tsx",
                "src/very/long/path/service/backend_service.ts",
                "src/very/long/path/data/repository_layer.ts",
                "src/very/long/path/data/db_model.ts",
            ]
            ir = self._basic_ir(repo, paths)
            ir["files"][file_id(paths[0])]["role"] = "entrypoint"
            ir["edges"]["file_dep"] = [
                {"from": file_id(paths[0]), "to": file_id(paths[1])},
                {"from": file_id(paths[1]), "to": file_id(paths[2])},
                {"from": file_id(paths[2]), "to": file_id(paths[3])},
            ]
            result = digest(
                ir,
                120,
                static_traces=True,
                include_routes=False,
                compress_paths=False,
                trace_depth=4,
                trace_max=6,
            )
            text = result.get("digest", "")
            self.assertIn("[STATIC_TRACES]", text)
            self.assertNotIn("FL1 ", text)

    def test_static_trace_start_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            paths = ["src/entry.ts", "src/service.ts", "src/data.ts"]
            ir = self._basic_ir(repo, paths)
            ir["files"][file_id("src/entry.ts")]["role"] = "entrypoint"
            ir["edges"]["file_dep"] = [
                {"from": file_id("src/entry.ts"), "to": file_id("src/service.ts")},
                {"from": file_id("src/service.ts"), "to": file_id("src/data.ts")},
            ]
            result = digest(
                ir,
                1200,
                static_traces=True,
                include_routes=False,
                compress_paths=False,
                trace_depth=3,
                trace_max=6,
                trace_start="src/service.ts",
            )
            aliases = result.get("aliases", {})
            start_alias = None
            for alias, node_id in aliases.items():
                if node_id == file_id("src/service.ts"):
                    start_alias = alias
                    break
            self.assertIsNotNone(start_alias)
            lines = result.get("digest", "").splitlines()
            trace_lines = [line for line in lines if line.startswith("ST")]
            self.assertTrue(trace_lines)
            tail = trace_lines[0].split("conf=", 1)[-1].strip()
            parts = tail.split()
            self.assertTrue(len(parts) >= 2)
            self.assertTrue(parts[1].startswith(str(start_alias)))

    def test_trace_graph_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            paths = [
                "src/entry.ts",
                "src/service.ts",
                "src/data.ts",
            ]
            ir = self._basic_ir(repo, paths)
            ir["files"][file_id("src/entry.ts")]["role"] = "entrypoint"
            ir["edges"]["file_dep"] = [
                {"from": file_id("src/entry.ts"), "to": file_id("src/service.ts")},
                {"from": file_id("src/service.ts"), "to": file_id("src/data.ts")},
            ]
            result = digest(
                ir,
                600,
                static_traces=True,
                include_routes=False,
                compress_paths=False,
                trace_depth=3,
                trace_max=6,
            )
            text = result.get("digest", "")
            self.assertIn("[TRACE_GRAPH]", text)
            self.assertIn("->", text)

    def test_entrypoints_exclude_middleware(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = "\n".join(
                [
                    "const app = new Hono()",
                    "app.use('/api', middleware)",
                    "app.get('/hello', (c) => {",
                    "  return c.json({ ok: true })",
                    "})",
                    "export const scheduled = () => {}",
                ]
            )
            write_file(root, "apps/server/src/index.ts", content)
            files = {
                "file:1": {"path": "apps/server/src/index.ts", "language": "ts"}
            }
            entry_lines, _entries, middleware = entrypoint_inventory_lines(
                files, prefix="", repo_root=root
            )
            joined = "\n".join(entry_lines)
            self.assertIn("get:/hello", joined)
            self.assertNotIn("use:/api", joined)
            self.assertTrue(any(item.get("label") == "use:/api" for item in middleware))

    def test_entry_details_preconditions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = "\n".join(
                [
                    "export async function listBookings() {",
                    "  await requireViewAccess()",
                    "  const rows = await db.query()",
                    "  return rows",
                    "}",
                ]
            )
            rel_path = "packages/api/src/services/bookings.ts"
            write_file(root, rel_path, content)
            entries = [
                {
                    "kind": "service",
                    "label": "listBookings",
                    "path": rel_path,
                    "line": 1,
                    "domain": "bookings",
                }
            ]
            detail_lines = entrypoint_detail_lines(
                entries, repo_root=root, prefix=""
            )
            detail_text = "\n".join(detail_lines)
            self.assertIn("pre=requireViewAccess", detail_text)

    def test_traceability_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = new_ir(repo, ["py"], {"src/app.py": "x", "docs/overview.md": "y"})
            fid = file_id("src/app.py")
            ir["files"][fid] = {
                "id": fid,
                "path": "src/app.py",
                "language": "py",
                "imports": [],
                "exports": [],
                "defines": [],
            }
            ir["meta"]["trace_links"] = [
                {
                    "doc_path": "docs/overview.md",
                    "target_id": fid,
                    "score": 1.0,
                    "reason": "name_overlap",
                }
            ]
            result = digest(ir, 2000, traceability=True, entity_graph=False)
            text = result.get("digest", "")
            self.assertIn("[TRACEABILITY]", text)
            self.assertIn("docs/overview.md", text)

    def test_entity_use_and_traceability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = new_ir(repo, ["ts"], {"src/user.ts": "x", "docs/overview.md": "y"})
            fid = file_id("src/user.ts")
            ir["files"][fid] = {
                "id": fid,
                "path": "src/user.ts",
                "language": "ts",
                "imports": [],
                "exports": [],
                "defines": [],
            }
            ent_id = "entity:User"
            ir["entities"][ent_id] = {
                "id": ent_id,
                "name": "User",
                "path": "prisma/schema.prisma",
                "fields": [{"name": "id", "type": "string"}],
            }
            ir["edges"]["entity_use"] = [{"from": fid, "to": ent_id, "kind": "uses"}]
            ir["meta"]["trace_links"] = [
                {
                    "doc_path": "docs/overview.md",
                    "target_id": fid,
                    "score": 1.0,
                    "reason": "name_overlap",
                }
            ]
            result = digest(ir, 2000, traceability=True, entity_graph=True)
            text = result.get("digest", "")
            self.assertIn("[ENTITIES]", text)
            self.assertIn("uses=F1", text)
            self.assertIn("[TRACEABILITY]", text)

    def test_flow_symbols(self) -> None:
        files = {
            "file:a.ts": {"path": "a.ts"},
            "file:b.ts": {"path": "b.ts"},
        }
        file_alias = {"file:a.ts": "F1", "file:b.ts": "F2"}
        edges = [{"from": "file:a.ts", "to": "file:b.ts", "weight": 1}]
        symbol_pairs = {("file:a.ts", "file:b.ts"): ("handleLogin", "validateCredentials")}
        lines = flow_lines(
            files=files,
            edges=edges,
            file_alias=file_alias,
            entrypoints=["file:a.ts"],
            route_entries=[],
            prefix="",
            flow_symbols=True,
            symbol_pairs=symbol_pairs,
            max_flows=1,
            max_depth=2,
        )
        joined = "\n".join(lines)
        self.assertIn("F1:handleLogin", joined)
        self.assertIn("F2:validateCredentials", joined)

    def test_call_chains_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            paths = ["src/main.ts", "src/service.ts", "src/repo.ts"]
            ir = self._basic_ir(repo, paths)
            main_fid = file_id("src/main.ts")
            service_fid = file_id("src/service.ts")
            repo_fid = file_id("src/repo.ts")

            ir["files"][main_fid]["role"] = "entrypoint"

            main_sid = "sym:src/main.ts#L1:handleRequest"
            service_sid = "sym:src/service.ts#L1:loadUser"
            repo_sid = "sym:src/repo.ts#L1:findById"

            ir["symbols"][main_sid] = {
                "id": main_sid,
                "name": "handleRequest",
                "kind": "function",
                "signature": "handleRequest()",
                "defined_in": {"path": "src/main.ts", "line": 1},
                "visibility": "public",
                "score": 2.0,
            }
            ir["symbols"][service_sid] = {
                "id": service_sid,
                "name": "loadUser",
                "kind": "function",
                "signature": "loadUser()",
                "defined_in": {"path": "src/service.ts", "line": 1},
                "visibility": "internal",
                "score": 1.5,
            }
            ir["symbols"][repo_sid] = {
                "id": repo_sid,
                "name": "findById",
                "kind": "function",
                "signature": "findById()",
                "defined_in": {"path": "src/repo.ts", "line": 1},
                "visibility": "internal",
                "score": 1.0,
            }

            ir["files"][main_fid]["exports"] = [main_sid]
            ir["files"][main_fid]["defines"] = [main_sid]
            ir["files"][service_fid]["exports"] = [service_sid]
            ir["files"][service_fid]["defines"] = [service_sid]
            ir["files"][repo_fid]["exports"] = [repo_sid]
            ir["files"][repo_fid]["defines"] = [repo_sid]

            ir["edges"]["symbol_ref"] = [
                {"from": main_sid, "to": service_sid, "kind": "call", "provenance": "codeql"},
                {"from": service_sid, "to": repo_sid, "kind": "call", "provenance": "codeql"},
            ]

            result = digest(
                ir,
                2000,
                include_routes=False,
                compress_paths=False,
                call_chains=True,
            )
            text = result.get("digest", "")
            self.assertIn("[CALL_CHAINS]", text)
            self.assertIn("CC1 depth=2", text)

    def test_call_chains_none_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = self._basic_ir(repo, ["src/main.ts", "src/service.ts"])
            ir["files"][file_id("src/main.ts")]["role"] = "entrypoint"
            result = digest(
                ir,
                1200,
                include_routes=False,
                compress_paths=False,
                call_chains=True,
            )
            text = result.get("digest", "")
            self.assertIn("[CALL_CHAINS]", text)
            self.assertIn("none_detected", text)

    def test_call_chains_expanded_scope_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            paths = ["src/main.ts", "src/service.ts"]
            ir = self._basic_ir(repo, paths)
            main_fid = file_id("src/main.ts")
            service_fid = file_id("src/service.ts")
            ir["files"][main_fid]["role"] = "entrypoint"

            main_sid = "sym:src/main.ts#L1:mainEntry"
            service_sid = "sym:src/service.ts#L1:doWork"
            ir["symbols"][main_sid] = {
                "id": main_sid,
                "name": "mainEntry",
                "kind": "function",
                "signature": "mainEntry()",
                "defined_in": {"path": "src/main.ts", "line": 1},
                "visibility": "public",
                "score": 1.0,
            }
            ir["symbols"][service_sid] = {
                "id": service_sid,
                "name": "doWork",
                "kind": "function",
                "signature": "doWork()",
                "defined_in": {"path": "src/service.ts", "line": 1},
                "visibility": "internal",
                "score": 1.0,
            }
            ir["files"][main_fid]["exports"] = [main_sid]
            ir["files"][main_fid]["defines"] = [main_sid]
            ir["files"][service_fid]["exports"] = [service_sid]
            ir["files"][service_fid]["defines"] = [service_sid]
            ir["edges"]["symbol_ref"] = [
                {"from": main_sid, "to": service_sid, "kind": "call", "provenance": "codeql"}
            ]

            # Force selected files to a single alias-bearing file so fallback is required.
            result = digest(
                ir,
                1200,
                include_routes=False,
                compress_paths=False,
                call_chains=True,
                max_files=1,
            )
            text = result.get("digest", "")
            self.assertIn("[CALL_CHAINS]", text)
            self.assertIn("scope=expanded", text)
            self.assertIn("service.ts", text)

    def test_focus_depth(self) -> None:
        ir = {
            "edges": {
                "file_dep": [
                    {"from": "file:a.ts", "to": "file:b.ts"},
                    {"from": "file:b.ts", "to": "file:c.ts"},
                ]
            }
        }
        focus = {"file:a.ts"}
        depth1 = expand_focus_neighbors(ir, focus, depth=1)
        depth2 = expand_focus_neighbors(ir, focus, depth=2)
        depth3 = expand_focus_neighbors(ir, focus, depth=3)
        self.assertEqual(depth1, {"file:a.ts"})
        self.assertEqual(depth2, {"file:a.ts", "file:b.ts"})
        self.assertEqual(depth3, {"file:a.ts", "file:b.ts", "file:c.ts"})

    def test_entity_fields_types(self) -> None:
        entities = [
            {
                "name": "User",
                "path": "models/user.ts",
                "fields": [
                    {"name": "id", "type": "string"},
                    {"name": "email", "type": "string"},
                    {"name": "teamId", "type": "string"},
                ],
                "relations": ["Team.id"],
            }
        ]
        lines, _aliases = entity_section_lines(entities, prefix="", file_alias={})
        joined = "\n".join(lines)
        self.assertIn("id:string(pk)", joined)
        self.assertIn("teamId:string", joined)
        self.assertIn("rels=Team.id", joined)

    def test_architecture_lines(self) -> None:
        arch = {
            "monorepo": {"type": "pnpm", "packages": 3, "markers": ["pnpm-workspace.yaml"]},
            "frameworks": ["nextjs"],
            "orms": ["prisma"],
            "layout": "clean-architecture",
            "layout_hints": ["domain", "infra"],
        }
        overview = architecture_overview_line(arch)
        self.assertIn("pnpm monorepo", overview or "")
        lines = architecture_section_lines(arch)
        joined = "\n".join(lines)
        self.assertIn("[ARCHITECTURE]", joined)
        self.assertIn("frameworks=nextjs", joined)
        self.assertIn("orms=prisma", joined)

    def test_type_hierarchy_lines(self) -> None:
        ir = {
            "symbols": {
                "sym:a": {"name": "User"},
                "sym:b": {"name": "BaseModel"},
            },
            "edges": {
                "type_ref": [
                    {"from": "sym:a", "to": "sym:b", "kind": "extends", "from_path": "models/user.ts"}
                ]
            },
        }
        lines = type_hierarchy_lines(ir, prefix="", file_alias={})
        joined = "\n".join(lines)
        self.assertIn("[TYPE_HIERARCHY]", joined)
        self.assertIn("User -extends-> BaseModel", joined)

    def test_semantic_comprehension_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            paths = ["src/main.ts", "src/retry.ts", "tests/main.test.ts"]
            ir = self._basic_ir(repo, paths)
            main_fid = file_id("src/main.ts")
            retry_fid = file_id("src/retry.ts")
            test_fid = file_id("tests/main.test.ts")
            ir["files"][main_fid]["role"] = "entrypoint"
            ir["files"][test_fid]["role"] = "test"

            retry_sid = "sym:src/retry.ts#L10:withRetry"
            ir["symbols"][retry_sid] = {
                "id": retry_sid,
                "name": "withRetry",
                "kind": "function",
                "signature": "withRetry(exponentialBackoff)",
                "doc_1l": "Retries with exponential backoff",
                "defined_in": {"path": "src/retry.ts", "line": 10},
                "visibility": "public",
                "score": 1.5,
            }
            ir["files"][retry_fid]["exports"] = [retry_sid]
            ir["files"][retry_fid]["defines"] = [retry_sid]

            ir["meta"]["test_mapping"] = [
                {
                    "test": "tests/main.test.ts",
                    "targets": ["src/main.ts"],
                    "symbols": ["main"],
                }
            ]
            ir["meta"]["test_summary"] = {
                "orphan_tests": ["tests/orphan.test.ts"],
                "untested_entrypoints": ["src/main.ts"],
            }
            ir["meta"]["quality"] = {
                "coupling": {
                    "fan_in": [{"id": main_fid, "count": 4}],
                }
            }

            result = digest(
                ir,
                2400,
                include_routes=False,
                compress_paths=False,
            )
            text = result.get("digest", "")
            self.assertIn("[CAPABILITIES]", text)
            self.assertIn("retry_backoff", text)
            self.assertIn("retry.ts:10", text)
            self.assertIn("[CAPABILITY_GAPS]", text)
            self.assertIn("[TEST_CONFIDENCE]", text)
            self.assertIn("[CHANGE_RISK]", text)

    def test_confidence_gate_passes_for_high_confidence_non_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = self._basic_ir(repo, ["src/main.ts"])
            ir["meta"]["quality"] = {
                "ast_edge_confidence": "high",
                "codeql_mode": "off",
                "codeql_ran": False,
            }
            result = digest(ir, 3000, include_routes=False, compress_paths=False)
            text = result.get("digest", "")
            gate = result.get("confidence_gate", {})
            self.assertIn("[CONFIDENCE_GATE]", text)
            self.assertIn("status=pass", text)
            self.assertEqual(gate.get("status"), "pass")

    def test_confidence_gate_requires_codeql_on_for_medium_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = self._basic_ir(repo, ["src/main.ts"])
            ir["meta"]["quality"] = {
                "ast_edge_confidence": "medium",
                "codeql_mode": "auto",
                "codeql_ran": False,
            }
            ir["meta"]["codeql"] = {"status": "skipped"}
            first = digest(ir, 3000, include_routes=False, compress_paths=False)
            first_text = first.get("digest", "")
            self.assertIn("status=fail", first_text)
            self.assertIn("rerun with --codeql on", first_text)

            ir["meta"]["quality"]["codeql_mode"] = "on"
            ir["meta"]["quality"]["codeql_ran"] = True
            ir["meta"]["codeql"]["status"] = "ok"
            second = digest(ir, 3000, include_routes=False, compress_paths=False)
            second_text = second.get("digest", "")
            self.assertIn("status=pass", second_text)
            self.assertEqual(second.get("confidence_gate", {}).get("status"), "pass")

    def test_confidence_gate_fails_when_target_stack_marked_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            ir = self._basic_ir(repo, ["src/main.ts"])
            ir["meta"]["quality"] = {
                "ast_edge_confidence": "high",
                "codeql_mode": "off",
                "codeql_ran": False,
            }
            ir["meta"]["unsupported_languages"] = {"python": 2}
            result = digest(ir, 3000, include_routes=False, compress_paths=False)
            text = result.get("digest", "")
            gate = result.get("confidence_gate", {})
            self.assertIn("status=fail", text)
            self.assertIn("unsupported_hits=python", text)
            self.assertEqual(gate.get("status"), "fail")


if __name__ == "__main__":
    unittest.main()
