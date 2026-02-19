from __future__ import annotations

import ast
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, TYPE_CHECKING

from utils import ToolState, run_cmd, run_cmd_logged

if TYPE_CHECKING:
    from . import IndexOptions


def ensure_codeql_packs(repo: Path, warnings: List[str], tools: ToolState) -> None:
    cmd = [
        "codeql",
        "pack",
        "download",
        "codeql/javascript-queries",
        "codeql/javascript-all",
        "codeql/python-queries",
        "codeql/python-all",
    ]
    result = run_cmd(cmd, cwd=repo, warnings=warnings, tools=tools, capture=True)
    if result and result.returncode != 0:
        warnings.append("CodeQL pack download failed; queries may not run")


def write_codeql_queries(out_dir: Path) -> Dict[str, Path]:
    queries_dir = out_dir / "codeql" / "queries"
    js_dir = queries_dir / "js"
    py_dir = queries_dir / "py"
    js_dir.mkdir(parents=True, exist_ok=True)
    py_dir.mkdir(parents=True, exist_ok=True)

    (js_dir / "qlpack.yml").write_text(
        "\n".join(
            [
                "name: codebase-map-js-queries",
                "version: 0.0.0",
                "dependencies:",
                "  codeql/javascript-all: \"*\"",
            ]
        ),
        encoding="utf-8",
    )
    (py_dir / "qlpack.yml").write_text(
        "\n".join(
            [
                "name: codebase-map-py-queries",
                "version: 0.0.0",
                "dependencies:",
                "  codeql/python-all: \"*\"",
            ]
        ),
        encoding="utf-8",
    )

    js_call = js_dir / "js_callgraph.ql"
    js_call.write_text(
        "\n".join(
            [
                "/**",
                " * @name Call graph edges",
                " * @id js/meta/callgraph-edges",
                " */",
                "import javascript",
                "private import semmle.javascript.dataflow.internal.FlowSteps as FlowSteps",
                "",
                "predicate isProjectFile(File f) {",
                "  not f.getRelativePath().matches(\"%/node_modules/%\") and",
                "  not f.getRelativePath().matches(\"%/dist/%\") and",
                "  not f.getRelativePath().matches(\"%/build/%\") and",
                "  not f.getRelativePath().matches(\"%/.next/%\")",
                "}",
                "",
                "from DataFlow::InvokeNode invoke, Function callee, Function caller",
                "where",
                "  FlowSteps::calls(invoke, callee) and",
                "  caller = invoke.getInvokeExpr().getEnclosingFunction() and",
                "  isProjectFile(caller.getFile()) and",
                "  isProjectFile(callee.getFile())",
                "select",
                "  caller.getFile().getRelativePath(),",
                "  caller.getLocation().getStartLine(),",
                "  caller.getName(),",
                "  callee.getFile().getRelativePath(),",
                "  callee.getLocation().getStartLine(),",
                "  callee.getName()",
            ]
        ),
        encoding="utf-8",
    )

    js_flow = js_dir / "js_dataflow.ql"
    js_flow.write_text(
        "\n".join(
            [
                "/**",
                " * @name Local dataflow edges",
                " * @id js/meta/dataflow-edges",
                " */",
                "import javascript",
                "",
                "predicate isProjectFile(File f) {",
                "  not f.getRelativePath().matches(\"%/node_modules/%\") and",
                "  not f.getRelativePath().matches(\"%/dist/%\") and",
                "  not f.getRelativePath().matches(\"%/build/%\") and",
                "  not f.getRelativePath().matches(\"%/.next/%\")",
                "}",
                "",
                "from DataFlow::Node pred, DataFlow::Node succ",
                "where",
                "  DataFlow::localFlowStep(pred, succ) and",
                "  exists(pred.getLocation()) and",
                "  exists(succ.getLocation()) and",
                "  isProjectFile(pred.getLocation().getFile()) and",
                "  isProjectFile(succ.getLocation().getFile())",
                "select",
                "  pred.getLocation().getFile().getRelativePath(),",
                "  pred.getLocation().getStartLine(),",
                "  succ.getLocation().getFile().getRelativePath(),",
                "  succ.getLocation().getStartLine()",
            ]
        ),
        encoding="utf-8",
    )

    py_call = py_dir / "py_callgraph.ql"
    py_call.write_text(
        "\n".join(
            [
                "/**",
                " * @name Call graph edges",
                " * @id py/meta/callgraph-edges",
                " */",
                "import python",
                "import semmle.python.dataflow.new.internal.DataFlowDispatch",
                "",
                "predicate isProjectFile(File f) {",
                "  not f.getRelativePath().matches(\"%/site-packages/%\") and",
                "  not f.getRelativePath().matches(\"%/.venv/%\") and",
                "  not f.getRelativePath().matches(\"%/venv/%\") and",
                "  not f.getRelativePath().matches(\"%/dist/%\") and",
                "  not f.getRelativePath().matches(\"%/build/%\")",
                "}",
                "",
                "predicate callsByName(Call call, Function callee) {",
                "  exists(Name target |",
                "    call.getFunc() = target and",
                "    callee.getName() = target.getId()",
                "  )",
                "  or",
                "  exists(Attribute target |",
                "    call.getFunc() = target and",
                "    callee.getName() = target.getName()",
                "  )",
                "}",
                "",
                "from Function caller, Function callee, Call call",
                "where",
                "  call.getScope() = caller and",
                "  callsByName(call, callee) and",
                "  isProjectFile(caller.getLocation().getFile()) and",
                "  isProjectFile(callee.getLocation().getFile())",
                "select",
                "  caller.getLocation().getFile().getRelativePath(),",
                "  caller.getLocation().getStartLine(),",
                "  caller.getQualifiedName(),",
                "  callee.getLocation().getFile().getRelativePath(),",
                "  callee.getLocation().getStartLine(),",
                "  callee.getQualifiedName()",
            ]
        ),
        encoding="utf-8",
    )

    py_flow = py_dir / "py_dataflow.ql"
    py_flow.write_text(
        "\n".join(
            [
                "/**",
                " * @name Local dataflow edges",
                " * @id py/meta/dataflow-edges",
                " */",
                "import python",
                "import semmle.python.dataflow.new.DataFlow",
                "",
                "predicate isProjectFile(File f) {",
                "  not f.getRelativePath().matches(\"%/site-packages/%\") and",
                "  not f.getRelativePath().matches(\"%/.venv/%\") and",
                "  not f.getRelativePath().matches(\"%/venv/%\") and",
                "  not f.getRelativePath().matches(\"%/dist/%\") and",
                "  not f.getRelativePath().matches(\"%/build/%\")",
                "}",
                "",
                "from Assign assign, Expr fromExpr, Expr toExpr",
                "where",
                "  fromExpr = assign.getValue() and",
                "  toExpr = assign.getATarget() and",
                "  exists(fromExpr.getLocation()) and",
                "  exists(toExpr.getLocation()) and",
                "  isProjectFile(fromExpr.getLocation().getFile()) and",
                "  isProjectFile(toExpr.getLocation().getFile())",
                "select",
                "  fromExpr.getLocation().getFile().getRelativePath(),",
                "  fromExpr.getLocation().getStartLine(),",
                "  toExpr.getLocation().getFile().getRelativePath(),",
                "  toExpr.getLocation().getStartLine()",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "js_call": js_call,
        "js_flow": js_flow,
        "py_call": py_call,
        "py_flow": py_flow,
        "js_root": js_dir,
        "py_root": py_dir,
    }


def parse_codeql_callgraph(path: Path) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    if not path.exists():
        return edges
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 6:
                continue
            caller_path, caller_line, caller_name, callee_path, callee_line, callee_name = row[:6]
            if not caller_name or not callee_name:
                continue
            try:
                caller_line_i = int(caller_line)
            except ValueError:
                caller_line_i = 0
            try:
                callee_line_i = int(callee_line)
            except ValueError:
                callee_line_i = 0
            edges.append(
                {
                    "caller_path": caller_path,
                    "caller_line": caller_line_i,
                    "caller_name": caller_name,
                    "callee_path": callee_path,
                    "callee_line": callee_line_i,
                    "callee_name": callee_name,
                }
            )
    return edges


def parse_codeql_dataflow(path: Path) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    if not path.exists():
        return edges
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            from_path, from_line, to_path, to_line = row[:4]
            try:
                from_line_i = int(from_line)
            except ValueError:
                from_line_i = 0
            try:
                to_line_i = int(to_line)
            except ValueError:
                to_line_i = 0
            edges.append(
                {
                    "from_path": from_path,
                    "from_line": from_line_i,
                    "to_path": to_path,
                    "to_line": to_line_i,
                }
            )
    return edges


def count_codeql_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _iter_python_files(repo: Path, max_files: int = 5000) -> List[Tuple[str, Path]]:
    skip_dirs = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "codeql",
    }
    files: List[Tuple[str, Path]] = []
    for root, dirs, names in os.walk(repo):
        dirs[:] = [name for name in dirs if name not in skip_dirs and not name.startswith(".beads")]
        for name in names:
            if not name.endswith(".py"):
                continue
            full_path = Path(root) / name
            try:
                rel_path = full_path.relative_to(repo).as_posix()
            except ValueError:
                continue
            if "/.tmp-codebase-map-" in rel_path or rel_path.startswith(".tmp-codebase-map-"):
                continue
            if rel_path.startswith(".code-state/"):
                continue
            files.append((rel_path, full_path))
            if len(files) >= max_files:
                return files
    return files


def build_python_ast_fallback_edges(
    repo: Path,
    *,
    max_files: int = 5000,
    max_call_edges: int = 20000,
    max_flow_edges: int = 30000,
) -> Dict[str, List[Dict[str, Any]]]:
    function_defs: Dict[str, List[Tuple[str, int, str]]] = {}
    call_sites: List[Tuple[str, int, str, str]] = []
    flow_pairs: Set[Tuple[str, int, str, int]] = set()

    class Analyzer(ast.NodeVisitor):
        def __init__(self, rel_path: str) -> None:
            self.rel_path = rel_path
            self.scope_stack: List[Tuple[str, int]] = []

        def _qual_name_for(self, name: str) -> str:
            if not self.scope_stack:
                return name
            return ".".join([part for part, _ in self.scope_stack] + [name])

        def _current_scope(self, fallback_line: int) -> Tuple[str, int]:
            if not self.scope_stack:
                return ("<module>", max(1, fallback_line))
            scope_name = ".".join(part for part, _ in self.scope_stack)
            scope_line = self.scope_stack[-1][1]
            return (scope_name, max(1, scope_line))

        def _record_function(self, node: ast.AST) -> None:
            if not hasattr(node, "name"):
                return
            name = str(getattr(node, "name", "") or "")
            if not name:
                return
            line = int(getattr(node, "lineno", 0) or 0)
            qual_name = self._qual_name_for(name)
            function_defs.setdefault(name, []).append((self.rel_path, line, qual_name))
            self.scope_stack.append((name, line))
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record_function(node)

        def visit_Call(self, node: ast.Call) -> None:
            callee_name = ""
            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee_name = node.func.attr
            if callee_name:
                call_line = int(getattr(node, "lineno", 0) or 0)
                caller_name, caller_line = self._current_scope(call_line)
                call_sites.append((self.rel_path, caller_line, caller_name, callee_name))
            self.generic_visit(node)

        def _add_flow_pair(self, value_node: ast.AST, target_node: ast.AST, stmt_line: int) -> None:
            from_line = int(getattr(value_node, "lineno", 0) or stmt_line or 0)
            to_line = int(getattr(target_node, "lineno", 0) or stmt_line or 0)
            if from_line <= 0 or to_line <= 0:
                return
            flow_pairs.add((self.rel_path, from_line, self.rel_path, to_line))

        def visit_Assign(self, node: ast.Assign) -> None:
            stmt_line = int(getattr(node, "lineno", 0) or 0)
            for target in node.targets:
                self._add_flow_pair(node.value, target, stmt_line)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            stmt_line = int(getattr(node, "lineno", 0) or 0)
            if node.value is not None:
                self._add_flow_pair(node.value, node.target, stmt_line)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            stmt_line = int(getattr(node, "lineno", 0) or 0)
            self._add_flow_pair(node.value, node.target, stmt_line)
            self.generic_visit(node)

    for rel_path, full_path in _iter_python_files(repo, max_files=max_files):
        try:
            source = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=rel_path)
        except (SyntaxError, ValueError):
            continue
        Analyzer(rel_path).visit(tree)

    call_edges: List[Dict[str, Any]] = []
    seen_call_edges: Set[Tuple[str, int, str, str, int, str]] = set()
    for caller_path, caller_line, caller_name, callee_name in call_sites:
        candidates = function_defs.get(callee_name, [])
        if not candidates:
            continue
        same_file = [entry for entry in candidates if entry[0] == caller_path]
        selected = same_file if same_file else candidates
        for callee_path, callee_line, callee_qual_name in selected[:3]:
            edge_key = (
                caller_path,
                caller_line,
                caller_name,
                callee_path,
                callee_line,
                callee_qual_name,
            )
            if edge_key in seen_call_edges:
                continue
            seen_call_edges.add(edge_key)
            call_edges.append(
                {
                    "caller_path": caller_path,
                    "caller_line": caller_line,
                    "caller_name": caller_name,
                    "callee_path": callee_path,
                    "callee_line": callee_line,
                    "callee_name": callee_qual_name,
                    "provenance": "codeql-fallback",
                }
            )
            if len(call_edges) >= max_call_edges:
                break
        if len(call_edges) >= max_call_edges:
            break

    dataflow_edges: List[Dict[str, Any]] = []
    for from_path, from_line, to_path, to_line in sorted(flow_pairs):
        dataflow_edges.append(
            {
                "from_path": from_path,
                "from_line": from_line,
                "to_path": to_path,
                "to_line": to_line,
                "provenance": "codeql-fallback",
            }
        )
        if len(dataflow_edges) >= max_flow_edges:
            break

    return {"call_edges": call_edges, "dataflow_edges": dataflow_edges}


def run_codeql(
    repo: Path,
    out_dir: Path,
    has_js: bool,
    has_py: bool,
    options: "IndexOptions",
    warnings: List[str],
    tools: ToolState,
    *,
    python_fallback: bool = True,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "call_edges": [],
        "dataflow_edges": [],
        "counts": {},
        "artifacts": {},
        "ran": False,
    }
    ensure_codeql_packs(repo, warnings, tools)
    if options.codeql_db:
        options.codeql_db.mkdir(parents=True, exist_ok=True)

    codeql_dir = out_dir / "codeql"
    codeql_dir.mkdir(parents=True, exist_ok=True)
    queries = write_codeql_queries(out_dir)
    packages_dir = Path.home() / ".codeql" / "packages"
    search_paths: Dict[str, str] = {}
    if has_js:
        js_parts = [str(queries["js_root"])]
        if packages_dir.exists():
            js_parts.append(str(packages_dir))
        search_paths["js"] = os.pathsep.join(js_parts)
        js_pack_rc = run_cmd_logged(
            ["codeql", "pack", "install"],
            cwd=queries["js_root"],
            log_path=codeql_dir / "pack_install_js.log",
            warnings=warnings,
            tools=tools,
        )
        if js_pack_rc != 0:
            warnings.append("CodeQL JS pack install failed; see codeql/pack_install_js.log")
    if has_py:
        py_parts = [str(queries["py_root"])]
        if packages_dir.exists():
            py_parts.append(str(packages_dir))
        search_paths["py"] = os.pathsep.join(py_parts)
        py_pack_rc = run_cmd_logged(
            ["codeql", "pack", "install"],
            cwd=queries["py_root"],
            log_path=codeql_dir / "pack_install_py.log",
            warnings=warnings,
            tools=tools,
        )
        if py_pack_rc != 0:
            warnings.append("CodeQL PY pack install failed; see codeql/pack_install_py.log")

    js_db = None
    py_db = None
    if has_js:
        db_name = "codeql-js-db"
        js_db = (options.codeql_db / db_name) if options.codeql_db else (codeql_dir / db_name)
        if js_db.exists() and not options.reuse_codeql_db:
            for path in (js_db,):
                if path.is_dir():
                    for child in path.rglob("*"):
                        if child.is_file():
                            child.unlink(missing_ok=True)
        if not js_db.exists() or not options.reuse_codeql_db:
            cmd = [
                "codeql",
                "database",
                "create",
                str(js_db),
                "--language=javascript",
                f"--source-root={repo.as_posix()}",
            ]
            rc = run_cmd_logged(
                cmd,
                cwd=repo,
                log_path=codeql_dir / "db_create_js.log",
                warnings=warnings,
                tools=tools,
            )
            if rc != 0:
                warnings.append("CodeQL JS DB create failed; see codeql/db_create_js.log")
                js_db = None

    if has_py:
        db_name = "codeql-py-db"
        py_db = (options.codeql_db / db_name) if options.codeql_db else (codeql_dir / db_name)
        if py_db.exists() and not options.reuse_codeql_db:
            for path in (py_db,):
                if path.is_dir():
                    for child in path.rglob("*"):
                        if child.is_file():
                            child.unlink(missing_ok=True)
        if not py_db.exists() or not options.reuse_codeql_db:
            cmd = [
                "codeql",
                "database",
                "create",
                str(py_db),
                "--language=python",
                f"--source-root={repo.as_posix()}",
            ]
            rc = run_cmd_logged(
                cmd,
                cwd=repo,
                log_path=codeql_dir / "db_create_py.log",
                warnings=warnings,
                tools=tools,
            )
            if rc != 0:
                warnings.append("CodeQL PY DB create failed; see codeql/db_create_py.log")
                py_db = None

    if has_js and js_db:
        js_call_csv = codeql_dir / "js_callgraph.csv"
        js_call_bqrs = codeql_dir / "js_callgraph.bqrs"
        js_flow_csv = codeql_dir / "js_dataflow.csv"
        js_flow_bqrs = codeql_dir / "js_dataflow.bqrs"
        cmd = [
            "codeql",
            "query",
            "run",
            str(queries["js_call"]),
            "--database",
            str(js_db),
            "--output",
            str(js_call_bqrs),
        ]
        if "js" in search_paths:
            cmd.extend(["--search-path", search_paths["js"]])
        rc = run_cmd_logged(
            cmd,
            cwd=repo,
            log_path=codeql_dir / "run_js_call.log",
            warnings=warnings,
            tools=tools,
        )
        if rc == 0 and js_call_bqrs.exists():
            decode_cmd = [
                "codeql",
                "bqrs",
                "decode",
                str(js_call_bqrs),
                "--format=csv",
                "--no-titles",
                "--output",
                str(js_call_csv),
            ]
            decode_rc = run_cmd_logged(
                decode_cmd,
                cwd=repo,
                log_path=codeql_dir / "decode_js_call.log",
                warnings=warnings,
                tools=tools,
            )
            if decode_rc == 0 and js_call_csv.exists():
                result["call_edges"].extend(parse_codeql_callgraph(js_call_csv))
                result["counts"]["js_call_edges"] = count_codeql_rows(js_call_csv)
                result["artifacts"]["js_call"] = js_call_csv.as_posix()
            else:
                warnings.append("CodeQL JS call decode failed; see codeql/decode_js_call.log")
        else:
            warnings.append("CodeQL JS call graph failed; see codeql/run_js_call.log")
        cmd = [
            "codeql",
            "query",
            "run",
            str(queries["js_flow"]),
            "--database",
            str(js_db),
            "--output",
            str(js_flow_bqrs),
        ]
        if "js" in search_paths:
            cmd.extend(["--search-path", search_paths["js"]])
        rc = run_cmd_logged(
            cmd,
            cwd=repo,
            log_path=codeql_dir / "run_js_flow.log",
            warnings=warnings,
            tools=tools,
        )
        if rc == 0 and js_flow_bqrs.exists():
            decode_cmd = [
                "codeql",
                "bqrs",
                "decode",
                str(js_flow_bqrs),
                "--format=csv",
                "--no-titles",
                "--output",
                str(js_flow_csv),
            ]
            decode_rc = run_cmd_logged(
                decode_cmd,
                cwd=repo,
                log_path=codeql_dir / "decode_js_flow.log",
                warnings=warnings,
                tools=tools,
            )
            if decode_rc == 0 and js_flow_csv.exists():
                result["dataflow_edges"].extend(parse_codeql_dataflow(js_flow_csv))
                result["counts"]["js_dataflow_edges"] = count_codeql_rows(js_flow_csv)
                result["artifacts"]["js_flow"] = js_flow_csv.as_posix()
            else:
                warnings.append("CodeQL JS dataflow decode failed; see codeql/decode_js_flow.log")
        else:
            warnings.append("CodeQL JS dataflow failed; see codeql/run_js_flow.log")

    if has_py and py_db:
        py_call_csv = codeql_dir / "py_callgraph.csv"
        py_call_bqrs = codeql_dir / "py_callgraph.bqrs"
        py_flow_csv = codeql_dir / "py_dataflow.csv"
        py_flow_bqrs = codeql_dir / "py_dataflow.bqrs"
        cmd = [
            "codeql",
            "query",
            "run",
            str(queries["py_call"]),
            "--database",
            str(py_db),
            "--output",
            str(py_call_bqrs),
        ]
        if "py" in search_paths:
            cmd.extend(["--search-path", search_paths["py"]])
        rc = run_cmd_logged(
            cmd,
            cwd=repo,
            log_path=codeql_dir / "run_py_call.log",
            warnings=warnings,
            tools=tools,
        )
        if rc == 0 and py_call_bqrs.exists():
            decode_cmd = [
                "codeql",
                "bqrs",
                "decode",
                str(py_call_bqrs),
                "--format=csv",
                "--no-titles",
                "--output",
                str(py_call_csv),
            ]
            decode_rc = run_cmd_logged(
                decode_cmd,
                cwd=repo,
                log_path=codeql_dir / "decode_py_call.log",
                warnings=warnings,
                tools=tools,
            )
            if decode_rc == 0 and py_call_csv.exists():
                result["call_edges"].extend(parse_codeql_callgraph(py_call_csv))
                result["counts"]["py_call_edges"] = count_codeql_rows(py_call_csv)
                result["artifacts"]["py_call"] = py_call_csv.as_posix()
            else:
                warnings.append("CodeQL PY call decode failed; see codeql/decode_py_call.log")
        else:
            warnings.append("CodeQL PY call graph failed; see codeql/run_py_call.log")
        cmd = [
            "codeql",
            "query",
            "run",
            str(queries["py_flow"]),
            "--database",
            str(py_db),
            "--output",
            str(py_flow_bqrs),
        ]
        if "py" in search_paths:
            cmd.extend(["--search-path", search_paths["py"]])
        rc = run_cmd_logged(
            cmd,
            cwd=repo,
            log_path=codeql_dir / "run_py_flow.log",
            warnings=warnings,
            tools=tools,
        )
        if rc == 0 and py_flow_bqrs.exists():
            decode_cmd = [
                "codeql",
                "bqrs",
                "decode",
                str(py_flow_bqrs),
                "--format=csv",
                "--no-titles",
                "--output",
                str(py_flow_csv),
            ]
            decode_rc = run_cmd_logged(
                decode_cmd,
                cwd=repo,
                log_path=codeql_dir / "decode_py_flow.log",
                warnings=warnings,
                tools=tools,
            )
            if decode_rc == 0 and py_flow_csv.exists():
                result["dataflow_edges"].extend(parse_codeql_dataflow(py_flow_csv))
                result["counts"]["py_dataflow_edges"] = count_codeql_rows(py_flow_csv)
                result["artifacts"]["py_flow"] = py_flow_csv.as_posix()
            else:
                warnings.append("CodeQL PY dataflow decode failed; see codeql/decode_py_flow.log")
        else:
            warnings.append("CodeQL PY dataflow failed; see codeql/run_py_flow.log")

    py_call_count = int(result["counts"].get("py_call_edges", 0) or 0)
    py_flow_count = int(result["counts"].get("py_dataflow_edges", 0) or 0)
    needs_py_call_fallback = has_py and py_call_count == 0
    needs_py_flow_fallback = has_py and py_flow_count == 0
    if python_fallback and (needs_py_call_fallback or needs_py_flow_fallback):
        fallback_edges = build_python_ast_fallback_edges(repo)
        used_fallback = False
        if needs_py_call_fallback:
            fallback_calls = fallback_edges.get("call_edges", [])
            if isinstance(fallback_calls, list) and fallback_calls:
                result["call_edges"].extend(fallback_calls)
                result["counts"]["py_call_edges"] = len(fallback_calls)
                result["counts"]["py_call_edges_fallback"] = len(fallback_calls)
                result["artifacts"]["py_call_fallback"] = "python-ast"
                used_fallback = True
        if needs_py_flow_fallback:
            fallback_flow = fallback_edges.get("dataflow_edges", [])
            if isinstance(fallback_flow, list) and fallback_flow:
                result["dataflow_edges"].extend(fallback_flow)
                result["counts"]["py_dataflow_edges"] = len(fallback_flow)
                result["counts"]["py_dataflow_edges_fallback"] = len(fallback_flow)
                result["artifacts"]["py_flow_fallback"] = "python-ast"
                used_fallback = True
        if used_fallback:
            warnings.append(
                "CodeQL PY query coverage was empty; used Python AST fallback edges."
            )
    elif needs_py_call_fallback or needs_py_flow_fallback:
        warnings.append("CodeQL PY query coverage was empty; fallback disabled.")

    result["ran"] = bool(result["call_edges"] or result["dataflow_edges"])
    if js_db and not options.keep_codeql_db:
        for child in js_db.rglob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)
    if py_db and not options.keep_codeql_db:
        for child in py_db.rglob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)

    return result
