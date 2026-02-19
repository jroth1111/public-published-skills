# Module Map

- `../scripts/main.py`: CLI entry point; wires subcommands and stdout/JSON output.
- `../scripts/indexer/core.py`: repo discovery, AST extraction, CodeQL integration, and IR assembly.
- `../scripts/packer/digest_core.py`: digest/expand/find/deps views and token budgeting.
- `../scripts/ranker.py`: scoring for files and symbols.
- `../scripts/ir.py`: IR versioning helpers, ids, and serialization.
- `../scripts/utils.py`: tool execution, progress reporting, and version probes.
- `../scripts/_fs.py`: safe workspace writes and preview helpers.
- `../scripts/exporter.py`: graph export in JSON or GraphML formats.

These modules are intentionally narrow; shared data lives in the IR to keep cross-module coupling minimal.
