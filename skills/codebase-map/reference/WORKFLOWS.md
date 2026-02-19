# Workflows

## 1) Repo onboarding
```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced
```

## 2) API surface discovery
```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced --api-contracts
```

## 3) Data model understanding
```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced --entity-graph
```

## 4) Flow/path tracing (UI ↔ service ↔ DB)
```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced --static-traces
```

## 5) Targeted symbol query
```bash
python {baseDir}/scripts/main.py query --repo /path/to/repo --out codebase-map/my-repo --query auth --depth 2
```

## 6) Focused expansion
```bash
python {baseDir}/scripts/main.py expand --repo /path/to/repo --out codebase-map/my-repo --focus "checkout" --depth 2
```
