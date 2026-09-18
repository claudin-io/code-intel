# code-intel

Local hybrid code search for your workspace, as a plugin for **Claude Code**,
**Cursor** and **GitHub Copilot** (CLI and VS Code). It is the code
intelligence from [Claudinio Code](https://github.com/claudin-io/claudinio-code),
extracted into an MCP server so any agent can use it:

- **tree-sitter symbols** for 77 languages — functions, classes, methods, types, imports, and the call relations between them
- **BM25 full-text** (SQLite FTS5) over symbol names, signatures, code bodies, docs and paths
- **MiniLM embeddings** (all-MiniLM-L6-v2, ONNX, quantized, 23 MB) of every chunk
- **hybrid ranking**: reciprocal-rank fusion of the two, so an exact identifier and a description of behaviour both work
- a **file watcher** keeps the index current as you edit

Everything runs on your machine. The index and the model live in your OS
cache directory, never inside the repository, and no code is sent anywhere.

## Tools the agent gets

| Tool | Use |
|---|---|
| `semantic_search` | "where do we retry failed uploads?" — meaning **or** rare terms, across code and docs |
| `code_search` | partial symbol names / signatures, full-text |
| `symbol_lookup` | exact symbol name |
| `file_outline` | every symbol in a file with line ranges — read this before the file |
| `find_callers` | who calls a symbol (from recorded call relations) |
| `index_status` | phase and progress; the first scan takes seconds, embeddings a few minutes in the background |
| `open_workspace` | index another directory |

A `code-intel` skill ships with the plugin and teaches the agent when to
reach for which tool instead of grep.

## Install

The plugin is the same repository for all three hosts; each reads its own
manifest. Node ≥ 18 must be on `PATH` (all three hosts already need it).
On first run the launcher downloads the binary for your platform from the
matching [GitHub Release](https://github.com/claudin-io/code-intel/releases)
and verifies it against `SHA256SUMS`.

### Claude Code

```
/plugin marketplace add claudin-io/code-intel
/plugin install code-intel@claudinio
```

### GitHub Copilot CLI

```
copilot plugin marketplace add claudin-io/code-intel
copilot plugin install code-intel@claudinio
```

### VS Code (Copilot)

Add `claudin-io/code-intel` under **Settings → Chat › Plugins: Marketplaces**,
or run **Chat: Install Plugin From Source** and point it at this repository.

### Cursor

Cursor plugins install from the [Cursor marketplace](https://cursor.com/marketplace);
until this plugin is listed there, add the server by hand (below).

### Plain MCP (any client)

```json
{
  "mcpServers": {
    "code-intel": {
      "command": "node",
      "args": ["/path/to/code-intel/bin/launcher.mjs"]
    }
  }
}
```

`claude mcp add code-intel -- node /path/to/code-intel/bin/launcher.mjs`
does the same for Claude Code without the plugin. The server indexes the
directory it is started in (or the client's MCP roots); `--workspace <dir>`
and `CODE_INTEL_WORKSPACE` override that.

## Platforms

| Asset | Runs on |
|---|---|
| `linux-x64`, `linux-arm64` | glibc Linux |
| `linux-x64-baseline` | x86-64 CPUs without AVX2/BMI2 (picked automatically from `/proc/cpuinfo`) |
| `darwin-arm64`, `darwin-x64` | macOS |
| `win32-x64`, `win32-arm64` | Windows 10+ |
| `win32-x64-baseline` | pre-Haswell Windows; set `CODE_INTEL_VARIANT=baseline` |

The `-baseline` builds swap ONNX Runtime for the pure-Rust `candle` backend:
same model, same vectors, slower.

## Configuration

| Variable / flag | Meaning |
|---|---|
| `CODE_INTEL_WORKSPACE`, `--workspace <dir>` | directory to index (default: client roots, then cwd) |
| `CODE_INTEL_CACHE_DIR`, `--cache-dir <dir>` | where indexes, the model and the binary live |
| `CODE_INTEL_EMBEDDINGS=0`, `--no-embeddings` | lexical only, no model download |
| `CODE_INTEL_LOG` | tracing filter for stderr (`debug`, `info,ort=warn`, …) |
| `CODE_INTEL_BIN` | run this binary instead of the downloaded one |

`claudinio-code-intel index --workspace <dir>` builds the index and exits —
useful to warm a cache before an agent session.

## Build from source

```
cargo build --release -p claudinio-code-intel-mcp
# pre-AVX2 machines:
cargo build --release -p claudinio-code-intel-mcp --no-default-features --features embeddings-candle
```

`crates/code-intel` is the library (usable on its own); `crates/code-intel-mcp`
is the server. Tests: `cargo test --workspace` and `npm test`.

## License

MIT — same as Claudinio Code. The embedding model is
[Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
(Apache-2.0), downloaded at first run and pinned by sha256.
