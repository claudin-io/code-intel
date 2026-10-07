# code-intel

Local hybrid code search for your workspace, as a plugin for **Claude Code**,
**Cursor** and **GitHub Copilot** (CLI and VS Code). It is the code
intelligence from [Claudinio Code](https://github.com/claudin-io/claudinio-code),
extracted into an MCP server so any agent can use it:

- **tree-sitter symbols** for 77 languages — functions, classes, methods, types, imports, and the call relations between them
- **BM25 full-text** (SQLite FTS5) over symbol names, signatures, code bodies, docs and paths
- **embeddings** of every chunk — [EmbeddingGemma 2](#embedding-model) where the platform can run it, all-MiniLM-L6-v2 everywhere else
- **hybrid ranking**: reciprocal-rank fusion of the two, so an exact identifier and a description of behaviour both work
- **images and audio** of the project, findable by what they show or sound like (EmbeddingGemma 2) and always by file name
- a **file watcher** keeps the index current as you edit

Everything runs on your machine. The index and the model live in your OS
cache directory, never inside the repository, and no code is sent anywhere.

## Tools the agent gets

| Tool | Use |
|---|---|
| `semantic_search` | "where do we retry failed uploads?" — meaning **or** rare terms, across code and docs; matching images/audio come back under `media` |
| `code_search` | partial symbol names / signatures, full-text |
| `symbol_lookup` | exact symbol name |
| `file_outline` | every symbol in a file with line ranges — read this before the file |
| `find_callers` | who calls a symbol (from recorded call relations) |
| `index_status` | phase and progress, the embedding model in use, media counts; the first scan takes seconds, embeddings a few minutes in the background |
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
| `darwin-arm64` | Apple Silicon macOS |
| `darwin-x64` | Intel macOS (candle backend: ONNX Runtime ships no Intel Mac build) |
| `win32-x64`, `win32-arm64` | Windows 10+ |
| `win32-x64-baseline` | pre-Haswell Windows; set `CODE_INTEL_VARIANT=baseline` |

The `-baseline` and `darwin-x64` builds swap ONNX Runtime for the pure-Rust `candle` backend.
They run MiniLM (same model, same vectors as before, slower) and index media by file name only:
EmbeddingGemma 2 is published as ONNX graphs, which candle cannot load.

## Embedding model

| | EmbeddingGemma 2 | all-MiniLM-L6-v2 |
|---|---|---|
| Used by | ONNX Runtime builds (default) | candle builds; any build where the other fails to download or load |
| Understands | code and text in 100+ languages, images, audio | English text |
| Vectors | 768-d | 384-d |
| Download | 175 MB text model, **plus 109 MB only if the workspace has images, plus 189 MB only if it has audio** | 23 MB |

Which encoders to fetch is decided per workspace from the files it actually contains
(`.gitignore` respected): a text-only project downloads the text model and nothing else.
Images are `png jpg jpeg webp gif bmp`; audio is `wav mp3 flac ogg` (first 30 s of a clip).
SVG is indexed as code; video and AAC/M4A audio are not indexed. Up to 200 media files per workspace get a
content vector — encoding one image is seconds of CPU — and the rest stay findable by name.

If EmbeddingGemma 2 cannot be downloaded or loaded, the server logs why and carries on with
MiniLM. An index remembers which model built it; opening it with the other one re-embeds it
in the background rather than mixing vectors. `index_status` shows the model in use.

> **Status of this model.** The loading, batching, media decoding and indexing paths are
> covered by tests that run through ONNX Runtime against a stand-in model with the same
> contract, and the image and audio preprocessing is checked against the reference
> implementation's output. What has to come from the real weights is measured by two things
> that download them: `cargo test -p claudinio-code-intel --test gemma2_e2e -- --ignored --nocapture`
> (does it rank code, images and audio sensibly at all) and the eval below. Until the eval
> has been run, the two search thresholds for this model (`GEMMA2_MIN_COSINE_CANDIDATE`,
> `MEDIA_MIN_COSINE` in `db.rs`) are provisional — MiniLM's came out of a sweep over real
> queries, these have not yet.

### Evaluating a model

```
cargo run --release --example semantic_eval -- /path/to/claudinio-code --sweep --report eval-report.txt
```

indexes the workspace once per model and prints, for each: the rank of the expected file for
59 real queries (and that 5 off-topic ones return nothing), the raw cosine distributions the
vector gate is chosen from, a sweep over the fusion gates, how the workspace's images score
against code queries and against queries that describe them, and indexing speed. The query set
(`crates/code-intel/examples/semantic_eval_queries.json`) was written against the
[Claudinio Code](https://github.com/claudin-io/claudinio-code) repository; CI runs the same
eval on a pinned commit of it for every pull request and puts the report in the job summary.

## Configuration

| Variable / flag | Meaning |
|---|---|
| `CODE_INTEL_WORKSPACE`, `--workspace <dir>` | directory to index (default: client roots, then cwd) |
| `CODE_INTEL_CACHE_DIR`, `--cache-dir <dir>` | where indexes, the model and the binary live |
| `CODE_INTEL_EMBEDDINGS=0`, `--no-embeddings` | lexical only, no model download |
| `CODE_INTEL_MODEL` | `auto` (default: EmbeddingGemma 2, MiniLM as fallback), `minilm`, or `embeddinggemma2` (no fallback — for comparing the two) |
| `CODE_INTEL_THREADS` | threads one model run may use (default 2: indexing is a background job) |
| `CODE_INTEL_MEDIA=0` | do not index images and audio at all |
| `CODE_INTEL_MEDIA_MAX` | media files per workspace that get a content vector (default 200) |
| `CODE_INTEL_IMAGE_TOKENS` | detail per image: `70`, `140`, `280` (default), `560`, `1120` — fewer is faster |
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

MIT — same as Claudinio Code. The embedding models are
[onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX)
and [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
(both Apache-2.0), downloaded at first run and pinned by sha256.
