# code-intel

Local hybrid code search for your workspace, as a plugin for **Claude Code**,
**Cursor** and **GitHub Copilot** (CLI and VS Code). It is the code
intelligence from [Claudinio Code](https://github.com/claudin-io/claudinio-code),
extracted into an MCP server so any agent can use it:

- **tree-sitter symbols** for 77 languages — functions, classes, methods, types, imports, and the call relations between them
- **BM25 full-text** (SQLite FTS5) over symbol names, signatures, code bodies, docs and paths
- **embeddings** of every chunk: all-MiniLM-L6-v2 (23 MB) within minutes, then [EmbeddingGemma 2](#embedding-models) taking over in the background
- **hybrid ranking**: reciprocal-rank fusion of the two, so an exact identifier and a description of behaviour both work
- **images and audio** of the project, findable by what they show or sound like (EmbeddingGemma 2 again) and always by file name
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
| `index_status` | phase and progress, the embedding model search is using, how far the background upgrade has got, media counts; the first scan takes seconds, embeddings a few minutes in the background |
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
They run MiniLM (same model, same vectors, slower) and index media by file name only:
EmbeddingGemma 2 is published as ONNX graphs, which candle cannot load.

## Embedding models

Two models, and neither is waited for.

| | all-MiniLM-L6-v2 | EmbeddingGemma 2 |
|---|---|---|
| Role | the first index: code and docs searchable within minutes | the upgrade: the same code and docs again, in the background; and images and audio |
| Used by | every build | ONNX Runtime builds |
| Vectors | 384-d | 768-d |
| Download | 23 MB | 175 MB, **plus 109 MB if the workspace has images, plus 189 MB if it has audio** |
| Languages | English | 100+ |

**MiniLM first.** A workspace is scanned in seconds and embedded with MiniLM in minutes;
search is complete from then on. It is the model the index never goes without.

**EmbeddingGemma 2 in the background.** It ranks somewhat better and embeds some thirty times
slower, so it is given all the time it needs instead of being put in the way. Once MiniLM is
done the server downloads it, embeds every chunk a second time — into its own set of
vectors, next to MiniLM's — and, when that set covers the workspace, moves search over to it.
Nothing is asked of the agent or of you: `index_status` reports `upgrade.state` (`waiting`,
`loading`, `embedding`, `ready`) and `upgrade.searching`, and `embeddingModel` names the
model that ranked the last search.

What that costs, measured on the 15.7k chunks of
[Claudinio Code](https://github.com/claudin-io/claudinio-code):

| | all-MiniLM-L6-v2 | EmbeddingGemma 2 |
|---|---|---|
| Embedding the repository, GitHub-hosted Linux runner, 2 threads | 4 min (66 chunks/s) | 2 h 21 min (1.8 chunks/s) |
| The same, projected from 161 chunks of another repository on a laptop at 8 threads | — | about 40 min (6.6 chunks/s) |
| Encoding a query | 14 ms | 49 ms |
| Expected file ranked first / in the top 3 / in the top 15 (59 queries, in English) | 62% / 84% / 100% | 69% / 86% / 98% |
| Vectors on disk | 24 MB | 48 MB more |

The upgrade is built around that hour:

- it uses half the machine's physical cores (`CODE_INTEL_THREADS`), in the background, and stands aside
  whenever a first index or an edited file is waiting to be embedded with MiniLM; the model
  itself stays in memory for as long as the server runs;
- it records its progress file by file, so closing the editor halfway costs nothing — the
  next session carries on where this one stopped; and with two agents open on one repository,
  only one of the two servers does the work;
- once search has moved over, an edited file is re-embedded by both models: by MiniLM at
  once, by EmbeddingGemma 2 a moment later, and is found by its text in between;
- MiniLM's vectors are never dropped. If EmbeddingGemma 2 cannot be downloaded or loaded, or
  the build cannot run it, or its set has fallen far behind (a change of branch that rewrote
  half the repository), search uses MiniLM's and nothing else changes.

`CODE_INTEL_UPGRADE=0` turns the upgrade off: no second download, no background hour, text
search on MiniLM for good. `CODE_INTEL_MODEL=embeddinggemma2` is the opposite extreme —
EmbeddingGemma 2 for everything from the start, with search by meaning arriving only as fast
as that model embeds.

**Images and audio.** What to fetch for them is decided per workspace from the files it
actually contains (`.gitignore` respected): one with screenshots but no sounds never fetches
the audio encoder. Images are `png jpg jpeg webp gif bmp`; audio is `wav mp3 flac ogg` (first
30 s of a clip). SVG is indexed as code; video and AAC/M4A audio are not indexed. Up to 200
media files per workspace get a content vector — encoding one image is seconds of CPU — and
the rest stay findable by name. Media gets its vectors after the code has its own
(`index_status` → `media.state`: `loading`, `embedding`, `ready`); without the model, images
and audio are matched by file name instead of by content (`name-only`).

An index records which model wrote each kind of vector — the first text set, the upgrade set,
media. When one of them changes, that set is re-embedded in the background and the others are
left alone; vectors of two models are never compared.

> **How this is tested.** The loading, batching, media decoding and indexing paths are covered
> by tests that run through ONNX Runtime against stand-in models with the same contracts, and
> the image and audio preprocessing is checked against the reference implementation's output.
> What has to come from the real weights is measured by two things that download them:
> `cargo test -p claudinio-code-intel --test gemma2_e2e -- --ignored --nocapture` (does it rank
> code, images and audio sensibly at all) and the eval below. The two search thresholds
> specific to EmbeddingGemma 2 (`GEMMA2_MIN_COSINE_CANDIDATE`, `MEDIA_MIN_COSINE` in `db.rs`)
> are set from that eval's score distributions on one repository. The media threshold rests on
> 18 images, two descriptive queries and no real audio. The text one — which every search uses
> once the upgrade has taken over — has not had its effect on ranking swept yet: the eval that
> does it takes hours, and is run on request.

### Evaluating a model

```
cargo run --release --example semantic_eval -- /path/to/claudinio-code --sweep --report eval-report.txt
```

indexes the workspace twice — with MiniLM, which is what search runs on until the upgrade has
finished, then with EmbeddingGemma 2 for text, which is what it runs on afterwards — and
prints, for each: the rank of the expected file for 59 real queries (and
that 5 off-topic ones return nothing), the raw cosine distributions the vector gate is chosen
from, a sweep over the fusion gates, how the workspace's images score against code queries and
against queries that describe them, and indexing speed for text and for media separately. The query set
(`crates/code-intel/examples/semantic_eval_queries.json`) was written against the
[Claudinio Code](https://github.com/claudin-io/claudinio-code) repository; CI runs the same
eval on a pinned commit of it for every pull request and puts the report in the job summary.

## Configuration

| Variable / flag | Meaning |
|---|---|
| `CODE_INTEL_WORKSPACE`, `--workspace <dir>` | directory to index (default: client roots, then cwd) |
| `CODE_INTEL_CACHE_DIR`, `--cache-dir <dir>` | where indexes, the models and the binary live |
| `CODE_INTEL_EMBEDDINGS=0`, `--no-embeddings` | lexical only, no model download |
| `CODE_INTEL_MODEL` | `auto` (default: MiniLM first, EmbeddingGemma 2 taking over in the background, and for images and audio), `minilm` (MiniLM only; media by file name), or `embeddinggemma2` (EmbeddingGemma 2 for everything from the start; no fallback) |
| `CODE_INTEL_UPGRADE=0` | under `auto`, keep text search on MiniLM: no background upgrade |
| `CODE_INTEL_THREADS` | threads one model run may use (default: half the physical cores, at least 2, at most 8) |
| `CODE_INTEL_MEDIA=0` | do not index images and audio at all |
| `CODE_INTEL_MEDIA_MAX` | media files per workspace that get a content vector (default 200) |
| `CODE_INTEL_IMAGE_TOKENS` | detail per image: `70`, `140`, `280` (default), `560`, `1120` — fewer is faster |
| `CODE_INTEL_LOG` | tracing filter for stderr (`debug`, `info,ort=warn`, …) |
| `CODE_INTEL_BIN` | run this binary instead of the downloaded one |

`claudinio-code-intel index --workspace <dir>` builds the index and exits —
useful to warm a cache before an agent session. With `--upgrade` it also runs
the background upgrade to the end before exiting.

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
