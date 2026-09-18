---
name: code-intel
description: Navigate and search the workspace with the code-intel MCP tools (semantic_search, code_search, symbol_lookup, file_outline, find_callers) instead of grep/find when looking for where something is implemented, what a file contains, or who calls a symbol.
---

# Code intel — search the workspace by meaning, not by string

The `code-intel` MCP server keeps a local index of this workspace: every
symbol tree-sitter can extract (77 languages), a BM25 full-text index over
code bodies, docs and paths, and MiniLM embeddings of each chunk. Hybrid
search fuses the two. The index lives in the OS cache directory, never in the
repo, and nothing is sent anywhere.

## Pick the tool

| You want… | Use | Not |
|---|---|---|
| Code that *does* something ("where do we retry failed uploads?") | `semantic_search` | grep for guessed words |
| A rare term, error string or file name anywhere in code or docs | `semantic_search` | grep (it finds it too, but ranked) |
| A definition by partial name (`hourly budget`, `Reconciler`) | `code_search` | find/grep |
| A definition by exact name | `symbol_lookup` | grep |
| What a file contains before reading it | `file_outline` | reading the whole file |
| Who calls / references a function | `find_callers` | grep for the name |
| Whether the index is warm | `index_status` | guessing |

Order of preference for "where is X?": `semantic_search` → `code_search` →
grep as the fallback. Reach for grep when you need every textual occurrence
(a rename), a regex, or a file the index skips (generated bundles, >2 MB).

## How to query

- **English only.** The embedding model is English; translate the request
  first, even when the user wrote in another language.
- Describe the *behaviour* for `semantic_search`: "rate limit a client by
  hourly spend" beats "limit". Identifiers and file names work too — the BM25
  leg matches them exactly.
- `limit` defaults to 15/20; raise it for surveys, lower it for a quick check.
- Results carry `filePath`, `startLine`/`endLine` and, for the top hits, a
  `snippet`. Read the file at that range for the rest.

## Reading the results

`semantic_search` returns `{mode, note?, results}`.

- `mode: "hybrid"` — embeddings were used. `score` is relative confidence in
  (0, 1]; `matchType` says whether a hit came from both legs, only semantic
  or only lexical.
- `mode: "lexical-only"` — the model is still loading (first run downloads
  ~23 MB) or embeddings are still being generated. Keyword hits are still
  good; re-run later for the semantic ranking. `note` explains which.

## When the index is not ready

The first scan of a workspace takes seconds; embeddings run for a few minutes
in the background. A tool that answers `index not ready: N of M files
scanned` is telling you to call `index_status`, then retry, or use grep for
this one question. `phase: "failed"` with an `error` means the scan itself
broke — report it rather than retrying forever.

The index follows edits live (file watcher). If `index_status` reports a
`watcherWarning`, edits made in this session may not be reflected; re-run
the search after saving or restart the server.

## Which directory is indexed

Claude Code and Cursor tell the server the project root at startup. GitHub
Copilot does not — it starts the server inside the plugin's own folder and
sends no roots — so the server starts with **no workspace open** there and a
tool answers `no workspace open`. When that happens (or `index_status` returns
an empty list), pass the project's absolute path as `workspace` on the call;
it is opened and indexed on demand. Never pass the plugin folder itself.

To search a second directory, call `open_workspace` with its absolute path,
then pass `workspace` on later calls to choose which one answers.
