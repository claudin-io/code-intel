#!/usr/bin/env node
// Launcher for the `claudinio-code-intel` MCP server.
//
// Every host (Claude Code, Cursor, GitHub Copilot) runs this with Node and
// gets the same thing: the Rust binary for this platform, fetched once from
// the GitHub Release that matches package.json's version, sha256-verified
// against the release's SHA256SUMS, cached under the same directory the
// binary keeps its indexes and model, then exec'd with stdio inherited so the
// MCP channel passes straight through.
//
// Nothing here is specific to a host. Overrides, all optional:
//   CODE_INTEL_BIN        run this binary instead of downloading one (dev)
//   CODE_INTEL_CACHE_DIR  cache root (default: the OS cache dir + /claudinio-code-intel)
//   CODE_INTEL_VARIANT    "baseline" forces the pre-AVX2 (candle) build on x64
//   CODE_INTEL_RELEASE_BASE  alternative download base (mirrors, tests)

import { createHash } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(fs.readFileSync(path.join(here, "..", "package.json"), "utf8"));
const VERSION = pkg.version;
const REPO = "claudin-io/code-intel";
const RELEASE_BASE =
  process.env.CODE_INTEL_RELEASE_BASE || `https://github.com/${REPO}/releases/download/v${VERSION}`;
const BIN_NAME = process.platform === "win32" ? "claudinio-code-intel.exe" : "claudinio-code-intel";

function log(msg) {
  // stdout is the MCP transport; never write there.
  process.stderr.write(`[claudinio-code-intel] ${msg}\n`);
}

/** Mirrors Rust's `dirs::cache_dir()` so the binary and the launcher share one tree. */
export function defaultCacheDir(env = process.env, platform = process.platform, home = os.homedir()) {
  if (env.CODE_INTEL_CACHE_DIR) return env.CODE_INTEL_CACHE_DIR;
  let base;
  if (platform === "win32") base = env.LOCALAPPDATA || path.join(home, "AppData", "Local");
  else if (platform === "darwin") base = path.join(home, "Library", "Caches");
  else base = env.XDG_CACHE_HOME || path.join(home, ".cache");
  return path.join(base, "claudinio-code-intel");
}

/** True when this x86-64 CPU lacks the AVX2/BMI2 instructions the ORT build needs. */
export function needsBaseline(platform = process.platform, arch = process.arch, cpuinfo = null) {
  if (arch !== "x64") return false;
  if (platform !== "linux") return false; // Intel Macs all have AVX2; Windows: use CODE_INTEL_VARIANT
  try {
    const text = cpuinfo ?? fs.readFileSync("/proc/cpuinfo", "utf8");
    const flags = /^flags\s*:\s*(.*)$/m.exec(text)?.[1]?.split(/\s+/) ?? [];
    if (flags.length === 0) return false;
    return !(flags.includes("avx2") && flags.includes("bmi2"));
  } catch {
    return false;
  }
}

/** Release asset target for this machine, e.g. `linux-x64`, `darwin-arm64`, `win32-x64-baseline`. */
export function targetName(platform = process.platform, arch = process.arch, env = process.env, cpuinfo = null) {
  const supported = { "linux-x64": 1, "linux-arm64": 1, "darwin-x64": 1, "darwin-arm64": 1, "win32-x64": 1, "win32-arm64": 1 };
  const key = `${platform}-${arch}`;
  if (!supported[key]) throw new Error(`unsupported platform ${key} — build from source: cargo install --git https://github.com/${REPO}`);
  const baseline = env.CODE_INTEL_VARIANT === "baseline" || (env.CODE_INTEL_VARIANT !== "ort" && needsBaseline(platform, arch, cpuinfo));
  return baseline && arch === "x64" ? `${key}-baseline` : key;
}

async function fetchBytes(url) {
  const res = await fetch(url, { redirect: "follow", headers: { "user-agent": `claudinio-code-intel-launcher/${VERSION}` } });
  if (!res.ok) throw new Error(`GET ${url} → ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

function sha256(buf) {
  return createHash("sha256").update(buf).digest("hex");
}

/** Parse a `sha256sum`-style file into {filename: hex}. */
export function parseSums(text) {
  const out = {};
  for (const line of text.split(/\r?\n/)) {
    const m = /^([0-9a-f]{64})\s+\*?(.+)$/.exec(line.trim());
    if (m) out[m[2].trim()] = m[1];
  }
  return out;
}

async function install(binDir, target) {
  const asset = `claudinio-code-intel-${target}.tar.gz`;
  const url = `${RELEASE_BASE}/${asset}`;
  log(`first run: downloading ${asset} (v${VERSION})`);
  const [sumsText, archive] = await Promise.all([fetchBytes(`${RELEASE_BASE}/SHA256SUMS`), fetchBytes(url)]);
  const expected = parseSums(sumsText.toString("utf8"))[asset];
  if (!expected) throw new Error(`SHA256SUMS on the release has no entry for ${asset}`);
  const got = sha256(archive);
  if (got !== expected) throw new Error(`sha256 mismatch for ${asset}: expected ${expected}, got ${got}`);

  const staging = `${binDir}.part-${process.pid}`;
  fs.rmSync(staging, { recursive: true, force: true });
  fs.mkdirSync(staging, { recursive: true });
  const archivePath = path.join(staging, asset);
  fs.writeFileSync(archivePath, archive);
  // bsdtar ships with Windows 10+, GNU tar everywhere else.
  const tar = spawnSync("tar", ["-xzf", archivePath, "-C", staging], { stdio: ["ignore", "ignore", "inherit"] });
  if (tar.status !== 0) throw new Error(`tar failed (${tar.status ?? tar.signal}) extracting ${asset}`);
  fs.rmSync(archivePath);
  const bin = path.join(staging, BIN_NAME);
  if (!fs.existsSync(bin)) throw new Error(`${asset} did not contain ${BIN_NAME}`);
  if (process.platform !== "win32") fs.chmodSync(bin, 0o755);
  try {
    fs.renameSync(staging, binDir);
  } catch (e) {
    // Another host process won the race; its copy is the same bytes.
    if (fs.existsSync(path.join(binDir, BIN_NAME))) fs.rmSync(staging, { recursive: true, force: true });
    else throw e;
  }
  log(`installed to ${binDir}`);
}

async function resolveBinary() {
  if (process.env.CODE_INTEL_BIN) return process.env.CODE_INTEL_BIN;
  const target = targetName();
  const binDir = path.join(defaultCacheDir(), "bin", VERSION, target);
  const bin = path.join(binDir, BIN_NAME);
  if (!fs.existsSync(bin)) {
    fs.mkdirSync(path.dirname(binDir), { recursive: true });
    await install(binDir, target);
  }
  return bin;
}

async function main() {
  const bin = await resolveBinary();
  const child = spawn(bin, process.argv.slice(2), { stdio: "inherit", windowsHide: true });
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(sig, () => { try { child.kill(sig); } catch {} });
  }
  child.on("error", (e) => { log(`failed to start ${bin}: ${e.message}`); process.exit(1); });
  child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((e) => { log(e.message); process.exit(1); });
}
