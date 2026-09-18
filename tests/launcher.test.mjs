import { test } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { defaultCacheDir, needsBaseline, parseSums, targetName } from "../bin/launcher.mjs";

test("cache dir mirrors dirs::cache_dir() so launcher and binary share one tree", () => {
  assert.equal(defaultCacheDir({}, "linux", "/home/u"), "/home/u/.cache/claudinio-code-intel");
  assert.equal(defaultCacheDir({ XDG_CACHE_HOME: "/x" }, "linux", "/home/u"), "/x/claudinio-code-intel");
  assert.equal(defaultCacheDir({}, "darwin", "/Users/u"), "/Users/u/Library/Caches/claudinio-code-intel");
  assert.equal(defaultCacheDir({ LOCALAPPDATA: "C:\\Users\\u\\AppData\\Local" }, "win32", "C:\\Users\\u"),
    path.join("C:\\Users\\u\\AppData\\Local", "claudinio-code-intel"));
  assert.equal(defaultCacheDir({ CODE_INTEL_CACHE_DIR: "/custom" }, "linux", "/home/u"), "/custom");
});

test("a linux x64 CPU without AVX2/BMI2 gets the baseline (candle) build", () => {
  const haswell = "flags\t\t: fpu sse2 avx avx2 bmi1 bmi2\n";
  const celeron = "flags\t\t: fpu sse2 ssse3 sse4_1\n";
  assert.equal(needsBaseline("linux", "x64", haswell), false);
  assert.equal(needsBaseline("linux", "x64", celeron), true);
  assert.equal(needsBaseline("linux", "arm64", celeron), false);
  assert.equal(needsBaseline("darwin", "x64", celeron), false);
  assert.equal(targetName("linux", "x64", {}, celeron), "linux-x64-baseline");
  assert.equal(targetName("linux", "x64", {}, haswell), "linux-x64");
  assert.equal(targetName("linux", "x64", { CODE_INTEL_VARIANT: "ort" }, celeron), "linux-x64");
  assert.equal(targetName("win32", "x64", { CODE_INTEL_VARIANT: "baseline" }), "win32-x64-baseline");
});

test("target names match the release matrix", () => {
  assert.equal(targetName("darwin", "arm64", {}), "darwin-arm64");
  assert.equal(targetName("darwin", "x64", {}), "darwin-x64");
  assert.equal(targetName("linux", "arm64", {}), "linux-arm64");
  assert.equal(targetName("win32", "arm64", {}), "win32-arm64");
  assert.throws(() => targetName("freebsd", "x64", {}), /unsupported platform/);
});

test("SHA256SUMS parsing accepts both text and binary markers", () => {
  const sums = parseSums(`${"a".repeat(64)}  claudinio-code-intel-linux-x64.tar.gz\n${"b".repeat(64)} *claudinio-code-intel-win32-x64.tar.gz\n`);
  assert.equal(sums["claudinio-code-intel-linux-x64.tar.gz"], "a".repeat(64));
  assert.equal(sums["claudinio-code-intel-win32-x64.tar.gz"], "b".repeat(64));
});
