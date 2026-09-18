// The three hosts read three different manifest sets from this one repo.
// They must describe the same plugin — same name, same version as the
// crate and package.json, same launcher path — and the Agent Plugins ones
// must satisfy the published schemas (vendored in tests/schemas/, fetched
// 2026-09-18 from agent-plugins.org).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (p) => JSON.parse(fs.readFileSync(path.join(root, p), "utf8"));

const pkg = read("package.json");
const agent = read("plugin.json");
const agentMcp = read("mcp.json");
const claude = read(".claude-plugin/plugin.json");
const claudeMcp = read(".mcp.json");
const cursor = read(".cursor-plugin/plugin.json");
const markets = [".claude-plugin/marketplace.json", ".cursor-plugin/marketplace.json", ".github/plugin/marketplace.json"].map(read);

const cargoVersion = /^version\s*=\s*"([^"]+)"/m.exec(
  fs.readFileSync(path.join(root, "crates/code-intel-mcp/Cargo.toml"), "utf8"),
)[1];

test("one version everywhere: crate, package.json, every manifest, every marketplace", () => {
  for (const [what, v] of [
    ["crates/code-intel-mcp", cargoVersion],
    ["plugin.json", agent.version],
    [".claude-plugin/plugin.json", claude.version],
    [".cursor-plugin/plugin.json", cursor.version],
    ...markets.map((m, i) => [`marketplace #${i} plugin`, m.plugins[0].version]),
  ]) {
    assert.equal(v, pkg.version, `${what} must carry ${pkg.version}`);
  }
});

test("one plugin name everywhere", () => {
  for (const m of [agent, claude, cursor]) assert.equal(m.name, "code-intel");
  for (const m of markets) {
    assert.equal(m.plugins.length, 1);
    assert.equal(m.plugins[0].name, "code-intel");
    assert.equal(m.plugins[0].source, "./", "single-plugin repo: the plugin is the repo");
    assert.equal(m.name, "claudinio", "marketplace name is the install suffix: code-intel@claudinio");
  }
});

test("the three marketplaces are the same document", () => {
  const [a, b, c] = markets.map((m) => JSON.stringify(m));
  assert.equal(a, b);
  assert.equal(a, c);
});

test("every MCP config launches the same file through node, and that file exists", () => {
  for (const [what, cfg] of [["mcp.json", agentMcp], [".mcp.json", claudeMcp]]) {
    const server = cfg.mcpServers["code-intel"];
    assert.ok(server, `${what} declares the code-intel server`);
    assert.equal(server.type, "stdio");
    assert.equal(server.command, "node");
    assert.deepEqual(server.args, ["${CLAUDE_PLUGIN_ROOT}/bin/launcher.mjs"]);
  }
  assert.ok(fs.existsSync(path.join(root, "bin/launcher.mjs")));
  assert.equal(pkg.bin["claudinio-code-intel"], "bin/launcher.mjs");
  assert.equal(cursor.mcpServers, "./mcp.json");
  assert.equal(claude.mcpServers, "./.mcp.json");
});

test("Claude Code gets the project dir; no other host is handed an unexpanded variable", () => {
  assert.equal(claudeMcp.mcpServers["code-intel"].env.CODE_INTEL_WORKSPACE, "${CLAUDE_PROJECT_DIR}");
  assert.equal(agentMcp.mcpServers["code-intel"].env, undefined);
});

// Minimal JSON-schema checker: enough of draft 2020-12 for these two schemas
// (type, const, required, additionalProperties, properties, pattern, items,
// propertyNames.not.enum, $ref within $defs, oneOf, minLength).
function validate(schema, value, rootSchema = schema, where = "$") {
  const errors = [];
  const fail = (m) => errors.push(`${where}: ${m}`);
  if (schema.$ref) {
    const target = schema.$ref.replace(/^#\/\$defs\//, "");
    return validate(rootSchema.$defs[target], value, rootSchema, where);
  }
  if (schema.oneOf) {
    const ok = schema.oneOf.filter((s) => validate(s, value, rootSchema, where).length === 0);
    if (ok.length !== 1) fail(`oneOf matched ${ok.length} alternatives`);
    return errors;
  }
  if (schema.const !== undefined && value !== schema.const) fail(`expected const ${schema.const}`);
  if (schema.type) {
    const t = Array.isArray(value) ? "array" : value === null ? "null" : typeof value;
    if (t !== schema.type) fail(`expected ${schema.type}, got ${t}`);
  }
  if (schema.type === "string") {
    if (schema.pattern && !new RegExp(schema.pattern).test(value)) fail(`does not match ${schema.pattern}`);
    if (schema.minLength && value.length < schema.minLength) fail("too short");
  }
  if (schema.type === "array" && schema.items) {
    value.forEach((v, i) => errors.push(...validate(schema.items, v, rootSchema, `${where}[${i}]`)));
  }
  if (schema.type === "object" && value && typeof value === "object") {
    for (const r of schema.required ?? []) if (!(r in value)) fail(`missing required ${r}`);
    for (const [k, v] of Object.entries(value)) {
      if (schema.propertyNames?.not?.enum?.includes(k)) fail(`property name ${k} is forbidden`);
      if (schema.properties && k in schema.properties) {
        errors.push(...validate(schema.properties[k], v, rootSchema, `${where}.${k}`));
      } else if (schema.additionalProperties === false) {
        fail(`unexpected property ${k}`);
      } else if (schema.additionalProperties && typeof schema.additionalProperties === "object") {
        errors.push(...validate(schema.additionalProperties, v, rootSchema, `${where}.${k}`));
      }
    }
  }
  return errors;
}

test("plugin.json satisfies the Agent Plugins 1.0.0 plugin schema", () => {
  const schema = read("tests/schemas/plugin.schema.json");
  assert.deepEqual(validate(schema, agent), []);
  assert.equal(agent.$schema, "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json");
});

test("mcp.json satisfies the Agent Plugins 1.0.0 mcp schema", () => {
  const schema = read("tests/schemas/mcp.schema.json");
  assert.deepEqual(validate(schema, agentMcp), []);
});

test("the schema checker itself rejects what it should", () => {
  const schema = read("tests/schemas/mcp.schema.json");
  const bad = structuredClone(agentMcp);
  bad.mcpServers["code-intel"].cwd = "/absolute/not/allowed";
  assert.notDeepEqual(validate(schema, bad), []);
  const bad2 = structuredClone(agentMcp);
  bad2.mcpServers["code-intel"].env = { PLUGIN_ROOT: "x" };
  assert.notDeepEqual(validate(schema, bad2), []);
});

test("the skill is discoverable by all three hosts: skills/<name>/SKILL.md with frontmatter", () => {
  const skill = fs.readFileSync(path.join(root, "skills/code-intel/SKILL.md"), "utf8");
  const fm = /^---\n([\s\S]*?)\n---\n/.exec(skill);
  assert.ok(fm, "YAML frontmatter present");
  assert.match(fm[1], /^name: code-intel$/m);
  assert.match(fm[1], /^description: .{40,}/m);
  for (const tool of ["semantic_search", "code_search", "symbol_lookup", "file_outline", "find_callers", "index_status"]) {
    assert.ok(skill.includes(`\`${tool}\``), `skill teaches ${tool}`);
  }
  assert.match(skill, /English/, "the index is English-only and the skill must say so");
});

test("the skill names only tools the server actually registers", () => {
  const main = fs.readFileSync(path.join(root, "crates/code-intel-mcp/src/main.rs"), "utf8");
  const registered = [...main.matchAll(/name = "([a-z_]+)"/g)].map((m) => m[1]);
  const skill = fs.readFileSync(path.join(root, "skills/code-intel/SKILL.md"), "utf8");
  for (const named of skill.matchAll(/`([a-z]+_[a-z_]+)`/g)) {
    assert.ok(registered.includes(named[1]), `${named[1]} is mentioned in the skill but not a registered tool`);
  }
});
