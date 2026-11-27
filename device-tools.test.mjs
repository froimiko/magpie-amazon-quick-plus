import assert from "node:assert/strict"
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import test from "node:test"

import {
  buildPutOutpostEnvelope,
  buildToolResultEvent,
  capToolResult,
  defaultCapabilities,
  MAX_TOOL_RESULT_BYTES,
  nativeDeviceToolsEnabled,
  stableDeviceId,
  toolCallSessionId,
  unwrapToolArguments,
} from "./device-tools.mjs"
import { applyUnifiedDiff, createLocalDeviceToolExecutor } from "./device-exec.mjs"

test("device tools are off unless asked for", () => {
  assert.equal(nativeDeviceToolsEnabled({}), false)
  assert.equal(nativeDeviceToolsEnabled({ nativeDeviceTools: false }), false)
  assert.equal(nativeDeviceToolsEnabled({ nativeDeviceTools: true }), true)
  assert.equal(nativeDeviceToolsEnabled({ deviceTools: "1" }), true)
  assert.equal(nativeDeviceToolsEnabled({ deviceTools: "ON" }), true)
  assert.equal(nativeDeviceToolsEnabled({ deviceTools: "nope" }), false)
})

test("the registration envelope is the shape the cell accepts", () => {
  // Captured from the official outpost's own egress: eventType putOutpost with a
  // single putOutpost body carrying outpostId/name/outpostType/capabilities.
  const env = buildPutOutpostEnvelope({ outpostId: "abc", name: "box" })
  assert.equal(env.eventType, "putOutpost")
  const body = env.eventBody.putOutpost
  assert.equal(body.outpostId, "abc")
  assert.equal(body.name, "box")
  assert.equal(body.outpostType, "DESKTOP")
  assert.ok(Array.isArray(body.capabilities) && body.capabilities.length > 0)
  for (const group of body.capabilities) {
    assert.equal(typeof group.name, "string")
    assert.ok(Array.isArray(group.tools) && group.tools.length > 0)
    for (const t of group.tools) {
      assert.equal(typeof t.name, "string")
      assert.equal(t.inputSchema.type, "object")
      assert.equal(typeof t.isReadTool, "boolean")
    }
  }
  assert.throws(() => buildPutOutpostEnvelope({}), /outpostId/)
})

test("every advertised tool has an executor behind it", () => {
  // An advertised-but-unexecutable tool shows up as a failed tool_result, which
  // reads to the model as "the device is broken".
  const exec = createLocalDeviceToolExecutor({ roots: [tmpdir()] })
  const advertised = defaultCapabilities().flatMap((g) => g.tools.map((t) => t.name))
  for (const name of advertised) {
    assert.ok(exec.supportedTools.includes(name), "no executor for advertised tool " + name)
  }
})

test("stableDeviceId is deterministic and shaped like an id", () => {
  const a = stableDeviceId("account|machine|x")
  assert.equal(a, stableDeviceId("account|machine|x"))
  assert.notEqual(a, stableDeviceId("account|machine|y"))
  assert.match(a, /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/)
})

test("device tool arguments are unwrapped from the relay envelope", () => {
  // The official relay nests them as {params:{arguments:{...}}}
  assert.deepEqual(unwrapToolArguments({ params: { arguments: { path: "/x" } } }), { path: "/x" })
  assert.deepEqual(unwrapToolArguments({ path: "/x" }), { path: "/x" })
  assert.deepEqual(unwrapToolArguments(null), {})
  assert.equal(toolCallSessionId({ params: { _quickwork: { offload_session_id: "s1" } } }), "s1")
  assert.equal(toolCallSessionId({ thread_id: "t1" }), "t1")
  assert.equal(toolCallSessionId({}), "")
})

test("tool results are capped with a visible note, never silently", () => {
  const small = capToolResult({ ok: true })
  assert.equal(small.truncated, false)

  const big = capToolResult({ content: "x".repeat(MAX_TOOL_RESULT_BYTES) })
  assert.equal(big.truncated, true)
  assert.match(big.result.content, /\[TRUNCATED/)
  const bytes = Buffer.byteLength(JSON.stringify({ type: "tool_result", payload: { tool_call_id: "", result: big.result, error: null } }), "utf8")
  assert.ok(bytes <= MAX_TOOL_RESULT_BYTES, "capped result must fit the relay cap, got " + bytes)

  const bigOutput = capToolResult({ output: "y".repeat(MAX_TOOL_RESULT_BYTES) })
  assert.match(bigOutput.result.output, /\[TRUNCATED/)

  const bigString = capToolResult("z".repeat(MAX_TOOL_RESULT_BYTES))
  assert.match(bigString.result, /\[TRUNCATED/)

  const bigOpaque = capToolResult({ n: "q".repeat(MAX_TOOL_RESULT_BYTES) })
  assert.equal(bigOpaque.result.truncated, true)
})

test("the tool_result frame uses tool_call_id, not call_id", () => {
  const ev = buildToolResultEvent("call-1", { ok: true }, null)
  assert.deepEqual(ev, { type: "tool_result", payload: { tool_call_id: "call-1", result: { ok: true }, error: null } })
})

test("file tools stay inside the allowed roots", async () => {
  const root = mkdtempSync(join(tmpdir(), "magpie-dev-"))
  const outside = mkdtempSync(join(tmpdir(), "magpie-out-"))
  try {
    writeFileSync(join(root, "a.txt"), "hello\nworld\n", "utf8")
    const exec = createLocalDeviceToolExecutor({ roots: [root] })

    const listed = await exec({ toolName: "folder_list", arguments: { path: root } })
    assert.equal(listed.result.success, true)
    assert.ok(listed.result.entries.some((e) => e.name === "a.txt"))

    const read = await exec({ toolName: "file_read", arguments: { path: "a.txt" } })
    assert.equal(read.result.success, true)
    assert.match(read.result.content, /hello/)

    const denied = await exec({ toolName: "file_read", arguments: { path: join(outside, "x.txt") } })
    assert.equal(denied.result.success, false)
    assert.match(denied.result.error, /not in allowed folders/)

    const wroteOut = await exec({ toolName: "file_write", arguments: { path: join(outside, "x.txt"), content: "no" } })
    assert.equal(wroteOut.result.success, false)

    const wroteIn = await exec({ toolName: "file_write", arguments: { path: "b.txt", content: "fresh" } })
    assert.equal(wroteIn.result.success, true)
    assert.equal(readFileSync(join(root, "b.txt"), "utf8"), "fresh")
  } finally {
    rmSync(root, { recursive: true, force: true })
    rmSync(outside, { recursive: true, force: true })
  }
})

test("an unknown device tool fails loudly instead of pretending", async () => {
  const exec = createLocalDeviceToolExecutor({ roots: [tmpdir()] })
  const r = await exec({ toolName: "no_such_tool", arguments: {} })
  assert.equal(r.result.success, false)
  assert.match(r.result.error, /cannot execute/)
  assert.ok(Array.isArray(r.result.supported_tool))
})

test("shell tools can be disabled", async () => {
  const exec = createLocalDeviceToolExecutor({ roots: [tmpdir()], allowShell: false })
  const r = await exec({ toolName: "terminal_on_device", arguments: { command: "echo hi" } })
  assert.equal(r.result.success, false)
  assert.match(r.result.error, /disabled/)
})

test("unified diffs apply by context, tolerating off line numbers", () => {
  const src = ["a", "b", "c", "d"].join("\n")
  const ok = applyUnifiedDiff(src, ["@@ -1,2 +1,2 @@", " a", "-b", "+B"].join("\n"))
  assert.equal(ok.error, undefined)
  assert.equal(ok.text, ["a", "B", "c", "d"].join("\n"))

  // Wrong start line, right context: still applies (content-based match).
  const fuzz = applyUnifiedDiff(src, ["@@ -40,2 +40,2 @@", " c", "-d", "+D"].join("\n"))
  assert.equal(fuzz.error, undefined)
  assert.equal(fuzz.text, ["a", "b", "c", "D"].join("\n"))

  const missing = applyUnifiedDiff(src, ["@@ -1,1 +1,1 @@", "-zzz", "+yyy"].join("\n"))
  assert.match(missing.error, /Could not locate/)

  assert.match(applyUnifiedDiff(src, "").error, /empty/)
  assert.match(applyUnifiedDiff(src, "not a diff").error, /no hunks/)
})

test("file_edit writes through the diff", async () => {
  const root = mkdtempSync(join(tmpdir(), "magpie-edit-"))
  try {
    writeFileSync(join(root, "f.txt"), "one\ntwo\nthree\n", "utf8")
    const exec = createLocalDeviceToolExecutor({ roots: [root] })
    const r = await exec({ toolName: "file_edit", arguments: { path: "f.txt", diff: "@@ -1,1 +1,1 @@\n-one\n+ONE" } })
    assert.equal(r.result.success, true)
    assert.equal(readFileSync(join(root, "f.txt"), "utf8"), "ONE\ntwo\nthree\n")
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
