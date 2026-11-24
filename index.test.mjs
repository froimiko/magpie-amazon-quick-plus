import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import test from "node:test"

import {
  buildAidpInputFrame,
  buildGatewayInputFrame,
  cborDecode,
  cborEncode,
  decodeAidpFrame,
  decodeGatewayFrame,
  esDrain,
  esEncode,
} from "./codec.mjs"
import {
  browserSignIn,
  clearDesktopCache,
  desktopSignIn,
  deviceCodeSignIn,
  fetchQuickSettings,
  importSignIns,
  invokeQbsApi,
  MACOS_KEYCHAIN_TIMEOUT_MS,
  notDesktop,
  readChunkedCredential,
  readDesktop,
  renewOidc,
  signInsIn,
  uploadQuickAttachment,
  usageFromProfile,
} from "./credentials.mjs"
import {
  configModels,
  createThreadStore,
  createToolCallSplitter,
  executeSncsChat,
  MAX_USER_MESSAGE_FRAME_BYTES,
  modelsFromSettings,
  openaiToSncsRequest,
  resolveModelAndEffort,
  stripReasoningEnabled,
} from "./bridge.mjs"
import { createAmazonQuickPlugin } from "./index.mjs"
import * as opencode1 from "./opencode.mjs"
import opencode2 from "./opencode2.mjs"

test("EventStream + CBOR + AIDP/Gateway codecs roundtrip cleanly", () => {
  const payload = new TextEncoder().encode('{"ok":true}')
  const frame = esEncode({ ":event-type": "rawOutputEvent", ":message-type": "event" }, payload)

  // Split across boundary to verify incremental drain
  const part1 = frame.subarray(0, 14)
  const part2 = frame.subarray(14)
  const d1 = esDrain(part1, 0)
  assert.equal(d1.messages.length, 0)
  assert.equal(d1.consumed, 0)

  const combined = new Uint8Array(part1.length + part2.length + frame.length)
  combined.set(part1, 0)
  combined.set(part2, part1.length)
  combined.set(frame, frame.length)
  const d2 = esDrain(combined, 0)
  assert.equal(d2.messages.length, 2)
  assert.equal(d2.consumed, combined.length)
  assert.equal(d2.messages[0].headers[":event-type"], "rawOutputEvent")

  // CBOR roundtrip
  const sample = { a: 1, b: -42, c: true, d: null, e: [1, "hi"], bytes: new Uint8Array([1, 2, 3]) }
  const decodedCbor = cborDecode(cborEncode(sample))
  assert.equal(decodedCbor.a, 1)
  assert.equal(decodedCbor.b, -42)
  assert.equal(decodedCbor.c, true)
  assert.equal(decodedCbor.d, null)
  assert.deepEqual(decodedCbor.e, [1, "hi"])
  assert.deepEqual(Array.from(decodedCbor.bytes), [1, 2, 3])

  // AIDP outbound & inbound decode
  const outAidp = JSON.parse(Buffer.from(buildAidpInputFrame({ type: "ping", payload: { n: 7 } })).toString("utf8"))
  assert.equal(outAidp.eventType, "rawInputEvent")
  assert.deepEqual(JSON.parse(Buffer.from(outAidp.payload.payload, "base64").toString("utf8")), { type: "ping", payload: { n: 7 } })

  const aidpErr = decodeAidpFrame(
    { ":message-type": "exception", ":exception-type": "AccessDeniedException" },
    new TextEncoder().encode(JSON.stringify({ message: "Unauthorized token" })),
  )
  assert.deepEqual(aidpErr, { type: "error", code: "AccessDeniedException", message: "Unauthorized token" })

  // Gateway outbound -> EventStream -> Gateway inbound roundtrip
  const gwFrame = buildGatewayInputFrame({ type: "text_delta", payload: { delta: "hello" } })
  const gwDrained = esDrain(gwFrame, 0)
  assert.equal(gwDrained.messages.length, 1)
  const gwDecoded = decodeGatewayFrame(gwDrained.messages[0].headers, gwDrained.messages[0].payload)
  assert.deepEqual(gwDecoded, { type: "event", event: { type: "text_delta", payload: { delta: "hello" } } })
})

test("readChunkedCredential & readDesktop decode UTF-16-LE chunked secrets read-only", () => {
  const fullJson = JSON.stringify({
    access_token: "acc-123",
    refresh_token: "ref-123",
    access_token_expiry: 1800000000.5,
    client_id: "cid-1",
    issuer_url: "https://identitycenter.amazonaws.com/ssoins-abc",
    tenant_url: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
    email: "user@example.com",
  })
  const half = Math.ceil(fullJson.length / 2)
  const c0 = fullJson.slice(0, half)
  const c1 = fullJson.slice(half)
  const sha = createHash("sha256").update(Buffer.from(fullJson, "utf16le")).digest("hex")
  const manifest = JSON.stringify({ __chunks__: 2, __sha__: sha })

  const store = new Map([
    ["quickwork-social-social-test", manifest],
    ["quickwork-social-social-test_chunk_0", c0],
    ["quickwork-social-social-test_chunk_1", c1],
  ])
  assert.equal(readChunkedCredential((k) => store.get(k) ?? null, "quickwork-social-social-test"), fullJson)

  // Corrupted hash returns null
  store.set("quickwork-social-social-test", JSON.stringify({ __chunks__: 2, __sha__: "0000" }))
  assert.equal(readChunkedCredential((k) => store.get(k) ?? null, "quickwork-social-social-test"), null)
  store.set("quickwork-social-social-test", manifest)

  // Temp profiles dir test
  const dir = mkdtempSync(join(tmpdir(), "aq-test-"))
  try {
    writeFileSync(
      join(dir, "profiles.json"),
      JSON.stringify({
        version: 1,
        last_active: "social-test",
        entries: [
          {
            id: "social-test",
            provider: "social",
            display_name: "Demo User",
            login_metadata: {
              issuer_url: "https://identitycenter.amazonaws.com/ssoins-abc",
              tenant_url: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
              client_id: "cid-1",
            },
          },
        ],
      }),
    )

    const readKeyring = (name) => readChunkedCredential((k) => store.get(k) ?? null, name)
    const desktop = readDesktop({ dir, readKeyring, noCache: true })
    assert.equal(desktop.access, "acc-123")
    assert.equal(desktop.expires, 1800000000500)
    assert.equal(desktop.region, "us-east-1")
    assert.equal(desktop.accountId, "user@example.com")
    assert.equal(desktop.source, "desktop")

    const authMethod = desktopSignIn({ dir, readKeyring })
    return authMethod.callback().then((res) => {
      assert.equal(res.type, "success")
      assert.equal(res.source, "desktop")
      assert.equal(res.access, "")
      assert.equal(res.refresh, "")
      assert.equal(res.accountId, "user@example.com")
    })
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test("renewOidc deduplicates concurrent refreshes and handles importSignIns", async () => {
  let calls = 0
  const mockFetch = async (url, init) => {
    calls++
    assert.equal(url, "https://oidc.us-west-2.amazonaws.com/token")
    const body = JSON.parse(init.body)
    assert.equal(body.grantType, "refresh_token")
    assert.equal(body.refreshToken, "ref-old-1")
    return new Response(JSON.stringify({ accessToken: "acc-new", refreshToken: "ref-new", expiresIn: 3600 }), { status: 200 })
  }

  const baseAuth = {
    access: "acc-old",
    refresh: "ref-old-1",
    clientId: "cid-west",
    region: "us-west-2",
    tenantUrl: "https://qbs.dp.appintegrations.us-west-2.prod.plato.ai.aws.dev",
  }
  const [r1, r2] = await Promise.all([
    renewOidc(baseAuth, null, { fetch: mockFetch, now: 1_000_000 }),
    renewOidc(baseAuth, null, { fetch: mockFetch, now: 1_000_000 }),
  ])
  assert.equal(calls, 1)
  assert.equal(r1.access, "acc-new")
  assert.equal(r2.refresh, "ref-new")
  assert.equal(r1.expires, 1_000_000 + 3_600_000)

  // Import parser + renewal
  const parsed = signInsIn(
    JSON.stringify({
      access_token: "acc-exp",
      refresh_token: "ref-imp-2",
      client_id: "cid-west",
      tenant_url: "https://qbs.dp.appintegrations.us-west-2.prod.plato.ai.aws.dev",
    }),
  )
  assert.equal(parsed.length, 1)
  assert.equal(parsed[0].region, "us-west-2")

  const imported = await importSignIns(
    [
      JSON.stringify({
        access_token: "acc-exp",
        refresh_token: "ref-imp-2",
        client_id: "cid-west",
        tenant_url: "https://qbs.dp.appintegrations.us-west-2.prod.plato.ai.aws.dev",
      }),
    ],
    {
      fetch: async () => new Response(JSON.stringify({ accessToken: "acc-imp-fresh", refreshToken: "ref-imp-3", expiresIn: 1800 }), { status: 200 }),
    },
  )
  assert.equal(imported.length, 1)
  assert.equal(imported[0].type, "success")
  assert.equal(imported[0].access, "acc-imp-fresh")
})

test("invokeQbsApi and usageFromProfile support AIDP & Gateway transports", async () => {
  const aidpCalls = []
  const usageData = {
    monthly: { used: 25, limit: 100, label: "25 / 100 credits", percent: 25, resets_at: "2026-12-01T00:00:00Z" },
    session: { used: 2, limit: 20, label: "2 / 20 session", percent: 10, resume_in_minutes: 30 },
    overage_monthly: { used: 0, limit: 0, percent: 0 },
    subscription: { subscription_type: "FREE_TRIAL" },
  }
  const aidpOut = await invokeQbsApi(
    {
      providerType: "social",
      access: "tok-1",
      tenantUrl: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
    },
    "profile/usage",
    "GET",
    {},
    {
      fetch: async (url, init) => {
        aidpCalls.push({ url, body: JSON.parse(init.body) })
        return new Response(
          JSON.stringify({
            eventResponse: { invokeAgentResponse: JSON.stringify({ statusCode: 200, body: JSON.stringify(usageData) }) },
          }),
          { status: 200 },
        )
      },
    },
  )
  assert.equal(aidpCalls[0].url, "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev/integration/quick-work/qbs-event")
  assert.equal(aidpCalls[0].body.eventBody.invokeAgent.path, "profile/usage")

  const magpieUsage = usageFromProfile(aidpOut)
  assert.equal(magpieUsage.plan, "Free Trial")
  assert.equal(magpieUsage.windows.length, 2)
  assert.equal(magpieUsage.windows[0].name, "Monthly usage")
  assert.equal(magpieUsage.windows[0].used, 25)
  assert.equal(magpieUsage.windows[0].resetsAt, "2026-12-01T00:00:00.000Z")
  assert.equal(magpieUsage.windows[1].name, "Session usage")

  // Gateway transport
  const gwUrls = []
  await invokeQbsApi(
    {
      providerType: "gateway",
      awsAccountId: "123456789012",
      access: "tok-gw",
      tenantUrl: "https://tenant.desktop.enterprise.quick.aws.com",
    },
    "api/ws-rpc",
    "POST",
    { type: "get_settings", payload: {} },
    {
      fetch: async (url) => {
        gwUrls.push(url)
        if (url.endsWith("/v1/discovery")) {
          return new Response(JSON.stringify({ cellHost: "cell1.desktop.enterprise.quick.aws.com" }), { status: 200 })
        }
        return new Response(JSON.stringify({ frames: [{ type: "settings", payload: { advanced_mode_enabled: true } }] }), { status: 200 })
      },
    },
  )
  assert.ok(gwUrls.some((u) => u.includes("cell1.desktop.enterprise.quick.aws.com/accounts/123456789012/invocations/rpc")))
})

test("model mapping, tool-call splitter, and openaiToSncsRequest handle variants and chunk boundaries", () => {
  const dynModels = modelsFromSettings("amazon-quick-plus", {
    advanced_mode_enabled: true,
    model_slots: [{ id: "balanced", display_name: "Sonnet Custom", thinking: { mode: "optional", efforts: ["off", "high"], default_effort: "off" } }],
  })
  assert.equal(dynModels.balanced.name, "Sonnet Custom")
  assert.deepEqual(Object.keys(dynModels.balanced.variants), ["off", "high"])
  assert.ok(dynModels.advanced)

  assert.deepEqual(resolveModelAndEffort("amazon-quick-plus/balanced@high"), {
    modelId: "balanced",
    agentMode: "balanced",
    autoRouting: false,
    thinkingEffort: "high",
  })
  assert.deepEqual(resolveModelAndEffort("fast", { reasoning_effort: "high" }), {
    modelId: "fast",
    agentMode: "fast",
    autoRouting: false,
    thinkingEffort: "off",
  })
  assert.deepEqual(resolveModelAndEffort("amazon-quick-plus/smart[1m]"), {
    modelId: "smart",
    agentMode: "smart",
    autoRouting: false,
    thinkingEffort: "off",
  })
  assert.deepEqual(resolveModelAndEffort("amazon-quick-plus/smart@high[1m]"), {
    modelId: "smart",
    agentMode: "smart",
    autoRouting: false,
    thinkingEffort: "high",
  })
  assert.deepEqual(resolveModelAndEffort("claude-sonnet-5-5[1m]@medium"), {
    modelId: "balanced",
    agentMode: "balanced",
    autoRouting: false,
    thinkingEffort: "medium",
  })
  // Magpie's variant picker sends the effort in X-Magpie-Effort; the header
  // wins over the body, and the relay spellings map to Quick's levels
  assert.equal(resolveModelAndEffort("balanced", {}, "high").thinkingEffort, "high")
  assert.equal(resolveModelAndEffort("balanced", { reasoning_effort: "low" }, "max").thinkingEffort, "max")
  assert.equal(resolveModelAndEffort("balanced", { reasoning_effort: "xhigh" }).thinkingEffort, "max")
  assert.equal(resolveModelAndEffort("balanced", { reasoning_effort: "minimal" }).thinkingEffort, "low")
  assert.equal(resolveModelAndEffort("balanced", { reasoning_effort: "none" }).thinkingEffort, "off")
  assert.equal(resolveModelAndEffort("balanced", { reasoning: false }).thinkingEffort, "off")

  // OpenAI -> SNCS multi-turn + tools
  const sncs = openaiToSncsRequest({
    model: "smart",
    reasoning_effort: "medium",
    stream: true,
    tools: [{ type: "function", function: { name: "read_file", description: "Read", parameters: { type: "object" } } }],
    messages: [
      { role: "system", content: "Be concise." },
      { role: "user", content: "Check foo.txt" },
      { role: "assistant", content: null, tool_calls: [{ id: "c1", type: "function", function: { name: "read_file", arguments: '{"path":"foo.txt"}' } }] },
      { role: "tool", tool_call_id: "c1", content: "hello" },
    ],
  })
  assert.equal(sncs.userMessagePayload.agent_mode, "smart")
  assert.equal(sncs.userMessagePayload.thinking_effort, "medium")

  const sncsEffort = openaiToSncsRequest(
    { model: "smart", reasoning_effort: "low", stream: true, messages: [{ role: "user", content: "Hi" }] },
    { effortHeader: "max" },
  )
  assert.equal(sncsEffort.userMessagePayload.thinking_effort, "max")
  assert.equal(sncs.userMessagePayload.incognito, true)
  assert.equal(sncs.userMessagePayload.web_search_enabled, true)
  assert.equal(sncs.sessionConnectPayload.outpostId, undefined)
  assert.equal(sncs.userMessagePayload.outpostId, undefined)
  assert.ok(sncs.userMessagePayload.custom_prompt.includes("IDE / CLI Client Tool-Bridge Instructions"))
  assert.ok(sncs.userMessagePayload.custom_prompt.includes("Keep your AWS Quick assistant identity"))
  assert.ok(!sncs.userMessagePayload.custom_prompt.includes("untrusted user content"))
  assert.ok(sncs.userMessagePayload.custom_prompt.includes("read_file"))

  // Claude Code identity-clash system prompt should be softened without losing project instructions
  const ccSncs = openaiToSncsRequest({
    model: "balanced",
    tools: [{ type: "function", function: { name: "Write", description: "Write file", parameters: { type: "object" } } }],
    messages: [
      { role: "system", content: "x-anthropic-billing-header: cc_version=2.1\nYou are Claude Code, Anthropic's official CLI for Claude.\nAlways use TypeScript." },
      { role: "user", content: "Create hello.ts" },
    ],
  })
  assert.ok(!ccSncs.userMessagePayload.text.includes("You are Claude Code"))
  assert.ok(!ccSncs.userMessagePayload.text.includes("x-anthropic-billing-header"))
  assert.ok(ccSncs.userMessagePayload.text.includes("local CLI/IDE coding workspace"))
  assert.ok(ccSncs.userMessagePayload.text.includes("Always use TypeScript."))
  assert.ok(sncs.userMessagePayload.text.includes("<instructions>"))
  assert.ok(sncs.userMessagePayload.text.includes("Be concise."))
  assert.ok(sncs.userMessagePayload.text.includes("CLIENT TOOL CALLING PROTOCOL"))
  assert.ok(sncs.userMessagePayload.text.includes('<tool_result id="c1">\nhello\n</tool_result>'))
  assert.ok(sncs.userMessagePayload.text.endsWith("</client_tool_directive>"))
  assert.ok(sncs.userMessagePayload.text.includes("Available client tool names: read_file."))

  const sncsNoWeb = openaiToSncsRequest(
    { model: "fast", web_search_enabled: false, messages: [{ role: "user", content: "Ping" }] },
    { outpostId: "outpost-123" },
  )
  assert.equal(sncsNoWeb.userMessagePayload.web_search_enabled, false)
  assert.equal(sncsNoWeb.sessionConnectPayload.outpostId, "outpost-123")
  assert.equal(sncsNoWeb.userMessagePayload.outpostId, "outpost-123")

  // Stream splitter across partial tag prefixes
  const splitter = createToolCallSplitter(true)
  const chunks = ["Thinking... <", "t", "emp>not a tag</temp> <too", "l_call>{\"name\":\"read_file\",", "\"arguments\":{\"p\":1}}</tool", "_call> done."]
  let textOut = ""
  const calls = []
  for (const ch of chunks) {
    const res = splitter.push(ch)
    textOut += res.text
    calls.push(...res.toolCalls)
  }
  const end = splitter.flush()
  textOut += end.text
  calls.push(...end.toolCalls)
  assert.equal(textOut, "Thinking... <temp>not a tag</temp>  done.")
  assert.equal(calls.length, 1)
  assert.equal(calls[0].function.name, "read_file")
  assert.equal(calls[0].function.arguments, '{"p":1}')

  const fencedSplitter = createToolCallSplitter(true)
  const fencedOut = fencedSplitter.push('<tool_call>\n```json\n{"name":"write_file","arguments":{"path":"a.txt"}}\n```\n</tool_call>')
  const fencedEnd = fencedSplitter.flush()
  const fencedCalls = [...fencedOut.toolCalls, ...fencedEnd.toolCalls]
  assert.equal(fencedCalls.length, 1)
  assert.equal(fencedCalls[0].function.name, "write_file")
  assert.equal(fencedCalls[0].function.arguments, '{"path":"a.txt"}')

  // Large prompt with 30 verbose tool schemas + big system prompt + long older turns stays under 64 KB SNCS wire limit
  const verboseTools = Array.from({ length: 30 }, (_, i) => ({
    type: "function",
    function: {
      name: `tool_${i}`,
      description: `Detailed description ${"x".repeat(400)} for tool ${i}`,
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Full path ".repeat(30) },
          mode: { type: "string", enum: ["read", "write"] },
          tags: { type: "array", items: { type: "string" } },
        },
        required: ["path"],
        additionalProperties: false,
      },
    },
  }))
  const hugeReq = openaiToSncsRequest({
    model: "balanced",
    tools: verboseTools,
    messages: [
      { role: "system", content: "CORE_SYSTEM\n" + "Guide rule line.\n".repeat(2500) },
      { role: "user", content: "Step 1: inspect files" },
      { role: "assistant", content: null, tool_calls: [{ id: "old1", type: "function", function: { name: "tool_0", arguments: '{"path":"a"}' } }] },
      { role: "tool", tool_call_id: "old1", content: "FILE_LINE\n".repeat(2000) },
      { role: "user", content: "Final task: edit_file douban_spider.py now" },
    ],
  })
  const frameBytes = buildAidpInputFrame({ type: "user_message", payload: hugeReq.userMessagePayload }).byteLength
  assert.ok(frameBytes <= MAX_USER_MESSAGE_FRAME_BYTES, `Expected frameBytes ${frameBytes} <= ${MAX_USER_MESSAGE_FRAME_BYTES}`)
  assert.ok(hugeReq.userMessagePayload.text.includes("Final task: edit_file douban_spider.py now"))
  assert.ok(hugeReq.userMessagePayload.text.includes("tool_29(path*: string"))
  assert.ok(hugeReq.userMessagePayload.text.endsWith("</client_tool_directive>"))
})

test("CJK sessions get a language-consistency rule and reasoning_content can be stripped", async () => {
  const tools = [
    {
      type: "function",
      function: {
        name: "read_file",
        description: "Read a file",
        parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
      },
    },
  ]
  const ask = (content, locale) =>
    openaiToSncsRequest({ model: "smart", tools, messages: [{ role: "user", content }] }, { locale }).userMessagePayload
      .custom_prompt

  // Chinese locale injects the alignment rule naming the session language.
  assert.ok(ask("排查日志", "zh-CN").includes("Language Consistency"))
  assert.ok(ask("排查日志", "zh-CN").includes("Chinese"))
  // Japanese / Korean locales are named too.
  assert.ok(ask("ログを調べて", "ja-JP").includes("Japanese"))
  assert.ok(ask("로그를 확인해줘", "ko-KR").includes("Korean"))
  // English conversation keeps the previous prompt byte-for-byte (no reasoning-language constraint).
  const en = ask("find the error in the log", "en-US")
  assert.ok(!en.includes("Language Consistency"))
  // Chinese content on an en-US OS locale still aligns, including when accompanied by a long English stack trace.
  assert.ok(ask("请帮我排查日志中「用户状态异常，禁止登录」的报错原因，并给出修复建议", "en-US").includes("Language Consistency"))
  assert.ok(
    ask(
      "请帮我排查下面这段英文报错的原因并修复：\nTypeError: Cannot read properties of undefined (reading 'access') at executeSncsChat (bridge.mjs:1054)",
      "en-US",
    ).includes("Language Consistency"),
  )

  // stripReasoning: explicit option wins, env override applies otherwise.
  const saved = process.env.MAGPIE_QUICK_STRIP_REASONING_CONTENT
  try {
    delete process.env.MAGPIE_QUICK_STRIP_REASONING_CONTENT
    assert.equal(stripReasoningEnabled({}), false)
    assert.equal(stripReasoningEnabled({ stripReasoning: true }), true)
    assert.equal(stripReasoningEnabled({ stripReasoning: false }), false)
    process.env.MAGPIE_QUICK_STRIP_REASONING_CONTENT = "1"
    assert.equal(stripReasoningEnabled({}), true)
    assert.equal(stripReasoningEnabled({ stripReasoning: false }), false)
  } finally {
    if (saved === undefined) delete process.env.MAGPIE_QUICK_STRIP_REASONING_CONTENT
    else process.env.MAGPIE_QUICK_STRIP_REASONING_CONTENT = saved
  }

  const acct = {
    providerType: "social",
    access: "tok-live",
    tenantUrl: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
  }
  const makeWS = () =>
    makeMockWebSocketClass((ws) => {
      ws.readyState = 1
      ws.onopen?.()
      queueMicrotask(() => {
        ws.emitAidpEvent({ type: "session_established", payload: {} })
        ws.emitAidpEvent({ type: "message_start", payload: {} })
        ws.emitAidpEvent({ type: "thinking_delta", payload: { delta: "The user wants me to grep the log" } })
        ws.emitAidpEvent({ type: "text_delta", payload: { delta: "日志里没有匹配项。" } })
        ws.emitAidpEvent({ type: "message_complete", payload: {} })
      })
    })
  const body = { model: "fast", messages: [{ role: "user", content: "排查日志" }] }

  // Streamed: thinking stays server-side, answer still streams.
  const strippedSse = await (
    await executeSncsChat(acct, { ...body, stream: true }, { WebSocket: makeWS(), stateful: false, stripReasoning: true })
  ).text()
  assert.ok(!strippedSse.includes("reasoning_content"))
  assert.ok(strippedSse.includes("日志里没有匹配项。"))
  assert.ok(strippedSse.includes("data: [DONE]"))

  // Default: reasoning is still forwarded to the caller.
  const keptSse = await (
    await executeSncsChat(acct, { ...body, stream: true }, { WebSocket: makeWS(), stateful: false })
  ).text()
  assert.ok(keptSse.includes('"reasoning_content":"The user wants me to grep the log"'))

  // Non-streamed: the completed message carries no reasoning_content when stripped.
  const strippedJson = await (
    await executeSncsChat(acct, { ...body, stream: false }, { WebSocket: makeWS(), stateful: false, stripReasoning: true })
  ).json()
  assert.equal(strippedJson.choices[0].message.content, "日志里没有匹配项。")
  assert.equal(strippedJson.choices[0].message.reasoning_content, undefined)
  assert.equal(strippedJson.choices[0].finish_reason, "stop")
})

function makeMockWebSocketClass(responder) {
  return class MockWebSocket {
    constructor(url, init) {
      this.url = url
      this.init = init
      this.readyState = 0
      this.sent = []
      queueMicrotask(() => responder(this))
    }
    send(data) {
      this.sent.push(data)
    }
    close(code = 1000, reason = "") {
      this.readyState = 3
      this.onclose?.({ code, reason })
    }
    emitAidpEvent(obj) {
      const payload = Buffer.from(
        JSON.stringify({ payload: Buffer.from(JSON.stringify(obj), "utf8").toString("base64") }),
        "utf8",
      )
      const frame = esEncode({ ":event-type": "rawOutputEvent", ":message-type": "event" }, payload)
      this.onmessage?.({ data: frame })
    }
    emitGatewayEvent(obj) {
      const innerCbor = cborEncode({ payload: new TextEncoder().encode(JSON.stringify(obj)) })
      const frame = esEncode(
        { ":event-type": "rawOutputEvent", ":content-type": "application/cbor", ":message-type": "event" },
        innerCbor,
      )
      this.onmessage?.({ data: frame })
    }
  }
}

test("AmazonQuickAuthPlugin executes SSE streaming, 401 token retry, usage, and exports cleanly", async () => {
  let wsAttempts = 0
  const MockWS = makeMockWebSocketClass((ws) => {
    wsAttempts++
    if (ws.url.includes("expired-tok")) {
      ws.readyState = 3
      ws.onerror?.({ message: "401 Unauthorized" })
      ws.onclose?.({ code: 1008, reason: "401 Unauthorized" })
      return
    }
    const emit = (obj) =>
      ws.url.includes("ChatBidirectionalStream") ? ws.emitGatewayEvent(obj) : ws.emitAidpEvent(obj)
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      emit({ type: "session_established", payload: {} })
      emit({ type: "message_start", payload: {} })
      emit({ type: "tool_start", payload: { tool_name: "web_search", tool_id: "ws-1" } })
      emit({ type: "tool_result", payload: { tool_id: "ws-1", result: "ok" } })
      emit({ type: "thinking_delta", payload: { delta: "Let me think" } })
      emit({ type: "text_delta", payload: { delta: "Hello world!" } })
      emit({ type: "turn_usage", payload: { input_tokens: 10, output_tokens: 5, total_tokens: 15 } })
      emit({ type: "message_complete", payload: {} })
    })
  })

  let stored = {
    type: "oauth",
    access: "expired-tok",
    refresh: "ref-live",
    expires: Date.now() + 3600_000,
    clientId: "cid-1",
    region: "us-east-1",
    providerType: "social",
    tenantUrl: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
  }

  const pluginFactory = createAmazonQuickPlugin({
    WebSocket: MockWS,
    fetch: async (url) => {
      if (url.includes("oidc.us-east-1.amazonaws.com/token")) {
        return new Response(JSON.stringify({ accessToken: "fresh-tok", refreshToken: "ref-next", expiresIn: 3600 }), { status: 200 })
      }
      if (url.includes("qbs-event")) {
        return new Response(
          JSON.stringify({
            eventResponse: {
              invokeAgentResponse: JSON.stringify({
                statusCode: 200,
                body: JSON.stringify({
                  monthly: { used: 5, limit: 50, percent: 10, label: "5 / 50" },
                  subscription: { subscription_type: "PRO" },
                }),
              }),
            },
          }),
          { status: 200 },
        )
      }
      return new Response("{}", { status: 404 })
    },
  })

  const plugin = await pluginFactory({
    client: {
      auth: {
        set: async ({ body }) => {
          stored = { ...stored, ...body }
        },
      },
    },
  })

  // config hook
  const cfg = {}
  await plugin.config(cfg)
  assert.equal(cfg.provider["amazon-quick-plus"].npm, "@ai-sdk/openai-compatible")
  assert.equal(cfg.provider["amazon-quick-plus"].api, "https://quick.aws.com/v1")
  assert.equal(cfg.provider["amazon-quick-plus"].models.balanced.api.url, "https://quick.aws.com/v1")
  assert.equal(cfg.provider["amazon-quick-plus"].models.balanced.reasoning, true)
  assert.equal(cfg.provider["amazon-quick-plus"].models.fast.reasoning, false)
  assert.deepEqual(cfg.provider["amazon-quick-plus"].models.balanced.modalities.input, ["text", "image"])

  // a config-path list Magpie hands back without capabilities keeps the
  // plugin's reasoning/vision, while the user's name and limits stand
  const fallbackModels = await plugin.provider.models(
    {
      api: "https://quick.aws.com/v1",
      models: {
        balanced: {
          id: "balanced",
          name: "Balanced (configured)",
          limit: { context: 4096 },
          capabilities: { temperature: true, toolcall: true, input: { text: true }, output: { text: true } },
          variants: {},
        },
      },
    },
    {},
  )
  assert.equal(fallbackModels.balanced.capabilities.reasoning, true)
  assert.equal(fallbackModels.balanced.capabilities.input.image, true)
  assert.deepEqual(fallbackModels.balanced.variants.medium, { reasoningEffort: "medium" })
  assert.equal(fallbackModels.balanced.name, "Balanced (configured)")
  assert.equal(fallbackModels.balanced.limit.context, 4096)

  // loader SSE stream with 401 refresh recovery (plus Uint8Array body & non-empty baseURL)
  const loader = await plugin.auth.loader(async () => stored)
  assert.equal(loader.baseURL, "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev")
  const res = await loader.fetch(`${loader.baseURL}/chat/completions`, {
    method: "POST",
    body: new TextEncoder().encode(JSON.stringify({ model: "balanced@high", stream: true, messages: [{ role: "user", content: "Hi" }] })),
  })
  assert.equal(res.status, 200)
  assert.equal(res.headers.get("X-Magpie-Sign-In"), "renewed")
  assert.equal(wsAttempts, 2)
  const sseText = await res.text()
  assert.ok(sseText.includes('"reasoning_content":"Let me think"'))
  assert.ok(sseText.includes('"content":"Hello world!"'))
  assert.ok(sseText.includes('"finish_reason":"stop"'))
  assert.ok(sseText.includes("data: [DONE]"))

  // Non-streaming Gateway path
  const gwRes = await executeSncsChat(
    { providerType: "gateway", access: "tok-ok", tenantUrl: "https://gw.desktop.enterprise.quick.aws.com" },
    { model: "fast", stream: false, messages: [{ role: "user", content: "Ping" }] },
    { WebSocket: MockWS },
  )
  const gwJson = await gwRes.json()
  assert.equal(gwJson.choices[0].message.content, "Hello world!")
  assert.equal(gwJson.usage.total_tokens, 15)

  // Usage hook
  const usageRes = await plugin.auth.usage(async () => stored)
  assert.equal(usageRes.plan, "Pro")
  assert.equal(usageRes.windows[0].name, "Monthly usage")
  assert.equal(usageRes.signIn, "kept")

  // opencode1 & opencode2 entrypoint sanity
  assert.equal(typeof opencode1.AmazonQuickAuthPlugin, "function")
  assert.equal(typeof opencode2.id, "string")
  assert.equal(typeof opencode2.setup, "function")
  const addedMethods = []
  const addedProviders = []
  await opencode2.setup({
    integration: {
      transform: async (fn) =>
        fn({
          update: () => {},
          method: { update: (x) => addedMethods.push(x) },
        }),
      connection: { active: async () => undefined },
    },
    provider: {
      transform: async (fn) => fn({ add: (x) => addedProviders.push(x) }),
      reload: async () => {},
    },
    aisdk: { hook: async () => {} },
  })
  assert.equal(addedMethods.length, 4)
  assert.equal(addedMethods[0].integrationID, "amazon-quick-plus")
  assert.match(addedMethods[2].method.label, /设备码登录/)
  assert.equal(addedMethods[2].method.form?.every((f) => !f.required), true)
  assert.equal(addedProviders[0].info.id, "amazon-quick-plus")

  const pkgJson = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8"))
  assert.ok(pkgJson.magpie?.icon?.startsWith("data:image/png;base64,"))
  const decodedIcon = Buffer.from(pkgJson.magpie.icon.slice("data:image/png;base64,".length), "base64")
  assert.equal(decodedIcon.subarray(0, 4).toString("hex"), "89504e47")

  // Unrecoverable 401 behavior: desktop stays "kept", OAuth marks "expired"
  const RejectWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 3
    ws.onerror?.({ message: "401 Unauthorized" })
    ws.onclose?.({ code: 1008, reason: "401 Unauthorized" })
  })
  const rejectFactory = createAmazonQuickPlugin({
    WebSocket: RejectWS,
    fetch: async () => new Response(JSON.stringify({ error: "invalid_grant" }), { status: 400 }),
  })
  const rejectPlugin = await rejectFactory({})
  const oauthLoader = await rejectPlugin.auth.loader(async () => ({
    type: "oauth",
    access: "bad-tok",
    refresh: "bad-ref",
    clientId: "cid-1",
    region: "us-east-1",
    providerType: "social",
    tenantUrl: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
  }))
  const oauth401 = await oauthLoader.fetch("https://unused.local/chat/completions", {
    method: "POST",
    body: JSON.stringify({ model: "fast", messages: [{ role: "user", content: "Hi" }] }),
  })
  assert.equal(oauth401.status, 401)
  assert.equal(oauth401.headers.get("X-Magpie-Sign-In"), "expired")

  const dir401 = mkdtempSync(join(tmpdir(), "aq-401-"))
  try {
    writeFileSync(
      join(dir401, "profiles.json"),
      JSON.stringify({
        version: 1,
        last_active: "p1",
        entries: [{ id: "p1", provider: "social", login_metadata: { tenant_url: "https://qbs.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev" } }],
      }),
    )
    const desktopRejectFactory = createAmazonQuickPlugin({
      dir: dir401,
      WebSocket: RejectWS,
      readKeyring: () =>
        JSON.stringify({
          access_token: "desk-bad",
          tenant_url: "https://qbs.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
        }),
    })
    const desktopRejectPlugin = await desktopRejectFactory({})
    const deskLoader = await desktopRejectPlugin.auth.loader(async () => ({ type: "oauth", source: "desktop", profileId: "p1" }))
    const desk401 = await deskLoader.fetch("https://unused.local/chat/completions", {
      method: "POST",
      body: JSON.stringify({ model: "fast", messages: [{ role: "user", content: "Hi" }] }),
    })
    assert.equal(desk401.status, 401)
    assert.equal(desk401.headers.get("X-Magpie-Sign-In"), "kept")
  } finally {
    rmSync(dir401, { recursive: true, force: true })
  }
})

test("browserSignIn registers quicksight:read_write OIDC client and handles loopback callback", async () => {
  const dir = mkdtempSync(join(tmpdir(), "aq-browser-"))
  try {
    writeFileSync(
      join(dir, "profiles.json"),
      JSON.stringify({
        version: 1,
        last_active: "p1",
        entries: [
          {
            id: "p1",
            provider: "social",
            login_metadata: {
              issuer_url: "https://identitycenter.amazonaws.com/ssoins-7223e642b2afb64c",
              tenant_url: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
            },
          },
        ],
      }),
    )

    let registeredBody = null
    let tokenBody = null
    const mockFetch = async (url, init) => {
      if (url.endsWith("/client/register")) {
        registeredBody = JSON.parse(init.body)
        return new Response(JSON.stringify({ clientId: "cid-pub", clientSecret: "csec-1" }), { status: 200 })
      }
      if (url.endsWith("/token")) {
        tokenBody = JSON.parse(init.body)
        return new Response(JSON.stringify({ accessToken: "tok-browser", refreshToken: "ref-browser", expiresIn: 3600 }), { status: 200 })
      }
      return new Response("{}", { status: 404 })
    }

    const flow = await browserSignIn({}, { dir, fetch: mockFetch })
    assert.deepEqual(registeredBody.scopes, ["quicksight:read_write"])
    assert.ok(registeredBody.redirectUris[0].startsWith("http://127.0.0.1:"))

    const u = new URL(flow.url)
    assert.equal(u.origin, "https://oidc.us-east-1.amazonaws.com")
    assert.equal(u.searchParams.get("scope"), "quicksight:read_write")
    assert.equal(u.searchParams.get("prompt"), "login")
    const redirectUri = u.searchParams.get("redirect_uri")
    const state = u.searchParams.get("state")

    const cbPromise = flow.callback()
    const hit = await fetch(`${redirectUri}/?code=auth-code-99&state=${encodeURIComponent(state)}`)
    assert.equal(hit.status, 200)

    const result = await cbPromise
    assert.equal(result.type, "success")
    assert.equal(result.access, "tok-browser")
    assert.equal(result.refresh, "ref-browser")
    assert.equal(result.providerType, "social")
    assert.equal(result.tenantUrl, "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev")
    assert.equal(tokenBody.code, "auth-code-99")
    assert.ok(tokenBody.codeVerifier)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test("createToolCallSplitter parses Anthropic <function_calls><invoke> XML blocks and coerces schema types", () => {
  const tools = [
    {
      type: "function",
      function: {
        name: "read_file",
        parameters: { type: "object", properties: { path: { type: "string" }, limit: { type: "integer" }, raw: { type: "boolean" } } },
      },
    },
    {
      type: "function",
      function: {
        name: "write_file",
        parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } } },
      },
    },
  ]
  const sample =
    'Intro\n<function_calls><invoke name="read_file"><parameter name="path">123</parameter><parameter name="limit">50</parameter><parameter name="raw">true</parameter></invoke><invoke name="write_file"><parameter name="path">a.py</parameter><parameter name="content">\nprint("hi")\n</parameter></invoke></function_calls>\nDone.'
  const splitter = createToolCallSplitter(true, tools)
  let text = ""
  const outCalls = []
  for (let i = 0; i < sample.length; i += 2) {
    const r = splitter.push(sample.slice(i, i + 2))
    text += r.text
    outCalls.push(...r.toolCalls)
  }
  const end = splitter.flush()
  text += end.text
  outCalls.push(...end.toolCalls)
  assert.equal(text, "Intro\n\nDone.")
  assert.equal(outCalls.length, 2)
  assert.equal(outCalls[0].function.name, "read_file")
  assert.deepEqual(JSON.parse(outCalls[0].function.arguments), { path: "123", limit: 50, raw: true })
  assert.equal(outCalls[1].function.name, "write_file")
  assert.deepEqual(JSON.parse(outCalls[1].function.arguments), { path: "a.py", content: 'print("hi")' })
})

test("stateful thread_id incremental continuation reuses threads, isolates accounts, and recovers from stale errors", async () => {
  const store = createThreadStore()
  const sentFrames = []
  let failResumed = false

  const decodeOutboundAidp = (rawBytes) => {
    const outer = JSON.parse(Buffer.from(rawBytes).toString("utf8"))
    return JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
  }

  const MockWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      const frames = ws.sent.map(decodeOutboundAidp)
      sentFrames.push(frames)
      const connect = frames.find((f) => f.type === "session_connect")?.payload
      const userMsg = frames.find((f) => f.type === "user_message")?.payload
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      if (failResumed && connect?.focus_only) {
        failResumed = false
        ws.emitAidpEvent({ type: "error", payload: { message: "session not found" } })
        return
      }
      ws.emitAidpEvent({ type: "message_start", payload: {} })
      const reply =
        sentFrames.length === 1
          ? '<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>'
          : `Answer for ${userMsg.thread_id}`
      ws.emitAidpEvent({ type: "text_delta", payload: { delta: reply } })
      ws.emitAidpEvent({ type: "message_complete", payload: {} })
    })
  })

  const acctA = { providerType: "social", access: "tok-a", tenantUrl: "https://qbs-a.plato.ai.aws.dev", accountId: "acct-a" }
  const acctB = { providerType: "social", access: "tok-b", tenantUrl: "https://qbs-b.plato.ai.aws.dev", accountId: "acct-b" }
  const tools = [{ type: "function", function: { name: "read_file", parameters: { type: "object", properties: { path: { type: "string" } } } } }]
  const sys = { role: "system", content: "SYSTEM_PROMPT_V1" }
  const u1 = { role: "user", content: "Inspect a.txt" }

  // Turn 1: fresh thread
  const r1 = await (await executeSncsChat(acctA, { model: "fast", stream: false, tools, messages: [sys, u1] }, { WebSocket: MockWS, threadStore: store })).json()
  assert.equal(r1.choices[0].finish_reason, "tool_calls")
  const t1Connect = sentFrames[0].find((f) => f.type === "session_connect").payload
  const t1Msg = sentFrames[0].find((f) => f.type === "user_message").payload
  assert.equal(t1Connect.session_id, undefined)
  assert.ok(t1Msg.text.includes("SYSTEM_PROMPT_V1"))

  // Turn 2 (no sessionId header, IDE mutates assistant content): matches tool_call_id / head index, reuses thread_id and omits <instructions>
  const a1 = { ...r1.choices[0].message, content: "IDE reformatted assistant preface" }
  const tr1 = { role: "tool", tool_call_id: a1.tool_calls[0].id, content: "SECRET_LINE" }
  const r2 = await (await executeSncsChat(acctA, { model: "fast", stream: false, tools, messages: [sys, u1, a1, tr1] }, { WebSocket: MockWS, threadStore: store })).json()
  const t2Connect = sentFrames[1].find((f) => f.type === "session_connect").payload
  const t2Msg = sentFrames[1].find((f) => f.type === "user_message").payload
  assert.equal(t2Connect.session_id, t1Msg.thread_id)
  assert.equal(t2Connect.focus_only, true)
  assert.equal(t2Msg.thread_id, t1Msg.thread_id)
  assert.ok(!t2Msg.text.includes("SYSTEM_PROMPT_V1"))
  assert.ok(t2Msg.text.includes("SECRET_LINE"))
  assert.ok(t2Msg.text.includes("<client_tool_directive>"))
  assert.ok(t2Msg.text.includes("live execution outputs"))
  assert.ok(r2.choices[0].message.content.includes(t1Msg.thread_id))

  // Turn 3 with different account: must NOT reuse acctA's threadId
  const a2 = r2.choices[0].message
  const u2 = { role: "user", content: "Follow up on other account" }
  await executeSncsChat(acctB, { model: "fast", stream: false, tools, messages: [sys, u1, a1, tr1, a2, u2] }, { WebSocket: MockWS, threadStore: store })
  const t3Msg = sentFrames[2].find((f) => f.type === "user_message").payload
  assert.notEqual(t3Msg.thread_id, t1Msg.thread_id)
  assert.ok(t3Msg.text.includes("SYSTEM_PROMPT_V1"))

  // Turn 3 on acctA with stale server thread (stream: true): fails on resume and auto-retries fresh thread
  failResumed = true
  const sRes = await executeSncsChat(acctA, { model: "fast", stream: true, tools, messages: [sys, u1, a1, tr1, a2, { role: "user", content: "Continue" }] }, { WebSocket: MockWS, threadStore: store })
  const sseBody = await sRes.text()
  assert.equal(sentFrames.length, 5) // turn 4 resumed + rejected -> turn 5 auto-retried fresh!
  const retryMsg = sentFrames[4].find((f) => f.type === "user_message").payload
  assert.notEqual(retryMsg.thread_id, t1Msg.thread_id)
  assert.ok(retryMsg.text.includes("SYSTEM_PROMPT_V1"))
  assert.ok(sseBody.includes(retryMsg.thread_id))

  // Turn 4 on acctA with updated IDE dynamic context inside system prompt: STILL reuses retryMsg.thread_id and sends <system_context_update> diff!
  const sysDynamic = { role: "system", content: "SYSTEM_PROMPT_V1\nOpen file: app.py (line 42)" }
  await executeSncsChat(
    acctA,
    {
      model: "fast",
      stream: false,
      tools,
      messages: [sysDynamic, u1, a1, tr1, a2, { role: "user", content: "Continue" }, { role: "assistant", content: `Answer for ${retryMsg.thread_id}` }, { role: "user", content: "Check line 42" }],
    },
    { WebSocket: MockWS, threadStore: store },
  )
  const dynMsg = sentFrames[5].find((f) => f.type === "user_message").payload
  assert.equal(dynMsg.thread_id, retryMsg.thread_id)
  assert.ok(!dynMsg.text.includes("SYSTEM_PROMPT_V1"))
  assert.ok(dynMsg.text.includes("<system_context_update>\n+Open file: app.py (line 42)\n</system_context_update>"))
})

test("_chunked_frame reassembly, WebSocket connect/idle timeouts, uploadQuickAttachment, and image attachments work cleanly", async () => {
  // 1. Out-of-order _chunked_frame reassembly + text_complete
  const ChunkWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      ws.emitAidpEvent({ type: "message_start", payload: {} })
      const inner = JSON.stringify({ type: "text_complete", payload: { text: "Chunked reply ok!" } })
      const half = Math.ceil(inner.length / 2)
      ws.emitAidpEvent({ type: "_chunked_frame", chunk_id: "c9", chunk_index: 1, total_chunks: 2, data: inner.slice(half) })
      ws.emitAidpEvent({ type: "_chunked_frame", chunk_id: "c9", chunk_index: 0, total_chunks: 2, data: inner.slice(0, half) })
      ws.emitAidpEvent({ type: "message_complete", payload: {} })
    })
  })
  const acct = { providerType: "social", access: "tok", tenantUrl: "https://qbs-test.plato.ai.aws.dev" }
  const chunkRes = await executeSncsChat(acct, { model: "fast", stream: false, messages: [{ role: "user", content: "Hi" }] }, { WebSocket: ChunkWS, stateful: false })
  const chunkJson = await chunkRes.json()
  assert.equal(chunkJson.choices[0].message.content, "Chunked reply ok!")

  // 2. Connect timeout rejects cleanly with 504
  const HangWS = makeMockWebSocketClass(() => {})
  await assert.rejects(
    executeSncsChat(acct, { model: "fast", messages: [{ role: "user", content: "Hi" }] }, { WebSocket: HangWS, connectTimeoutMs: 40, stateful: false }),
    (e) => e.status === 504 && /timed out/i.test(e.message),
  )

  // 3. uploadQuickAttachment (AIDP) + multimodal image_url attachment in executeSncsChat
  const calls = []
  const mockFetch = async (url, init) => {
    calls.push({ url, method: init?.method || "GET", body: init?.body })
    if (url.includes("/integration/quick-work/upload-attachment")) {
      return new Response(JSON.stringify({ attachmentId: "att-123" }), { status: 200 })
    }
    if (url.includes("/integration/quick-work/attachment-dlp-status")) {
      return new Response(JSON.stringify({ dlpJobStatus: "COMPLETED", policyAction: "ALLOW" }), { status: 200 })
    }
    if (url.includes("/integration/quick-work/qbs-event")) {
      return new Response(JSON.stringify({ eventResponse: { invokeAgentResponse: JSON.stringify({ statusCode: 200, body: JSON.stringify({ path: "/workspace/image_001.png" }) }) } }), { status: 200 })
    }
    return new Response("{}", { status: 404 })
  }
  const uploaded = await uploadQuickAttachment(
    acct,
    { name: "shot.png", mimeType: "image/png", bytes: new Uint8Array([1, 2, 3, 4]), threadId: "thread-xyz" },
    { fetch: mockFetch, dlpPollMs: 0 },
  )
  assert.equal(uploaded.attachmentId, "att-123")
  assert.equal(uploaded.path, "/workspace/image_001.png")

  const sentPayloads = []
  const ImgWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      for (const raw of ws.sent) {
        const outer = JSON.parse(Buffer.from(raw).toString("utf8"))
        const inner = JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
        if (inner.type === "user_message") sentPayloads.push(inner.payload)
      }
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      ws.emitAidpEvent({ type: "text_delta", payload: { delta: "Saw image" } })
      ws.emitAidpEvent({ type: "message_complete", payload: {} })
    })
  })
  const b64Png = Buffer.from([137, 80, 78, 71]).toString("base64")
  const imgRes = await executeSncsChat(
    acct,
    {
      model: "fast",
      stream: false,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What is in this screenshot?" },
            { type: "image_url", image_url: { url: `data:image/png;base64,${b64Png}` } },
          ],
        },
      ],
    },
    { WebSocket: ImgWS, fetch: mockFetch, dlpPollMs: 0, stateful: false },
  )
  assert.equal((await imgRes.json()).choices[0].message.content, "Saw image")
  assert.equal(sentPayloads[0].attachments.length, 1)
  assert.equal(sentPayloads[0].attachments[0].type, "image")
  assert.equal(sentPayloads[0].attachments[0].attachmentId, "att-123")
  assert.equal(sentPayloads[0].attachments[0].path, "/workspace/image_001.png")
})

test("readDesktop mtime cache avoids repeated OS keyring calls and Gateway uploadQuickAttachment works", async () => {
  const dir = mkdtempSync(join(tmpdir(), "aq-cache-"))
  try {
    writeFileSync(
      join(dir, "profiles.json"),
      JSON.stringify({
        version: 1,
        last_active: "social-cache",
        entries: [
          {
            id: "social-cache",
            provider: "social",
            display_name: "Cache User",
            login_metadata: {
              tenant_url: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
              client_id: "cid-cache",
            },
          },
        ],
      }),
    )

    let osCalls = 0
    const secretJson = JSON.stringify({
      accessToken: "acc-cached",
      refreshToken: "ref-cached",
      expiresAt: "2026-11-22T12:00:00.000Z",
    })
    const mockExec = () => {
      osCalls++
      return secretJson
    }
    let mtimeMs = 1_000_000
    const mockStat = () => ({ mtimeMs })
    const baseNow = Date.parse("2026-11-22T11:30:00.000Z")

    const d1 = readDesktop({ dir, platform: "win32", execFileSync: mockExec, statSync: mockStat, now: baseNow, noCache: false })
    assert.equal(d1.access, "acc-cached")
    assert.equal(osCalls, 1)

    // 10s later (> 2s short TTL), same mtimeMs and unexpired token -> should reuse cache without spawning OS reader
    const d2 = readDesktop({ dir, platform: "win32", execFileSync: mockExec, statSync: mockStat, now: baseNow + 10_000 })
    assert.equal(d2.access, "acc-cached")
    assert.equal(osCalls, 1)

    // profiles.json mtime changed -> should re-read OS keyring
    mtimeMs = 2_000_000
    const d3 = readDesktop({ dir, platform: "win32", execFileSync: mockExec, statSync: mockStat, now: baseNow + 10_000 })
    assert.equal(d3.access, "acc-cached")
    assert.equal(osCalls, 2)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }

  // Gateway uploadQuickAttachment flow (PUT presigned-url + DLP describe-user-aware + files/upload)
  const gwCalls = []
  const gwFetch = async (url, init) => {
    gwCalls.push({ url, method: init?.method || "GET" })
    if (url.includes("/uploads/accounts/123456789012/attachments/presigned-url")) {
      return new Response(JSON.stringify({ attachmentId: "gw-att-1", dlpJobId: "dlp-job-9" }), { status: 200 })
    }
    if (url.includes("/dlp/internal/123456789012/data-loss-prevention/job/describe-user-aware")) {
      return new Response(JSON.stringify({ DlpJobStatus: "COMPLETED", PolicyAction: "ALLOW" }), { status: 200 })
    }
    if (url.includes("/invocations/rpc")) {
      return new Response(JSON.stringify({ statusCode: 200, body: JSON.stringify({ path: "/workspace/gw-file.txt" }) }), { status: 200 })
    }
    return new Response("{}", { status: 404 })
  }
  const gwAcct = {
    providerType: "gateway",
    access: "gw-tok",
    tenantUrl: "https://qbs.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
    gatewayHost: "cell-1.us-east-1.quick.aws.dev",
    awsAccountId: "123456789012",
    region: "us-east-1",
  }
  const gwUp = await uploadQuickAttachment(
    gwAcct,
    { name: "context.txt", mimeType: "text/plain", bytes: Buffer.from("hello gateway"), threadId: "t-gw" },
    { fetch: gwFetch, dlpPollMs: 0 },
  )
  assert.equal(gwUp.attachmentId, "gw-att-1")
  assert.equal(gwUp.path, "/workspace/gw-file.txt")
  assert.ok(gwCalls.some((c) => c.method === "PUT" && c.url.includes("/uploads/accounts/123456789012/attachments/presigned-url")))
  assert.ok(gwCalls.some((c) => c.method === "POST" && c.url.includes("/dlp/internal/123456789012/data-loss-prevention/job/describe-user-aware")))
  assert.ok(gwCalls.some((c) => c.method === "POST" && c.url.includes("/invocations/rpc")))
})

test("idleTimeoutMs, overflowUserText upload, fetchQuickSettings cache, and browserSignIn abandon cleanup work", async () => {
  const acct = { providerType: "social", access: "tok-extra", tenantUrl: "https://qbs-test.plato.ai.aws.dev", accountId: "extra-acct" }

  // 1. WebSocket idle timeout rejects cleanly with 504 after handshake
  const StalledWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      ws.emitAidpEvent({ type: "message_start", payload: {} })
      // Never sends message_complete
    })
  })
  await assert.rejects(
    executeSncsChat(acct, { model: "fast", stream: false, messages: [{ role: "user", content: "Hi" }] }, { WebSocket: StalledWS, idleTimeoutMs: 40, stateful: false }),
    (e) => e.status === 504 && /timed out/i.test(e.message),
  )

  // 2. Oversized user prompt triggers prompt_context.txt upload with MIME "text/plain" (no charset suffix)
  const sentUserPayloads = []
  let uploadedMime = ""
  let uploadedName = ""
  const mockUpload = async (_acct, file) => {
    uploadedName = file.name
    uploadedMime = file.mimeType
    return {
      type: "file",
      name: file.name,
      size: file.bytes.byteLength,
      mimeType: file.mimeType,
      path: `/workspace/attached_files/${file.name}`,
      attachmentId: "att-big-1",
      contentSha256: "abc123",
    }
  }
  const OverflowWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      for (const raw of ws.sent) {
        const outer = JSON.parse(Buffer.from(raw).toString("utf8"))
        const inner = JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
        if (inner.type === "user_message") sentUserPayloads.push(inner.payload)
      }
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      ws.emitAidpEvent({ type: "text_delta", payload: { delta: "Read overflow file" } })
      ws.emitAidpEvent({ type: "message_complete", payload: {} })
    })
  })
  const hugePrompt = "Intro " + "X".repeat(65_000) + " End"
  const overRes = await executeSncsChat(
    acct,
    { model: "fast", stream: false, messages: [{ role: "user", content: hugePrompt }] },
    { WebSocket: OverflowWS, uploadAttachment: mockUpload, stateful: false },
  )
  assert.equal((await overRes.json()).choices[0].message.content, "Read overflow file")
  assert.match(uploadedName, /^prompt_context_[a-z0-9]+\.txt$/i)
  assert.equal(uploadedMime, "text/plain")
  assert.equal(sentUserPayloads[0].attachments.length, 1)
  assert.equal(sentUserPayloads[0].attachments[0].attachmentId, "att-big-1")

  // 3. fetchQuickSettings reuses 2-min cache when cache is enabled
  let settingsHits = 0
  const sFetch = async () => {
    settingsHits++
    return new Response(JSON.stringify({ eventResponse: { invokeAgentResponse: JSON.stringify({ statusCode: 200, body: JSON.stringify({ frames: [{ type: "settings", payload: { db_limit: 5 } }] }) }) } }), { status: 200 })
  }
  const s1 = await fetchQuickSettings(acct, { fetch: sFetch, cache: true, now: 1_000_000 })
  const s2 = await fetchQuickSettings(acct, { fetch: sFetch, cache: true, now: 1_060_000 })
  assert.equal(s1.db_limit, 5)
  assert.equal(s2.db_limit, 5)
  assert.equal(settingsHits, 1)

  // 4. browserSignIn abandon timer closes local callback server cleanly
  const signInFetch = async (url) => {
    if (String(url).endsWith("/client/register")) {
      return new Response(JSON.stringify({ clientId: "cid-abandon", clientSecret: "sec" }), { status: 200 })
    }
    return new Response("{}", { status: 404 })
  }
  const flow = await browserSignIn(
    { tenantUrl: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev" },
    { fetch: signInFetch, abandonMs: 30, dir: join(tmpdir(), "aq-nonexistent-" + Date.now()) },
  )
  assert.match(flow.url, /^https:\/\/oidc\.us-east-1\.amazonaws\.com\/authorize\?/)
  const abRes = await flow.callback()
  assert.equal(abRes.type, "failed")
  assert.match(abRes.error, /timed out/i)
})

test("macOS Keychain timeout/denied negative cache, actionable errors, and browser-first sign-in methods", async () => {
  const dir = mkdtempSync(join(tmpdir(), "aq-mac-"))
  try {
    writeFileSync(
      join(dir, "profiles.json"),
      JSON.stringify({
        version: 1,
        last_active: "social-mac",
        entries: [
          {
            id: "social-mac",
            provider: "social",
            display_name: "Mac User",
            login_metadata: {
              tenant_url: "https://qbs-mac.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
              client_id: "cid-mac",
            },
          },
        ],
      }),
    )

    // 1. Timeout records negative cache so subsequent calls within 45s do NOT re-spawn `security`
    clearDesktopCache()
    let calls = 0
    let seenOpts = null
    const timeoutExec = (_cmd, _args, opts) => {
      calls++
      seenOpts = opts
      const err = new Error("spawnSync security ETIMEDOUT")
      err.code = "ETIMEDOUT"
      throw err
    }
    const t0 = 5_000_000
    assert.equal(readDesktop({ dir, platform: "darwin", execFileSync: timeoutExec, now: t0 }), null)
    assert.equal(calls, 1)
    assert.equal(seenOpts.timeout, MACOS_KEYCHAIN_TIMEOUT_MS)
    assert.deepEqual(seenOpts.stdio, ["ignore", "pipe", "pipe"])
    assert.match(notDesktop({ dir, platform: "darwin" }), /macOS Keychain authorization timed out/i)
    assert.match(notDesktop({ dir, platform: "darwin" }), /Always Allow/i)

    // Within 45s negative cache window -> returns null without invoking `security` again
    assert.equal(readDesktop({ dir, platform: "darwin", execFileSync: timeoutExec, now: t0 + 10_000 }), null)
    assert.equal(calls, 1)

    // After 45s window -> retries and classifies user denial cleanly
    const deniedExec = () => {
      calls++
      const err = new Error("Command failed")
      err.stderr = "security: SecKeychainSearchCopyNext: User interaction is not allowed."
      throw err
    }
    assert.equal(readDesktop({ dir, platform: "darwin", execFileSync: deniedExec, now: t0 + 50_000 }), null)
    assert.equal(calls, 2)
    assert.match(notDesktop({ dir, platform: "darwin" }), /macOS Keychain denied access/i)
    assert.match(notDesktop({ dir, platform: "darwin" }), /browser/i)

    // 2. Auth methods list puts browser OIDC first and labels macOS Keychain caveat on desktop sign-in
    const macPlugin = await createAmazonQuickPlugin({ platform: "darwin" })({})
    assert.equal(macPlugin.auth.methods[0].label, "Amazon Quick Plus 浏览器登录（推荐）")
    assert.match(macPlugin.auth.methods[1].label, /macOS 需钥匙串密码/)
    assert.match(macPlugin.auth.methods[2].label, /设备码登录/)
    assert.match(macPlugin.auth.methods[3].label, /企业版 \/ 自定义 OIDC/)
  } finally {
    clearDesktopCache()
    rmSync(dir, { recursive: true, force: true })
  }
})

test("deviceCodeSignIn supports headless/VPS OAuth device polling, slow_down, abort, and bare hostnames", async () => {
  const missingDir = join(tmpdir(), "aq-missing-dc-" + Date.now())

  // 1. Missing tenantUrl without local desktop profiles.json fails with actionable VPS hint
  await assert.rejects(
    deviceCodeSignIn({}, { dir: missingDir }),
    (e) => /Missing Amazon Quick Tenant URL/i.test(e.message) && /profiles\.json/i.test(e.message),
  )

  // 2. Happy path: bare hostnames normalized, polls authorization_pending -> slow_down -> success
  const seenCalls = []
  let pollAttempt = 0
  const mockFetch = async (url, init) => {
    const body = JSON.parse(init.body)
    seenCalls.push({ url: String(url), body })
    if (String(url).endsWith("/client/register")) {
      assert.deepEqual(body.grantTypes, ["urn:ietf:params:oauth:grant-type:device_code", "refresh_token"])
      assert.equal(body.issuerUrl, "https://identitycenter.amazonaws.com/ssoins-7223b3b7b80ce833")
      return new Response(JSON.stringify({ clientId: "cid-dev-1", clientSecret: "sec-dev-1" }), { status: 200 })
    }
    if (String(url).endsWith("/device_authorization")) {
      assert.equal(body.clientId, "cid-dev-1")
      assert.equal(body.clientSecret, "sec-dev-1")
      assert.equal(body.startUrl, "https://identitycenter.amazonaws.com/ssoins-7223b3b7b80ce833")
      return new Response(
        JSON.stringify({
          deviceCode: "dev-code-123",
          userCode: "ABCD-EFGH",
          verificationUri: "https://us-east-1.signin.aws/platform/authorize",
          verificationUriComplete: "https://us-east-1.signin.aws/platform/authorize?user_code=ABCD-EFGH",
          expiresIn: 600,
          interval: 1,
        }),
        { status: 200 },
      )
    }
    if (String(url).endsWith("/token")) {
      pollAttempt++
      assert.equal(body.grantType, "urn:ietf:params:oauth:grant-type:device_code")
      assert.equal(body.deviceCode, "dev-code-123")
      if (pollAttempt === 1) {
        return new Response(JSON.stringify({ error: "authorization_pending" }), { status: 400 })
      }
      if (pollAttempt === 2) {
        return new Response(JSON.stringify({ error: "slow_down" }), { status: 400 })
      }
      return new Response(
        JSON.stringify({ accessToken: "acc-dev-ok", refreshToken: "ref-dev-ok", expiresIn: 3600 }),
        { status: 200 },
      )
    }
    return new Response("{}", { status: 404 })
  }

  const flow = await deviceCodeSignIn(
    {
      tenantUrl: "qbs-cell9d81.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev/",
      issuerUrl: "identitycenter.amazonaws.com/ssoins-7223b3b7b80ce833/",
    },
    { fetch: mockFetch, dir: missingDir, pollIntervalMs: 5, slowDownStepMs: 5 },
  )
  assert.equal(flow.method, "auto")
  assert.equal(flow.url, "https://us-east-1.signin.aws/platform/authorize?user_code=ABCD-EFGH")
  assert.match(flow.instructions, /ABCD-EFGH/)

  const res = await flow.callback()
  assert.equal(res.type, "success")
  assert.equal(res.access, "acc-dev-ok")
  assert.equal(res.refresh, "ref-dev-ok")
  assert.equal(res.clientId, "cid-dev-1")
  assert.equal(res.clientSecret, "sec-dev-1")
  assert.equal(res.tenantUrl, "https://qbs-cell9d81.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev")
  assert.equal(res.issuerUrl, "https://identitycenter.amazonaws.com/ssoins-7223b3b7b80ce833")
  assert.equal(res.providerType, "social")
  assert.equal(pollAttempt, 3)

  // 3. Cancellation via AbortSignal stops polling immediately
  const ac = new AbortController()
  const abortFlow = await deviceCodeSignIn(
    { tenantUrl: "https://tenant.desktop.enterprise.quick.aws.com" },
    {
      fetch: async (url) => {
        if (String(url).endsWith("/client/register")) return new Response(JSON.stringify({ clientId: "cid-ab" }), { status: 200 })
        if (String(url).endsWith("/device_authorization")) {
          return new Response(JSON.stringify({ deviceCode: "dc-ab", verificationUriComplete: "https://example.aws/verify", expiresIn: 600, interval: 5 }), { status: 200 })
        }
        return new Response(JSON.stringify({ error: "authorization_pending" }), { status: 400 })
      },
      dir: missingDir,
      pollIntervalMs: 5000,
    },
  )
  setTimeout(() => ac.abort(), 15)
  const abRes = await abortFlow.callback(undefined, { signal: ac.signal })
  assert.equal(abRes.type, "failed")
  assert.match(abRes.error, /cancelled/i)

  // 4. access_denied returns clean failure
  const deniedFlow = await deviceCodeSignIn(
    { tenantUrl: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev" },
    {
      fetch: async (url) => {
        if (String(url).endsWith("/client/register")) return new Response(JSON.stringify({ clientId: "cid-den" }), { status: 200 })
        if (String(url).endsWith("/device_authorization")) {
          return new Response(JSON.stringify({ deviceCode: "dc-den", verificationUri: "https://example.aws/verify", expiresIn: 600, interval: 1 }), { status: 200 })
        }
        return new Response(JSON.stringify({ error: "access_denied" }), { status: 400 })
      },
      dir: missingDir,
      pollIntervalMs: 5,
    },
  )
  const denRes = await deniedFlow.callback()
  assert.equal(denRes.type, "failed")
  assert.match(denRes.error, /denied/i)
})

test("parallel tool_result blocks in current turn are protected, proportionally budgeted, and fully spilled to overflowUserText", () => {
  const seg1 = "SEG1_HEAD\n" + "A".repeat(22_000) + "\nSEG1_TAIL"
  const seg2 = "SEG2_HEAD\n" + "B".repeat(22_000) + "\nSEG2_TAIL"
  const seg3 = "SEG3_HEAD\n" + "C".repeat(22_000) + "\nSEG3_TAIL"
  const oldTool = "OLD_TOOL_" + "Z".repeat(5_000)

  const tools = [
    {
      type: "function",
      function: {
        name: "read_file",
        description: "Read file lines",
        parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
      },
    },
  ]

  // 1. Full-history request with older turns + null message + paired assistant tool_calls + 3 parallel read_file results
  const fullReq = openaiToSncsRequest({
    model: "balanced",
    tools,
    messages: [
      { role: "system", content: "You are a code auditor." },
      { role: "user", content: "Old turn question" },
      null,
      { role: "assistant", content: "Old answer", tool_calls: [{ id: "old_1", type: "function", function: { name: "read_file", arguments: '{"path":"old.mjs"}' } }] },
      { role: "tool", tool_call_id: "old_1", name: "read_file", content: oldTool },
      { role: "user", content: "Read bridge.mjs in 3 chunks" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "r1", type: "function", function: { name: "read_file", arguments: '{"path":"bridge.mjs","start":1,"end":555}' } },
          { id: "r2", type: "function", function: { name: "read_file", arguments: '{"path":"bridge.mjs","start":556,"end":1100}' } },
          { id: "r3", type: "function", function: { name: "read_file", arguments: '{"path":"bridge.mjs","start":1101,"end":1665}' } },
        ],
      },
      { role: "tool", tool_call_id: "r1", name: "read_file", content: seg1 },
      { role: "tool", tool_call_id: "r2", name: "read_file", content: seg2 },
      { role: "tool", tool_call_id: "r3", name: "read_file", content: seg3 },
    ],
  })

  const frameBytes = buildAidpInputFrame({ type: "user_message", payload: fullReq.userMessagePayload }).byteLength
  assert.ok(frameBytes <= MAX_USER_MESSAGE_FRAME_BYTES, `expected frame <= ${MAX_USER_MESSAGE_FRAME_BYTES}, got ${frameBytes}`)

  const outText = fullReq.userMessagePayload.text
  // Paired assistant tool_call block preserved; older turns dropped
  assert.ok(outText.includes('"start":1,"end":555'), "paired assistant tool_call block should survive Stage 2 shift")
  assert.ok(!outText.includes("OLD_TOOL_"), "older tool result should be dropped before touching current turn")

  // All 3 parallel tool results preserved inline with fair proportional budgets (far above old 600-char cap)
  const extractResult = (id) => {
    const m = outText.match(new RegExp(`<tool_result id="${id}" name="read_file">[\\s\\S]*?</tool_result>`))
    assert.ok(m, `missing inline tool_result for ${id}`)
    return m[0]
  }
  const inline1 = extractResult("r1")
  const inline2 = extractResult("r2")
  const inline3 = extractResult("r3")
  assert.ok(inline1.length > 8_000, `r1 should get proportional budget, got ${inline1.length}`)
  assert.ok(inline2.length > 8_000, `r2 should get proportional budget, got ${inline2.length}`)
  assert.ok(inline3.length > 8_000, `r3 should get proportional budget, got ${inline3.length}`)
  assert.ok(Math.abs(inline1.length - inline3.length) < 1_000, `equal-sized parallel results should get balanced budgets: ${inline1.length} vs ${inline3.length}`)

  // Informative truncation markers report original length and smaller line-range re-read hint,
  // WITHOUT prematurely claiming prompt_context.txt before upload succeeds
  for (const [id, block] of [["r1", inline1], ["r2", inline2], ["r3", inline3]]) {
    assert.match(block, /trimmed \d+ of \d+ chars to fit 64KB frame/, `${id} should report original/omitted length`)
    assert.ok(!block.includes("prompt_context.txt"), `${id} must not claim prompt_context.txt before upload succeeds`)
    assert.match(block, /smaller line range/i, `${id} should hint to re-read with smaller line range`)
  }

  // Full protected turn (all 3 tool results) spilled intact into overflowUserText
  assert.ok(fullReq.overflowUserText.includes(seg1), "overflowUserText must contain full seg1")
  assert.ok(fullReq.overflowUserText.includes(seg2), "overflowUserText must contain full seg2")
  assert.ok(fullReq.overflowUserText.includes(seg3), "overflowUserText must contain full seg3")

  // 2. Mixed-size parallel tool results on resume: short block stays 100% intact while large blocks split remaining budget
  const shortOut = "On branch main\nnothing to commit, working tree clean"
  const resumeReq = openaiToSncsRequest(
    { model: "balanced", tools, messages: [{ role: "user", content: "ignored on resume" }] },
    {
      resumeState: {
        threadId: "thread-par-1",
        workingSessionId: "ws-par-1",
        deltaMessages: [
          { role: "tool", tool_call_id: "p1", name: "read_file", content: "L1_" + "X".repeat(30_000) },
          { role: "tool", tool_call_id: "p2", name: "run_cmd", content: shortOut },
          { role: "tool", tool_call_id: "p3", name: "read_file", content: "L2_" + "Y".repeat(30_000) },
        ],
      },
    },
  )
  const resumeBytes = buildAidpInputFrame({ type: "user_message", payload: resumeReq.userMessagePayload }).byteLength
  assert.ok(resumeBytes <= MAX_USER_MESSAGE_FRAME_BYTES, `resume frame <= ${MAX_USER_MESSAGE_FRAME_BYTES}, got ${resumeBytes}`)
  assert.ok(resumeReq.userMessagePayload.text.includes(shortOut), "small parallel tool result must remain completely untruncated")
  assert.ok(resumeReq.overflowUserText.includes("X".repeat(30_000)), "resume overflowUserText must preserve full p1")
  assert.ok(resumeReq.overflowUserText.includes("Y".repeat(30_000)), "resume overflowUserText must preserve full p3")
  assert.ok(resumeReq.overflowUserText.includes(shortOut), "resume overflowUserText must preserve p2")
})

test("prompt_context.txt attachment note is added only after upload succeeds and final frame stays within 62 KB", async () => {
  assert.equal(MAX_USER_MESSAGE_FRAME_BYTES, 62_000)
  const acct = { providerType: "social", access: "tok-att", tenantUrl: "https://qbs-test.plato.ai.aws.dev", accountId: "att-acct" }
  const tools = [
    {
      type: "function",
      function: { name: "read_file", description: "Read", parameters: { type: "object", properties: { path: { type: "string" } } } },
    },
  ]
  const body = {
    model: "balanced",
    stream: false,
    tools,
    messages: [
      { role: "user", content: "Audit file" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "c1", type: "function", function: { name: "read_file", arguments: '{"path":"a.mjs"}' } },
          { id: "c2", type: "function", function: { name: "read_file", arguments: '{"path":"b.mjs"}' } },
        ],
      },
      { role: "tool", tool_call_id: "c1", name: "read_file", content: "A".repeat(32_000) },
      { role: "tool", tool_call_id: "c2", name: "read_file", content: "B".repeat(32_000) },
    ],
  }

  const runWithUploader = async (uploadAttachment) => {
    const sentFrames = []
    const MockWS = makeMockWebSocketClass((ws) => {
      ws.readyState = 1
      ws.onopen?.()
      queueMicrotask(() => {
        for (const raw of ws.sent) {
          const outer = JSON.parse(Buffer.from(raw).toString("utf8"))
          const inner = JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
          if (inner.type === "user_message") sentFrames.push({ bytes: Buffer.from(raw).byteLength, payload: inner.payload })
        }
        ws.emitAidpEvent({ type: "session_established", payload: {} })
        ws.emitAidpEvent({ type: "text_delta", payload: { delta: "ok" } })
        ws.emitAidpEvent({ type: "message_complete", payload: {} })
      })
    })
    await executeSncsChat(acct, body, { WebSocket: MockWS, uploadAttachment, stateful: false })
    assert.equal(sentFrames.length, 1)
    return sentFrames[0]
  }

  // 1. Upload succeeds -> prompt_context.txt attached, note placed before <client_tool_directive>, wire frame <= 62KB
  const okFrame = await runWithUploader(async (_a, file) => ({
    type: "file",
    name: file.name,
    size: file.bytes.byteLength,
    mimeType: file.mimeType,
    path: `/workspace/attached_files/${file.name}`,
    attachmentId: "att-ok-1",
    contentSha256: "sha-ok",
  }))
  assert.ok(okFrame.bytes <= MAX_USER_MESSAGE_FRAME_BYTES, `final frame with attachment + note must be <= ${MAX_USER_MESSAGE_FRAME_BYTES}, got ${okFrame.bytes}`)
  assert.match(okFrame.payload.attachments?.[0]?.name || "", /^tool_results_[a-z0-9]+\.txt$/i)
  assert.match(okFrame.payload.text, /attached in your Quick workspace at \/workspace\/attached_files\/tool_results_[a-z0-9]+\.txt/i)
  assert.match(okFrame.payload.text, /full content in attached file tool_results_[a-z0-9]+\.txt lines \d+-\d+/i)
  assert.ok(!okFrame.payload.text.includes("smaller line range"), "when attachment uploaded, must not contradict it with re-read hint")
  assert.match(okFrame.payload.text, /built-in workspace file reader/i)
  assert.ok(okFrame.payload.text.endsWith("</client_tool_directive>"), "note must be inserted before trailing <client_tool_directive>")

  // 2. Upload fails -> no attachment and NO misleading attachment claim in text
  const failFrame = await runWithUploader(async () => {
    throw new Error("DLP upload failed")
  })
  assert.ok(failFrame.bytes <= MAX_USER_MESSAGE_FRAME_BYTES)
  assert.equal(failFrame.payload.attachments, undefined)
  assert.ok(!failFrame.payload.text.includes("attached_files/"), "must NOT mention attached file when upload fails")
  assert.match(failFrame.payload.text, /smaller line range/i)

  // 3. Tool-less overflow whose body literally contains "<client_tool_directive>" (e.g. reading bridge.mjs)
  // must place the attachment note strictly AFTER the body, NEVER spliced into the middle of the text
  const sourceCodeWithLiteral =
    'const tag = "<client_tool_directive>"\n' + "X".repeat(65_000) + "\nconst endMarker = 'EOF'"
  const literalFrames = []
  const LiteralWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      for (const raw of ws.sent) {
        const outer = JSON.parse(Buffer.from(raw).toString("utf8"))
        const inner = JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
        if (inner.type === "user_message") literalFrames.push(inner.payload)
      }
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      ws.emitAidpEvent({ type: "text_delta", payload: { delta: "ok" } })
      ws.emitAidpEvent({ type: "message_complete", payload: {} })
    })
  })
  await executeSncsChat(
    acct,
    { model: "balanced", stream: false, messages: [{ role: "user", content: sourceCodeWithLiteral }] },
    {
      WebSocket: LiteralWS,
      stateful: false,
      uploadAttachment: async (_a, file) => ({
        type: "file",
        name: file.name,
        size: file.bytes.byteLength,
        mimeType: file.mimeType,
        path: `/workspace/attached_files/${file.name}`,
        attachmentId: "att-lit-1",
        contentSha256: "sha-lit",
      }),
    },
  )
  assert.equal(literalFrames.length, 1)
  const litText = literalFrames[0].text
  assert.ok(litText.includes('const tag = "<client_tool_directive>"'), "source literal must remain untouched")
  assert.ok(
    litText.indexOf("const endMarker = 'EOF'") < litText.indexOf("attached in your Quick workspace"),
    "attachment note must be placed strictly after body, not spliced at literal <client_tool_directive>",
  )
})

test("resumed thread sends incremental <tools_update> when toolset changes across turns and compacts large tool deltas", async () => {
  const store = createThreadStore()
  const acct = { providerType: "social", access: "tok-tools", tenantUrl: "https://qbs-test.plato.ai.aws.dev", accountId: "tools-acct" }
  const readTool = {
    type: "function",
    function: {
      name: "read_file",
      description: "Read local file",
      parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
    },
  }
  const writeTool = {
    type: "function",
    function: {
      name: "write_file",
      description: "Write local file",
      parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
    },
  }
  const mcpTool = {
    type: "function",
    function: {
      name: "mcp_search",
      description: "Search docs via MCP",
      parameters: { type: "object", properties: { query: { type: "string" }, limit: { type: "integer" } }, required: ["query"] },
    },
  }

  const sentPayloads = []
  const MockWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      for (const raw of ws.sent) {
        const outer = JSON.parse(Buffer.from(raw).toString("utf8"))
        const inner = JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
        if (inner.type === "user_message") sentPayloads.push(inner.payload)
      }
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      ws.emitAidpEvent({ type: "text_delta", payload: { delta: "done" } })
      ws.emitAidpEvent({ type: "message_complete", payload: {} })
    })
  })

  // Turn 1: initial thread with [read_file, write_file]
  await executeSncsChat(
    acct,
    {
      model: "balanced",
      stream: false,
      tools: [readTool, writeTool],
      messages: [
        { role: "system", content: "Workspace agent" },
        { role: "user", content: "Start auditing" },
      ],
    },
    { WebSocket: MockWS, threadStore: store, sessionId: "sess-tools-1" },
  )
  assert.equal(sentPayloads.length, 1)
  const tid = sentPayloads[0].thread_id
  assert.ok(sentPayloads[0].text.includes("<instructions>"))
  assert.ok(!sentPayloads[0].text.includes("<tools_update>"))

  // Turn 2: resumed on same thread, write_file removed & mcp_search added (e.g., Zed profile switch / Claude ToolSearch)
  await executeSncsChat(
    acct,
    {
      model: "balanced",
      stream: false,
      tools: [readTool, mcpTool],
      messages: [
        { role: "system", content: "Workspace agent" },
        { role: "user", content: "Start auditing" },
        { role: "assistant", content: "done" },
        { role: "user", content: "Now search docs with MCP" },
      ],
    },
    { WebSocket: MockWS, threadStore: store, sessionId: "sess-tools-1" },
  )
  assert.equal(sentPayloads.length, 2)
  assert.equal(sentPayloads[1].thread_id, tid, "should preserve 1M server thread_id when tools change")
  assert.ok(!sentPayloads[1].text.includes("<instructions>"), "resumed turn should not resend full <instructions>")
  const tuMatch = sentPayloads[1].text.match(/<tools_update>[\s\S]*?<\/tools_update>/)
  assert.ok(tuMatch, "resumed turn must emit <tools_update> when toolsSig changes")
  assert.ok(tuMatch[0].includes("mcp_search"), "<tools_update> must include newly added tool schema")
  assert.ok(tuMatch[0].includes('"limit":{"type":"integer"}'), "<tools_update> must include parameter schema of new tool")
  assert.ok(tuMatch[0].includes("Removed client tools (no longer available): write_file"), "<tools_update> must list removed tools")
  assert.ok(!tuMatch[0].includes("Read local file"), "<tools_update> should not redundantly resend unchanged read_file schema")

  // Turn 3: same updated toolset -> no redundant <tools_update>
  await executeSncsChat(
    acct,
    {
      model: "balanced",
      stream: false,
      tools: [readTool, mcpTool],
      messages: [
        { role: "system", content: "Workspace agent" },
        { role: "user", content: "Start auditing" },
        { role: "assistant", content: "done" },
        { role: "user", content: "Now search docs with MCP" },
        { role: "assistant", content: "done" },
        { role: "user", content: "Summarize findings" },
      ],
    },
    { WebSocket: MockWS, threadStore: store, sessionId: "sess-tools-1" },
  )
  assert.equal(sentPayloads.length, 3)
  assert.equal(sentPayloads[2].thread_id, tid)
  assert.ok(!sentPayloads[2].text.includes("<tools_update>"), "unchanged toolset should not re-emit <tools_update>")

  // Turn 4: tool-less sidecar request (e.g. title generation) runs stateless on a fresh thread
  // without polluting or overwriting the tool-bearing main thread
  await executeSncsChat(
    acct,
    {
      model: "balanced",
      stream: false,
      tools: [],
      messages: [
        { role: "system", content: "Workspace agent" },
        { role: "user", content: "Start auditing" },
        { role: "assistant", content: "done" },
        { role: "user", content: "Now search docs with MCP" },
        { role: "assistant", content: "done" },
        { role: "user", content: "Summarize findings" },
        { role: "assistant", content: "done" },
        { role: "user", content: "Generate a concise 3-7 word title" },
      ],
    },
    { WebSocket: MockWS, threadStore: store, sessionId: "sess-tools-1" },
  )
  assert.equal(sentPayloads.length, 4)
  assert.notEqual(sentPayloads[3].thread_id, tid, "tool-less sidecar must not attach to tool-bearing main thread")
  assert.ok(!sentPayloads[3].text.includes("<tools_update>"), "stateless sidecar should not emit <tools_update>")

  // Turn 5: subsequent tool-bearing turn still resumes the original main thread cleanly
  await executeSncsChat(
    acct,
    {
      model: "balanced",
      stream: false,
      tools: [readTool, mcpTool],
      messages: [
        { role: "system", content: "Workspace agent" },
        { role: "user", content: "Start auditing" },
        { role: "assistant", content: "done" },
        { role: "user", content: "Now search docs with MCP" },
        { role: "assistant", content: "done" },
        { role: "user", content: "Summarize findings" },
        { role: "assistant", content: "done" },
        { role: "user", content: "Continue main audit" },
      ],
    },
    { WebSocket: MockWS, threadStore: store, sessionId: "sess-tools-1" },
  )
  assert.equal(sentPayloads.length, 5)
  assert.equal(sentPayloads[4].thread_id, tid, "main thread must remain intact in cache after sidecar ran")
  assert.ok(!sentPayloads[4].text.includes("<tools_update>"))

  // Large <tools_update> (50 verbose MCP tools added on resume -> > 62 KB raw) gets compacted cleanly by Stage 1 under 62 KB
  const manyVerboseTools = Array.from({ length: 50 }, (_, i) => ({
    type: "function",
    function: {
      name: `mcp_bulk_${i}`,
      description: `Bulk MCP tool ${"D".repeat(500)} #${i}`,
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Long param doc ".repeat(40) },
          mode: { type: "string", enum: ["fast", "deep"] },
        },
        required: ["query"],
      },
    },
  }))
  const compactResume = openaiToSncsRequest(
    { model: "balanced", tools: [readTool, ...manyVerboseTools], messages: [{ role: "user", content: "ignored" }] },
    {
      resumeState: {
        threadId: "thread-bulk-1",
        workingSessionId: "ws-bulk-1",
        toolsChanged: true,
        prevToolMap: { read_file: "prev-hash" },
        deltaMessages: [{ role: "user", content: "Use mcp_bulk_0" }],
      },
    },
  )
  const bulkBytes = buildAidpInputFrame({ type: "user_message", payload: compactResume.userMessagePayload }).byteLength
  assert.ok(bulkBytes <= MAX_USER_MESSAGE_FRAME_BYTES, `compacted <tools_update> frame must be <= ${MAX_USER_MESSAGE_FRAME_BYTES}, got ${bulkBytes}`)
  assert.ok(compactResume.userMessagePayload.text.includes("<tools_update>"))
  assert.ok(compactResume.userMessagePayload.text.includes("mcp_bulk_0(query*: string"))
})

test("tool-less title sidecar does not hijack tc: or main thread, and inflight lock serializes concurrent streaming turns", async () => {
  const store = createThreadStore()
  const acct = { providerType: "social", access: "tok-lock", tenantUrl: "https://qbs-test.plato.ai.aws.dev", accountId: "lock-acct" }
  const readTool = {
    type: "function",
    function: { name: "read_file", description: "Read file", parameters: { type: "object", properties: { path: { type: "string" } } } },
  }

  // 1. Seed a tool-bearing main thread that emitted 2 tool calls
  let activeToolCallIds = []
  let mainThreadId = ""
  const seedCalls = []
  const SeedWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      for (const raw of ws.sent) {
        const outer = JSON.parse(Buffer.from(raw).toString("utf8"))
        const inner = JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
        if (inner.type === "user_message") seedCalls.push(inner.payload)
      }
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      ws.emitAidpEvent({
        type: "text_delta",
        payload: {
          delta:
            '<tool_call>{"name":"read_file","arguments":{"path":"a.mjs"}}</tool_call>' +
            '<tool_call>{"name":"read_file","arguments":{"path":"b.mjs"}}</tool_call>',
        },
      })
      ws.emitAidpEvent({ type: "message_complete", payload: {} })
    })
  })
  const seedRes = await executeSncsChat(
    acct,
    { model: "balanced", stream: false, tools: [readTool], messages: [{ role: "user", content: "Inspect a.mjs and b.mjs" }] },
    { WebSocket: SeedWS, threadStore: store, sessionId: "zed-sess-1" },
  )
  const seedJson = await seedRes.json()
  activeToolCallIds = seedJson.choices[0].message.tool_calls.map((c) => c.id)
  assert.equal(activeToolCallIds.length, 2)
  mainThreadId = seedCalls[0].thread_id

  // 2. Zed fires title-gen sidecar (tools: [], 2 tool_results + trailing user prompt) AND main continuation concurrently
  const secondRoundPayloads = []
  const FastWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      for (const raw of ws.sent) {
        const outer = JSON.parse(Buffer.from(raw).toString("utf8"))
        const inner = JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
        if (inner.type === "user_message") secondRoundPayloads.push(inner.payload)
      }
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      ws.emitAidpEvent({ type: "text_delta", payload: { delta: "ok" } })
      ws.emitAidpEvent({ type: "message_complete", payload: {} })
    })
  })
  const assistantToolMsg = {
    role: "assistant",
    content: null,
    tool_calls: activeToolCallIds.map((id, i) => ({
      id,
      type: "function",
      function: { name: "read_file", arguments: JSON.stringify({ path: i === 0 ? "a.mjs" : "b.mjs" }) },
    })),
  }
  const toolResultMsgs = [
    { role: "tool", tool_call_id: activeToolCallIds[0], name: "read_file", content: "FILE_A_BODY" },
    { role: "tool", tool_call_id: activeToolCallIds[1], name: "read_file", content: "FILE_B_BODY" },
  ]
  const [titleRes, mainRes] = await Promise.all([
    executeSncsChat(
      acct,
      {
        model: "balanced",
        stream: false,
        tools: [],
        messages: [
          { role: "user", content: "Inspect a.mjs and b.mjs" },
          assistantToolMsg,
          ...toolResultMsgs,
          { role: "user", content: "Generate a concise 3-7 word title" },
        ],
      },
      { WebSocket: FastWS, threadStore: store, sessionId: "zed-sess-1" },
    ),
    executeSncsChat(
      acct,
      {
        model: "balanced",
        stream: false,
        tools: [readTool],
        messages: [
          { role: "user", content: "Inspect a.mjs and b.mjs" },
          assistantToolMsg,
          ...toolResultMsgs,
        ],
      },
      { WebSocket: FastWS, threadStore: store, sessionId: "zed-sess-1" },
    ),
  ])
  assert.equal(titleRes.status, 200)
  assert.equal(mainRes.status, 200)
  const titlePayload = secondRoundPayloads.find((p) => p.text.includes("Generate a concise 3-7 word title"))
  const mainPayload = secondRoundPayloads.find((p) => !p.text.includes("Generate a concise 3-7 word title"))
  assert.ok(titlePayload && mainPayload)
  assert.notEqual(titlePayload.thread_id, mainThreadId, "title-gen sidecar must run on isolated stateless thread")
  assert.equal(mainPayload.thread_id, mainThreadId, "main continuation must resume mainThreadId cleanly")
  assert.ok(!mainPayload.text.includes("Generate a concise 3-7 word title"))

  // 3. Inflight lock holds across full SSE stream until message_complete, then second request re-checks cache
  const events = []
  let finishStreamA = null
  let wsCount = 0
  const ControlledWS = makeMockWebSocketClass((ws) => {
    const idx = ++wsCount
    events.push(`ws_open_${idx}`)
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      if (idx === 1) {
        ws.emitAidpEvent({ type: "text_delta", payload: { delta: "stream-A-part1" } })
        finishStreamA = () => {
          events.push("stream_A_completing")
          ws.emitAidpEvent({ type: "text_delta", payload: { delta: "-part2" } })
          ws.emitAidpEvent({ type: "message_complete", payload: {} })
        }
      } else {
        ws.emitAidpEvent({ type: "text_delta", payload: { delta: "reply-B" } })
        ws.emitAidpEvent({ type: "message_complete", payload: {} })
      }
    })
  })

  // Start streaming Request A on mainThreadId
  const reqAMsgs = [
    { role: "user", content: "Inspect a.mjs and b.mjs" },
    assistantToolMsg,
    ...toolResultMsgs,
    { role: "assistant", content: "ok" },
    { role: "user", content: "Turn A" },
  ]
  const resA = await executeSncsChat(
    acct,
    { model: "balanced", stream: true, tools: [readTool], messages: reqAMsgs },
    { WebSocket: ControlledWS, threadStore: store, sessionId: "zed-sess-1" },
  )
  assert.equal(resA.status, 200)
  // resA has ALREADY resolved (stream started!), but message_complete has NOT arrived yet.
  // Launch concurrent Request B targeting the same mainThreadId.
  const reqBPromise = executeSncsChat(
    acct,
    {
      model: "balanced",
      stream: false,
      tools: [readTool],
      messages: [...reqAMsgs, { role: "assistant", content: "stream-A-part1-part2" }, { role: "user", content: "Turn B" }],
    },
    { WebSocket: ControlledWS, threadStore: store, sessionId: "zed-sess-1" },
  )
  await new Promise((r) => setTimeout(r, 25))
  assert.equal(wsCount, 1, "Request B MUST wait for Request A's SSE stream to finish before opening its WebSocket")

  // Now complete Request A's SSE stream -> Request B wakes up, re-looks up cache, and opens ws #2
  finishStreamA()
  const [textA, resB] = await Promise.all([resA.text(), reqBPromise])
  assert.ok(textA.includes("stream-A-part1"))
  assert.ok(textA.includes("-part2"))
  assert.equal((await resB.json()).choices[0].message.content, "reply-B")
  assert.deepEqual(events, ["ws_open_1", "stream_A_completing", "ws_open_2"])
})

test("CJK language detection strips <context>/code blocks and persists across resumed tool-result turns", async () => {
  const store = createThreadStore()
  const acct = { providerType: "social", access: "tok-lang", tenantUrl: "https://qbs-test.plato.ai.aws.dev", accountId: "lang-acct" }
  const readTool = {
    type: "function",
    function: { name: "read_file", description: "Read file", parameters: { type: "object", properties: { path: { type: "string" } } } },
  }

  // 1. Short Chinese user prose wrapped in large English <context> and fenced code still detects "zh"
  const userWithContext =
    "<context>\nActive file: src/bridge.mjs\n" +
    "export function longEnglishBoilerplateIdentifier() { return true }\n".repeat(20) +
    "</context>\n```js\nfunction parseToolBlock(openTag, inner) { return inner }\n```\n请帮我检查这段代码"
  const sentPayloads = []
  let turn = 0
  const MockWS = makeMockWebSocketClass((ws) => {
    turn++
    const curTurn = turn
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      for (const raw of ws.sent) {
        const outer = JSON.parse(Buffer.from(raw).toString("utf8"))
        const inner = JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
        if (inner.type === "user_message") sentPayloads.push(inner.payload)
      }
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      ws.emitAidpEvent({
        type: "text_delta",
        payload: {
          delta:
            curTurn === 1
              ? '<tool_call>{"name":"read_file","arguments":{"path":"bridge.mjs"}}</tool_call>'
              : "检查完成",
        },
      })
      ws.emitAidpEvent({ type: "message_complete", payload: {} })
    })
  })

  const r1 = await executeSncsChat(
    acct,
    { model: "balanced", stream: false, tools: [readTool], messages: [{ role: "user", content: userWithContext }] },
    { WebSocket: MockWS, threadStore: store, sessionId: "sess-lang-1", locale: "en-US" },
  )
  const j1 = await r1.json()
  const tcId = j1.choices[0].message.tool_calls[0].id
  assert.match(sentPayloads[0].custom_prompt, /4\. Language Consistency:[\s\S]*Chinese/)

  // 2. Turn 2 resumes with ONLY pure English code in tool_result -> cached lang="zh" still preserves rule #4
  await executeSncsChat(
    acct,
    {
      model: "balanced",
      stream: false,
      tools: [readTool],
      messages: [
        { role: "user", content: userWithContext },
        {
          role: "assistant",
          content: null,
          tool_calls: [{ id: tcId, type: "function", function: { name: "read_file", arguments: '{"path":"bridge.mjs"}' } }],
        },
        { role: "tool", tool_call_id: tcId, name: "read_file", content: "export function pureEnglishCode() {\n  return 42\n}\n".repeat(40) },
      ],
    },
    { WebSocket: MockWS, threadStore: store, sessionId: "sess-lang-1", locale: "en-US" },
  )
  assert.equal(sentPayloads.length, 2)
  assert.equal(sentPayloads[1].thread_id, sentPayloads[0].thread_id)
  assert.match(
    sentPayloads[1].custom_prompt,
    /4\. Language Consistency:[\s\S]*Chinese/,
    "resumed English-only tool_result turn must preserve cached Chinese language consistency rule",
  )
})

test("line-boundary truncation, dropped-history marker + attachment, frame budget hint, and -old/+new systemDeltaText work", async () => {
  const readTool = {
    type: "function",
    function: { name: "read_file", description: "Read file", parameters: { type: "object", properties: { path: { type: "string" } } } },
  }

  // 1. Line-boundary truncation does not split mid-token like `${a?.` and attachment line ranges match exact content
  const codeLines = Array.from({ length: 1200 }, (_, i) => `const line_${String(i + 1).padStart(4, "0")} = \`val_\${a?.b?.c_${i}}\``).join("\n")
  const req = openaiToSncsRequest({
    model: "balanced",
    tools: [readTool],
    messages: [
      { role: "user", content: "Old Q1 " + "H".repeat(25_000) },
      { role: "assistant", content: "Old A1 " + "K".repeat(25_000) },
      { role: "user", content: "Read file" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "lb1", type: "function", function: { name: "read_file", arguments: '{"path":"big.mjs"}' } }],
      },
      { role: "tool", tool_call_id: "lb1", name: "read_file", content: codeLines },
    ],
  })

  // Frame budget hint present in tool protocol
  assert.match(req.userMessagePayload.text, /~55 KB[\s\S]*~250 lines/)
  // Stage 2 omitted marker present AND dropped earlier history spilled into overflowUserText on new thread
  assert.match(req.userMessagePayload.text, /\[Earlier \d+ conversation message\(s\) omitted to fit 64KB frame\]/)
  assert.ok(req.overflowUserText.includes("=== DROPPED EARLIER HISTORY"), "dropped history should be preserved in overflow attachment on new thread")
  assert.ok(req.overflowUserText.includes("Old Q1 "), "untruncated dropped history must be in overflow attachment")
  assert.ok(req.overflowUserText.includes("=== TOOL RESULT #1"), "overflow attachment must have structured section headers")

  // Verify line-boundary snap: text immediately before `...[trimmed` ends with a newline, and text immediately after `]...` starts with a full line
  const trimMatch = req.textWithAttachmentNote.match(/([^\n]*)\n\.\.\.\[trimmed \d+ of \d+ chars; full content in attached file (tool_results_[a-z0-9]+\.txt) lines (\d+)-(\d+)\]\.\.\.\n([^\n]*)/i)
  assert.ok(trimMatch, "with-attachment text must include exact attachment filename and line range")
  assert.match(trimMatch[1], /^const line_\d{4} = `val_\$\{a\?\.b\?\.c_\d+\}`$/, "head cut must land cleanly on a full line boundary")
  assert.match(trimMatch[5], /^const line_\d{4} = `val_\$\{a\?\.b\?\.c_\d+\}`$/, "tail cut must land cleanly on a full line boundary")
  const attachLines = req.overflowUserText.split("\n")
  const startLine = Number(trimMatch[3])
  const endLine = Number(trimMatch[4])
  assert.ok(attachLines[startLine - 1].startsWith('<tool_result id="lb1"'), `line ${startLine} should start tool_result`)
  assert.equal(attachLines[endLine - 1], "</tool_result>", `line ${endLine} should close tool_result`)

  // 2. systemDeltaText: -old / +new diff format AND fresh thread fallback when prevSys="" and nextSys > 20k chars
  const store = createThreadStore()
  const acct = { providerType: "social", access: "tok-sys", tenantUrl: "https://qbs-test.plato.ai.aws.dev", accountId: "sys-acct" }
  const sent = []
  const MockWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      for (const raw of ws.sent) {
        const outer = JSON.parse(Buffer.from(raw).toString("utf8"))
        const inner = JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
        if (inner.type === "user_message") sent.push(inner.payload)
      }
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      ws.emitAidpEvent({ type: "text_delta", payload: { delta: "ok" } })
      ws.emitAidpEvent({ type: "message_complete", payload: {} })
    })
  })

  // Turn 1 with rule A
  await executeSncsChat(
    acct,
    {
      model: "fast",
      stream: false,
      messages: [
        { role: "system", content: "Header\nRule: Always use tabs\nFooter" },
        { role: "user", content: "Hi" },
      ],
    },
    { WebSocket: MockWS, threadStore: store, sessionId: "sys-diff-1" },
  )
  // Turn 2 replaces rule A with rule B -> <system_context_update> shows BOTH -old and +new
  await executeSncsChat(
    acct,
    {
      model: "fast",
      stream: false,
      messages: [
        { role: "system", content: "Header\nRule: Always use 2 spaces\nFooter" },
        { role: "user", content: "Hi" },
        { role: "assistant", content: "ok" },
        { role: "user", content: "Next" },
      ],
    },
    { WebSocket: MockWS, threadStore: store, sessionId: "sys-diff-1" },
  )
  assert.equal(sent[1].thread_id, sent[0].thread_id)
  assert.ok(sent[1].text.includes("<system_context_update>\n-Rule: Always use tabs\n+Rule: Always use 2 spaces\n</system_context_update>"))

  // Turn 1 with NO system prompt, then Turn 2 adds >20k system prompt -> must open fresh thread (NOT silently drop!)
  await executeSncsChat(
    acct,
    { model: "fast", stream: false, messages: [{ role: "user", content: "No sys initially" }] },
    { WebSocket: MockWS, threadStore: store, sessionId: "sys-big-2" },
  )
  const hugeSys = "IMPORTANT_RULE_START\n" + "S".repeat(22_000) + "\nIMPORTANT_RULE_END"
  await executeSncsChat(
    acct,
    {
      model: "fast",
      stream: false,
      messages: [
        { role: "system", content: hugeSys },
        { role: "user", content: "No sys initially" },
        { role: "assistant", content: "ok" },
        { role: "user", content: "Now follow huge system prompt" },
      ],
    },
    { WebSocket: MockWS, threadStore: store, sessionId: "sys-big-2" },
  )
  assert.notEqual(sent[3].thread_id, sent[2].thread_id, "must open fresh thread when >20k system prompt appears from empty prevSys")
  assert.ok(sent[3].text.includes("IMPORTANT_RULE_START"), "new thread must include full <instructions> with new system prompt")
})

test("expectedAssistantSig rejects edited/regenerated replies, thread-level LRU preserves sessions across 10 tool calls, and mid-stream cancel/error invalidates thread", async () => {
  const acct = { providerType: "social", access: "tok-lru", tenantUrl: "https://qbs-test.plato.ai.aws.dev", accountId: "lru-acct" }
  const readTool = {
    type: "function",
    function: { name: "read_file", description: "Read", parameters: { type: "object", properties: { path: { type: "string" } } } },
  }

  // 1. Thread-level LRU: maxEntries=2 with 10 parallel tool_call_ids on thread 2 must NOT evict thread 1
  const lruStore = createThreadStore(2)
  const sentLru = []
  let lruTurn = 0
  const LruWS = makeMockWebSocketClass((ws) => {
    const t = ++lruTurn
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      for (const raw of ws.sent) {
        const outer = JSON.parse(Buffer.from(raw).toString("utf8"))
        const inner = JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
        if (inner.type === "user_message") sentLru.push(inner.payload)
      }
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      const delta =
        t === 2
          ? Array.from({ length: 10 }, (_, i) => `<tool_call>{"name":"read_file","arguments":{"path":"f${i}.mjs"}}</tool_call>`).join("")
          : `reply-${t}`
      ws.emitAidpEvent({ type: "text_delta", payload: { delta } })
      ws.emitAidpEvent({ type: "message_complete", payload: {} })
    })
  })

  await executeSncsChat(
    acct,
    { model: "fast", stream: false, messages: [{ role: "user", content: "Thread 1 hello" }] },
    { WebSocket: LruWS, threadStore: lruStore, sessionId: "lru-s1" },
  )
  const tid1 = sentLru[0].thread_id

  // Thread 2 emits 10 parallel tool calls (10 tc: index entries) -> still counts as only 1 thread in LRU
  await executeSncsChat(
    acct,
    { model: "fast", stream: false, tools: [readTool], messages: [{ role: "user", content: "Thread 2 ten tools" }] },
    { WebSocket: LruWS, threadStore: lruStore, sessionId: "lru-s2" },
  )
  assert.equal(lruStore.size, 2, "LRU must count threads, not secondary index keys")

  // Thread 1 must STILL be cached and resumable!
  await executeSncsChat(
    acct,
    {
      model: "fast",
      stream: false,
      messages: [
        { role: "user", content: "Thread 1 hello" },
        { role: "assistant", content: "reply-1" },
        { role: "user", content: "Thread 1 follow-up" },
      ],
    },
    { WebSocket: LruWS, threadStore: lruStore, sessionId: "lru-s1" },
  )
  assert.equal(sentLru[2].thread_id, tid1, "thread 1 must not be evicted by thread 2's 10 tool calls")

  // 2. Edited/regenerated assistant reply fails expectedAssistantSig check and starts a fresh thread
  await executeSncsChat(
    acct,
    {
      model: "fast",
      stream: false,
      messages: [
        { role: "user", content: "Thread 1 hello" },
        { role: "assistant", content: "reply-1" },
        { role: "user", content: "Thread 1 follow-up" },
        { role: "assistant", content: "USER_EDITED_DIFFERENT_REPLY" },
        { role: "user", content: "Continue from edited reply" },
      ],
    },
    { WebSocket: LruWS, threadStore: lruStore, sessionId: "lru-s1" },
  )
  assert.notEqual(sentLru[3].thread_id, tid1, "edited assistant reply must not resume stale thread")

  // 3. Mid-stream cancel() and mid-stream failEarly() invalidate the dirty thread
  const streamStore = createThreadStore()
  const streamSent = []
  let streamIdx = 0
  let emitMidStreamError = null
  const StreamWS = makeMockWebSocketClass((ws) => {
    const idx = ++streamIdx
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      for (const raw of ws.sent) {
        const outer = JSON.parse(Buffer.from(raw).toString("utf8"))
        const inner = JSON.parse(Buffer.from(outer.payload.payload, "base64").toString("utf8"))
        if (inner.type === "user_message") streamSent.push(inner.payload)
      }
      ws.emitAidpEvent({ type: "session_established", payload: {} })
      if (idx === 1 || idx === 3) {
        ws.emitAidpEvent({ type: "text_delta", payload: { delta: `ok-${idx}` } })
        ws.emitAidpEvent({ type: "message_complete", payload: {} })
      } else if (idx === 2) {
        // Start streaming, then hang until client cancels
        ws.emitAidpEvent({ type: "text_delta", payload: { delta: "partial-before-cancel" } })
      } else if (idx === 4) {
        // Start streaming, then fail mid-stream
        ws.emitAidpEvent({ type: "text_delta", payload: { delta: "partial-before-error" } })
        emitMidStreamError = () => ws.emitAidpEvent({ type: "error", payload: { message: "mid-stream boom" } })
      } else {
        ws.emitAidpEvent({ type: "text_delta", payload: { delta: `ok-${idx}` } })
        ws.emitAidpEvent({ type: "message_complete", payload: {} })
      }
    })
  })

  // Seed thread A
  await executeSncsChat(
    acct,
    { model: "fast", stream: false, messages: [{ role: "user", content: "Stream A" }] },
    { WebSocket: StreamWS, threadStore: streamStore, sessionId: "st-1" },
  )
  const tidA = streamSent[0].thread_id

  // Turn 2 resumes tidA in stream mode, then client cancels mid-stream -> must invalidate tidA
  const cancelRes = await executeSncsChat(
    acct,
    {
      model: "fast",
      stream: true,
      messages: [
        { role: "user", content: "Stream A" },
        { role: "assistant", content: "ok-1" },
        { role: "user", content: "Will cancel" },
      ],
    },
    { WebSocket: StreamWS, threadStore: streamStore, sessionId: "st-1" },
  )
  const reader = cancelRes.body.getReader()
  await reader.read()
  await reader.cancel()

  // Turn 3 after cancel must NOT reuse dirty tidA
  await executeSncsChat(
    acct,
    {
      model: "fast",
      stream: false,
      messages: [
        { role: "user", content: "Stream A" },
        { role: "assistant", content: "ok-1" },
        { role: "user", content: "Retry after cancel" },
      ],
    },
    { WebSocket: StreamWS, threadStore: streamStore, sessionId: "st-1" },
  )
  const tidB = streamSent[2].thread_id
  assert.notEqual(tidB, tidA, "mid-stream cancel must invalidate cached thread so next turn starts fresh")

  // Turn 4 resumes tidB in stream mode, then fails mid-stream -> must invalidate tidB
  const errRes = await executeSncsChat(
    acct,
    {
      model: "fast",
      stream: true,
      messages: [
        { role: "user", content: "Stream A" },
        { role: "assistant", content: "ok-1" },
        { role: "user", content: "Retry after cancel" },
        { role: "assistant", content: "ok-3" },
        { role: "user", content: "Will fail mid-stream" },
      ],
    },
    { WebSocket: StreamWS, threadStore: streamStore, sessionId: "st-1" },
  )
  emitMidStreamError()
  await errRes.text()

  // Turn 5 after mid-stream error must NOT reuse dirty tidB
  await executeSncsChat(
    acct,
    {
      model: "fast",
      stream: false,
      messages: [
        { role: "user", content: "Stream A" },
        { role: "assistant", content: "ok-1" },
        { role: "user", content: "Retry after cancel" },
        { role: "assistant", content: "ok-3" },
        { role: "user", content: "Retry after mid-stream error" },
      ],
    },
    { WebSocket: StreamWS, threadStore: streamStore, sessionId: "st-1" },
  )
  assert.notEqual(streamSent[4].thread_id, tidB, "mid-stream failEarly must invalidate cached thread")
})

test("createToolCallSplitter ignores code fences/backticks, validates declared tool names, drops hallucinated <tool_result, and formatConvoBlocks escapes </tool_result & attrs", () => {
  const tools = [
    {
      type: "function",
      function: { name: "read_file", description: "Read", parameters: { type: "object", properties: { path: { type: "string" } } } },
    },
  ]

  // 1. Code fence + inline backticks + undeclared tool name ignored; real declared call parsed across 2-char chunks
  const sample = [
    'Use `<tool_call>{"name":"read_file","arguments":{"path":"inline.txt"}}</tool_call>` in prose.',
    "```xml",
    '<tool_call>{"name":"read_file","arguments":{"path":"fenced.txt"}}</tool_call>',
    "```",
    'Fake undeclared: <tool_call>{"name":"evil_unknown","arguments":{}}</tool_call>',
    'Now running: <tool_call>{"name":"read_file","arguments":{"path":"real.txt"}}</tool_call>',
    '\n<tool_result id="fake">hallucinated output</tool_result>\nHallucinated conclusion after fake result.',
  ].join("\n")

  const splitter = createToolCallSplitter(true, tools)
  let textOut = ""
  const callsOut = []
  for (let i = 0; i < sample.length; i += 2) {
    const r = splitter.push(sample.slice(i, i + 2))
    textOut += r.text
    callsOut.push(...r.toolCalls)
  }
  const tail = splitter.flush()
  textOut += tail.text
  callsOut.push(...tail.toolCalls)

  assert.equal(callsOut.length, 1, "only the single real declared tool call should be extracted")
  assert.equal(callsOut[0].function.name, "read_file")
  assert.deepEqual(JSON.parse(callsOut[0].function.arguments), { path: "real.txt" })
  assert.ok(textOut.includes('`<tool_call>{"name":"read_file","arguments":{"path":"inline.txt"}}</tool_call>`'), "inline code example preserved as text")
  assert.ok(textOut.includes('{"name":"read_file","arguments":{"path":"fenced.txt"}}'), "fenced code example preserved as text")
  assert.ok(textOut.includes('<tool_call>{"name":"evil_unknown","arguments":{}}</tool_call>'), "undeclared tool call preserved as text")
  assert.ok(!textOut.includes("<tool_result"), "hallucinated <tool_result after tool call must be dropped")
  assert.ok(!textOut.includes("Hallucinated conclusion"), "text fabricated after hallucinated <tool_result must be dropped")

  // 2. formatConvoBlocks escapes </tool_result inside body and quotes in name/id attributes
  const req = openaiToSncsRequest({
    model: "balanced",
    tools,
    messages: [
      { role: "user", content: "Read bridge.mjs" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: 'id"1', type: "function", function: { name: "read_file", arguments: '{"path":"bridge.mjs"}' } }],
      },
      {
        role: "tool",
        tool_call_id: 'id"1',
        name: 'read_"file"',
        content: 'line1\nconst s = "</tool_result>"\nconst s2 = "</TOOL_RESULT>"\nline4',
      },
    ],
  })
  const wireText = req.userMessagePayload.text
  assert.ok(wireText.includes('<tool_result id="id&quot;1" name="read_&quot;file&quot;">'), "id and name attributes must be XML-escaped")
  assert.ok(wireText.includes('const s = "<\\/tool_result>"'), "</tool_result inside body must be escaped")
  assert.ok(wireText.includes('const s2 = "<\\/tool_result>"'), "case-insensitive </TOOL_RESULT inside body must be escaped")
  // Only the real closing tag should match unescaped `</tool_result>`
  assert.equal((wireText.match(/<\/tool_result>/gi) || []).length, 1, "only the real wrapper closing </tool_result> should remain unescaped")
})
