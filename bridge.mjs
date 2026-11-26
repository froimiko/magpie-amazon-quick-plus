import { createHash, randomUUID } from "node:crypto"
import {
  closeSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
  writeSync,
} from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"
import {
  buildAidpInputFrame,
  buildGatewayInputFrame,
  decodeAidpFrame,
  decodeGatewayFrame,
  esDrain,
} from "./codec.mjs"
import {
  APP_VERSION,
  hostFromUrl,
  isGatewayProvider,
  quickworkDir,
  surfaceTypeForPlatform,
  uploadQuickAttachment,
} from "./credentials.mjs"
import { agentModeEnabled, DEFAULT_AGENT_STORE, ensureCarrierAgent } from "./agents.mjs"

export const DEFAULT_API_URL = "https://quick.aws.com/v1"
export const DEFAULT_CONTEXT_LIMIT = 1_000_000
export const DEFAULT_OUTPUT_LIMIT = 64_000

const EFFORT_LEVELS = ["off", "low", "medium", "high", "max"]
const EFFORT_ALIASES = new Map([
  ["none", "off"],
  ["disabled", "off"],
  ["false", "off"],
  ["minimal", "low"],
  ["med", "medium"],
  ["xhigh", "max"],
  ["enabled", "high"],
  ["true", "high"],
])

export const DEFAULT_MODELS = {
  auto: {
    id: "auto",
    name: "Amazon Quick Auto",
    agentMode: "smart",
    autoRouting: true,
    reasoning: true,
    efforts: ["off", "low", "medium", "high", "max"],
    defaultEffort: "medium",
    context: DEFAULT_CONTEXT_LIMIT,
    output: DEFAULT_OUTPUT_LIMIT,
  },
  fast: {
    id: "fast",
    name: "Amazon Quick Fast (Haiku 5.5)",
    agentMode: "fast",
    autoRouting: false,
    reasoning: false,
    efforts: [],
    defaultEffort: "off",
    context: DEFAULT_CONTEXT_LIMIT,
    output: 32_000,
  },
  balanced: {
    id: "balanced",
    name: "Amazon Quick Balanced (Sonnet 5.5)",
    agentMode: "balanced",
    autoRouting: false,
    reasoning: true,
    efforts: ["off", "low", "medium", "high", "max"],
    defaultEffort: "off",
    context: DEFAULT_CONTEXT_LIMIT,
    output: DEFAULT_OUTPUT_LIMIT,
  },
  smart: {
    id: "smart",
    name: "Amazon Quick Smart (Opus 5.5)",
    agentMode: "smart",
    autoRouting: false,
    reasoning: true,
    efforts: ["off", "low", "medium", "high", "max"],
    defaultEffort: "off",
    context: DEFAULT_CONTEXT_LIMIT,
    output: DEFAULT_OUTPUT_LIMIT,
  },
  advanced: {
    id: "advanced",
    name: "Amazon Quick Advanced (Deep Reasoning)",
    agentMode: "advanced",
    autoRouting: false,
    reasoning: true,
    efforts: ["low", "medium", "high", "max"],
    defaultEffort: "medium",
    context: DEFAULT_CONTEXT_LIMIT,
    output: DEFAULT_OUTPUT_LIMIT,
  },
}

const KNOWN_ALIASES = new Map([
  ["auto", "auto"],
  ["fast", "fast"],
  ["balanced", "balanced"],
  ["smart", "smart"],
  ["advanced", "advanced"],
  ["us.anthropic.claude-haiku-5-5", "fast"],
  ["claude-haiku-5-5", "fast"],
  ["us.anthropic.claude-sonnet-5-5", "balanced"],
  ["claude-sonnet-5-5", "balanced"],
  ["us.anthropic.claude-opus-5-5", "smart"],
  ["claude-opus-5-5", "smart"],
])

function variantsForSpec(spec) {
  if (!spec.reasoning || !spec.efforts?.length) return {}
  const out = {}
  for (const level of spec.efforts) {
    out[level] = level === "off" ? { reasoning: false } : { reasoningEffort: level }
  }
  return out
}

export function buildProviderModelEntry(providerID, spec, baseURL = DEFAULT_API_URL) {
  const apiUrl = baseURL || DEFAULT_API_URL
  const cap = {
    temperature: true,
    reasoning: !!spec.reasoning,
    attachment: true,
    toolcall: true,
    input: { text: true, audio: false, image: true, video: false, pdf: false },
    output: { text: true, audio: false, image: false, video: false, pdf: false },
    interleaved: false,
  }
  return {
    id: spec.id,
    providerID,
    name: spec.name,
    api: { id: spec.id, url: apiUrl, npm: "@ai-sdk/openai-compatible" },
    // A config model is read in OpenCode's config shape (top-level
    // reasoning/attachment/tool_call/modalities, provider for its route);
    // the provider.models hook is read through capabilities below. Both
    // are given so magpie keeps the model's reasoning and image support
    // when it merges the config hook's list.
    provider: { npm: "@ai-sdk/openai-compatible", api: apiUrl },
    temperature: true,
    reasoning: !!spec.reasoning,
    attachment: true,
    tool_call: true,
    modalities: { input: ["text", "image"], output: ["text"] },
    status: "active",
    headers: {},
    options: {},
    cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
    limit: { context: spec.context || DEFAULT_CONTEXT_LIMIT, output: spec.output || DEFAULT_OUTPUT_LIMIT },
    capabilities: cap,
    release_date: "",
    variants: variantsForSpec(spec),
  }
}

export function configModels(providerID = "amazon-quick-plus", baseURL = DEFAULT_API_URL) {
  const out = {}
  for (const spec of Object.values(DEFAULT_MODELS)) {
    out[spec.id] = buildProviderModelEntry(providerID, spec, baseURL)
  }
  return out
}

/**
 * Derive dynamic provider models from Amazon Quick's `get_settings` payload when available.
 */
export function modelsFromSettings(providerID, settings, baseURL = DEFAULT_API_URL) {
  if (!settings || typeof settings !== "object") return configModels(providerID, baseURL)
  const slots = Array.isArray(settings.model_slots) ? settings.model_slots : null
  const advancedEnabled = settings.advanced_mode_enabled === true
  const specs = { ...DEFAULT_MODELS }
  if (!advancedEnabled && !slots?.some((s) => s?.id === "advanced")) {
    delete specs.advanced
  }
  if (slots?.length) {
    for (const s of slots) {
      if (!s || typeof s !== "object" || typeof s.id !== "string" || !s.id) continue
      if (s.enabled === false) {
        delete specs[s.id]
        continue
      }
      const base = specs[s.id] ?? {
        id: s.id,
        name: s.label || s.name || `Amazon Quick ${s.id}`,
        agentMode: s.agent_mode || s.id,
        autoRouting: s.id === "auto",
        reasoning: s.thinking?.mode !== "none",
        efforts: Array.isArray(s.thinking?.efforts) ? s.thinking.efforts : EFFORT_LEVELS,
        defaultEffort: s.thinking?.default_effort || "off",
        context: Number(s.context_window) || DEFAULT_CONTEXT_LIMIT,
        output: Number(s.max_output_tokens) || DEFAULT_OUTPUT_LIMIT,
      }
      specs[s.id] = {
        ...base,
        name: s.display_name || s.label || base.name,
        reasoning: s.thinking ? s.thinking.mode !== "none" : base.reasoning,
        efforts: Array.isArray(s.thinking?.efforts) ? s.thinking.efforts : base.efforts,
        defaultEffort: s.thinking?.default_effort ?? base.defaultEffort,
      }
    }
  }
  const out = {}
  for (const spec of Object.values(specs)) {
    out[spec.id] = buildProviderModelEntry(providerID, spec, baseURL)
  }
  return out
}

/**
 * Map an OpenAI-style `model` id (including `amazon-quick-plus/...` prefix or `@effort` suffix),
 * Magpie's X-Magpie-Effort header and body reasoning hints into SNCS
 * `{ modelId, agentMode, autoRouting, thinkingEffort }`.
 */
export function resolveModelAndEffort(requestedModel, body = {}, effortHeader = "") {
  let raw = String(requestedModel || "balanced").trim()
  if (raw.includes("/")) raw = raw.split("/").pop()
  raw = raw.replace(/\[[^\]]+\]/g, "").trim()
  let atEffort = ""
  const atIdx = raw.lastIndexOf("@")
  if (atIdx > 0) {
    atEffort = raw.slice(atIdx + 1).toLowerCase()
    raw = raw.slice(0, atIdx)
  }
  const canonical = KNOWN_ALIASES.get(raw.toLowerCase()) || raw.toLowerCase()
  const spec = DEFAULT_MODELS[canonical] ?? {
    id: canonical || "balanced",
    agentMode: ["fast", "balanced", "smart", "advanced"].includes(canonical) ? canonical : "balanced",
    autoRouting: canonical === "auto",
    reasoning: canonical !== "fast",
    efforts: EFFORT_LEVELS,
    defaultEffort: canonical === "advanced" ? "medium" : "off",
  }

  // Magpie tells a plugin the effort its variant picker asked in
  // X-Magpie-Effort; the body is what the client itself said. The header
  // wins, as the picker's choice is the one the user made.
  const rawEffort = String(
    effortHeader ||
      atEffort ||
      (body?.reasoning === false || body?.thinking?.type === "disabled" ? "off" : "") ||
      body?.reasoning_effort ||
      body?.reasoning?.effort ||
      body?.thinking?.effort ||
      "",
  )
    .trim()
    .toLowerCase()
  let requestedEffort = EFFORT_ALIASES.get(rawEffort) ?? rawEffort

  if (!spec.reasoning) {
    requestedEffort = "off"
  } else if (!requestedEffort || !EFFORT_LEVELS.includes(requestedEffort)) {
    requestedEffort = spec.defaultEffort || "off"
  } else if (spec.efforts?.length && !spec.efforts.includes(requestedEffort)) {
    requestedEffort = requestedEffort === "off" && spec.efforts.length ? spec.efforts[0] : spec.defaultEffort || "medium"
  }

  return {
    modelId: spec.id,
    agentMode: spec.agentMode,
    autoRouting: !!spec.autoRouting,
    thinkingEffort: requestedEffort || "off",
  }
}

function textFromContent(content, includeImagePlaceholder = true) {
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return content == null ? "" : String(content)
  const parts = []
  for (const part of content) {
    if (!part || typeof part !== "object") continue
    if (part.type === "text" || part.type === "input_text") parts.push(String(part.text ?? ""))
    else if (part.type === "image_url" && includeImagePlaceholder) {
      parts.push("[Image attachment]")
    }
  }
  return parts.join("\n")
}

function digestContent(content) {
  if (typeof content === "string") return content.trim()
  if (!Array.isArray(content)) return content == null ? "" : String(content).trim()
  return content
    .map((part) => {
      if (!part || typeof part !== "object") return ""
      if (part.type === "text" || part.type === "input_text") return String(part.text ?? "")
      if (part.type === "image_url") {
        const u = typeof part.image_url === "string" ? part.image_url : part.image_url?.url
        return `[image:${sha256Hex(u || "")}]`
      }
      return ""
    })
    .filter(Boolean)
    .join("\n")
    .trim()
}

function collectImageUrls(msgs) {
  const urls = []
  for (const m of Array.isArray(msgs) ? msgs : []) {
    if (!Array.isArray(m?.content)) continue
    for (const part of m.content) {
      if (part?.type !== "image_url") continue
      const u = typeof part.image_url === "string" ? part.image_url : part.image_url?.url
      if (typeof u === "string" && u.trim()) urls.push(u.trim())
    }
  }
  return urls
}

async function resolveImageBytes(urlStr, fetchImpl, signal) {
  if (urlStr.startsWith("data:")) {
    const m = urlStr.match(/^data:([^;,]+)?(?:;base64)?,([\s\S]*)$/i)
    if (!m) return null
    const mimeType = (m[1] || "image/png").toLowerCase()
    const isB64 = /;base64,/i.test(urlStr)
    const bytes = isB64 ? Buffer.from(m[2], "base64") : Buffer.from(decodeURIComponent(m[2]), "utf8")
    const ext = mimeType.split("/")[1]?.replace(/[^a-z0-9]/gi, "") || "png"
    return { mimeType, bytes, ext }
  }
  if (/^https?:\/\//i.test(urlStr)) {
    const res = await fetchImpl(urlStr, { signal: signal ?? AbortSignal.timeout(15000) })
    if (!res.ok) return null
    const buf = new Uint8Array(await res.arrayBuffer())
    const mimeType = (res.headers.get("content-type") || "image/png").split(";")[0].trim().toLowerCase()
    const ext = mimeType.split("/")[1]?.replace(/[^a-z0-9]/gi, "") || "png"
    return { mimeType, bytes: buf, ext }
  }
  return null
}

/**
 * Budget for one outbound `user_message`, measured in the bytes the SERVER counts:
 * the UTF-8 length of `JSON.stringify({ type, payload })`.
 *
 * The limit used to be applied to the base64-wrapped wire frame
 * (`buildAidpInputFrame`). Base64 inflates by ~4/3, so a 62,000 "frame" budget only
 * allowed ~46.5 KB of real payload — the server accepts far more than that.
 *
 * Measured against the live service (2026-11-26, us-east-1, social):
 *   63,472 B payload -> accepted
 *   63,857 B payload -> rejected (bare `error` event)
 *   both ASCII and CJK rejected at the same PAYLOAD byte count, so the server
 *   measures payload bytes, not characters and not base64.
 * 64,000 is therefore the highest round number that stays under the observed
 * ceiling, leaving ~140 B of headroom.
 */
export const MAX_USER_MESSAGE_PAYLOAD_BYTES = 64_000

/**
 * @deprecated Kept for callers and tests written against the old name. This is now
 * a derived wire-frame figure, NOT the budget the pipeline measures against.
 */
export const MAX_USER_MESSAGE_FRAME_BYTES = Math.ceil((MAX_USER_MESSAGE_PAYLOAD_BYTES * 4) / 3) + 64
const OVERFLOW_ATTACHMENT_RESERVE_BYTES = 800
// Minimal room Stage 5's system-file pointer needs in the frame. The overflow
// note's own size is measured per turn (see measureNoteBudget) instead of being
// guessed: it embeds the attachment path and two line numbers, so its length
// follows the filename and the range rather than any fixed constant.
const SYSTEM_POINTER_RESERVE_BYTES = 420

/**
 * Frame bytes the overflow note will occupy for this turn, measured from the real
 * note text rather than assumed. Line numbers are rendered at their widest so the
 * estimate cannot come out short of what composeText later emits.
 */
function measureNoteBudget(attachmentName) {
  const widest = { name: attachmentName, startLine: 999_999, endLine: 9_999_999 }
  return Buffer.byteLength(buildOverflowAttachmentNote(attachmentName, widest), "utf8")
}
// How much of a trimmed system prompt still rides inline as an anchor.
const SYSTEM_ANCHOR_CHARS = 2048
// Smallest inline system text worth keeping once the attachment pointer must fit.
const SYSPOINTER_MIN_ANCHOR_CHARS = 256
const SYSTEM_ATTACHMENT_NAME = "system_instructions.txt"

const WEB_SEARCH_ENV = "MAGPIE_QUICK_WEB_SEARCH"

/**
 * Default for `web_search_enabled` when the request body does not state one.
 *
 * Quick's own cloud search answers before the caller's tool chain can, so an
 * operator may want it off for every turn instead of per request. The body (and
 * `opts.webSearchEnabled`) always wins; this only supplies the fallback via the
 * `MAGPIE_QUICK_WEB_SEARCH` env switch, defaulting to on like the desktop app.
 */
export function webSearchDefaultEnabled() {
  const raw = String(process.env?.[WEB_SEARCH_ENV] ?? "").trim().toLowerCase()
  if (!raw) return true
  return !["0", "off", "no", "false", "disabled", "disable", "none"].includes(raw)
}

function buildOverflowAttachmentNote(attachmentName, droppedRange = null) {
  const droppedPart = droppedRange
    ? ` Omitted earlier history is in lines ${droppedRange.startLine}-${droppedRange.endLine}.`
    : ""
  return `[Note: Full untruncated content is attached in your Quick workspace at /workspace/attached_files/${attachmentName} (workspace-relative path: attached_files/${attachmentName}).${droppedPart} Exception to the no-container-tools rule: use your built-in workspace file reader to read this attachment directly instead of re-reading those lines via local <tool_call>.]`
}

/**
 * Inline placeholder left behind when whole history blocks are shifted out of the
 * turn. It names the payload budget the pipeline actually enforces instead of the
 * old, misleading "64KB frame" wording.
 */
function omittedMarkerText(count) {
  return `[Earlier ${count} conversation message(s) omitted to fit the ${Math.round(MAX_USER_MESSAGE_PAYLOAD_BYTES / 1000)}KB payload budget]`
}

function snapHeadToLineBoundary(s, targetHead) {
  if (targetHead <= 0 || targetHead >= s.length) return targetHead
  const minHead = Math.max(16, targetHead - Math.min(240, Math.floor(targetHead * 0.25)))
  const nl = s.lastIndexOf("\n", targetHead)
  if (nl >= minHead) return nl
  return targetHead
}

function snapTailToLineBoundary(s, targetTailStart) {
  if (targetTailStart <= 0 || targetTailStart >= s.length) return targetTailStart
  const maxTailStart = Math.min(
    s.length - 16,
    targetTailStart + Math.min(240, Math.floor((s.length - targetTailStart) * 0.25)),
  )
  const nl = s.indexOf("\n", targetTailStart)
  if (nl !== -1 && nl + 1 <= maxTailStart) return nl + 1
  return targetTailStart
}

function truncateMiddle(str, maxChars, optsOrMarker = undefined) {
  const s = String(str ?? "")
  if (s.length <= maxChars) return s
  const opts = typeof optsOrMarker === "string" ? { marker: optsOrMarker } : optsOrMarker || {}
  const makeMarker = (omitted) => {
    if (opts.marker) return opts.marker
    if (opts.attachmentRef) {
      return `\n...[trimmed ${omitted} of ${s.length} chars; full content in attached file ${opts.attachmentRef.name} lines ${opts.attachmentRef.startLine}-${opts.attachmentRef.endLine}]...\n`
    }
    const base = `trimmed ${omitted} of ${s.length} chars to fit the ${Math.round(MAX_USER_MESSAGE_PAYLOAD_BYTES / 1000)}KB payload budget`
    const hints = []
    if (opts.isToolResult) hints.push("re-read with a smaller line range if exact omitted lines are needed")
    return `\n...[${base}${hints.length ? `; ${hints.join("; ")}` : ""}]...\n`
  }
  let marker = makeMarker(Math.max(1, s.length - maxChars))
  if (maxChars <= marker.length + 16) {
    marker = `\n...[trimmed ${Math.max(1, s.length - maxChars)}/${s.length} chars]...\n`
  }
  if (maxChars <= marker.length + 16) {
    return s.slice(0, Math.max(0, maxChars))
  }
  const kept = maxChars - marker.length
  const headTarget = Math.floor(kept * 0.65)
  const tailTarget = kept - headTarget
  let headEnd = snapHeadToLineBoundary(s, headTarget)
  let tailStart = snapTailToLineBoundary(s, s.length - tailTarget)
  if (headEnd >= tailStart) {
    headEnd = headTarget
    tailStart = s.length - tailTarget
  }
  const omitted = Math.max(1, tailStart - headEnd)
  const exactMarker = opts.marker ? marker : makeMarker(omitted)
  if (headEnd + exactMarker.length + (s.length - tailStart) <= maxChars) {
    marker = exactMarker
  }
  return s.slice(0, headEnd) + marker + s.slice(tailStart)
}

function describeBlockForHeader(block, index) {
  const txt = String(block?.text || "")
  const m = txt.match(/^<tool_result\b([^>]*)>/i)
  if (m) return `TOOL RESULT #${index + 1} (${m[1].trim()})`
  if (block?.role === "assistant") return `ASSISTANT #${index + 1}`
  if (block?.role === "user") return `USER #${index + 1}`
  return `SECTION #${index + 1}`
}

/**
 * Rebuild the overflow attachment: system instructions first (they are the one
 * thing the frame can never carry whole), then dropped history, then the
 * protected current turn. Line numbers are exact so the inline note can point
 * at a range instead of asking the model to search.
 */
function buildOverflowDocument(droppedBlocks, protectedBlocks, attachmentName, systemBlock = null) {
  const lines = []
  let droppedRange = null
  let sysRange = null
  const sysText = String(systemBlock?.text || "")
  if (sysText) {
    if (lines.length) lines.push("")
    lines.push("=== SYSTEM INSTRUCTIONS (complete, verbatim) ===")
    const startLine = lines.length + 1
    lines.push(...sysText.split("\n"))
    sysRange = { name: attachmentName, startLine, endLine: lines.length }
  }
  if (droppedBlocks.length) {
    lines.push(`=== DROPPED EARLIER HISTORY (${droppedBlocks.length} message(s)) ===`)
    const startLine = lines.length + 1
    for (let i = 0; i < droppedBlocks.length; i++) {
      const b = droppedBlocks[i]
      if (!b?.text) continue
      if (i > 0) lines.push("")
      lines.push(...String(b.text).split("\n"))
    }
    const endLine = lines.length
    droppedRange = { name: attachmentName, startLine, endLine }
  }
  const protectedRanges = []
  for (let i = 0; i < protectedBlocks.length; i++) {
    const b = protectedBlocks[i]
    const raw = String(b?.text || "")
    if (!raw) {
      protectedRanges.push(null)
      continue
    }
    if (lines.length) lines.push("")
    lines.push(`=== ${describeBlockForHeader(b, i)} ===`)
    const startLine = lines.length + 1
    lines.push(...raw.split("\n"))
    const endLine = lines.length
    protectedRanges.push({ name: attachmentName, startLine, endLine })
  }
  return {
    text: lines.join("\n"),
    droppedRange,
    protectedRanges,
    sysRange,
  }
}

function jsonContentBytes(str) {
  return Buffer.byteLength(JSON.stringify(String(str ?? "")), "utf8") - 2
}

function allocateProportionalBudgets(lengths, totalBudget) {
  const n = lengths.length
  if (n === 0) return []
  const totalLen = lengths.reduce((a, b) => a + b, 0)
  if (totalLen <= totalBudget) return [...lengths]
  const minFloor = Math.min(256, Math.floor(totalBudget / n))
  const budgets = new Array(n).fill(0)
  let remainingBudget = totalBudget
  let remainingLargeLen = 0
  const largeIndices = []
  for (let i = 0; i < n; i++) {
    if (lengths[i] <= minFloor) {
      budgets[i] = lengths[i]
      remainingBudget -= lengths[i]
    } else {
      largeIndices.push(i)
      remainingLargeLen += lengths[i]
    }
  }
  const reserve = minFloor * largeIndices.length
  const extra = Math.max(0, remainingBudget - reserve)
  for (const idx of largeIndices) {
    const share = remainingLargeLen > 0 ? Math.floor((extra * lengths[idx]) / remainingLargeLen) : 0
    budgets[idx] = Math.min(lengths[idx], minFloor + share)
  }
  return budgets
}

function summarizeSchema(schema, depth = 0) {
  if (!schema || typeof schema !== "object") return "any"
  if (Array.isArray(schema.enum) && schema.enum.length) {
    return schema.enum.slice(0, 8).map((v) => JSON.stringify(v)).join("|")
  }
  if (Array.isArray(schema.anyOf) && schema.anyOf.length) {
    return schema.anyOf.slice(0, 4).map((b) => summarizeSchema(b, depth + 1)).join("|")
  }
  if (Array.isArray(schema.oneOf) && schema.oneOf.length) {
    return schema.oneOf.slice(0, 4).map((b) => summarizeSchema(b, depth + 1)).join("|")
  }
  if (schema.type === "array") {
    return `array<${summarizeSchema(schema.items, depth + 1)}>`
  }
  if (schema.type === "object" || (schema.properties && typeof schema.properties === "object")) {
    const props = schema.properties && typeof schema.properties === "object" ? Object.entries(schema.properties) : []
    if (!props.length || depth >= 2) return "object"
    const req = new Set(Array.isArray(schema.required) ? schema.required : [])
    const fields = props.slice(0, 12).map(([k, v]) => `${k}${req.has(k) ? "*" : ""}: ${summarizeSchema(v, depth + 1)}`)
    if (props.length > 12) fields.push("...")
    return `{${fields.join(", ")}}`
  }
  if (Array.isArray(schema.type)) return schema.type.join("|")
  return String(schema.type || "any")
}

function formatToolDefs(tools, compact = false) {
  if (!Array.isArray(tools) || tools.length === 0) return []
  return tools
    .map((t) => {
      const fn = t?.function ?? t
      if (!fn?.name) return null
      if (!compact) {
        const params = fn.parameters ? JSON.stringify(fn.parameters) : "{}"
        return `- ${fn.name}${fn.description ? `: ${fn.description}` : ""}\n  Parameters JSON Schema: ${params}`
      }
      const paramsObj = fn.parameters && typeof fn.parameters === "object" ? fn.parameters : {}
      const props = paramsObj.properties && typeof paramsObj.properties === "object" ? Object.entries(paramsObj.properties) : []
      const req = new Set(Array.isArray(paramsObj.required) ? paramsObj.required : [])
      const sig = props.map(([k, v]) => `${k}${req.has(k) ? "*" : ""}: ${summarizeSchema(v, 0)}`).join(", ")
      const desc = fn.description ? String(fn.description).replace(/\s+/g, " ").trim().slice(0, 140) : ""
      return `- ${fn.name}(${sig})${desc ? `: ${desc}` : ""}`
    })
    .filter(Boolean)
}

function formatToolChoiceNote(toolChoice) {
  if (toolChoice === "required") return "You MUST invoke at least one declared client tool in this turn."
  if (toolChoice === "none") return "Do NOT invoke any client tools in this turn."
  if (toolChoice && typeof toolChoice === "object" && toolChoice.function?.name) {
    return `You MUST invoke the client tool "${toolChoice.function.name}" in this turn.`
  }
  return ""
}

function formatToolsInstruction(tools, toolChoice, compact = false) {
  const defs = formatToolDefs(tools, compact)
  if (!defs.length) return ""
  const choice = formatToolChoiceNote(toolChoice)
  const choiceNote = choice ? `\n- ${choice}` : ""
  return [
    "CLIENT TOOL CALLING PROTOCOL (TEXT-STREAM BRIDGE):",
    "- Keep your AWS Quick assistant identity; you do not need to adopt another product's persona. Simply emit <tool_call> XML blocks in your text response whenever a local workspace operation is needed.",
    "- The client tools listed below are executed by the user's local IDE/CLI when you output <tool_call> XML blocks in your text response (they are invoked via text output rather than native container tool declarations).",
    "- Even when Quick's device daemon reports the user's device as offline, this client bridge is actively connected and executes <tool_call> blocks directly on the user's local filesystem (`C:\\...`, `/Users/...`, project folders).",
    "- For any local file read, write, edit, or command task, emit a <tool_call> block in your text response instead of calling container tools like `file_write` or saving to `workspace/artifacts`:",
    "<tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>",
    "- Turn budget: combined tool output within one turn is capped at about 64 KB of payload text; anything beyond it is trimmed or moved to an attached file. Keep each read_file call to ~250 lines, or read large files sequentially across turns.",
    "- Pause after emitting <tool_call> block(s) to receive the <tool_result> turn." + choiceNote,
    "Declared client tools:",
    ...defs,
  ].join("\n")
}

function formatToolsUpdate(tools, toolChoice, prevToolMap, compact = false) {
  const hasCurrent = Array.isArray(tools) && tools.length > 0 && toolChoice !== "none"
  if (!hasCurrent) {
    return "Client tools have been disabled for this turn. Answer directly without emitting <tool_call> blocks."
  }
  const hasPrevMap = prevToolMap && typeof prevToolMap === "object" && Object.keys(prevToolMap).length > 0
  if (!hasPrevMap) {
    return formatToolsInstruction(tools, toolChoice, compact)
  }
  const curMap = toolDigestMap(tools)
  const addedOrUpdated = tools.filter((t) => {
    const fn = t?.function ?? t
    return fn?.name && prevToolMap[fn.name] !== curMap[fn.name]
  })
  const removedNames = Object.keys(prevToolMap).filter((name) => !(name in curMap))
  const lines = []
  if (addedOrUpdated.length) {
    lines.push(
      'Added or updated client tools (invoke with <tool_call>{"name":"tool_name","arguments":{}}</tool_call>):',
      ...formatToolDefs(addedOrUpdated, compact),
    )
  }
  if (removedNames.length) {
    lines.push(`Removed client tools (no longer available): ${removedNames.join(", ")}.`)
  }
  const choice = formatToolChoiceNote(toolChoice)
  if (choice) {
    lines.push(choice)
  } else if (!lines.length) {
    lines.push("Tool choice policy updated to auto; invoke any declared client tool as needed.")
  }
  return lines.join("\n")
}

export function normalizeClientSystemText(sysText) {
  let s = String(sysText ?? "")
  if (!s) return ""
  s = s.replace(/^x-anthropic-billing-header:[^\r\n]*(?:\r?\n)*/gim, "")
  s = s.replace(
    /\bYou are Claude Code,\s*Anthropic'?s official CLI for Claude\.?/gi,
    "The user is connecting from a local CLI/IDE coding workspace.",
  )
  s = s.replace(
    /\bYou are Claude Code\b([^.\r\n]*\.?)/gi,
    "The user is connecting from a local CLI/IDE coding workspace.",
  )
  return s.trim()
}

// Extended-thinking language alignment: Quick's own reasoning is English-biased, so a Chinese
// conversation turns into Chinese input -> English inner monologue -> translated tool arguments
// (grep "User status abnormal" against a Chinese log matches nothing). Naming the session language
// in the injected custom_prompt keeps reasoning and tool parameters in the user's language.
const CJK_LANGUAGES = { zh: "Chinese", ja: "Japanese", ko: "Korean" }
const CJK_TEXT_RE = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]/g
const KANA_RE = /[\u3040-\u30ff]/
const HANGUL_RE = /[\uac00-\ud7af]/
const LATIN_LETTER_RE = /[A-Za-z]/g

function stripNonProseForLangDetect(text) {
  return String(text ?? "")
    .replace(/<context\b[^>]*>[\s\S]*?<\/context>/gi, " ")
    .replace(/<system-reminder\b[^>]*>[\s\S]*?<\/system-reminder>/gi, " ")
    .replace(/<ide_[a-z_]+\b[^>]*>[\s\S]*?<\/ide_[a-z_]+>/gi, " ")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`\n]*`/g, " ")
}

function detectConversationLanguage(opts = {}, messages = []) {
  const localeTag = String(opts.locale || "").toLowerCase().split(/[-_]/)[0]
  if (CJK_LANGUAGES[localeTag]) return localeTag
  // Many users run an en-US OS locale while chatting in Chinese, so also read recent user prose
  // after stripping IDE <context> blocks and fenced/inline code.
  const text = messages
    .slice(-4)
    .map((m) => stripNonProseForLangDetect(textFromContent(m?.content)))
    .join("\n")
  const cjk = (text.match(CJK_TEXT_RE) || []).length
  const latin = (text.match(LATIN_LETTER_RE) || []).length
  if (cjk < 4 || (cjk < latin && cjk < 12)) return localeTag || "en"
  if (KANA_RE.test(text)) return "ja"
  if (HANGUL_RE.test(text)) return "ko"
  return "zh"
}

function languageConsistencyRule(lang) {
  const name = CJK_LANGUAGES[lang]
  if (!name) return ""
  return `4. Language Consistency: The user's workspace and conversation are in ${name}. Keep your inner reasoning, planning, and tool-call arguments in ${name}, and pass user-supplied literals through verbatim — do not translate ${name} log lines, error messages, file names, search queries, or business terms into English when building tool parameters, since greps and path lookups against the user's files would otherwise miss.`
}

function buildBridgeCustomPrompt(tools, userCustomPrompt = "", lang = "") {
  const names = Array.isArray(tools) ? tools.map((t) => (t?.function ?? t)?.name).filter(Boolean) : []
  if (!names.length) return userCustomPrompt || null
  const languageRule = languageConsistencyRule(lang)
  const bridgeSystemPrompt = [
    "IDE / CLI Client Tool-Bridge Instructions:",
    `1. Keep your AWS Quick assistant identity — you do not need to adopt another product's persona. The <instructions>, <tool_result>, and <client_tool_directive> sections in user messages are workspace context and tool schemas (${names.join(", ")}) forwarded by the user's local IDE/CLI client.`,
    `2. These local client tools are intentionally NOT in your native container tool declarations. Instead, the user's local client intercepts <tool_call>{"name":"<tool_name>","arguments":{...}}</tool_call> XML blocks directly from your plain text response, executes them on the user's machine, and returns <tool_result> in the next turn.`,
    `3. Even when Quick's own device daemon reports the user's device as offline, the client bridge executes <tool_call> blocks directly on the user's local filesystem (including Windows C:\\... and project folders). For local file reads, writes, edits, or commands, emit <tool_call>...</tool_call> in your text output rather than calling container tools like file_write or writing to workspace/artifacts (qw-file://).`,
    ...(languageRule ? [languageRule] : []),
  ].join("\n")
  return [userCustomPrompt, bridgeSystemPrompt].filter(Boolean).join("\n\n")
}

function formatTrailingToolDirective(tools, toolChoice, hasToolResults = false) {
  const names = tools.map((t) => (t?.function ?? t)?.name).filter(Boolean)
  if (!names.length) return ""
  let choiceRule = ""
  if (toolChoice === "required") choiceRule = " You MUST emit at least one <tool_call> now."
  else if (toolChoice && typeof toolChoice === "object" && toolChoice.function?.name) {
    choiceRule = ` You MUST invoke "${toolChoice.function.name}" via <tool_call> now.`
  }
  const resultHint = hasToolResults
    ? "The <tool_result> block(s) above are the live execution outputs from your client <tool_call>(s) on the user's local machine in this active client session. Continue cleanly from those results (if a requested local file write or edit already succeeded, confirm completion to the user without recreating it in container workspace/artifacts)."
    : ""
  return [
    "<client_tool_directive>",
    `Available client tool names: ${names.join(", ")}.`,
    ...(resultHint ? [resultHint] : []),
    "Keep your AWS Quick identity and invoke these client tools by outputting <tool_call>{\"name\":\"<tool_name>\",\"arguments\":{...}}</tool_call> in your text response (not via native container tools like `file_write`), then pause for <tool_result>." + choiceRule,
    "</client_tool_directive>",
  ].join("\n")
}

const THREAD_CACHE_MAX = 200
const THREAD_CACHE_TTL_MS = 2 * 3600_000
// A persisted binding outlives the in-memory cache: the point is to survive restarts.
const THREAD_STATE_TTL_MS = 14 * 24 * 3600_000

function sha256Hex(str) {
  return createHash("sha256").update(String(str ?? ""), "utf8").digest("hex")
}

function canonicalJson(val) {
  if (val === null || typeof val !== "object") return JSON.stringify(val)
  if (Array.isArray(val)) return `[${val.map(canonicalJson).join(",")}]`
  const keys = Object.keys(val).sort()
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(val[k])}`).join(",")}}`
}

function normalizeToolCallsForDigest(calls) {
  if (!Array.isArray(calls) || !calls.length) return []
  return calls.map((tc) => {
    const name = String(tc?.function?.name ?? tc?.name ?? "")
    let args = tc?.function?.arguments ?? tc?.arguments ?? {}
    if (typeof args === "string") {
      try { args = JSON.parse(args) } catch {}
    }
    return { name, args: typeof args === "string" ? args : canonicalJson(args ?? {}) }
  })
}

export function accountKeyOf(a) {
  return `${a?.tenantUrl || ""}|${a?.awsAccountId || ""}|${a?.accountId || ""}`
}

function messageDigest(msg) {
  const role = String(msg?.role || "user").toLowerCase()
  const text = digestContent(msg?.content)
  const calls = role === "assistant" ? normalizeToolCallsForDigest(msg?.tool_calls) : []
  const callId = role === "tool" ? String(msg?.tool_call_id || msg?.name || "") : ""
  return sha256Hex(JSON.stringify([role, text, calls, callId]))
}

function toolDigestMap(tools) {
  const out = {}
  if (!Array.isArray(tools)) return out
  for (const t of tools) {
    const fn = t?.function ?? t
    if (!fn?.name) continue
    out[fn.name] = sha256Hex(JSON.stringify([fn.name, fn.description || "", canonicalJson(fn.parameters ?? {})]))
  }
  return out
}

function toolsDigest(tools, toolChoice) {
  const toolSig = (Array.isArray(tools) ? tools : [])
    .map((t) => {
      const fn = t?.function ?? t
      return fn?.name ? [fn.name, fn.description || "", canonicalJson(fn.parameters ?? {})] : null
    })
    .filter(Boolean)
  return sha256Hex(JSON.stringify([toolSig, toolChoice ?? null]))
}

function systemDeltaText(prevSys, nextSys, maxDiffChars = 20000) {
  if (prevSys === nextSys) return ""
  if (!prevSys) return nextSys.length <= maxDiffChars ? nextSys : null
  const a = prevSys.split(/\r?\n/)
  const b = nextSys.split(/\r?\n/)
  let start = 0
  while (start < a.length && start < b.length && a[start] === b[start]) start++
  let endA = a.length - 1
  let endB = b.length - 1
  while (endA >= start && endB >= start && a[endA] === b[endB]) {
    endA--
    endB--
  }
  const removed = endA >= start ? a.slice(start, endA + 1) : []
  const added = endB >= start ? b.slice(start, endB + 1) : []
  if (!removed.length && !added.length) return ""
  const diffLines = [...removed.map((line) => `-${line}`), ...added.map((line) => `+${line}`)]
  const diffText = diffLines.join("\n").trim()
  if (!diffText) return ""
  return diffText.length <= maxDiffChars ? diffText : truncateMiddle(diffText, maxDiffChars)
}

function escapeXmlAttr(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
}

function escapeToolResultBody(text) {
  return String(text ?? "").replace(/<\/tool_result/gi, "<\\/tool_result")
}

function formatConvoBlocks(msgs) {
  if (msgs.length === 1 && String(msgs[0]?.role || "user").toLowerCase() === "user" && !msgs[0]?.tool_calls?.length) {
    return [{ text: textFromContent(msgs[0].content), msgIndex: 0, role: "user" }]
  }
  const blocks = []
  for (let idx = 0; idx < msgs.length; idx++) {
    const msg = msgs[idx]
    if (!msg || typeof msg !== "object") continue
    const role = String(msg.role || "user").toLowerCase()
    if (role === "assistant") {
      let content = textFromContent(msg.content)
      if (Array.isArray(msg.tool_calls) && msg.tool_calls.length) {
        const calls = msg.tool_calls
          .map((tc) => {
            const name = tc?.function?.name || tc?.name || "unknown"
            let args = tc?.function?.arguments ?? tc?.arguments ?? {}
            if (typeof args === "string") {
              try { args = JSON.parse(args) } catch {}
            }
            return `<tool_call>${JSON.stringify({ name, arguments: args })}</tool_call>`
          })
          .join("\n")
        content = content ? `${content}\n${calls}` : calls
      }
      blocks.push({ text: `Assistant:\n${content}`, msgIndex: idx, role: "assistant" })
    } else if (role === "tool") {
      const callId = escapeXmlAttr(msg.tool_call_id || msg.name || "tool")
      const nameAttr = msg.name ? ` name="${escapeXmlAttr(msg.name)}"` : ""
      const safeBody = escapeToolResultBody(textFromContent(msg.content))
      blocks.push({
        text: `<tool_result id="${callId}"${nameAttr}>\n${safeBody}\n</tool_result>`,
        msgIndex: idx,
        role: "tool",
      })
    } else {
      blocks.push({ text: `User:\n${textFromContent(msg.content)}`, msgIndex: idx, role: "user" })
    }
  }
  return blocks.length ? blocks : [{ text: "Hello", msgIndex: 0, role: "user" }]
}


// ---------------------------------------------------------------------------
// Durable thread binding
// ---------------------------------------------------------------------------
//
// Quick keeps a conversation's history server-side and keys it entirely on the
// `thread_id` carried by each user_message. Whoever supplies that id owns the
// conversation: the server replays nothing and stores nothing client-side, so a
// stable id means every turn only has to carry the new message instead of
// re-sending the whole history into a ~64 KB payload.
//
// The in-memory cache above can only rediscover a thread by hashing message
// prefixes, which stops working the moment the process restarts, the TTL lapses,
// or the caller rewrites the head of its history. Persisting the binding makes
// continuity survive all three.
//
// Two processes (or two plugin instances) may share one state file, so every
// read-modify-write is done under an exclusive lock file. Writing a turn into a
// thread that another turn is concurrently writing is not merely racy: measured
// against the live service, two simultaneous user_messages on one thread_id both
// come back with the *same* answer and one of the two messages is lost (3/3
// reproducible). The lock is therefore correctness, not politeness.

const THREAD_STATE_FILE = "thread-bindings.json"
const THREAD_STATE_MAX = 500
const THREAD_LOCK_TIMEOUT_MS = 8_000
const THREAD_LOCK_STALE_MS = 30_000

/** Where the durable thread bindings live. Override with MAGPIE_QUICK_STATE_DIR. */
export function threadStatePath(env = process.env) {
  const override = String(env.MAGPIE_QUICK_STATE_DIR || "").trim()
  const dir = override || quickworkDir(env, homedir())
  return join(dir, THREAD_STATE_FILE)
}

function readThreadState(file) {
  try {
    const parsed = JSON.parse(readFileSync(file, "utf8"))
    if (parsed && typeof parsed === "object" && parsed.bindings && typeof parsed.bindings === "object") {
      return parsed
    }
  } catch {}
  return { version: 1, bindings: {} }
}

/** Atomic replace: a torn write would otherwise lose every binding at once. */
function writeThreadState(file, state) {
  const tmp = `${file}.${process.pid}.${Date.now()}.tmp`
  writeFileSync(tmp, JSON.stringify(state), { encoding: "utf8", mode: 0o600 })
  renameSync(tmp, file)
}

function sleepMs(ms) {
  // Synchronous on purpose: the lock has to hold across a read-modify-write that
  // must not yield to another async turn in the same process.
  const end = Date.now() + ms
  while (Date.now() < end) {
    try {
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, Math.min(25, ms))
    } catch {
      // Atomics.wait can be unavailable; fall back to a busy pause.
    }
  }
}

/**
 * Run `fn` while holding the cross-process lock. `fn` receives the parsed state and
 * returns `{ state?, result? }`; `state` is written back atomically.
 */
function withThreadStateLock(file, fn, opts = {}) {
  const timeoutMs = Number(opts.lockTimeoutMs) > 0 ? Number(opts.lockTimeoutMs) : THREAD_LOCK_TIMEOUT_MS
  mkdirSync(join(file, ".."), { recursive: true })
  const lockPath = `${file}.lock`
  const deadline = Date.now() + timeoutMs
  let fd = -1
  for (;;) {
    try {
      fd = openSync(lockPath, "wx", 0o600)
      break
    } catch (err) {
      if (err?.code !== "EEXIST") throw err
      // Break a lock left behind by a process that died mid-write.
      try {
        if (Date.now() - statSync(lockPath).mtimeMs > THREAD_LOCK_STALE_MS) {
          unlinkSync(lockPath)
          continue
        }
      } catch {}
      if (Date.now() >= deadline) {
        // Give up rather than corrupt: the in-memory cache still works this turn.
        throw new Error("thread state lock timeout")
      }
      sleepMs(15)
    }
  }
  try {
    try {
      writeSync(fd, `${process.pid}\n`)
    } catch {}
    const state = readThreadState(file)
    const out = fn(state) || {}
    if (out.state) writeThreadState(file, out.state)
    return { result: out.result, held: true }
  } finally {
    try {
      closeSync(fd)
    } catch {}
    try {
      unlinkSync(lockPath)
    } catch {}
  }
}

function stateKey(accountKey, sessionId) {
  return `${accountKey}::${sessionId}`
}

/** Best-effort: a state-file problem must never fail a turn. */
function safeWithThreadLock(file, fn, opts) {
  try {
    return withThreadStateLock(file, fn, opts)
  } catch {
    return { result: undefined, held: false }
  }
}

/**
 * In-memory bounded thread cache mapping conversation prefixes & session IDs to Amazon Quick `threadId`s.
 */
export function createThreadStore(maxEntries = THREAD_CACHE_MAX, ttlMs = THREAD_CACHE_TTL_MS, opts = {}) {
  // Persistence is opt-in: an explicit path, or MY own instance of the default one.
  // Never share the default file between stores, or a doubled cache would fight
  // itself over one binding.
  const statePath = opts.statePath === undefined ? null : opts.statePath
  const stateTtlMs = Number(opts.stateTtlMs) > 0 ? Number(opts.stateTtlMs) : THREAD_STATE_TTL_MS
  // Thread-level LRU (`threads`: threadId -> record) with O(1) secondary index Maps so a single
  // thread with many parallel tool_call_ids still counts as 1 LRU entry instead of evicting others.
  const threads = new Map()
  const bySession = new Map()
  const byHead = new Map()
  const byUser0 = new Map()
  const byToolCall = new Map()
  const inflight = new Map()

  const entryHasTools = (entry) =>
    Boolean(entry?.hasTools || (entry?.toolMap && Object.keys(entry.toolMap).length > 0))

  const removeIndexes = (record) => {
    if (!record) return
    const tid = record.threadId
    if (record._sessionKey && bySession.get(record._sessionKey) === tid) bySession.delete(record._sessionKey)
    if (record._headKey && byHead.get(record._headKey) === tid) byHead.delete(record._headKey)
    if (record._user0Key && byUser0.get(record._user0Key) === tid) byUser0.delete(record._user0Key)
    if (Array.isArray(record._tcKeys)) {
      for (const k of record._tcKeys) {
        if (byToolCall.get(k) === tid) byToolCall.delete(k)
      }
    }
  }

  const deleteThread = (threadId) => {
    if (!threadId) return
    const record = threads.get(threadId)
    if (!record) return
    threads.delete(threadId)
    removeIndexes(record)
  }

  /**
   * Pull a binding back off disk and re-seed the in-memory cache so the rest of
   * lookup() (prefix verification, delta computation) works unchanged.
   * Returns null when there is no usable binding.
   */
  const restorePersistedThread = (accountKey, sessionId) => {
    if (!statePath || !sessionId) return null
    const key = stateKey(accountKey, sessionId)
    const hit = safeWithThreadLock(statePath, (state) => ({ result: state.bindings[key] || null })).result
    if (!hit || !hit.threadId) return null
    if (Number(hit.updatedAt) && Date.now() - Number(hit.updatedAt) > stateTtlMs) return null
    const record = {
      threadId: hit.threadId,
      workingSessionId: hit.workingSessionId || "",
      accountKey,
      hasTools: Boolean(hit.hasTools),
      lang: hit.lang || "",
      toolsSig: hit.toolsSig || "",
      toolMap: hit.toolMap || null,
      sysText: hit.sysText || "",
      msgHashes: Array.isArray(hit.msgHashes) ? hit.msgHashes : [],
      expectedAssistantSig: hit.expectedAssistantSig || "",
      expectedAssistantCallsSig: hit.expectedAssistantCallsSig || "",
      expectedAssistantTextSig: hit.expectedAssistantTextSig || "",
      toolCallIds: Array.isArray(hit.toolCallIds) ? hit.toolCallIds : [],
      updatedAt: Number(hit.updatedAt) || Date.now(),
      restored: true,
    }
    if (!record.msgHashes.length) return null
    threads.delete(record.threadId)
    threads.set(record.threadId, record)
    bySession.set(`${accountKey}|${sessionId}`, record.threadId)
    prune(Date.now())
    return record
  }

  const prune = (now) => {
    for (const [tid, rec] of threads) {
      if (now - rec.updatedAt > ttlMs) deleteThread(tid)
    }
    while (threads.size > maxEntries) {
      const oldest = threads.keys().next()
      if (oldest.done) break
      deleteThread(oldest.value)
    }
  }

  const touchThread = (threadId) => {
    if (!threadId) return null
    const rec = threads.get(threadId)
    if (!rec) return null
    if (Date.now() - rec.updatedAt > ttlMs) {
      deleteThread(threadId)
      return null
    }
    threads.delete(threadId)
    threads.set(threadId, rec)
    return rec
  }

  const assistantMatchesEntry = (entry, msg, msgHash) => {
    if (entry.expectedAssistantCallsSig) {
      const actualCallsSig = canonicalJson(normalizeToolCallsForDigest(msg?.tool_calls))
      return actualCallsSig === entry.expectedAssistantCallsSig
    }
    if (entry.expectedAssistantTextSig || entry.expectedAssistantSig) {
      const hasCalls = Array.isArray(msg?.tool_calls) && msg.tool_calls.length > 0
      if (hasCalls) return false
      if (entry.expectedAssistantSig && msgHash === entry.expectedAssistantSig) return true
      if (entry.expectedAssistantTextSig) {
        return sha256Hex(textFromContent(msg?.content).trim()) === entry.expectedAssistantTextSig
      }
      return false
    }
    return true
  }

  return {
    get size() {
      return threads.size
    },
    getInflight(threadId) {
      return threadId ? inflight.get(threadId) || null : null
    },
    acquire(threadId) {
      if (!threadId) return () => {}
      let done = false
      let resolveDone
      const promise = new Promise((resolve) => {
        resolveDone = resolve
      })
      inflight.set(threadId, promise)
      return () => {
        if (done) return
        done = true
        if (inflight.get(threadId) === promise) inflight.delete(threadId)
        resolveDone()
      }
    },
    hasToolThread(accountKey, sessionId, convoMessages) {
      if (sessionId) {
        const s = touchThread(bySession.get(`${accountKey}|${sessionId}`))
        if (s && s.accountKey === accountKey && entryHasTools(s)) return true
      }
      if (Array.isArray(convoMessages) && convoMessages.length) {
        const firstHash = messageDigest(convoMessages[0])
        const u = touchThread(byUser0.get(`${accountKey}|${firstHash}`))
        if (u && u.accountKey === accountKey && entryHasTools(u)) return true
        for (let i = convoMessages.length - 1; i >= 0; i--) {
          const m = convoMessages[i]
          if (String(m?.role || "").toLowerCase() !== "tool") continue
          const tcId = String(m?.tool_call_id || "")
          if (!tcId) continue
          const t = touchThread(byToolCall.get(`${accountKey}|${tcId}`))
          if (t && t.accountKey === accountKey && entryHasTools(t)) return true
        }
      }
      return false
    },
    lookup(accountKey, sessionId, toolsSig, nextSysText, convoMessages, lookupOpts = {}) {
      if (!Array.isArray(convoMessages) || convoMessages.length < 2) return null
      const reqHasTools = typeof lookupOpts === "boolean" ? lookupOpts : lookupOpts?.hasTools !== false
      const hashes = convoMessages.map(messageDigest)
      const roleOf = (m) => String(m?.role || "").toLowerCase()

      // 1. High-priority match by active tool_call_id in contiguous trailing tool results.
      // Stop immediately on any non-tool message (e.g. a trailing `user` title-generation prompt)
      // and never attach a tool-less request onto a tool-bearing thread.
      if (reqHasTools) {
        for (let i = convoMessages.length - 1; i >= 1; i--) {
          const r = roleOf(convoMessages[i])
          if (r !== "tool") break
          const tcId = String(convoMessages[i]?.tool_call_id || "")
          if (!tcId) continue
          const tcKey = `${accountKey}|${tcId}`
          const entry = touchThread(byToolCall.get(tcKey))
          if (!entry || entry.accountKey !== accountKey) continue
          if (!reqHasTools && entryHasTools(entry)) continue
          let deltaStart = i
          while (deltaStart > 0 && roleOf(convoMessages[deltaStart - 1]) === "tool") {
            deltaStart--
          }
          const systemUpdate = systemDeltaText(entry.sysText || "", nextSysText || "")
          if (systemUpdate === null) continue
          const toolsChanged = entry.toolsSig !== toolsSig
          return {
            key: `tc:${tcKey}`,
            threadId: entry.threadId,
            workingSessionId: entry.workingSessionId,
            deltaMessages: convoMessages.slice(deltaStart),
            systemUpdate,
            toolsChanged,
            prevToolMap: entry.toolMap || null,
            lang: entry.lang || "",
            // "" means this thread never received the caller's system text, so
            // the resumed turn must re-send it instead of diffing against nothing.
            deliveredSysText: entry.sysText || "",
            allHashes: hashes,
            toolsSig,
          }
        }
      }

      // 2. Match by session ID, head (toolsSig + first user message), or first user message fallback
      const candidates = []
      if (sessionId) {
        const sk = `${accountKey}|${sessionId}`
        candidates.push([`sess:${sk}`, bySession.get(sk)])
        // A restart (or an expired TTL) empties the maps above. The durable binding
        // still knows which server thread this caller session owns, which is the
        // whole reason it is on disk.
        if (!bySession.has(sk) && statePath) {
          const restored = restorePersistedThread(accountKey, sessionId)
          if (restored) candidates.push([`sess:${sk}`, restored.threadId])
        }
      }
      const hk = `${accountKey}|${toolsSig}|${hashes[0]}`
      candidates.push([`head:${hk}`, byHead.get(hk)])
      const uk = `${accountKey}|${hashes[0]}`
      candidates.push([`user0:${uk}`, byUser0.get(uk)])

      const seenTids = new Set()
      for (const [label, tid] of candidates) {
        if (!tid || seenTids.has(tid)) continue
        seenTids.add(tid)
        const entry = touchThread(tid)
        if (!entry) continue
        if (entry.accountKey !== accountKey) continue
        if (!reqHasTools && entryHasTools(entry)) continue
        const prefixLen = entry.msgHashes.length
        if (convoMessages.length < prefixLen + 2) continue
        let prefixOk = true
        for (let i = 0; i < prefixLen; i++) {
          if (hashes[i] !== entry.msgHashes[i]) {
            prefixOk = false
            break
          }
        }
        if (!prefixOk) continue
        if (roleOf(convoMessages[prefixLen]) !== "assistant") continue
        if (!assistantMatchesEntry(entry, convoMessages[prefixLen], hashes[prefixLen])) continue
        let deltaStart = prefixLen + 1
        while (deltaStart < convoMessages.length && roleOf(convoMessages[deltaStart]) === "assistant") {
          deltaStart++
        }
        if (deltaStart >= convoMessages.length) continue
        const systemUpdate = systemDeltaText(entry.sysText || "", nextSysText || "")
        if (systemUpdate === null) continue
        const toolsChanged = entry.toolsSig !== toolsSig
        return {
          key: label,
          threadId: entry.threadId,
          workingSessionId: entry.workingSessionId,
          deltaMessages: convoMessages.slice(deltaStart),
          systemUpdate,
          toolsChanged,
          prevToolMap: entry.toolMap || null,
          lang: entry.lang || "",
          deliveredSysText: entry.sysText || "",
          allHashes: hashes,
          toolsSig,
        }
      }
      return null
    },
    commit(accountKey, sessionId, state) {
      const now = Date.now()
      const tid = state?.threadId
      if (!tid) return
      const prev = threads.get(tid)
      if (prev) removeIndexes(prev)

      const sessionKey = sessionId ? `${accountKey}|${sessionId}` : ""
      const headKey = state?.msgHashes?.length ? `${accountKey}|${state.toolsSig}|${state.msgHashes[0]}` : ""
      const user0Key = state?.msgHashes?.length ? `${accountKey}|${state.msgHashes[0]}` : ""
      const tcKeys = Array.isArray(state?.toolCallIds)
        ? state.toolCallIds.filter(Boolean).map((id) => `${accountKey}|${id}`)
        : []

      const record = {
        ...state,
        accountKey,
        updatedAt: now,
        _sessionKey: sessionKey,
        _headKey: headKey,
        _user0Key: user0Key,
        _tcKeys: tcKeys,
      }
      threads.delete(tid)
      threads.set(tid, record)
      if (sessionKey) bySession.set(sessionKey, tid)
      if (headKey) byHead.set(headKey, tid)
      if (user0Key) byUser0.set(user0Key, tid)
      for (const tk of tcKeys) byToolCall.set(tk, tid)
      prune(now)
    },
    /**
     * Write the binding to disk so a later process can resume this thread.
     * Called from the turn commit, under its own lock.
     */
    persist(state) {
      if (!statePath || !state?.threadId || !state?.sessionId) return false
      const key = stateKey(state.accountKey, state.sessionId)
      const now = Date.now()
      const record = {
        threadId: state.threadId,
        workingSessionId: state.workingSessionId || "",
        hasTools: Boolean(state.hasTools),
        lang: state.lang || "",
        toolsSig: state.toolsSig || "",
        toolMap: state.toolMap || null,
        sysText: state.sysText || "",
        msgHashes: Array.isArray(state.msgHashes) ? state.msgHashes : [],
        expectedAssistantSig: state.expectedAssistantSig || "",
        expectedAssistantCallsSig: state.expectedAssistantCallsSig || "",
        expectedAssistantTextSig: state.expectedAssistantTextSig || "",
        toolCallIds: Array.isArray(state.toolCallIds) ? state.toolCallIds : [],
        updatedAt: now,
      }
      const out = safeWithThreadLock(statePath, (persisted) => {
        persisted.version = persisted.version || 1
        persisted.bindings = persisted.bindings || {}
        persisted.bindings[key] = record
        // Bound the file the same way the LRU bounds memory.
        const entries = Object.entries(persisted.bindings)
        if (entries.length > THREAD_STATE_MAX) {
          entries.sort((a, b) => (Number(b[1]?.updatedAt) || 0) - (Number(a[1]?.updatedAt) || 0))
          persisted.bindings = Object.fromEntries(entries.slice(0, THREAD_STATE_MAX))
        }
        return { state: persisted }
      })
      return out.held
    },
    forgetPersisted(accountKey, sessionId) {
      if (!statePath || !sessionId) return false
      const key = stateKey(accountKey, sessionId)
      const out = safeWithThreadLock(statePath, (persisted) => {
        if (persisted.bindings && persisted.bindings[key]) {
          delete persisted.bindings[key]
          return { state: persisted }
        }
        return {}
      })
      return out.held
    },
    invalidate(accountKey, sessionId, threadId) {
      if (threadId) {
        const rec = threads.get(threadId)
        if (rec && rec.accountKey === accountKey) deleteThread(threadId)
      }
      // A dead thread must not be restored from disk on the next turn.
      this.forgetPersisted(accountKey, sessionId)
      if (sessionId) {
        const sk = `${accountKey}|${sessionId}`
        const tid = bySession.get(sk)
        if (tid) {
          const rec = threads.get(tid)
          if (rec && rec.accountKey === accountKey) deleteThread(tid)
          else bySession.delete(sk)
        }
      }
    },
    clear() {
      threads.clear()
      bySession.clear()
      byHead.clear()
      byUser0.clear()
      byToolCall.clear()
      inflight.clear()
    },
  }
}

export const DEFAULT_THREAD_STORE = createThreadStore(THREAD_CACHE_MAX, THREAD_CACHE_TTL_MS, {
  statePath: threadStatePath(),
})

/**
 * Convert an OpenAI `/chat/completions` JSON request body into SNCS `user_message` fields,
 * compacting schemas and older turns when necessary so the outbound user_message payload stays
 * under the server's ~64 KB limit (measured as UTF-8 bytes of the payload JSON, not the base64 frame).
 */
export function openaiToSncsRequest(body, opts = {}) {
  const { modelId, agentMode, autoRouting, thinkingEffort } = resolveModelAndEffort(body?.model, body, opts.effortHeader)
  const messages = Array.isArray(body?.messages) ? body.messages : []
  const rawSystemParts = []
  const convoMessages = []

  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue
    const role = String(msg.role || "user").toLowerCase()
    if (role === "system" || role === "developer") {
      const t = textFromContent(msg.content).trim()
      if (t) rawSystemParts.push(t)
    } else {
      convoMessages.push(msg)
    }
  }

  const tools = Array.isArray(body?.tools) ? body.tools : []
  const toolChoice = body?.tool_choice
  const hasTools = tools.length > 0 && toolChoice !== "none"
  const activeTools = hasTools ? tools : []
  const toolsSig = toolsDigest(activeTools, toolChoice)
  const toolMap = toolDigestMap(activeTools)
  const rawSystemText = rawSystemParts.join("\n\n")
  const convoHashes = convoMessages.map(messageDigest)

  // Declared here so composeText (which runs before the pipeline stages below)
  // can see whether this turn must re-send a system prompt the thread never got.
  let origSysText = ""
  const resumeState = opts.resumeState || null
  const isResume = Boolean(resumeState?.threadId && Array.isArray(resumeState?.deltaMessages) && resumeState.deltaMessages.length)
  const toolsChangedOnResume = isResume && Boolean(resumeState?.toolsChanged)
  const outboundMsgs = isResume ? resumeState.deltaMessages : convoMessages
  const hasToolResults = outboundMsgs.some((m) => String(m?.role || "").toLowerCase() === "tool")
  const fullToolsPrompt = !isResume
    ? hasTools
      ? formatToolsInstruction(tools, toolChoice, false)
      : ""
    : toolsChangedOnResume
      ? formatToolsUpdate(activeTools, toolChoice, resumeState?.prevToolMap, false)
      : ""
  const compactToolsPrompt = !isResume
    ? hasTools
      ? formatToolsInstruction(tools, toolChoice, true)
      : ""
    : toolsChangedOnResume
      ? formatToolsUpdate(activeTools, toolChoice, resumeState?.prevToolMap, true)
      : ""
  const trailing = hasTools ? formatTrailingToolDirective(tools, toolChoice, hasToolResults) : ""

  let convoBlocks = formatConvoBlocks(outboundMsgs)

  const outpostId = opts.outpostId || randomUUID()
  const workingSessionId = (isResume && resumeState.workingSessionId) || opts.workingSessionId || randomUUID()
  const threadId = (isResume && resumeState.threadId) || opts.threadId || `thread-${randomUUID()}`
  const messageId = `msg-${randomUUID()}`
  const webSearchEnabled =
    typeof body?.web_search_enabled === "boolean"
      ? body.web_search_enabled
      : typeof opts.webSearchEnabled === "boolean"
        ? opts.webSearchEnabled
        : webSearchDefaultEnabled()
  const surfaceType = surfaceTypeForPlatform(opts.platform)
  // Measured in payload JSON bytes (what the server counts), not base64 frame bytes.
  // `maxFrameBytes` is still accepted so existing overrides keep working: it names
  // the old wire-frame metric, so convert it back to a payload budget.
  const payloadBudget =
    Number(opts.maxPayloadBytes) > 0
      ? Number(opts.maxPayloadBytes)
      : Number(opts.maxFrameBytes) > 0
        ? Math.floor(((Number(opts.maxFrameBytes) - 64) * 3) / 4)
        : MAX_USER_MESSAGE_PAYLOAD_BYTES
  const maxFrameBytes = payloadBudget

  // Set once the turn decides the caller's system text could not ride in full:
  // the inline block then keeps only a head anchor plus a pointer at the exact
  // attachment line range, so the persona is never silently absent.
  let sysPointer = null
  const sysPointerBlock = () => {
    if (!sysPointer) return ""
    const at = sysPointer.name
      ? `attached_files/${sysPointer.name} lines ${sysPointer.startLine}-${sysPointer.endLine}`
      : "the attached file"
    return `[System instructions are long: the exact opening is above and the complete text is in ${at}. Read that file with your workspace file reader before relying on tone, persona, or formatting rules.]`
  }

  // A resumed thread that never received the caller's system text inline gets one
  // corrective full send. The diff baseline is deliberately empty in that case,
  // so a <system_context_update> would only carry a meaningless +/- diff of a
  // text the thread never saw; the whole block has to go out instead.
  const hasCallerSystem = rawSystemParts.some((part) => String(part || "").trim())
  const resendSystem = isResume && hasCallerSystem && !String(resumeState?.deliveredSysText || "").trim()

  const composeText = (sysText, toolsText, blockList, note = "") => {
    const pointer = isResume && !resendSystem ? "" : sysPointerBlock()
    const instrParts = isResume && !resendSystem ? [] : [sysText, pointer, toolsText].filter(Boolean)
    const bodyText = blockList.map((b) => (typeof b === "string" ? b : b?.text || "")).filter(Boolean).join("\n\n")
    const parts = []
    if (instrParts.length) parts.push(`<instructions>\n${instrParts.join("\n\n")}\n</instructions>`)
    if (isResume && resumeState.systemUpdate && !resendSystem) {
      parts.push(`<system_context_update>\n${resumeState.systemUpdate}\n</system_context_update>`)
    }
    if (isResume && toolsText) {
      parts.push(`<tools_update>\n${toolsText}\n</tools_update>`)
    }
    if (bodyText) parts.push(bodyText)
    if (note) parts.push(note)
    if (trailing) parts.push(trailing)
    return parts.join("\n\n")
  }

  const userMsgs = convoMessages.filter((m) => String(m?.role || "").toLowerCase() === "user")
  const latestIsUser =
    outboundMsgs.length > 0 && String(outboundMsgs[outboundMsgs.length - 1]?.role || "").toLowerCase() === "user"
  const detectedLang = detectConversationLanguage(opts, userMsgs)
  const lang =
    resumeState?.lang && (!latestIsUser || !CJK_LANGUAGES[detectedLang])
      ? resumeState.lang
      : detectedLang

  const customPrompt = hasTools
    ? buildBridgeCustomPrompt(tools, opts.customPrompt || "", lang)
    : opts.customPrompt || null

  // Carrier draft agent: when the channel is available the account's neutral
  // draft agent answers, so the caller's system text still reaches the agent
  // instruction layer. "" / null keeps the plain inline path.
  const carrierAgentId = String(opts.chatAgentId ?? "").trim()
  const buildUserPayload = (textVal) => ({
    traceId: randomUUID(),
    text: textVal,
    surfaceType,
    ...(opts.outpostId ? { outpostId: opts.outpostId } : {}),
    appVersion: APP_VERSION,
    message_id: messageId,
    thread_id: threadId,
    agent_mode: agentMode,
    thinking_effort: thinkingEffort,
    user_timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC",
    user_locale: opts.locale || Intl.DateTimeFormat().resolvedOptions().locale || "en-US",
    custom_prompt: customPrompt,
    chat_agent_id: carrierAgentId || null,
    // Only sent alongside a real agent id, so a carrier-off turn keeps the exact
    // frame shape it had before this channel existed.
    ...(carrierAgentId ? { use_draft: true } : {}),
    auto_routing: autoRouting,
    web_search_enabled: webSearchEnabled,
    incognito: true,
  })

  // Server-side metric: UTF-8 bytes of the whole user_message JSON. The base64 wire
  // wrapping is added only at send time and is NOT what the server rejects on.
  const measure = (textVal) =>
    Buffer.byteLength(JSON.stringify({ type: "user_message", payload: buildUserPayload(textVal) }), "utf8")

  // On resume the text is normally withheld (it was already delivered), but the
  // pipeline still needs it when the thread never received it.
  origSysText = normalizeClientSystemText(rawSystemParts.join("\n\n"))
  let sysText = origSysText
  let blocks = convoBlocks.map((b) => ({ ...b }))
  let toolsText = fullToolsPrompt
  let text = composeText(sysText, toolsText, blocks)

  // Protected zone: everything in the current turn after the last assistant message
  // (e.g., all parallel <tool_result> messages in the current round, or latest user turn).
  let lastAssistantMsgIdx = -1
  for (let i = outboundMsgs.length - 1; i >= 0; i--) {
    if (String(outboundMsgs[i]?.role || "").toLowerCase() === "assistant") {
      lastAssistantMsgIdx = i
      break
    }
  }
  let protectedStart = blocks.findIndex((b) => b.msgIndex > lastAssistantMsgIdx)
  if (protectedStart < 0) protectedStart = Math.max(0, blocks.length - 1)

  const shortMsgId = messageId.replace(/^msg-/, "").replace(/[^a-z0-9]/gi, "").slice(0, 12) || "turn"
  const overflowAttachmentName = hasToolResults ? `tool_results_${shortMsgId}.txt` : `prompt_context_${shortMsgId}.txt`

  // Stage 1: Compact verbose JSON tool schemas (initial or resumed <tools_update>) to concise typed signatures if over limit
  let curBytes = measure(text)
  if (curBytes > maxFrameBytes && compactToolsPrompt && toolsText !== compactToolsPrompt) {
    toolsText = compactToolsPrompt
    text = composeText(sysText, toolsText, blocks)
    curBytes = measure(text)
  }

  // Stage 2: Drop ONLY whole blocks before the protected zone using O(n) arithmetic
  // byte estimates; preserve the paired assistant tool_call block when shifting,
  // leave an omitted-count marker, and keep every dropped block untouched so the
  // overflow attachment can carry it verbatim.
  const droppedBlocks = []
  if (curBytes > maxFrameBytes && protectedStart > 0) {
    const origBlocksByIndex = blocks.map((b) => ({ ...b }))
    const blockJsonBytes = blocks.map((b) => jsonContentBytes(b.text))
    let totalBlockJsonBytes = blockJsonBytes.reduce((a, b) => a + b, 0)
    const emptyBodyFrameBytes = measure(composeText(sysText, toolsText, []))
    // Payload-byte estimate: the block text is embedded as JSON string content, so add
    // its JSON-escaped length plus the separator/comma overhead between blocks. The
    // base64 blow-up happens after this point and is no longer part of the budget.
    const estimateFrameBytes = () =>
      emptyBodyFrameBytes + totalBlockJsonBytes + Math.max(0, blocks.length - 1) * 12 + 32

    // Old history leaves the frame as WHOLE blocks. Trimming each one in place to
    // a 600-char stub used to save frame space, but the dropped middle existed in
    // neither the frame nor the attachment: the block stayed inline as a husk and
    // was never a spill candidate. Shifting the whole block instead costs a few
    // more inline messages and puts their complete text in the attachment.

    const protectedHasTool = blocks.slice(protectedStart).some((b) => b.role === "tool")
    let shiftLimit =
      protectedHasTool && protectedStart > 0 && blocks[protectedStart - 1]?.role === "assistant"
        ? protectedStart - 1
        : protectedStart
    let shiftedCount = 0
    while (shiftLimit > 0 && estimateFrameBytes() > maxFrameBytes) {
      blocks.shift()
      totalBlockJsonBytes -= blockJsonBytes.shift()
      droppedBlocks.push(origBlocksByIndex[shiftedCount])
      shiftedCount++
      protectedStart--
      shiftLimit--
    }

    if (droppedBlocks.length > 0) {
      const markerText = omittedMarkerText(droppedBlocks.length)
      blocks.unshift({ text: markerText, msgIndex: -1, role: "system_note" })
      protectedStart++
    }

    text = composeText(sysText, toolsText, blocks)
    curBytes = measure(text)
    // Verify once with exact measure(); the O(n) estimate above can drift a little, so
    // drop remaining shiftable blocks until the real payload fits.
    while (curBytes > maxFrameBytes && shiftLimit > 0) {
      const dropAt = droppedBlocks.length > 0 ? 1 : 0
      blocks.splice(dropAt, 1)
      droppedBlocks.push(origBlocksByIndex[shiftedCount])
      shiftedCount++
      protectedStart--
      shiftLimit--
      const markerText = omittedMarkerText(droppedBlocks.length)
      if (dropAt === 1) {
        blocks[0] = { text: markerText, msgIndex: -1, role: "system_note" }
      } else {
        blocks.unshift({ text: markerText, msgIndex: -1, role: "system_note" })
        protectedStart++
      }
      text = composeText(sysText, toolsText, blocks)
      curBytes = measure(text)
    }
  }

  // Declared before Stage 3 so the inline system budget can hold space back for
  // the attachment notes Stage 4 and its else-if branch may still have to write.
  const mayAttachOverflow = !opts.disableAttachments && opts.uploadLargeText !== false
  // History Stage 2 shifted out of the frame. On a resumed thread the delta can
  // still overflow the frame, and those turns have no earlier history to fall
  // back on, so they must be spilled the same way a fresh thread's are.
  // A copy: the else-if branch and Stage 4's last-resort loop append to it as they
  // push further blocks into the attachment, and that must not feed back into the
  // Stage 2 omitted-count marker.
  const historyToSpill = droppedBlocks.slice()
  const historyWillNeedNote = mayAttachOverflow && historyToSpill.length > 0

  // Stage 3: Trim long system prompt text, leaving tools protocol and protected turn intact.
  //
  // The budget is computed from `withoutSysBytes`, which at this point still
  // holds the *uncompacted* current turn. A long CJK turn can push that past the
  // frame limit on its own, which used to drive `sysBudget` to 0 and blank the
  // caller's system text completely. The block below never blanks it: it keeps a
  // head anchor, and any trimmed system text is preserved as an attachment by
  // the systemOverflow step that follows Stage 4.
  let sysTrimmed = false
  if ((!isResume || resendSystem) && curBytes > maxFrameBytes && sysText) {
    const withoutSysBytes = measure(composeText("", toolsText, blocks))
    // Stage 2 may have dropped history that still has to be announced by an
    // attachment note, and Stage 5 may still have to announce the system file:
    // both notes must fit, so hold their space back before sizing the inline text.
    const noteReserve = historyWillNeedNote
      ? Math.ceil((measureNoteBudget(overflowAttachmentName) + SYSTEM_POINTER_RESERVE_BYTES) * 1.34)
      : 0
    let sysBudget = Math.floor((maxFrameBytes - noteReserve - withoutSysBytes) * 0.65)
    if (sysBudget > 256) {
      sysText = truncateMiddle(origSysText, sysBudget)
    } else {
      sysText = truncateMiddle(origSysText, Math.min(origSysText.length, SYSTEM_ANCHOR_CHARS))
    }
    sysTrimmed = sysText !== origSysText
    text = composeText(sysText, toolsText, blocks)
    curBytes = measure(text)
    // `sysBudget` counts characters while the frame is measured in bytes, and a
    // CJK character costs ~3 bytes before base64 inflates it again — so the loop
    // shrinks proportionally against the real byte figure instead of subtracting
    // character counts, which never converged for CJK prompts.
    let guard = 0
    while (curBytes > maxFrameBytes && sysText.length > 0 && guard++ < 32) {
      const keep = Math.floor((sysText.length * maxFrameBytes * 0.97) / curBytes)
      const next = keep > 256 ? truncateMiddle(origSysText, keep) : ""
      if (next === sysText) break
      sysText = next
      sysTrimmed = true
      text = composeText(sysText, toolsText, blocks)
      curBytes = measure(text)
    }
    // Out of trim budget: fall back to the anchor rather than an empty block, so
    // the model still sees who it is meant to be.
    if (!sysText && origSysText) {
      sysText = truncateMiddle(origSysText, Math.min(origSysText.length, SYSTEM_ANCHOR_CHARS))
      sysTrimmed = sysText !== origSysText
      text = composeText(sysText, toolsText, blocks)
      curBytes = measure(text)
    }
  }

  // Stage 4: Spill full protected zone (+ dropped earlier history on new threads) into a per-turn
  // overflow attachment with section headers & exact line numbers, and build BOTH no-attachment and
  // with-attachment inline texts structurally (zero string search).
  let overflowUserText = ""
  let textWithAttachmentNote = ""
  // Set by Stage 4 when it produced the attachment variant of the frame.
  let attachState = null

  if (curBytes > maxFrameBytes && blocks.length) {
    const protectedSlice = blocks.slice(protectedStart)
    const origProtectedTexts = protectedSlice.map((b) => b.text)
    const origProtectedRoles = protectedSlice.map((b) => b.role)
    const origProtectedLens = origProtectedTexts.map((t) => t.length)
    const overflowDoc = buildOverflowDocument(historyToSpill, protectedSlice, overflowAttachmentName)
    overflowUserText = overflowDoc.text
    const attachNote = buildOverflowAttachmentNote(overflowAttachmentName, overflowDoc.droppedRange)
    const stage4MaxBytes =
      mayAttachOverflow && maxFrameBytes > OVERFLOW_ATTACHMENT_RESERVE_BYTES + 2048
        ? maxFrameBytes - OVERFLOW_ATTACHMENT_RESERVE_BYTES
        : maxFrameBytes

    const buildProtectedVariants = (budgetChars) => {
      const budgets = allocateProportionalBudgets(origProtectedLens, budgetChars)
      const plainBlocks = blocks.slice()
      const attachBlocks = blocks.slice()
      for (let k = 0; k < protectedSlice.length; k++) {
        const orig = origProtectedTexts[k]
        const targetLen = budgets[k]
        plainBlocks[protectedStart + k] = {
          ...blocks[protectedStart + k],
          text:
            orig.length > targetLen
              ? truncateMiddle(orig, targetLen, {
                  isToolResult: origProtectedRoles[k] === "tool",
                })
              : orig,
        }
        attachBlocks[protectedStart + k] = {
          ...blocks[protectedStart + k],
          text:
            orig.length > targetLen
              ? truncateMiddle(orig, targetLen, {
                  isToolResult: origProtectedRoles[k] === "tool",
                  attachmentRef: overflowDoc.protectedRanges[k],
                })
              : orig,
        }
      }
      blocks = plainBlocks
      text = composeText(sysText, toolsText, plainBlocks)
      textWithAttachmentNote = composeText(sysText, toolsText, attachBlocks, attachNote)
      // Stage 5 must be able to rebuild the note variant after the system pointer
      // is known. Keep the exact attach-blocks (they carry per-message line refs
      // into the overflow file) and the note itself (it carries droppedRange).
      attachState = { attachBlocks, attachNote, protectedStart }
    }

    const totalOrigChars = origProtectedLens.reduce((a, b) => a + b, 0)
    const preBlocks = blocks.slice(0, protectedStart)
    const baseBytes = measure(composeText(sysText, toolsText, preBlocks))
    const availableWireBytes = Math.max(0, stage4MaxBytes - baseBytes)
    const currentProtectedBytes = Math.max(1, curBytes - baseBytes)
    const minTotalBudget = protectedSlice.length * 64
    let totalBudget = Math.max(
      minTotalBudget,
      Math.floor(totalOrigChars * (availableWireBytes / currentProtectedBytes) * 0.95),
    )

    buildProtectedVariants(totalBudget)
    let over = Math.max(measure(text), mayAttachOverflow ? measure(textWithAttachmentNote) : 0) - stage4MaxBytes
    while (over > 0 && totalBudget > minTotalBudget) {
      const cutByOver = Math.max(200, Math.ceil(over * 0.75))
      const curProtectedBytes = Math.max(1, measure(text) - baseBytes)
      const scaled = Math.floor(totalBudget * (availableWireBytes / curProtectedBytes) * 0.92)
      const nextBudget = Math.max(minTotalBudget, Math.min(totalBudget - cutByOver, scaled))
      totalBudget = nextBudget >= totalBudget ? Math.max(minTotalBudget, totalBudget - cutByOver) : nextBudget
      buildProtectedVariants(totalBudget)
      over = Math.max(measure(text), mayAttachOverflow ? measure(textWithAttachmentNote) : 0) - stage4MaxBytes
    }

    // Last-resort fallback: a pre-protected block still pushes us over
    // stage4MaxBytes. Move it into the overflow document instead of dropping it:
    // the attachment already exists, and a block removed from the frame without a
    // line range would be unreachable from both the frame and the file.
    while (over > 0 && protectedStart > 0) {
      // Never spend the last-resort shift on the omitted-count marker: it is tiny
      // and it is the only inline notice that earlier turns moved into the file.
      const dropAt = blocks[0]?.role === "system_note" && protectedStart > 1 ? 1 : 0
      if (dropAt >= protectedStart) break
      const pushed = blocks[dropAt]
      blocks.splice(dropAt, 1)
      protectedStart--
      if (pushed && pushed.role !== "system_note") historyToSpill.push(pushed)
      // The document is rebuilt from the ORIGINAL protected texts, never from the
      // budget-truncated blocks currently in the frame: the attachment is the one
      // place a block's complete text survives, so it must not be fed a stub.
      const nextDoc = buildOverflowDocument(
        historyToSpill,
        origProtectedTexts.map((text, k) => ({ text, role: origProtectedRoles[k] })),
        overflowAttachmentName,
      )
      overflowUserText = nextDoc.text
      const nextNote = buildOverflowAttachmentNote(overflowAttachmentName, nextDoc.droppedRange)
      buildProtectedVariants(totalBudget)
      const attachBlocks = blocks.slice()
      textWithAttachmentNote = composeText(sysText, toolsText, attachBlocks, nextNote)
      attachState = { attachBlocks, attachNote: nextNote, protectedStart }
      over = Math.max(measure(text), mayAttachOverflow ? measure(textWithAttachmentNote) : 0) - stage4MaxBytes
    }
  } else if (historyWillNeedNote) {
    // History was dropped in Stage 2 without Stage 4 needing to truncate anything.
    // The note announcing the spill still has to fit, and this branch used to give up
    // when it did not — the history then vanished from the frame and from every
    // attachment. Take the room from what the frame still carries instead:
    //   1. shrink the inline system text (its complete copy becomes
    //      system_instructions.txt in Stage 5), then
    //   2. move the oldest pre-protected block into the attachment itself.
    // Nothing is mutated until a frame that fits has actually been assembled.
    let trialSys = sysText
    let trialBlocks = blocks.slice()
    let trialProtected = protectedStart
    let spill = historyToSpill.slice()
    let doc = buildOverflowDocument(spill, [], overflowAttachmentName)
    let note = buildOverflowAttachmentNote(overflowAttachmentName, doc.droppedRange)
    let candidate = composeText(trialSys, toolsText, trialBlocks, note)
    const noteLimit = maxFrameBytes - 500
    let guard = 0
    let shrunkSys = false
    while (measure(candidate) > noteLimit && guard++ < 64) {
      const over = measure(candidate) - noteLimit
      if (trialSys.length > SYSPOINTER_MIN_ANCHOR_CHARS) {
        const keep = Math.max(SYSPOINTER_MIN_ANCHOR_CHARS, trialSys.length - Math.max(256, Math.ceil(over * 1.4)))
        const next = truncateMiddle(origSysText, keep)
        if (next !== trialSys) {
          trialSys = next
          shrunkSys = true
          candidate = composeText(trialSys, toolsText, trialBlocks, note)
          continue
        }
      }
      if (trialProtected <= 0) break
      // Step over the omitted-count marker rather than consuming it: it is tiny,
      // and it is the only inline notice left if this turn's upload fails and the
      // frame ships without the attachment note.
      const from = trialBlocks[0]?.role === "system_note" && trialProtected > 1 ? 1 : 0
      if (from >= trialProtected) break
      const moved = trialBlocks[from]
      trialBlocks = [...trialBlocks.slice(0, from), ...trialBlocks.slice(from + 1)]
      trialProtected--
      if (moved && moved.role !== "system_note") spill = [...spill, moved]
      doc = buildOverflowDocument(spill, [], overflowAttachmentName)
      note = buildOverflowAttachmentNote(overflowAttachmentName, doc.droppedRange)
      candidate = composeText(trialSys, toolsText, trialBlocks, note)
    }
    if (measure(candidate) <= noteLimit) {
      if (shrunkSys) {
        // The inline text now differs from the caller's original, so Stage 5 must
        // still publish the full copy and the delivery bookkeeping must know it.
        sysText = trialSys
        sysTrimmed = sysText !== origSysText
      }
      blocks = trialBlocks
      protectedStart = trialProtected
      // Recompute the plain variant last, so its byte figure reflects the blocks
      // this branch actually settled on.
      text = composeText(sysText, toolsText, blocks)
      curBytes = measure(text)
      overflowUserText = doc.text
      textWithAttachmentNote = candidate
      // Stage 5 rebuilds the note variant from this state; keeping it (instead of
      // regenerating a note with droppedRange: null) is what preserves the exact
      // line range of the spilled history.
      attachState = { attachBlocks: trialBlocks.slice(), attachNote: note, protectedStart: trialProtected }
    }
  }

  // Stage 5: system preservation. Stage 3 can only trim the caller's system text
  // (a long CJK turn consumes the whole frame on its own), and the trim is lossy:
  // truncateMiddle drops the middle. Whenever it was trimmed, the full original
  // text becomes its own attachment. This runs even when the protected turn was
  // short enough that Stage 4 never engaged — the large-system-plus-one-line-user
  // case, where the trimmed middle would otherwise vanish with no attachment.
  let sysAttachmentText = ""
  let sysAttachmentRange = null
  if ((!isResume || resendSystem) && origSysText && sysTrimmed && mayAttachOverflow) {
    sysAttachmentText = origSysText
    sysAttachmentRange = { name: SYSTEM_ATTACHMENT_NAME, startLine: 1, endLine: origSysText.split("\n").length }
    sysPointer = sysAttachmentRange
    // Rebuild BOTH variants from the same preserved state, so the pointer lands in
    // the plain frame and in the attachment variant without discarding the
    // per-message line refs and the droppedRange note Stage 4 produced.
    const rebuild = () => {
      text = composeText(sysText, toolsText, blocks)
      curBytes = measure(text)
      if (attachState) {
        textWithAttachmentNote = composeText(sysText, toolsText, attachState.attachBlocks, attachState.attachNote)
      } else if (textWithAttachmentNote) {
        textWithAttachmentNote = composeText(sysText, toolsText, blocks, buildOverflowAttachmentNote(overflowAttachmentName, null))
      }
    }
    rebuild()
    // The pointer must always reach the frame: an uploaded system_instructions.txt
    // the model is never told about is a wasted upload and a silent persona loss.
    // Shrink the inline system text (down to the anchor) until the pointer fits.
    let guard = 0
    while (
      guard++ < 32 &&
      (curBytes > maxFrameBytes || (textWithAttachmentNote && measure(textWithAttachmentNote) > maxFrameBytes))
    ) {
      if (sysText.length <= SYSPOINTER_MIN_ANCHOR_CHARS) break
      const keep = Math.max(SYSPOINTER_MIN_ANCHOR_CHARS, Math.floor((sysText.length * maxFrameBytes * 0.97) / Math.max(1, curBytes)))
      const next = keep < sysText.length ? truncateMiddle(origSysText, keep) : ""
      if (!next || next === sysText) break
      sysText = next
      rebuild()
    }
  }
  // Delivered means: no system text at all, or it rode inline whole on a fresh
  // thread, or the complete text is in an attachment the turn uploads.
  // `systemDeliveredInline` is the baseline a later resume may diff against: a
  // diff only makes sense when the server thread itself holds the full text.
  // An attachment delivery satisfies the model for this turn but leaves the
  // thread without a baseline, so the next resume re-sends it.
  const systemDeliveredInline = !origSysText || (!sysTrimmed && (!isResume || resendSystem))
  const systemDelivered = systemDeliveredInline || Boolean(sysAttachmentText)

  return {
    modelId,
    hasTools,
    isResume,
    lang,
    toolsSig,
    toolMap,
    rawSystemText,
    // Whether THIS turn actually put the caller's system text in front of the
    // model (inline whole, or complete in an attachment it uploads). The thread
    // store keeps this per thread so a later resume can re-send what was missed.
    systemDelivered,
    systemDeliveredInline,
    sysTextInFrame: sysText,
    // Server-visible size of the user_message about to be sent, in payload JSON bytes.
    payloadBytes: measure(text),
    convoHashes,
    imageUrls: collectImageUrls(outboundMsgs),
    overflowUserText,
    sysAttachmentText,
    sysAttachmentRange,
    overflowAttachmentName,
    textWithAttachmentNote,
    stream: body?.stream === true,
    outpostId,
    workingSessionId,
    threadId,
    messageId,
    sessionConnectPayload: {
      surfaceType,
      ...(opts.outpostId ? { outpostId: opts.outpostId } : {}),
      ...(isResume ? { session_id: threadId, focus_only: true, incognito: true } : {}),
      appVersion: APP_VERSION,
      acp_first_text: 1,
      user_notification: 1,
      subtask_progress: 1,
      external_agent_sign_in: 1,
      recents_promotion: 1,
    },
    userMessagePayload: buildUserPayload(text),
  }
}

const TAG_PAIRS = [
  ["<tool_call>", "</tool_call>"],
  ["<function_calls>", "</function_calls>"],
]

function longestSuffixPrefix(haystack, needle) {
  const max = Math.min(haystack.length, needle.length - 1)
  for (let len = max; len > 0; len--) {
    if (haystack.endsWith(needle.slice(0, len))) return len
  }
  return 0
}

function maxOpenTagHold(haystack) {
  let hold = 0
  for (const [open] of TAG_PAIRS) {
    const h = longestSuffixPrefix(haystack, open)
    if (h > hold) hold = h
  }
  return hold
}

function buildParamTypeMap(tools) {
  const map = new Map()
  if (!Array.isArray(tools)) return map
  for (const t of tools) {
    const fn = t?.function ?? t
    if (!fn?.name) continue
    const propTypes = new Map()
    const props = fn.parameters?.properties
    if (props && typeof props === "object") {
      for (const [k, v] of Object.entries(props)) {
        if (v && typeof v === "object" && v.type) propTypes.set(k, v.type)
      }
    }
    map.set(fn.name, propTypes)
  }
  return map
}

function coerceXmlParamValue(raw, expectedType) {
  let v = raw
  if (v.startsWith("\r\n")) v = v.slice(2)
  else if (v.startsWith("\n")) v = v.slice(1)
  if (v.endsWith("\r\n")) v = v.slice(0, -2)
  else if (v.endsWith("\n")) v = v.slice(0, -1)

  const wantsString = expectedType === "string" || (Array.isArray(expectedType) && expectedType.includes("string") && expectedType.length === 1)
  if (wantsString) return v

  const t = v.trim()
  const wantsJson =
    expectedType === "object" ||
    expectedType === "array" ||
    expectedType === "number" ||
    expectedType === "integer" ||
    expectedType === "boolean" ||
    Array.isArray(expectedType)
  if (wantsJson) {
    try {
      return JSON.parse(t)
    } catch {}
    return v
  }
  if ((t.startsWith("{") && t.endsWith("}")) || (t.startsWith("[") && t.endsWith("]"))) {
    try {
      return JSON.parse(t)
    } catch {}
  }
  return v
}

function parseInvokeBlocks(inner, paramTypeMap) {
  const out = []
  const invokeRe = /<invoke\b[^>]*\bname\s*=\s*(?:"([^"]+)"|'([^']+)')[^>]*>([\s\S]*?)<\/invoke>/gi
  let m
  while ((m = invokeRe.exec(inner))) {
    const name = (m[1] ?? m[2] ?? "").trim()
    if (!name) continue
    const body = m[3] || ""
    const typeMap = paramTypeMap.get(name)
    const args = {}
    const paramRe = /<parameter\b[^>]*\bname\s*=\s*(?:"([^"]+)"|'([^']+)')[^>]*>([\s\S]*?)<\/parameter>/gi
    let pm
    while ((pm = paramRe.exec(body))) {
      const key = (pm[1] ?? pm[2] ?? "").trim()
      if (!key) continue
      args[key] = coerceXmlParamValue(pm[3] ?? "", typeMap?.get(key))
    }
    out.push({ name, arguments: JSON.stringify(args) })
  }
  return out
}

function parseToolCallJson(rawJson) {
  let trimmed = rawJson.trim()
  const fence = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i)
  if (fence) trimmed = fence[1].trim()
  try {
    const parsed = JSON.parse(trimmed)
    if (parsed && typeof parsed.name === "string" && parsed.name) {
      const args = parsed.arguments === undefined ? {} : parsed.arguments
      return {
        name: parsed.name,
        arguments: typeof args === "string" ? args : JSON.stringify(args ?? {}),
      }
    }
  } catch {}
  return null
}

function parseToolBlock(openTag, inner, paramTypeMap) {
  if (openTag === "<tool_call>") {
    const jsonCall = parseToolCallJson(inner)
    if (jsonCall) return [jsonCall]
  }
  const xmlCalls = parseInvokeBlocks(inner, paramTypeMap)
  return xmlCalls.length ? xmlCalls : null
}

/**
 * Stateful splitter for `<tool_call>...</tool_call>` and `<function_calls><invoke>...</function_calls>`
 * blocks inside streamed text chunks. Ignores tags inside Markdown fenced code blocks or inline backticks,
 * validates tool names against declared tools when provided, and drops hallucinated `<tool_result` blocks
 * emitted after a tool call.
 */
export function createToolCallSplitter(enabled, tools = []) {
  const paramTypeMap = buildParamTypeMap(tools)
  let buf = ""
  let activePair = null
  let toolIndex = 0
  const calls = []

  // Markdown code-span / code-fence tracking outside active tool blocks
  let inFence = false
  let fenceLen = 0
  let inlineTicks = 0
  let atLineStart = true
  let prevCharWasNewline = false

  // Post-tool-call hallucinated `<tool_result` guard
  let afterToolCall = false
  let discardHallucinated = false

  const HALLUCINATED_TAG = "<tool_result"

  const advancePlainChar = (ch) => {
    if (ch === "\n") {
      if (prevCharWasNewline && inlineTicks > 0) inlineTicks = 0
      prevCharWasNewline = true
      atLineStart = true
    } else if (ch === "\r") {
      // keep newline state
    } else {
      prevCharWasNewline = false
      if (ch !== " " && ch !== "\t") atLineStart = false
    }
  }

  const processChunk = (delta, flushAll = false) => {
    const textPieces = []
    const toolCalls = []
    if (!enabled) {
      if (delta) textPieces.push(delta)
      return { text: textPieces.join(""), toolCalls }
    }
    if (discardHallucinated) {
      return { text: "", toolCalls }
    }
    buf += delta

    while (buf.length > 0) {
      if (discardHallucinated) {
        buf = ""
        break
      }

      if (activePair) {
        const [openTag, closeTag] = activePair
        const idx = buf.indexOf(closeTag)
        if (idx >= 0) {
          const inner = buf.slice(0, idx)
          buf = buf.slice(idx + closeTag.length)
          activePair = null
          const parsedList = parseToolBlock(openTag, inner, paramTypeMap)
          if (parsedList) {
            const validCalls = paramTypeMap.size ? parsedList.filter((p) => paramTypeMap.has(p.name)) : parsedList
            if (validCalls.length > 0) {
              for (const parsed of validCalls) {
                const call = {
                  index: toolIndex++,
                  id: `call_${randomUUID().replace(/-/g, "").slice(0, 24)}`,
                  type: "function",
                  function: parsed,
                }
                calls.push(call)
                toolCalls.push(call)
              }
              afterToolCall = true
              atLineStart = false
              prevCharWasNewline = false
            } else {
              textPieces.push(openTag + inner + closeTag)
              atLineStart = false
              prevCharWasNewline = false
            }
          } else {
            textPieces.push(openTag + inner + closeTag)
            atLineStart = false
            prevCharWasNewline = false
          }
          continue
        }
        if (flushAll) {
          textPieces.push(openTag + buf)
          buf = ""
          activePair = null
        }
        break
      }

      // If we just emitted a tool call and are outside code blocks, check whether the model
      // starts fabricating a `<tool_result...>` block right after optional whitespace.
      if (afterToolCall && !inFence && inlineTicks === 0) {
        let wsEnd = 0
        while (wsEnd < buf.length && /\s/.test(buf[wsEnd])) wsEnd++
        if (wsEnd === buf.length) {
          if (flushAll) {
            for (let k = 0; k < buf.length; k++) advancePlainChar(buf[k])
            textPieces.push(buf)
            buf = ""
            afterToolCall = false
          }
          break
        }
        const rest = buf.slice(wsEnd)
        const lowerRest = rest.toLowerCase()
        if (lowerRest.startsWith(HALLUCINATED_TAG)) {
          const nextCh = rest[HALLUCINATED_TAG.length]
          if (nextCh === undefined && !flushAll) break
          if (nextCh === undefined || /[\s>/]/.test(nextCh)) {
            discardHallucinated = true
            buf = ""
            break
          }
        } else if (!flushAll && HALLUCINATED_TAG.startsWith(lowerRest)) {
          // Partial `<tool_result` prefix right after tool call — hold until more arrives
          break
        }
        // Check if another real tool call immediately follows whitespace; otherwise normal text resumes
        const isNextToolOpen = TAG_PAIRS.some(([open]) => rest.startsWith(open))
        const isPartialToolOpen = !flushAll && TAG_PAIRS.some(([open]) => open.startsWith(rest))
        if (isPartialToolOpen && !isNextToolOpen) break
        if (!isNextToolOpen) afterToolCall = false
      }

      // Scan outside activePair, tracking Markdown fences (```) and inline backticks (`)
      let i = 0
      let emittedEnd = 0
      let needMore = false

      while (i < buf.length) {
        const ch = buf[i]

        if (ch === "`") {
          let end = i + 1
          while (end < buf.length && buf[end] === "`") end++
          if (end === buf.length && !flushAll) {
            // Trailing backticks may continue in the next chunk
            needMore = true
            break
          }
          const count = end - i
          if (inFence) {
            if (atLineStart && count >= fenceLen) {
              inFence = false
              fenceLen = 0
            }
          } else if (inlineTicks > 0) {
            if (count === inlineTicks) inlineTicks = 0
          } else if (atLineStart && count >= 3) {
            inFence = true
            fenceLen = count
          } else {
            inlineTicks = count
          }
          prevCharWasNewline = false
          atLineStart = false
          i = end
          emittedEnd = end
          continue
        }

        if (!inFence && inlineTicks === 0 && ch === "<") {
          let matchedPair = null
          for (const pair of TAG_PAIRS) {
            if (buf.startsWith(pair[0], i)) {
              matchedPair = pair
              break
            }
          }
          if (matchedPair) {
            if (i > 0) textPieces.push(buf.slice(0, i))
            buf = buf.slice(i + matchedPair[0].length)
            activePair = matchedPair
            emittedEnd = -1
            break
          }
          if (!flushAll) {
            const tail = buf.slice(i)
            const partial = TAG_PAIRS.some(([open]) => open.startsWith(tail))
            if (partial) {
              needMore = true
              break
            }
          }
        }

        advancePlainChar(ch)
        i++
        emittedEnd = i
      }

      if (emittedEnd === -1) {
        // Entered activePair — loop again to look for closeTag
        continue
      }
      if (emittedEnd > 0) {
        textPieces.push(buf.slice(0, emittedEnd))
        buf = buf.slice(emittedEnd)
      }
      if (needMore || buf.length === 0) break
    }

    return { text: textPieces.join(""), toolCalls }
  }

  return {
    push(chunk) {
      return processChunk(chunk, false)
    },
    flush() {
      return processChunk("", true)
    },
    get calls() {
      return calls
    },
  }
}

function isAuthErrorText(msg) {
  return /\b(401|403|unauthorized|unauthenticated|forbidden|invalid.?token|expired.?token|jwt)\b/i.test(String(msg || ""))
}

// Error text that can only mean the referenced draft agent is gone, not that the
// transport or the model failed. A websocket reset, an idle timeout or a generic
// upstream error matches none of these, so it must not cost us a live agent id.
const AGENT_DEAD_RE =
  /\b(agent|draft)\b[^\n]{0,40}\b(not[ _-]?found|does not exist|no longer exists|deleted|removed|unknown|invalid)\b|\b(invalid|unknown|missing)[ _-]?(chat[ _-]?)?agent[ _-]?id\b/i

function agentFailureIsAboutAgent(err) {
  const text = [err?.message, err?.code, err?.errorMessage, err?.cause?.message]
    .map((v) => String(v ?? ""))
    .filter(Boolean)
    .join(" ")
  if (!text.trim()) return false
  if (isAuthErrorText(text)) return false
  return AGENT_DEAD_RE.test(text)
}

const STRIP_REASONING_ENV = "MAGPIE_QUICK_STRIP_REASONING_CONTENT"

/**
 * Whether to withhold `reasoning_content` from the client.
 * Layered models (a Quick planner feeding Haiku / Flash / local executors) get dragged into English by a
 * long upstream thinking stream, so the bridge can keep the reasoning server-side and emit only the answer.
 * Enabled by `opts.stripReasoning` (boolean) or the `MAGPIE_QUICK_STRIP_REASONING_CONTENT` env override.
 */
export function stripReasoningEnabled(opts = {}) {
  if (typeof opts.stripReasoning === "boolean") return opts.stripReasoning
  const raw = String(process.env[STRIP_REASONING_ENV] ?? "").trim().toLowerCase()
  return raw === "1" || raw === "true" || raw === "yes" || raw === "on"
}

/**
 * Execute an SNCS chat turn over WebSocket (`AIDP` or `Gateway`), returning either a streaming SSE Response
 * or a completed OpenAI JSON Response. Reuses cached `threadId` incrementally when the prefix matches,
 * transparently falling back to a fresh `threadId` if a resumed thread fails before streaming begins.
 */
function waitForInflight(pending, signal) {
  if (!signal) return pending
  if (signal.aborted) {
    const err = new Error("Request aborted")
    err.status = 499
    return Promise.reject(err)
  }
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      const err = new Error("Request aborted")
      err.status = 499
      reject(err)
    }
    signal.addEventListener("abort", onAbort, { once: true })
    pending.then(
      () => {
        signal.removeEventListener("abort", onAbort)
        resolve()
      },
      () => {
        signal.removeEventListener("abort", onAbort)
        resolve()
      },
    )
  })
}

export async function executeSncsChat(a, openaiBody, opts = {}) {
  const WS = opts.WebSocket ?? globalThis.WebSocket
  if (typeof WS !== "function") {
    throw new Error("Global WebSocket is unavailable in this runtime (requires Node 22+ / Electron)")
  }
  const threadStore = opts.threadStore === undefined ? DEFAULT_THREAD_STORE : opts.threadStore
  const stateful = opts.stateful !== false && !opts.disableResume && !opts.threadId && !!threadStore
  const accountKey = accountKeyOf(a)

  // Resolve (once per account) the neutral draft agent that carries the turn's
  // instructions into Quick's agent layer. Any failure yields "" and the turn
  // continues on the plain inline <instructions> path.
  const agentStore = opts.agentStore === undefined ? DEFAULT_AGENT_STORE : opts.agentStore
  let chatAgentId = String(opts.chatAgentId ?? "").trim()
  if (!chatAgentId && agentModeEnabled(opts)) {
    chatAgentId = await ensureCarrierAgent(a, { ...opts, accountKey })
  }
  const turnOpts = chatAgentId ? { ...opts, chatAgentId } : opts

  let resumeState = null
  let releaseInflight = null

  if (stateful) {
    const rawMessages = Array.isArray(openaiBody?.messages) ? openaiBody.messages : []
    const sysParts = []
    const convo = []
    for (const m of rawMessages) {
      if (!m || typeof m !== "object") continue
      const r = String(m.role || "user").toLowerCase()
      if (r === "system" || r === "developer") {
        const t = textFromContent(m.content).trim()
        if (t) sysParts.push(t)
      } else {
        convo.push(m)
      }
    }
    const tools = Array.isArray(openaiBody?.tools) ? openaiBody.tools : []
    const toolChoice = openaiBody?.tool_choice
    const hasTools = tools.length > 0 && toolChoice !== "none"
    const toolsSig = toolsDigest(hasTools ? tools : [], toolChoice)
    const sysJoined = sysParts.join("\n\n")
    const sessionId = opts.sessionId || ""

    // Tool-less sidecar requests (e.g. Zed title generation / summarization) sharing a tool-bearing
    // main conversation MUST stay completely stateless so they neither pollute the server thread
    // nor overwrite the main thread's cache entries.
    if (!hasTools && threadStore.hasToolThread?.(accountKey, sessionId, convo)) {
      return await runSncsTurn(a, openaiBody, {
        ...turnOpts,
        stateful: false,
        resumeState: null,
        threadStore: null,
        accountKey,
        WS,
      })
    }

    const doLookup = () => threadStore.lookup(accountKey, sessionId, toolsSig, sysJoined, convo, { hasTools })
    resumeState = doLookup()
    while (resumeState?.threadId) {
      const pending = threadStore.getInflight?.(resumeState.threadId)
      if (!pending) break
      await waitForInflight(pending, opts.signal)
      resumeState = doLookup()
    }
    if (resumeState?.threadId && threadStore.acquire) {
      releaseInflight = threadStore.acquire(resumeState.threadId)
    }
  }

  try {
    return await runSncsTurn(a, openaiBody, {
      ...turnOpts,
      resumeState,
      accountKey,
      threadStore,
      WS,
      onTurnDone: releaseInflight,
    })
  } catch (err) {
    const canRetryFresh = Boolean(resumeState && err?.status !== 401 && !opts.signal?.aborted)
    if (canRetryFresh) {
      threadStore.invalidate(accountKey, opts.sessionId || "", resumeState.threadId)
    }
    releaseInflight?.()

    // Only a failure that actually names the draft agent proves the cached id is
    // dead (deleted, quota-cleaned, created under a previous sign-in). A dropped
    // websocket, an idle timeout or a generic upstream 5xx says nothing about it,
    // and treating those as "dead agent" used to discard a live id on the first
    // network blip — every later turn then silently fell back to the inline path.
    const authLike = err?.status === 401 || err?.status === 403
    const agentDied = chatAgentId && !authLike && agentFailureIsAboutAgent(err)
    if (agentDied) {
      try {
        agentStore?.forget?.(accountKey)
      } catch {}
    }
    // One retry, on a fresh thread, for the same conversation. The carrier id is
    // kept unless it was proven dead, so a dropped socket or an idle timeout can
    // no longer demote the whole session to the inline <instructions> path: the
    // conversation restarts, the surviving agent still answers it.
    if (!opts.signal?.aborted && !authLike && (canRetryFresh || chatAgentId)) {
      return await runSncsTurn(a, openaiBody, {
        ...turnOpts,
        ...(agentDied ? { chatAgentId: null } : {}),
        resumeState: null,
        accountKey,
        threadStore,
        WS,
      })
    }
    throw err
  }
}

/**
 * Remove the "the rest is in attached_files/..." pointer from a composed frame.
 * Used when that upload failed: pointing the model at a file that is not there
 * is worse than saying nothing, because it will try to read it and stall.
 */
/**
 * Remove the "the rest is in attached_files/..." overflow note. Used when that
 * upload failed: the note claims a file the turn does not carry.
 */
function stripOverflowNote(text, attachmentName) {
  const s = String(text || "")
  const needle = attachmentName || "prompt_context.txt"
  if (!s.includes(needle)) return s
  return s
    .split("\n\n")
    .filter((part) => !(part.trimStart().startsWith("[Note:") && part.includes(needle)))
    .join("\n\n")
}

function stripSystemPointer(text) {
  const s = String(text || "")
  const marker = "[System instructions are long:"
  if (!s.includes(marker)) return s
  return s
    .split("\n\n")
    .filter((part) => !part.trimStart().startsWith(marker))
    .join("\n\n")
}

function serializeWireAttachment(att) {
  return {
    type: att.type,
    name: att.name,
    size: att.size,
    mimeType: att.mimeType,
    path: att.path,
    attachmentId: att.attachmentId,
    contentSha256: att.contentSha256,
  }
}

async function prepareTurnAttachments(a, reqPlan, opts) {
  if (opts.disableAttachments) return []
  const hasImages = Array.isArray(reqPlan.imageUrls) && reqPlan.imageUrls.length > 0
  const hasOverflow = Boolean(reqPlan.overflowUserText) && opts.uploadLargeText !== false
  const hasSysAttachment = Boolean(reqPlan.sysAttachmentText) && opts.uploadLargeText !== false
  if (!hasImages && !hasOverflow && !hasSysAttachment) return []

  const uploader = opts.uploadAttachment ?? uploadQuickAttachment
  const fetchImpl = opts.fetch ?? fetch
  const tasks = []

  if (hasImages) {
    reqPlan.imageUrls.forEach((url, idx) => {
      tasks.push(
        (async () => {
          const img = await resolveImageBytes(url, fetchImpl, opts.signal)
          if (!img?.bytes?.byteLength) return null
          const name = `image_${String(idx + 1).padStart(3, "0")}.${img.ext}`
          const uploaded = await uploader(
            a,
            { name, mimeType: img.mimeType, bytes: img.bytes, threadId: reqPlan.threadId },
            opts,
          )
          return serializeWireAttachment(uploaded)
        })(),
      )
    })
  }

  if (hasOverflow) {
    const attachName = reqPlan.overflowAttachmentName || "prompt_context.txt"
    tasks.push(
      (async () => {
        const bytes = Buffer.from(reqPlan.overflowUserText, "utf8")
        const uploaded = await uploader(
          a,
          { name: attachName, mimeType: "text/plain", bytes, threadId: reqPlan.threadId },
          opts,
        )
        return serializeWireAttachment(uploaded)
      })(),
    )
  }

  // The caller's system text, when Stage 3 could only trim it. Uploaded under its
  // own name so the inline pointer can name a line range in it.
  if (hasSysAttachment) {
    const sysName = reqPlan.sysAttachmentRange?.name || "system_instructions.txt"
    tasks.push(
      (async () => {
        const bytes = Buffer.from(reqPlan.sysAttachmentText, "utf8")
        const uploaded = await uploader(
          a,
          { name: sysName, mimeType: "text/plain", bytes, threadId: reqPlan.threadId },
          opts,
        )
        return serializeWireAttachment(uploaded)
      })(),
    )
  }

  const settled = await Promise.allSettled(tasks)
  const attachments = []
  for (const r of settled) {
    if (r.status === "fulfilled" && r.value) {
      attachments.push(r.value)
    } else if (r.status === "rejected" && r.reason?.status === 401) {
      throw r.reason
    }
  }
  return attachments
}

async function runSncsTurn(a, openaiBody, ctx) {
  const { WS, accountKey, threadStore, onTurnDone, ...opts } = ctx
  const gateway = isGatewayProvider(a.providerType)
  const stripReasoning = stripReasoningEnabled(opts)
  const reqPlan = openaiToSncsRequest(openaiBody, opts)
  const wsHost = a.wsHost || hostFromUrl(a.tenantUrl)
  if (!wsHost) throw new Error("Missing Amazon Quick wsHost / tenantUrl")

  const attachments = await prepareTurnAttachments(a, reqPlan, opts)
  if (attachments.length) {
    reqPlan.userMessagePayload.attachments = attachments
    const expectedOverflowName = reqPlan.overflowAttachmentName || "prompt_context.txt"
    if (reqPlan.textWithAttachmentNote && attachments.some((att) => att?.name === expectedOverflowName)) {
      reqPlan.userMessagePayload.text = reqPlan.textWithAttachmentNote
    }
  }
  // The overflow attachment now carries an unbounded amount of history, so a
  // failed upload is a realistic outcome (a large file can outrun the upload
  // timeout). When it does not arrive, the frame must say so: history already
  // shifted out of the frame is gone, and a bare "[Earlier N ... omitted]" marker
  // reads as if it were merely trimmed. Name the failure so the model asks for
  // the missing turn instead of pretending it saw it.
  const overflowName = reqPlan.overflowAttachmentName || "prompt_context.txt"
  const overflowMissing =
    Boolean(reqPlan.overflowUserText) &&
    opts.uploadLargeText !== false &&
    !opts.disableAttachments &&
    !attachments.some((att) => att?.name === overflowName)
  if (overflowMissing) {
    const notice =
      `[Earlier conversation content could not be attached this turn (${overflowName} upload failed), so it is unavailable. If it matters, ask the user to repeat or resend the relevant part.]`
    const withNotice = (textVal) => (String(textVal || "").trim() ? `${textVal}\n\n${notice}` : notice)
    reqPlan.userMessagePayload.text = withNotice(reqPlan.userMessagePayload.text)
    if (reqPlan.textWithAttachmentNote) {
      reqPlan.textWithAttachmentNote = stripOverflowNote(withNotice(reqPlan.textWithAttachmentNote), overflowName)
    }
  }
  // A pointer to a file the turn failed to upload would be worse than no pointer:
  // drop the claim so the model is not told to read something that is not there.
  const sysName = reqPlan.sysAttachmentRange?.name
  if (sysName && !attachments.some((att) => att?.name === sysName)) {
    reqPlan.userMessagePayload.text = stripSystemPointer(reqPlan.userMessagePayload.text)
    if (reqPlan.textWithAttachmentNote) {
      reqPlan.textWithAttachmentNote = stripSystemPointer(reqPlan.textWithAttachmentNote)
    }
    // The model keeps only the anchor now, so the thread has no full text and
    // the next resume must send it again.
    reqPlan.systemDelivered = false
    reqPlan.systemDeliveredInline = false
  }

  const surface = surfaceTypeForPlatform(opts.platform)
  const encodeOutbound = gateway ? buildGatewayInputFrame : buildAidpInputFrame
  const decodeInbound = gateway ? decodeGatewayFrame : decodeAidpFrame

  const wsUrl = gateway
    ? `wss://${wsHost}/ws/service/SpaceNeedleConversationService/operation/ChatBidirectionalStream`
    : `wss://${wsHost}/streaming/quickwork-chat-bidirectional-stream` +
      `?surfaceType=${encodeURIComponent(surface)}` +
      `&token=${encodeURIComponent(a.access)}` +
      `&outpostId=${encodeURIComponent(reqPlan.outpostId)}` +
      `&workingSessionId=${encodeURIComponent(reqPlan.workingSessionId)}` +
      `&channel=prod`

  const wsInit = gateway
    ? {
        headers: {
          Authorization: `Bearer ${a.access}`,
          "X-Surface-Type": surface,
          "X-Outpost-Id": reqPlan.outpostId,
          "X-Agent-Service-Type": "UNIFIED_QUICK",
          "X-Working-Session-Id": reqPlan.workingSessionId,
        },
      }
    : undefined

  const completionId = `chatcmpl-${randomUUID().replace(/-/g, "").slice(0, 24)}`
  const created = Math.floor(Date.now() / 1000)

  return new Promise((resolve, reject) => {
    const ws = wsInit ? new WS(wsUrl, wsInit) : new WS(wsUrl)
    ws.binaryType = "arraybuffer"

    let handshakeReady = false
    let finished = false
    let pingTimer = null
    let connectTimer = null
    let idleTimer = null
    let rxBuffer = new Uint8Array(0)
    let accumulatedContent = ""
    let accumulatedReasoning = ""
    let sawReasoning = false
    let sawTextDelta = false
    let usageInfo = null
    let sseController = null
    let settledInitial = false
    const chunkBuffers = new Map()
    const splitter = createToolCallSplitter(reqPlan.hasTools, openaiBody?.tools)
    const sseEnc = new TextEncoder()
    const connectTimeoutMs = Number(opts.connectTimeoutMs) > 0 ? Number(opts.connectTimeoutMs) : 15_000
    const idleTimeoutMs = Number(opts.idleTimeoutMs) > 0 ? Number(opts.idleTimeoutMs) : 90_000

    const cleanup = () => {
      if (pingTimer) {
        clearInterval(pingTimer)
        pingTimer = null
      }
      if (connectTimer) {
        clearTimeout(connectTimer)
        connectTimer = null
      }
      if (idleTimer) {
        clearTimeout(idleTimer)
        idleTimer = null
      }
      chunkBuffers.clear()
      if (opts.signal && onAbort) {
        opts.signal.removeEventListener("abort", onAbort)
      }
      try {
        if (ws.readyState === 0 || ws.readyState === 1) ws.close(1000, "done")
      } catch {}
    }

    const resetIdleTimer = () => {
      if (idleTimer) clearTimeout(idleTimer)
      if (finished) return
      idleTimer = setTimeout(() => {
        failEarly("Amazon Quick WebSocket response timed out", 504)
      }, idleTimeoutMs)
      idleTimer.unref?.()
    }

    connectTimer = setTimeout(() => {
      failEarly("Amazon Quick WebSocket connection timed out", 504)
    }, connectTimeoutMs)
    connectTimer.unref?.()

    const emitSse = (obj) => {
      if (!sseController) return
      sseController.enqueue(sseEnc.encode(`data: ${JSON.stringify(obj)}\n\n`))
    }

    const emitDeltaChunk = (delta, finishReason = null) => {
      emitSse({
        id: completionId,
        object: "chat.completion.chunk",
        created,
        model: reqPlan.modelId,
        choices: [{ index: 0, delta, finish_reason: finishReason }],
      })
    }

    const failEarly = (message, status = 502) => {
      if (finished) return
      finished = true
      cleanup()
      const err = new Error(message)
      err.status = isAuthErrorText(message) ? 401 : status
      err.invalidToken = err.status === 401
      if (!settledInitial) {
        settledInitial = true
        reject(err)
      } else {
        // Stream already began (server thread holds a half-finished turn): invalidate cached thread
        // before releasing the inflight lock so the next request starts clean instead of duplicating turns.
        if (threadStore && opts.stateful !== false) {
          threadStore.invalidate(accountKey, opts.sessionId || "", reqPlan.threadId)
        }
        if (sseController) {
          try {
            sseController.enqueue(sseEnc.encode(`data: ${JSON.stringify({ error: { message, type: "upstream_error" } })}\n\n`))
            sseController.enqueue(sseEnc.encode("data: [DONE]\n\n"))
            sseController.close()
          } catch {}
        }
        onTurnDone?.()
      }
    }

    const commitThreadState = () => {
      if (!threadStore || opts.stateful === false || !reqPlan.convoHashes?.length) return
      const assistantContent = splitter.calls.length && !accumulatedContent ? null : accumulatedContent
      const expectedAssistantSig = messageDigest({
        role: "assistant",
        content: assistantContent,
        tool_calls: splitter.calls,
      })
      const expectedAssistantCallsSig = splitter.calls.length
        ? canonicalJson(normalizeToolCallsForDigest(splitter.calls))
        : ""
      const expectedAssistantTextSig = !splitter.calls.length
        ? sha256Hex(textFromContent(assistantContent).trim())
        : ""
      const toolCallIds = splitter.calls.map((c) => c?.id).filter(Boolean)
      const commitState = {
        threadId: reqPlan.threadId,
        workingSessionId: reqPlan.workingSessionId,
        hasTools: Boolean(reqPlan.hasTools),
        lang: reqPlan.lang || "",
        toolsSig: reqPlan.toolsSig,
        toolMap: reqPlan.toolMap,
        // The diff baseline must be what the model actually saw, not what the
        // caller wrote: recording rawSystemText after Stage 3 trimmed it made
        // every later resume compute an empty diff and never re-send the
        // instructions. When the full text only reached an attachment, the
        // baseline stays empty so the next turn re-sends it inline.
        sysText: reqPlan.systemDeliveredInline ? reqPlan.rawSystemText : "",
        msgHashes: reqPlan.convoHashes,
        expectedAssistantSig,
        expectedAssistantCallsSig,
        expectedAssistantTextSig,
        toolCallIds,
      }
      threadStore.commit(accountKey, opts.sessionId || "", commitState)
      // Only a successful turn earns a durable binding: persisting on failure would
      // pin later turns to a thread the server never really advanced.
      threadStore.persist?.({ ...commitState, accountKey, sessionId: opts.sessionId || "" })
    }

    const completeSuccess = () => {
      if (finished) return
      finished = true
      const flushed = splitter.flush()
      if (flushed.text) accumulatedContent += flushed.text
      const finishReason = splitter.calls.length > 0 ? "tool_calls" : "stop"
      commitThreadState()
      const openAiUsage = usageInfo
        ? {
            prompt_tokens: Number(usageInfo.input_tokens) || 0,
            completion_tokens: Number(usageInfo.output_tokens) || 0,
            total_tokens:
              Number(usageInfo.total_tokens) ||
              (Number(usageInfo.input_tokens) || 0) + (Number(usageInfo.output_tokens) || 0),
            ...(Number(usageInfo.cache_read_tokens) > 0
              ? { prompt_tokens_details: { cached_tokens: Number(usageInfo.cache_read_tokens) } }
              : {}),
          }
        : undefined

      if (reqPlan.stream) {
        if (!settledInitial) openSseStream()
        if (flushed.text) emitDeltaChunk({ content: flushed.text })
        for (const tc of flushed.toolCalls) emitDeltaChunk({ tool_calls: [tc] })
        emitSse({
          id: completionId,
          object: "chat.completion.chunk",
          created,
          model: reqPlan.modelId,
          choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
          ...(openAiUsage ? { usage: openAiUsage } : {}),
        })
        try {
          sseController.enqueue(sseEnc.encode("data: [DONE]\n\n"))
          sseController.close()
        } catch {}
      } else if (!settledInitial) {
        settledInitial = true
        const msg = {
          role: "assistant",
          content: splitter.calls.length && !accumulatedContent ? null : accumulatedContent,
          ...(accumulatedReasoning ? { reasoning_content: accumulatedReasoning } : {}),
          ...(splitter.calls.length
            ? {
                tool_calls: splitter.calls.map(({ id, type, function: fn }) => ({
                  id,
                  type,
                  function: fn,
                })),
              }
            : {}),
        }
        resolve(
          new Response(
            JSON.stringify({
              id: completionId,
              object: "chat.completion",
              created,
              model: reqPlan.modelId,
              choices: [{ index: 0, message: msg, finish_reason: finishReason }],
              ...(openAiUsage ? { usage: openAiUsage } : {}),
            }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
        )
      }
      cleanup()
      onTurnDone?.()
    }

    const openSseStream = () => {
      if (settledInitial) return
      settledInitial = true
      const stream = new ReadableStream({
        start(controller) {
          sseController = controller
          emitDeltaChunk({ role: "assistant", content: "" })
        },
        cancel() {
          if (!finished) {
            finished = true
            if (threadStore && opts.stateful !== false) {
              threadStore.invalidate(accountKey, opts.sessionId || "", reqPlan.threadId)
            }
          }
          cleanup()
          onTurnDone?.()
        },
      })
      resolve(
        new Response(stream, {
          status: 200,
          headers: {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            Connection: "keep-alive",
          },
        }),
      )
    }

    const onAbort = () => failEarly("Request aborted", 499)
    if (opts.signal) {
      if (opts.signal.aborted) {
        onAbort()
        return
      }
      opts.signal.addEventListener("abort", onAbort, { once: true })
    }

    const emitParsedText = (chunkText) => {
      if (!chunkText) return
      const out = splitter.push(chunkText)
      if (out.text) {
        accumulatedContent += out.text
        if (reqPlan.stream) emitDeltaChunk({ content: out.text })
      }
      if (reqPlan.stream) {
        for (const tc of out.toolCalls) emitDeltaChunk({ tool_calls: [tc] })
      }
    }

    const handleChunkedFrame = (ev) => {
      const p = ev.payload ?? ev
      const chunkId = typeof p?.chunk_id === "string" ? p.chunk_id : ""
      const idx = Number(p?.chunk_index)
      const total = Number(p?.total_chunks)
      const data = typeof p?.data === "string" ? p.data : ""
      if (!chunkId || !Number.isInteger(idx) || !Number.isInteger(total) || total <= 0 || total > 512 || idx < 0 || idx >= total) {
        return
      }
      let entry = chunkBuffers.get(chunkId)
      if (!entry || entry.total !== total) {
        entry = { chunks: new Array(total).fill(null), received: 0, total }
        chunkBuffers.set(chunkId, entry)
      }
      if (entry.chunks[idx] === null) entry.received++
      entry.chunks[idx] = data
      if (entry.received === total) {
        chunkBuffers.delete(chunkId)
        try {
          handleSncsEvent(JSON.parse(entry.chunks.join("")))
        } catch {}
      }
    }

    const handleSncsEvent = (ev) => {
      if (!ev || typeof ev !== "object") return
      const type = ev.type
      if (type === "_chunked_frame") {
        handleChunkedFrame(ev)
        return
      }
      if (type === "keepalive" || type === "pong") return
      if (type === "session_established" || type === "connected") {
        handshakeReady = true
        if (reqPlan.stream && !reqPlan.isResume && !settledInitial) openSseStream()
        return
      }
      if (
        type === "message_start" ||
        type === "thinking_delta" ||
        type === "text_delta" ||
        type === "tool_start"
      ) {
        handshakeReady = true
        if (reqPlan.stream && !settledInitial) openSseStream()
      }
      if (
        type === "tool_start" ||
        type === "tool_arguments" ||
        type === "tool_progress" ||
        type === "tool_result" ||
        type === "tool_error"
      ) {
        return
      }
      if (type === "thinking_delta") {
        const d = ev.payload?.delta ?? ev.payload?.text ?? ""
        if (d) {
          sawReasoning = true
          if (!stripReasoning) {
            accumulatedReasoning += d
            if (reqPlan.stream) emitDeltaChunk({ reasoning_content: d })
          }
        }
        return
      }
      if (type === "text_delta") {
        const d = ev.payload?.delta ?? ""
        if (d) {
          sawTextDelta = true
          emitParsedText(d)
        }
        return
      }
      if (type === "text_complete") {
        const full = ev.payload?.text
        if (typeof full === "string" && !sawTextDelta && full) {
          emitParsedText(full)
        }
        return
      }
      if (type === "turn_usage") {
        usageInfo = ev.payload ?? ev
        return
      }
      if (type === "guardrail_blocked") {
        const msg = ev.payload?.message || "Response blocked by Amazon Quick guardrail"
        failEarly(msg, 400)
        return
      }
      if (type === "error") {
        const msg = ev.payload?.message || ev.payload?.error || ev.message || "Amazon Quick WebSocket error"
        const rawStatus = Number(ev.payload?.status)
        // An oversized user_message is rejected with a bare `error` event whose payload is
        // empty — no message, no status. Measured ceiling: ~64 KB of payload JSON (see
        // MAX_USER_MESSAGE_PAYLOAD_BYTES). Report that as a request problem instead of a
        // generic upstream 502, so a caller can tell "your turn was too big" apart from
        // "the service broke".
        const emptyPayload = !ev.payload || Object.keys(ev.payload).length === 0
        const oversize = emptyPayload && reqPlan.payloadBytes > MAX_USER_MESSAGE_PAYLOAD_BYTES * 0.9
        const status =
          rawStatus || (oversize || /size limit|too large|too long|invalid/i.test(msg) ? 400 : 502)
        failEarly(
          oversize
            ? `Amazon Quick rejected this turn: its user_message payload was ${reqPlan.payloadBytes} bytes, above the service limit. Shorten the conversation or reduce attached context.`
            : msg,
          status,
        )
        return
      }
      if (type === "message_complete" || type === "stream_ended") {
        completeSuccess()
      }
    }

    ws.onopen = () => {
      if (finished) return
      if (connectTimer) {
        clearTimeout(connectTimer)
        connectTimer = null
      }
      resetIdleTimer()
      ws.send(encodeOutbound({ type: "session_connect", payload: reqPlan.sessionConnectPayload }))
      ws.send(encodeOutbound({ type: "user_message", payload: reqPlan.userMessagePayload }))
      pingTimer = setInterval(() => {
        try {
          if (ws.readyState === 1) {
            ws.send(encodeOutbound({ type: "ping", payload: { timestamp: Date.now() } }))
          }
        } catch {}
      }, 25000)
      pingTimer.unref?.()
    }

    ws.onmessage = (ev) => {
      if (finished) return
      resetIdleTimer()
      const data = ev.data
      if (typeof data === "string") {
        try {
          handleSncsEvent(JSON.parse(data))
        } catch {}
        return
      }
      const chunk = data instanceof Uint8Array ? data : new Uint8Array(data)
      if (!rxBuffer.length) rxBuffer = chunk
      else {
        const merged = new Uint8Array(rxBuffer.length + chunk.length)
        merged.set(rxBuffer, 0)
        merged.set(chunk, rxBuffer.length)
        rxBuffer = merged
      }
      const { messages, consumed } = esDrain(rxBuffer, 0)
      if (consumed > 0) rxBuffer = rxBuffer.subarray(consumed)
      for (const frame of messages) {
        const decoded = decodeInbound(frame.headers, frame.payload)
        if (!decoded) continue
        if (decoded.type === "error") {
          failEarly(decoded.message, isAuthErrorText(decoded.code + " " + decoded.message) ? 401 : 502)
          return
        }
        handleSncsEvent(decoded.event)
      }
    }

    ws.onerror = (ev) => {
      if (finished) return
      const msg = ev?.message || ev?.error?.message || "Amazon Quick WebSocket connection failed"
      failEarly(msg, handshakeReady ? 502 : 401)
    }

    ws.onclose = (ev) => {
      if (finished) return
      if (ev.code === 1000 && (accumulatedContent || sawReasoning || splitter.calls.length)) {
        completeSuccess()
        return
      }
      const reason = ev.reason || `WebSocket closed (${ev.code})`
      const likelyAuth = !handshakeReady || ev.code === 1008 || ev.code === 4401 || ev.code === 4403 || isAuthErrorText(reason)
      failEarly(reason, likelyAuth ? 401 : 502)
    }
  })
}
