/**
 * Native device tools: make Quick believe a real desktop device is attached, and
 * hand the model's device tool calls to the local IDE/CLI instead of returning
 * XML in the text stream.
 *
 * HOW THE CHAIN WORKS (all verified on a live account, 2026-11-27):
 *
 *   1. The client REGISTERS the device with the cell:
 *
 *        POST <tenantUrl>/integration/quick-work/qbs-event
 *        {"eventType":"putOutpost","eventBody":{"putOutpost":{
 *            outpostId, name, outpostType:"DESKTOP",
 *            capabilities:[{name, tools:[{name, description, inputSchema, isReadTool}]}]}}}
 *
 *      No signature, no attestation, no device certificate — just the account's
 *      own Bearer token. This is the whole "device registration" that
 *      qw_core/outposts/turn_scope.py gates on.
 *
 *   2. The turn must carry the SAME outpostId in BOTH the WS query string and
 *      the user_message PAYLOAD. The query string is what makes the cell show
 *      that id as ONLINE for the turn; the payload copy is what makes it
 *      consider the device for tool routing. The official renderer spreads
 *      ff() into the payload, which puts outpostId there.
 *
 *      Both are necessary but not sufficient: the account must also have room
 *      for a new device row — see "KEEP THE DEVICE COUNT SANE" below, which is
 *      the actual root cause found after a long detour.
 *
 *   3. Only then does the server emit tool_call frames, e.g.
 *        {"type":"tool_call","payload":{"tool_call_id","tool_name","arguments"}}
 *      These are the device-executed tools. tool_start/tool_result are the
 *      CLOUD sandbox and must not be confused with them.
 *
 *   4. The tool runs locally; the result goes back up the same socket:
 *        {"type":"tool_result","payload":{"tool_call_id","result","error"}}
 *      (60 KB cap; the official client truncates result.content/result.output.)
 *
 * MEASURED GOTCHAS — do not "fix" these without re-measuring:
 *
 *   - KEEP THE DEVICE COUNT SANE (root cause found 2026-11-28). A whole round
 *     of experiments concluded "a self-made id can never work, only the id the
 *     official client created does". That was WRONG, and the truth was dull:
 *     the test account had 325 outpost records left over from probing, and the
 *     cell stops routing device tools to new devices while that table is that
 *     full. After deleting 324 of them with
 *         {"eventType":"unregisterOutpost",
 *          "eventBody":{"unregisterOutpost":{"outpostId":"<id>"}}}
 *     a brand-new uuid scored 4/4, then 3/3 and 3/3, with no official process
 *     running at all. Raising the count back to 25 still worked 100%, so the
 *     limit is not "one device" - it is hundreds of zombie rows.
 *
 *     THIS is why stableDeviceId() below is a hard requirement, not a nicety:
 *     a per-turn random id would add a new row every turn, fill the account,
 *     and silently switch device tools off. Never make the id random.
 *
 *   - Consequences: nothing else gates this. The capability manifest size is
 *     irrelevant, the device name is irrelevant, the official custom headers
 *     are irrelevant, and registerOutpostTools / putDeviceState / createSpace
 *     / graph endpoint bindings change nothing. All of those were "confirmed"
 *     while the account was polluted and are recorded as retracted in
 *     quick-lab/docs/NATIVE-DEVICE-TOOLS.md section 0.2. Do not re-run them.
 *
 *   - NEVER use these as a success signal: putOutpost's returned "status"
 *     (always OFFLINE, even for a working device), the model-facing
 *     list_outposts tool (a current_turn_access view that returns {"outposts":[]}
 *     for a self-made id even while it is ONLINE and executing device tools),
 *     or tool_start/tool_progress/tool_result frames (those are the CLOUD
 *     sandbox). The only real signal is a "tool_call" frame.
 *
 *   - Advertising exactly the tools this client can really run is still right,
 *     but for honesty, not for gating: an unknown name just fails at execution.
 */

export const OUTPOST_TYPE_DESKTOP = "DESKTOP"

/** Result payloads above this are truncated, matching the official relay cap. */
export const MAX_TOOL_RESULT_BYTES = 60 * 1024

const TRUNCATION_NOTE = "\n\n[TRUNCATED — result too large for device relay]"

const tool = (name, description, properties = {}, required = [], isReadTool = true) => ({
  name,
  description,
  inputSchema: {
    type: "object",
    properties,
    ...(required.length ? { required } : {}),
  },
  isReadTool: !!isReadTool,
})

const str = (description) => ({ type: "string", description })

/**
 * The device capability manifest sent at registration.
 *
 * The names are the ones the model already looks for on a desktop device, so it
 * reaches for them without extra prompting. Keep this list in sync with what the
 * host client can actually execute — an advertised-but-unexecutable tool turns
 * into a failed tool_result, which reads to the model as "the device is broken".
 */
export function defaultCapabilities() {
  return [
    {
      name: "file_operations",
      tools: [
        tool("folder_list", "List the contents of a directory on the user's machine. If path is empty, lists the accessible folders.", { path: str("Directory path. Relative paths resolve against the workspace."), show_hidden: { type: "boolean", description: "Include dotfiles" } }),
        tool("file_matches", "Find paths matching a glob on the user's machine.", { path: str("Absolute path or glob") }, ["path"]),
        tool("ripgrep", "Search file contents with ripgrep on the user's machine.", { patterns: { type: "array", items: { type: "string" }, description: "Regex patterns" }, paths: { type: "array", items: { type: "string" }, description: "Files or directories to search" }, options: { type: "object", description: "Extra ripgrep flags" }, timeout: { type: "number", description: "Seconds" } }, ["patterns"]),
        tool("file_type", "Report whether a path is a file or directory, with size and timestamps.", { path: str("Path") }, ["path"]),
        tool("file_read", "Read a text file from the user's machine.", { path: str("Path"), offset: { type: "number", description: "First line (1-based)" }, limit: { type: "number", description: "How many lines" } }, ["path"]),
        tool("file_edit", "Edit a file by applying a unified diff.", { path: str("Path"), diff: str("Unified diff") }, ["path", "diff"], false),
        tool("file_write", "Write text to a file on the user's machine.", { path: str("Path"), content: str("Full file content") }, ["path", "content"], false),
        tool("file_move", "Move or rename a file or directory.", { source: str("Source"), destination: str("Destination") }, ["source", "destination"], false),
        tool("file_copy", "Copy a file or directory.", { source: str("Source"), destination: str("Destination") }, ["source", "destination"], false),
        tool("folder_create", "Create a directory.", { path: str("Directory path") }, ["path"], false),
        tool("file_delete", "Delete a file or directory.", { path: str("Path") }, ["path"], false),
      ],
    },
    {
      name: "code_execution",
      tools: [
        tool("terminal_on_device", "Run a shell command on the user's machine and return its output.", { command: str("Shell command"), cwd: str("Working directory"), timeout: { type: "number", description: "Seconds" } }, ["command"], false),
        tool("run_python_on_device", "Run Python on the user's machine and return its output.", { code: str("Python source"), timeout: { type: "number", description: "Seconds" } }, ["code"], false),
      ],
    },
  ]
}

function utf8Length(s) {
  return typeof TextEncoder === "function" ? new TextEncoder().encode(s).length : String(s).length * 3
}

/**
 * Fit a tool result under the relay cap the way the official client does: keep the
 * structure, truncate the one big string field, and say that it was cut — never
 * hand back a silently shortened payload.
 */
export function capToolResult(result, error = null, maxBytes = MAX_TOOL_RESULT_BYTES) {
  const payload = JSON.stringify({ type: "tool_result", payload: { tool_call_id: "", result, error } })
  if (utf8Length(payload) <= maxBytes) return { result, error, truncated: false }

  const originalBytes = utf8Length(payload)
  const budget = Math.floor(maxBytes * 0.83)
  const cut = (s) => {
    const text = typeof s === "string" ? s : JSON.stringify(s)
    let lo = 0
    let hi = text.length
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2)
      if (utf8Length(text.slice(0, mid)) <= budget) lo = mid
      else hi = mid - 1
    }
    return text.slice(0, lo) + TRUNCATION_NOTE
  }

  let capped
  if (result && typeof result === "object" && "content" in result) {
    capped = { ...result, content: cut(result.content) }
  } else if (result && typeof result === "object") {
    const key = ["output", "text", "stdout"].find((k) => typeof result[k] === "string")
    capped = key
      ? { ...result, [key]: cut(result[key]) }
      : { truncated: true, error: "Result too large (" + originalBytes + " bytes) for device relay" }
  } else if (typeof result === "string") {
    capped = cut(result)
  } else {
    capped = { truncated: true, error: "Result too large (" + originalBytes + " bytes) for device relay" }
  }
  return { result: capped, error, truncated: true, originalBytes }
}

/**
 * The official relay wraps each device call as
 *   arguments = { params: { name, arguments: {...}, _quickwork: {...} } }
 * while older/simpler emitters send the arguments object directly. Unwrap both so
 * a tool handler sees the plain arguments either way.
 */
export function unwrapToolArguments(args) {
  if (!args || typeof args !== "object") return {}
  const nested = args.params && args.params.arguments
  if (nested && typeof nested === "object" && !Array.isArray(nested)) return nested
  return args
}

/** The _quickwork block carries the offload session; useful for scratch dirs. */
export function toolCallSessionId(args) {
  const q = args && args.params && args.params._quickwork
  if (q && typeof q === "object" && typeof q.offload_session_id === "string") return q.offload_session_id
  if (args && typeof args === "object" && typeof args.thread_id === "string") return args.thread_id
  return ""
}

/**
 * Build the putOutpost envelope. Kept as its own function so it can be replayed
 * in tests without a network.
 */
export function buildPutOutpostEnvelope({ outpostId, name, outpostType = OUTPOST_TYPE_DESKTOP, capabilities = defaultCapabilities() }) {
  if (!outpostId) throw new Error("buildPutOutpostEnvelope: outpostId is required")
  return {
    eventType: "putOutpost",
    eventBody: {
      putOutpost: {
        outpostId,
        name: name || "magpie-device",
        outpostType,
        capabilities,
      },
    },
  }
}

/** Build the frame that answers a device tool call. */
export function buildToolResultEvent(toolCallId, result, error = null) {
  return { type: "tool_result", payload: { tool_call_id: toolCallId, result, error } }
}

/**
 * A device id that stays the same across turns and restarts for one account on
 * one machine. The cloud keys its registration on this id, so a fresh one per
 * turn would look like a brand-new device every time and leave the model looking
 * at a list of offline ghosts.
 */
export function stableDeviceId(seed) {
  const s = String(seed || "magpie")
  let h1 = 0x811c9dc5
  let h2 = 0xc2b2ae35
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i)
    h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0
    h2 = Math.imul(h2 + c, 0x85ebca6b) >>> 0
  }
  const hex = (n, w) => (n >>> 0).toString(16).padStart(w, "0").slice(-w)
  // Shape it like a UUIDv4 so it reads as a device id everywhere it is logged.
  return [
    hex(h1, 8),
    hex(h2 >>> 16, 4),
    "4" + hex(h2, 3),
    "a" + hex(h1 >>> 20, 3),
    hex(h1, 4) + hex(h2 >>> 8, 8),
  ].join("-")
}

/**
 * Is this device-tool advertisement allowed to run at all?
 * Off unless explicitly enabled: it changes what the model is offered on every
 * turn, so it must be an opt-in, not a surprise.
 */
export function nativeDeviceToolsEnabled(opts = {}) {
  const v = opts.nativeDeviceTools ?? opts.deviceTools ?? false
  if (v === true) return true
  const s = String(v).toLowerCase()
  return s === "1" || s === "true" || s === "yes" || s === "on"
}
