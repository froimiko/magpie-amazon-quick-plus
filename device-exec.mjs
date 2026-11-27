/**
 * Local executor for the device tools this plugin advertises.
 *
 * The contract the server expects back is a JSON object in tool_result.result.
 * Shapes follow what the official outpost returns, because the model has been
 * trained on those: listings carry entries, failures carry success:false plus a
 * human-readable error instead of throwing.
 *
 * Safety: every path is resolved and checked against the allowed roots this
 * process was given; nothing outside them is read or written. That is not
 * belt-and-braces — the model is remote, and this code runs as the user.
 */
import { createHash } from "node:crypto"
import {
  copyFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs"
import { homedir } from "node:os"
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path"
import { execFile } from "node:child_process"

const MAX_READ_BYTES = 2 * 1024 * 1024
const MAX_LIST_ENTRIES = 500
const DEFAULT_TIMEOUT_MS = 60_000
const GLOB_SPECIALS = /[.+^()|[\]{}$]/g

function expandHome(p) {
  const s = String(p ?? "").trim()
  if (!s) return s
  if (s === "~") return homedir()
  if (s.startsWith("~/") || s.startsWith("~\\")) return join(homedir(), s.slice(2))
  return s
}

function globToRegExp(pattern) {
  const escaped = pattern.replace(GLOB_SPECIALS, "\\$&")
  return new RegExp("^" + escaped.replace(/\*/g, ".*").replace(/\?/g, ".") + "$", "i")
}

export function createLocalDeviceToolExecutor(opts = {}) {
  const roots = (Array.isArray(opts.roots) && opts.roots.length ? opts.roots : [opts.cwd || process.cwd(), homedir()])
    .map((r) => resolve(expandHome(r)))
  const allowShell = opts.allowShell !== false
  const maxOutput = Number(opts.maxOutputBytes) > 0 ? Number(opts.maxOutputBytes) : 200_000

  const insideRoot = (abs) => {
    for (const root of roots) {
      if (abs === root) return true
      const rel = relative(root, abs)
      if (rel && !rel.startsWith(".." + sep) && rel !== ".." && !isAbsolute(rel)) return true
    }
    return false
  }

  const guarded = (rawPath, o = {}) => {
    const p = expandHome(rawPath)
    if (!p) return { error: "path is required" }
    const abs = resolve(isAbsolute(p) ? p : join(roots[0], p))
    if (!insideRoot(abs)) {
      return { error: "Access denied. Path is not in allowed folders: " + JSON.stringify(roots) }
    }
    if (o.forWrite && !existsSync(dirname(abs))) {
      return { error: "Parent directory does not exist: " + dirname(abs) }
    }
    return { abs }
  }

  const runCommand = (command, cwd, timeoutMs) =>
    new Promise((res) => {
      const isWin = process.platform === "win32"
      const shell = isWin ? "cmd.exe" : "/bin/sh"
      const args = isWin ? ["/d", "/s", "/c", command] : ["-c", command]
      execFile(shell, args, { cwd, timeout: timeoutMs, maxBuffer: maxOutput, windowsHide: true }, (err, stdout, stderr) => {
        res({
          success: !err,
          output: String(stdout || "").slice(0, maxOutput),
          stderr: String(stderr || "").slice(0, maxOutput),
          exit_code: err && typeof err.code === "number" ? err.code : 0,
          ...(err && typeof err.code !== "number" ? { error: String(err.message || err) } : {}),
        })
      })
    })

  const handlers = {
    folder_list(args) {
      const rawPath = String(args.path ?? "").trim()
      if (!rawPath) return { success: true, allowed_folders: roots }
      const g = guarded(rawPath)
      if (g.error) return { success: false, error: g.error, file_op_failure_reason: "path_not_allowed" }
      if (!existsSync(g.abs)) return { success: false, error: "Path does not exist: " + g.abs, path: g.abs }
      const showHidden = args.show_hidden === true
      const entries = readdirSync(g.abs, { withFileTypes: true })
        .filter((d) => showHidden || !d.name.startsWith("."))
        .slice(0, MAX_LIST_ENTRIES)
        .map((d) => {
          const full = join(g.abs, d.name)
          let size = null
          let mtime = null
          try {
            const st = statSync(full)
            size = st.size
            mtime = st.mtimeMs
          } catch {}
          return {
            name: d.name,
            path: full,
            type: d.isDirectory() ? "directory" : d.isFile() ? "file" : "other",
            size,
            mtime_ms: mtime,
          }
        })
      return { success: true, path: g.abs, entries, count: entries.length }
    },

    file_read(args) {
      const g = guarded(args.path)
      if (g.error) return { success: false, error: g.error }
      if (!existsSync(g.abs)) return { success: false, error: "File does not exist: " + g.abs }
      const st = statSync(g.abs)
      if (st.isDirectory()) return { success: false, error: "Path is a directory: " + g.abs }
      if (st.size > MAX_READ_BYTES) {
        return {
          success: false,
          error: "File is too large to read inline (" + st.size + " bytes, limit " + MAX_READ_BYTES + "). Read it with a shell command instead.",
          path: g.abs,
          size: st.size,
        }
      }
      const text = readFileSync(g.abs, "utf8")
      const offset = Number(args.offset) > 0 ? Math.floor(Number(args.offset)) : 1
      const limit = Number(args.limit) > 0 ? Math.floor(Number(args.limit)) : null
      const all = text.split("\n")
      const from = offset - 1
      const slice = limit ? all.slice(from, from + limit) : all.slice(from)
      return { success: true, path: g.abs, content: slice.join("\n"), total_lines: all.length, offset }
    },

    file_type(args) {
      const g = guarded(args.path)
      if (g.error) return { success: false, error: g.error }
      if (!existsSync(g.abs)) return { success: true, exists: false, path: g.abs }
      const st = lstatSync(g.abs)
      return {
        success: true,
        exists: true,
        path: g.abs,
        type: st.isDirectory() ? "directory" : st.isFile() ? "file" : "other",
        size: st.size,
        mtime_ms: st.mtimeMs,
      }
    },

    file_matches(args) {
      const g = guarded(args.path)
      if (g.error) return { success: false, error: g.error, matches: [] }
      const base = existsSync(g.abs) ? g.abs : dirname(g.abs)
      const pattern = g.abs.split(/[\\/]/).pop() || "*"
      let re
      try {
        re = globToRegExp(pattern)
      } catch {
        return { success: false, error: "Invalid pattern", matches: [] }
      }
      const out = []
      const walk = (dir, depth) => {
        if (depth > 6 || out.length >= MAX_LIST_ENTRIES) return
        let items
        try {
          items = readdirSync(dir, { withFileTypes: true })
        } catch {
          return
        }
        for (const d of items) {
          if (out.length >= MAX_LIST_ENTRIES) return
          if (d.name === "node_modules") continue
          const full = join(dir, d.name)
          if (re.test(d.name)) out.push(full)
          if (d.isDirectory()) walk(full, depth + 1)
        }
      }
      walk(base, 0)
      return { success: true, matches: out, count: out.length }
    },

    file_write(args) {
      const g = guarded(args.path, { forWrite: true })
      if (g.error) return { success: false, error: g.error }
      const body = String(args.content ?? "")
      writeFileSync(g.abs, body, "utf8")
      return { success: true, path: g.abs, bytes: Buffer.byteLength(body, "utf8") }
    },

    file_edit(args) {
      const g = guarded(args.path)
      if (g.error) return { success: false, error: g.error }
      if (!existsSync(g.abs)) return { success: false, error: "File does not exist: " + g.abs }
      const next = applyUnifiedDiff(readFileSync(g.abs, "utf8"), String(args.diff ?? ""))
      if (next.error) return { success: false, error: next.error }
      writeFileSync(g.abs, next.text, "utf8")
      return { success: true, path: g.abs }
    },

    file_move(args) {
      const s = guarded(args.source)
      if (s.error) return { success: false, error: s.error }
      const d = guarded(args.destination, { forWrite: true })
      if (d.error) return { success: false, error: d.error }
      renameSync(s.abs, d.abs)
      return { success: true, from: s.abs, to: d.abs }
    },

    file_copy(args) {
      const s = guarded(args.source)
      if (s.error) return { success: false, error: s.error }
      const d = guarded(args.destination, { forWrite: true })
      if (d.error) return { success: false, error: d.error }
      copyFileSync(s.abs, d.abs)
      return { success: true, from: s.abs, to: d.abs }
    },

    folder_create(args) {
      const g = guarded(args.path, { forWrite: true })
      if (g.error) return { success: false, error: g.error }
      mkdirSync(g.abs, { recursive: true })
      return { success: true, path: g.abs }
    },

    file_delete(args) {
      const g = guarded(args.path)
      if (g.error) return { success: false, error: g.error }
      rmSync(g.abs, { recursive: true, force: true })
      return { success: true, path: g.abs }
    },

    ripgrep(args) {
      const patterns = (Array.isArray(args.patterns) ? args.patterns : [args.patterns]).map(String).filter(Boolean)
      if (!patterns.length) return { success: false, error: "patterns is required" }
      const targets = []
      for (const p of Array.isArray(args.paths) && args.paths.length ? args.paths : [roots[0]]) {
        const g = guarded(p)
        if (g.error) return { success: false, error: g.error }
        targets.push(g.abs)
      }
      const parts = ["rg", "-n", "--no-heading"]
      for (const pat of patterns) parts.push("-e", pat)
      parts.push("--")
      for (const t of targets) parts.push(t)
      return runCommand(parts.map(quote).join(" "), roots[0], Number(args.timeout) * 1000 || DEFAULT_TIMEOUT_MS)
    },

    terminal_on_device(args) {
      if (!allowShell) return { success: false, error: "Shell execution is disabled on this client." }
      const command = String(args.command ?? "").trim()
      if (!command) return { success: false, error: "command is required" }
      let cwd = roots[0]
      const cwdRaw = String(args.cwd ?? "").trim()
      if (cwdRaw) {
        const g = guarded(cwdRaw)
        if (g.error) return { success: false, error: g.error }
        cwd = g.abs
      }
      return runCommand(command, cwd, Number(args.timeout) * 1000 || DEFAULT_TIMEOUT_MS)
    },

    run_python_on_device(args) {
      if (!allowShell) return { success: false, error: "Shell execution is disabled on this client." }
      const code = String(args.code ?? "")
      if (!code.trim()) return { success: false, error: "code is required" }
      const py = process.platform === "win32" ? "python" : "python3"
      const tmp = join(roots[0], ".magpie-run-" + createHash("sha1").update(code).digest("hex").slice(0, 10) + ".py")
      try {
        writeFileSync(tmp, code, "utf8")
        return runCommand(quote(py) + " " + quote(tmp), roots[0], Number(args.timeout) * 1000 || DEFAULT_TIMEOUT_MS)
      } finally {
        try {
          rmSync(tmp, { force: true })
        } catch {}
      }
    },
  }

  const execute = async ({ toolName, arguments: args }) => {
    const fn = handlers[toolName]
    if (!fn) {
      return {
        result: {
          success: false,
          error: "This client cannot execute the device tool '" + toolName + "'.",
          supported_tool: Object.keys(handlers),
        },
      }
    }
    try {
      return { result: await fn(args && typeof args === "object" ? args : {}) }
    } catch (e) {
      return { result: { success: false, error: String(e && e.message ? e.message : e) } }
    }
  }
  execute.supportedTools = Object.keys(handlers)
  return execute
}

function quote(s) {
  const str = String(s)
  if (process.platform === "win32") return '"' + str.replace(/"/g, '\\"') + '"'
  return "'" + str.replace(/'/g, "'\\''") + "'"
}

/**
 * Apply the unified diff the file_edit tool is documented to take.
 * Content-based with a small fuzz: line numbers need not be exact, context lines
 * decide the location. Anything ambiguous returns an error rather than guessing —
 * a wrong edit is worse than a refused one.
 */
export function applyUnifiedDiff(original, diffText) {
  const text = String(diffText ?? "")
  if (!text.trim()) return { error: "diff is empty" }
  const lines = original.split("\n")
  const hunks = []
  let cur = null
  for (const raw of text.split("\n")) {
    if (raw.startsWith("@@")) {
      const m = raw.match(/@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@/)
      if (!m) return { error: "Malformed hunk header: " + raw }
      cur = { oldStart: Number(m[1]), lines: [] }
      hunks.push(cur)
      continue
    }
    if (!cur) continue
    if (raw.startsWith("---") || raw.startsWith("+++") || raw.startsWith("\\")) continue
    cur.lines.push(raw)
  }
  if (!hunks.length) return { error: "diff has no hunks" }

  let offset = 0
  for (const h of hunks) {
    const want = []
    for (const l of h.lines) {
      if (l.startsWith("-") || l.startsWith(" ")) want.push(l.slice(1))
      else if (l === "") want.push("")
    }
    const targetIdx = h.oldStart - 1 + offset
    let at
    if (want.length) {
      at = findBlock(lines, want, targetIdx)
      if (at < 0) at = findBlock(lines, want, 0)
      if (at < 0) return { error: "Could not locate the context for hunk @@" + h.oldStart }
    } else {
      at = Math.max(0, targetIdx)
    }
    const replacement = []
    for (const l of h.lines) {
      if (l.startsWith("+") || l.startsWith(" ")) replacement.push(l.slice(1))
      else if (l === "") replacement.push("")
    }
    lines.splice(at, want.length, ...replacement)
    offset += replacement.length - want.length
  }
  return { text: lines.join("\n") }
}

function findBlock(lines, want, from) {
  const start = Math.max(0, Math.min(from, lines.length - want.length))
  for (let i = start; i + want.length <= lines.length; i++) {
    let ok = true
    for (let k = 0; k < want.length; k++) {
      if (lines[i + k].trim() !== want[k].trim()) {
        ok = false
        break
      }
    }
    if (ok) return i
  }
  return -1
}
