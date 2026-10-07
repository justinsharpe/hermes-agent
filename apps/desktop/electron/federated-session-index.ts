/**
 * federated-session-index.ts
 *
 * Persistent federated conversation index for the multi-connection registry
 * (L4 unified-inbox fix 1, "federated conversation index" — see
 * viiy-hq/research/tool-teardowns/hermes-openclaw-synthesis-2026-10-07/L4-desktop-unified-inbox.md,
 * §5 bullet 1, fixing M2's amnesia half + M3's cold-instance half).
 *
 * The unified Sessions list is assembled read-time only: every refresh re-reads
 * each reachable backend and the union lives solely in renderer memory
 * (electron/main.ts mergeRemoteProfileSessions). A cold or unreachable instance
 * contributes zero rows, and a gateway switch empties the list until the merge
 * re-reads. This module is the accumulating local cache that survives both: the
 * SAME rows `fetchRegistrySessionRows` (profile-session-routing.ts) already
 * stamps with `connection_id` + `profile` are remembered per
 * `(connectionId, profile)` owner, so the sidebar keeps history for instances
 * that are not currently readable — and opens still route exactly as today
 * (owner routing is unchanged; the rows carry the same tags).
 *
 * Additive/option-gated (upstream constraint): mergeSessionIndexRows() only
 * EXTENDS the live merged list — it never mutates, reorders, or replaces rows
 * the read-time merge produced. A registry with one connection is never
 * enumerated (the enumerator lives in main.ts and skips single-connection
 * registries), so single-connection behavior is byte-identical.
 *
 * Storage: one JSON file beside connections.json (userData/
 * session-index.json), written atomically (tmp + rename, same pattern as
 * pool-limits.json). Deliberately NOT SQLite: the index is a bounded cache
 * (MAX_OWNERS owners × MAX_ROWS_PER_OWNER rows), a rewrite is O(kB), and the
 * desktop main process already owns several atomic-JSON stores — this keeps the
 * zero-native-dep posture of the electron/ layer. JSON also unit-tests without
 * a database.
 *
 * Kept electron-free (no `import 'electron'`) so it unit-tests under the
 * `electron` vitest project's node environment — same pattern as
 * connection-registry.ts.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

/** One indexed conversation. Only the fields the sidebar needs to paint a row
 * and route an open; the transcript itself is always fetched live from the
 * owning host. */
export interface IndexedSessionRow {
  id: string
  profile: string
  connection_id: string
  title?: null | string
  /** Unix seconds — the same key the read-time merge sorts by. */
  last_active?: number
  started_at?: number
  preview?: null | string
  [key: string]: unknown
}

/** File schema. `version` gates shape changes; readers reject unknown versions
 * rather than guessing (the index is a rebuildable cache — a rejected file is
 * simply treated as absent and re-accumulated). */
export interface FederatedSessionIndex {
  version: 1
  /** `connectionId::profile` → newest-first rows. */
  owners: Record<string, IndexedSessionRow[]>
}

export const FEDERATED_SESSION_INDEX_VERSION = 1
export const SESSION_INDEX_FILE_NAME = 'session-index.json'

/** Bound the cache: an owner's rows are replaced wholesale by a fresh read
 * (mergeSessionIndexRows keeps only MAX_ROWS_PER_OWNER newest), and an owner
 * idle for MAX_OWNER_IDLE_MS is dropped wholesale on the next write. Prevents
 * a removed connection's history from lingering forever and keeps the file
 * bounded no matter how many pages the enumerator ever read. */
export const MAX_ROWS_PER_OWNER = 200
export const MAX_OWNERS = 50
export const MAX_OWNER_IDLE_MS = 14 * 24 * 60 * 60 * 1000 // 14 days

/** Row recency — the same key mergeRemoteProfileSessions sorts by. */
function rowRecency(row: IndexedSessionRow): number {
  return Math.max(Number(row.last_active) || 0, Number(row.started_at) || 0)
}

function ownerKey(connectionId: string, profile: string): string {
  return `${connectionId}::${profile}`
}

function rowsOfSessionList(data: unknown): IndexedSessionRow[] {
  if (!data || typeof data !== 'object' || !('sessions' in data)) {
    return []
  }

  return Array.isArray((data as { sessions?: unknown }).sessions) ? ((data as { sessions: unknown[] }).sessions as IndexedSessionRow[]) : []
}

function rowOwnerOf(row: unknown): null | { connectionId: string; profile: string } {
  if (!row || typeof row !== 'object') {
    return null
  }

  const session = row as Record<string, unknown>
  const connectionId = typeof session.connection_id === 'string' ? session.connection_id.trim() : ''
  const profile = typeof session.profile === 'string' && session.profile.trim() ? session.profile.trim() : 'default'

  return connectionId ? { connectionId, profile } : null
}

/** Parse + shape-validate a session-index.json payload. Unknown version or
 * malformed owners → null (callers treat as absent; the cache rebuilds). */
export function parseSessionIndex(data: unknown): FederatedSessionIndex | null {
  if (!data || typeof data !== 'object' || (data as { version?: unknown }).version !== FEDERATED_SESSION_INDEX_VERSION) {
    return null
  }

  const owners: Record<string, IndexedSessionRow[]> = {}

  if ((data as { owners?: unknown }).owners && typeof (data as { owners?: unknown }).owners === 'object') {
    for (const [key, value] of Object.entries((data as { owners: Record<string, unknown> }).owners)) {
      if (Array.isArray(value)) {
        owners[key] = value.filter(row => rowOwnerOf(row) !== null)
      }
    }
  }

  return { version: FEDERATED_SESSION_INDEX_VERSION, owners }
}

/** Newest-first per owner, at most MAX_ROWS_PER_OWNER. Sort BEFORE slicing:
 * an input page is not guaranteed recency-ordered, so slicing first would cap
 * by input position and drop the newest rows. */
function normalizeOwnerRows(rows: IndexedSessionRow[]): IndexedSessionRow[] {
  return [...rows].sort((a, b) => rowRecency(b) - rowRecency(a)).slice(0, MAX_ROWS_PER_OWNER)
}

/** Drop owners whose newest row is older than MAX_OWNER_IDLE_MS, and keep at
 * most MAX_OWNERS owners (newest activity wins). */
export function pruneSessionIndex(index: FederatedSessionIndex, now = Date.now()): FederatedSessionIndex {
  const nowSeconds = Math.floor(now / 1000)
  const owners = Object.entries(index.owners).filter(([, rows]) => {
    if (!rows.length) {
      return false
    }

    const newest = rows.reduce((max, row) => Math.max(max, rowRecency(row)), 0)

    return newest >= nowSeconds - MAX_OWNER_IDLE_MS / 1000
  })

  owners.sort((a, b) => rowRecency(b[1][0]) - rowRecency(a[1][0]))

  return { version: FEDERATED_SESSION_INDEX_VERSION, owners: Object.fromEntries(owners.slice(0, MAX_OWNERS)) }
}

/**
 * Remember the owner-tagged rows of one fresh registry read. Rows WITHOUT a
 * `connection_id` (the primary aggregate's own local rows) are ignored: they
 * are always re-readable from the primary backend and never need cold-cache
 * amnesia repair. A successful read REPLACES that owner's previous rows (a
 * read is authoritative for its owner); a `[]` page still counts as fresh and
 * clears the owner. `now` stamps recency for pruning.
 */
export function mergeSessionIndexRows(
  index: FederatedSessionIndex,
  connectionId: string,
  sessions: IndexedSessionRow[] | { sessions?: unknown[] } | null | undefined,
  now = Date.now()
): FederatedSessionIndex {
  const rows = Array.isArray(sessions) ? sessions : rowsOfSessionList(sessions)
  const owners = { ...index.owners }

  // Group by the row's own profile tag (fetchRegistrySessionRows stamps it);
  // a shared-host read spans several profiles in one page.
  const byProfile = new Map<string, IndexedSessionRow[]>()

  for (const row of rows) {
    const owner = rowOwnerOf(row)

    if (!owner || owner.connectionId !== connectionId) {
      continue
    }

    const list = byProfile.get(owner.profile) ?? []

    list.push(row)
    byProfile.set(owner.profile, list)
  }

  for (const [profile, profileRows] of byProfile) {
    owners[ownerKey(connectionId, profile)] = normalizeOwnerRows(profileRows)
  }

  // A fresh read that returned NO rows for a profile still proves that
  // profile readable-now: clear any stale indexed rows for profiles of this
  // connection that the read did not mention (archive/delete on the host).
  for (const key of Object.keys(owners)) {
    if (key.startsWith(`${connectionId}::`) && !byProfile.has(key.slice(connectionId.length + 2))) {
      delete owners[key]
    }
  }

  return pruneSessionIndex({ version: FEDERATED_SESSION_INDEX_VERSION, owners }, now)
}

/** All indexed rows across every owner, newest-first — the cold-instance
 * amnesia repair the unified list splices in when a source is not readable. */
export function indexedSessionRows(index: FederatedSessionIndex): IndexedSessionRow[] {
  const rows: IndexedSessionRow[] = []

  for (const ownerRows of Object.values(index.owners)) {
    rows.push(...ownerRows)
  }

  return rows.sort((a, b) => rowRecency(b) - rowRecency(a))
}

/** Owner keys present in the index (used to compute which registry
 * connections are cold-but-remembered). */
export function indexedOwnerKeys(index: FederatedSessionIndex): Set<string> {
  return new Set(Object.keys(index.owners))
}

/**
 * File-backed store. One instance lives in Electron main; `dir` is userData.
 * All operations are best-effort and never throw: the index is a cache, and a
 * read-only/full disk must not break the session list (same posture as the
 * registry's own write-failure handling).
 */
export class FederatedSessionIndexStore {
  private readonly filePath: string
  private cache: FederatedSessionIndex | null = null

  constructor(dir: string) {
    this.filePath = path.join(dir, SESSION_INDEX_FILE_NAME)
  }

  load(): FederatedSessionIndex {
    if (this.cache) {
      return this.cache
    }

    let parsed: unknown = null

    try {
      parsed = JSON.parse(fs.readFileSync(this.filePath, 'utf8'))
    } catch {
      // Absent or corrupt → empty (a rebuildable cache).
    }

    this.cache = parsed ? parseSessionIndex(parsed) : null

    return (this.cache ??= { version: FEDERATED_SESSION_INDEX_VERSION, owners: {} })
  }

  remember(connectionId: string, sessions: IndexedSessionRow[] | { sessions?: unknown[] } | null | undefined): void {
    try {
      const next = mergeSessionIndexRows(this.load(), connectionId, sessions)
      this.persist(next)
    } catch {
      // Best-effort by contract.
    }
  }

  rows(): IndexedSessionRow[] {
    return indexedSessionRows(this.load())
  }

  ownerKeys(): Set<string> {
    return indexedOwnerKeys(this.load())
  }

  clear(): void {
    this.cache = { version: FEDERATED_SESSION_INDEX_VERSION, owners: {} }

    try {
      fs.rmSync(this.filePath, { force: true })
    } catch {
      // Best-effort.
    }
  }

  /** Atomic tmp+rename write (same pattern as persistPoolLimits). */
  private persist(index: FederatedSessionIndex): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true })
    const tmp = `${this.filePath}.tmp`
    fs.writeFileSync(tmp, JSON.stringify(index), 'utf8')
    fs.renameSync(tmp, this.filePath)
    this.cache = index
  }
}