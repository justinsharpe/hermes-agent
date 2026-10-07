/**
 * registry-session-enumerator.ts
 *
 * Bounded background enumeration of registry connections (L4 unified-inbox
 * fix 2 — L4-desktop-unified-inbox.md §5 bullet 2, fixing M3/M4's silent
 * absence). The unified list today reads only already-pooled backends
 * (main.ts pooledRegistrySessionSources: 'a sidebar refresh must never dial or
 * spawn a backend'), so any gateway without a warm backend contributes zero
 * rows with no distinction between 'no conversations' and 'not fetched'.
 *
 * This module schedules ONE low-rate passive-pass per registered connection,
 * staggered, and only for multi-connection registries — the single-connection
 * case (the only one upstream guarantees byte-identical) never runs a pass,
 * because the primary backend already serves its rows through the normal
 * aggregate path.
 *
 * The dialing itself is inverted-only-for-this-lane, exactly as the L4 doc
 * prescribes: dispatchRegistryApiRequest treats `passive: true` as 'never
 * dial'; the enumerator instead asks its caller to dial with NORMAL
 * (non-passive) semantics but background spawn priority, through the same
 * claim-guarded ensureRegistryBackend path an interactive open uses. It is
 * bounded in every dimension:
 *
 *  - one in-flight pass per connection at a time (per-connection lock)
 *  - interval per pass (default 5 minutes) + stagger so passes never coincide
 *  - serialized queue: at most one pass runs at any moment across the fleet
 *  - bounded fetch window per pass (page cap, 10s timeout per page)
 *  - errors are swallowed and remembered; a pass never throws to the host
 *
 * Kept electron-free (no 'import electron') so it unit-tests under the
 * `electron` vitest project; main.ts injects dial/fetch/persist callbacks
 * (the FederatedSessionIndexStore from L4 fix 1 is the sink).
 */

/** Which connections a pass should cover, and how to stagger them. Pure so the
 * scheduler's tick math is unit-testable without a timer. */
export interface EnumerationTarget {
  connectionId: string
  /** Kinds 'ssh' and 'local' scope one backend per profile; shared
   * remote/cloud hosts serve every profile from one backend. */
  kind: string
}

export const ENUMERATION_INTERVAL_MS = 5 * 60 * 1000
/** Upper bound on rows asked of one backend per pass. Matches the read-time
 * merge's remote page shape (profile-session-routing REMOTE_SESSION_PAGE_LIMIT
 * * a small multiple), keeping a pass comparable to one sidebar refresh. */
export const ENUMERATION_PAGE_LIMIT = 100
export const ENUMERATION_FETCH_TIMEOUT_MS = 10_000

/** Stagger the fleet: pass N of connection i starts at
 * i * STAGGER_MS so two connections never dial simultaneously even
 * under a serialized queue that runs instantly. */
export const STAGGER_MS = 5_000

/** Host contract injected by main.ts. Every leg is bounded by the caller;
 * the enumerator only owns scheduling/locking. */
export interface EnumerationHost {
  /** List the registered connections (main.ts: readDesktopConnectionsRegistry). */
  connections: () => EnumerationTarget[]
  /**
   * Read one connection's cross-profile session rows, DIALING if no backend
   * is pooled (background spawn priority). Returns the owner-tagged rows on
   * success, null on any failure (dead host, auth failure, timeout). The
   * implementation must never resolve null slowly — the enumerator's own
   * queue keeps passes serialized regardless.
   */
  fetchSessions: (connectionId: string) => Promise<{ sessions: unknown[] } | null>
  /** Persist a successful read into the federated index (L4 fix 1). */
  remember: (connectionId: string, sessions: unknown[]) => void
  /** setTimeout/clearTimeout pair — injectable for tests. */
  schedule: (fn: () => void, ms: number) => unknown
  cancel: (handle: unknown) => void
  /** Test seam; defaults to Date.now(). */
  now?: () => number
}

/**
 * The scheduler. One instance lives in Electron main; `start()` is idempotent
 * and `stop()` fully tears down (quit / registry edit path).
 */
export class RegistrySessionEnumerator {
  private readonly host: EnumerationHost
  private readonly intervalMs: number
  private readonly staggerMs: number
  private readonly queue: string[] = []
  private readonly inFlight = new Set<string>()
  private running = false
  private timer: unknown = null
  private drainTimer: unknown = null
  private lastPassAt = new Map<string, number>()

  constructor(host: EnumerationHost, intervalMs = ENUMERATION_INTERVAL_MS, staggerMs = STAGGER_MS) {
    this.host = host
    this.intervalMs = intervalMs
    this.staggerMs = staggerMs
  }

  /** Whether a pass is currently executing (unit-test observable). */
  get busy(): boolean {
    return this.inFlight.size > 0
  }

  start(): void {
    if (this.running) {
      return
    }

    this.running = true
    this.tick()
    this.timer = this.host.schedule(() => {
      this.tick()
      if (this.running) {
        this.timer = this.host.schedule(this.tickLoop, this.intervalMs)
      }
    }, this.intervalMs)
  }

  stop(): void {
    this.running = false
    this.queue.length = 0

    if (this.timer !== null) {
      this.host.cancel(this.timer)
      this.timer = null
    }

    if (this.drainTimer !== null) {
      this.host.cancel(this.drainTimer)
      this.drainTimer = null
    }
  }

  /** One scheduling decision, pure enough to test: enqueue every
   * non-in-flight connection whose last pass is older than the interval,
   * then drain staggered. Single-connection registries (or empty ones) are
   * skipped entirely — the primary aggregate already covers that topology. */
  private tick = (): void => {
    const now = this.host.now?.() ?? Date.now()
    const targets = this.host.connections()

    if (targets.length < 2) {
      return
    }

    for (const target of targets) {
      const last = this.lastPassAt.get(target.connectionId)

      if (last !== undefined && now - last < this.intervalMs) {
        continue
      }

      if (this.inFlight.has(target.connectionId) || this.queue.includes(target.connectionId)) {
        continue
      }

      this.queue.push(target.connectionId)
    }

    this.drain()
  }

  private tickLoop = (): void => {
    this.tick()

    if (this.running) {
      this.timer = this.host.schedule(this.tickLoop, this.intervalMs)
    }
  }

  /** Serialized staggered drain: at most one pass executes at a time; the
   * next is scheduled staggerMs later. A connection removed from the
   * registry between enqueue and execution is skipped. */
  private drain = (): void => {
    if (!this.running || this.drainTimer !== null) {
      return
    }

    const next = this.queue.shift()

    if (next === undefined) {
      return
    }

    const stillRegistered = this.host.connections().some(target => target.connectionId === next)

    if (!stillRegistered) {
      this.drain()
      return
    }

    this.inFlight.add(next)
    this.lastPassAt.set(next, this.host.now?.() ?? Date.now())

    const settle = () => {
      this.inFlight.delete(next)

      if (this.queue.length) {
        this.drainTimer = this.host.schedule(() => {
          this.drainTimer = null
          this.drain()
        }, this.staggerMs)
      }
    }

    void this.host
      .fetchSessions(next)
      .then(result => {
        if (result?.sessions?.length) {
          this.host.remember(next, result.sessions)
        }
      })
      .catch(() => undefined)
      .finally(settle)
  }
}