import { describe, expect, it, vi } from 'vitest'

import { RegistrySessionEnumerator, type EnumerationHost } from './registry-session-enumerator'

interface Harness {
  host: EnumerationHost
  timers: Array<{ fire: () => void; ms: number; cancelled: boolean }>
  fetchSessions: ReturnType<typeof vi.fn>
  remember: ReturnType<typeof vi.fn>
  connections: Array<{ connectionId: string; kind: string }>
}

function harness(connections: Array<{ connectionId: string; kind: string }>): Harness {
  const timers: Harness['timers'] = []
  const fetchSessions = vi.fn(async () => ({ sessions: [{ id: 's1', profile: 'default', connection_id: 'homelab' }] }))
  const remember = vi.fn()

  const host: EnumerationHost = {
    connections: () => connections,
    fetchSessions,
    remember,
    schedule: (fn, ms) => {
      const timer = { fire: fn, ms, cancelled: false }
      timers.push(timer)

      return timer
    },
    cancel: handle => {
      ;(handle as Harness['timers'][number]).cancelled = true
    },
    now: () => 1_000_000
  }

  return { host, timers, fetchSessions, remember, connections }
}

function fireTimers(h: Harness) {
  for (const timer of [...h.timers]) {
    if (!timer.cancelled) {
      timer.fire()
    }
  }
}

/** Flush all pending microtasks (one macrotask boundary), so the enumerator's
 * fetch→remember→settle chain completes before assertions. */
async function settle(): Promise<void> {
  await new Promise(resolve => setImmediate(resolve))
}

function fireDueTimers(h: Harness) {
  for (const timer of [...h.timers]) {
    if (!timer.cancelled && timer.ms <= 5_000) {
      timer.fire()
    }
  }
}

describe('RegistrySessionEnumerator', () => {
  it('never enumerates a single-connection registry (upstream: single-connection behavior identical)', async () => {
    const h = harness([{ connectionId: 'local', kind: 'local' }])
    const enumerator = new RegistrySessionEnumerator(h.host)
    enumerator.start()
    fireTimers(h)
    await settle()

    expect(h.fetchSessions).not.toHaveBeenCalled()
  })

  it('enumerates each registered connection once, staggered and serialized', async () => {
    const h = harness([
      { connectionId: 'local', kind: 'local' },
      { connectionId: 'homelab', kind: 'ssh' }
    ])
    const enumerator = new RegistrySessionEnumerator(h.host)
    enumerator.start()

    // First tick fires on start(): homelab enqueued (local skipped? no —
    // local is a registered connection too and its backend dial is the
    // primary; both are enumerated; the primary dial is a no-op reuse).
    await settle()

    // First pass runs immediately; the second is staggered.
    fireDueTimers(h)
    await settle()

    expect(h.fetchSessions).toHaveBeenCalledTimes(2)
    expect(h.remember).toHaveBeenCalledTimes(2)
  })

  it('a failed fetch is swallowed and remembered nothing (dead gateway contributes no rows)', async () => {
    const h = harness([
      { connectionId: 'local', kind: 'local' },
      { connectionId: 'deadbox', kind: 'ssh' }
    ])
    // Reject the DEADBOX leg specifically (the queue is serialized, so a
    // `mockRejectedValueOnce` would reject the first pass instead).
    h.fetchSessions.mockImplementation(async (connectionId: string) => {
      if (connectionId === 'deadbox') {
        throw new Error('unreachable')
      }

      return { sessions: [{ id: 's1', profile: 'default', connection_id: 'local' }] }
    })
    const enumerator = new RegistrySessionEnumerator(h.host)
    enumerator.start()

    await settle()
    fireDueTimers(h)
    await settle()

    // The dead gateway itself contributed nothing; the OTHER connection's
    // (successful) read is remembered independently.
    expect(h.fetchSessions).toHaveBeenCalledWith('deadbox')
    expect(h.remember).not.toHaveBeenCalledWith('deadbox', expect.anything())
    expect(h.remember).toHaveBeenCalledTimes(1)
  })

  it('skips a connection removed between enqueue and execution', async () => {
    const h = harness([
      { connectionId: 'local', kind: 'local' },
      { connectionId: 'homelab', kind: 'ssh' },
      { connectionId: 'oldbox', kind: 'ssh' }
    ])
    const enumerator = new RegistrySessionEnumerator(h.host)
    enumerator.start()

    // Remove oldbox before the staggered leg fires.
    const old = h.connections
    h.host.connections = () => old.filter(c => c.connectionId !== 'oldbox')

    await settle()
    fireDueTimers(h)
    await settle()

    expect(h.fetchSessions).not.toHaveBeenCalledWith('oldbox')
  })

  it('respects the per-connection interval: a second tick inside the window is a no-op', async () => {
    const h = harness([
      { connectionId: 'local', kind: 'local' },
      { connectionId: 'homelab', kind: 'ssh' }
    ])
    let clock = 1_000_000
    h.host.now = () => clock
    const enumerator = new RegistrySessionEnumerator(h.host)
    enumerator.start()

    await settle()
    fireDueTimers(h)
    await settle()

    expect(h.fetchSessions).toHaveBeenCalledTimes(2)

    // A manual tick 1s later: interval not elapsed, nothing enqueued.
    clock += 1_000
    ;(enumerator as unknown as { tick: () => void }).tick()
    await Promise.resolve()

    expect(h.fetchSessions).toHaveBeenCalledTimes(2)

    // After the interval elapses, the next pass runs again.
    clock += 5 * 60 * 1000
    ;(enumerator as unknown as { tick: () => void }).tick()
    await settle()
    fireDueTimers(h)
    await settle()

    expect(h.fetchSessions).toHaveBeenCalledTimes(4)
  })

  it('stop() cancels pending timers and drains the queue', () => {
    const h = harness([
      { connectionId: 'local', kind: 'local' },
      { connectionId: 'homelab', kind: 'ssh' }
    ])
    const enumerator = new RegistrySessionEnumerator(h.host)
    enumerator.start()
    enumerator.stop()

    for (const timer of h.timers) {
      expect(timer.cancelled).toBe(true)
    }
  })
})