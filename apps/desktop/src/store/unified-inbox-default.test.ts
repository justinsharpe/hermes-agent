import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { atom } from 'nanostores'

// Keep profile.ts's side-effecting imports inert (same pattern as
// profile.test.ts): the gateway socket layer and REST client must not run in
// a unit test.
const $gateway = atom<unknown>({ id: 'live-socket', connectionState: 'open' })

vi.mock('@/store/gateway', () => ({
  $gateway,
  activeGateway: () => null,
  activeGatewayConnectionId: () => null,
  activeGatewayProfileKey: () => 'default',
  ensureGatewayForProfile: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined)
}))
vi.mock('@/store/pool-limits', async () => {
  const { atom } = await import('nanostores')

  return { $poolLimits: atom({ idleMs: 600_000, maxBackends: 3 }) }
})
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const {
  $showAllProfiles,
  applyUnifiedInboxDefault,
  setUnifiedInboxPreference,
  setShowAllProfiles,
  storedUnifiedInboxPreference
} = await import('./profile')

const UNIFIED_KEY = 'hermes.desktop.unifiedInboxEnabled'
const SHOW_ALL_KEY = 'hermes.desktop.showAllProfiles'

describe('applyUnifiedInboxDefault (L4 fix 3 — unified-default for multi-connection registries)', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setShowAllProfiles(false)
  })

  afterEach(() => {
    window.localStorage.clear()
  })

  it('defaults the unified view ON for a multi-connection registry when no preference exists', () => {
    expect(storedUnifiedInboxPreference()).toBeNull()

    applyUnifiedInboxDefault(2)

    expect($showAllProfiles.get()).toBe(true)
  })

  it('leaves a single-connection registry on the legacy scope (upstream: single-connection behavior identical)', () => {
    applyUnifiedInboxDefault(1)

    expect($showAllProfiles.get()).toBe(false)
  })

  it('an explicit opt-out preference wins over the multi-connection default', () => {
    setUnifiedInboxPreference(false)

    applyUnifiedInboxDefault(3)

    expect($showAllProfiles.get()).toBe(false)
    expect(storedUnifiedInboxPreference()).toBe(false)
  })

  it('an explicit opt-in preference is never flipped back off, even for a single connection', () => {
    setUnifiedInboxPreference(true)
    setShowAllProfiles(true)

    applyUnifiedInboxDefault(1)

    expect($showAllProfiles.get()).toBe(true)
  })

  it('is idempotent: re-publishing the same registry does not toggle the scope', () => {
    applyUnifiedInboxDefault(2)
    expect($showAllProfiles.get()).toBe(true)

    applyUnifiedInboxDefault(2)
    expect($showAllProfiles.get()).toBe(true)
  })

  it('clearing the preference restores the default-on behavior', () => {
    setUnifiedInboxPreference(false)
    applyUnifiedInboxDefault(2)
    expect($showAllProfiles.get()).toBe(false)

    setUnifiedInboxPreference(null)
    setShowAllProfiles(false)
    applyUnifiedInboxDefault(2)

    expect($showAllProfiles.get()).toBe(true)
    expect(window.localStorage.getItem(UNIFIED_KEY)).toBeNull()
    expect(window.localStorage.getItem(SHOW_ALL_KEY)).toBe('true')
  })
})