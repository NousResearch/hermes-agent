/**
 * Host-internal profile bridge store.
 *
 * Mirrors the live management-profile scope (`profile`, `currentProfile`,
 * `profiles`) out of React's `ProfileProvider` into a plain module-level store
 * so the plugin SDK's `profileScope` surface (see `plugins/registry.ts`) can
 * expose it to NON-React consumers (plain-JS plugin bundles that cannot use a
 * hook) as well as React plugins.
 *
 * Why a store and not a snapshot captured in `exposePluginSDK()`:
 * `exposePluginSDK()` runs once at boot, outside any React render, so a value
 * read there would freeze and never reflect a switcher flip. `ProfileProvider`
 * writes into this store via an effect; the SDK getters read from it on access
 * and `subscribe(cb)` fires once immediately (bootstrap) and then on change.
 *
 * This module is host-internal: it is imported only by `registry.ts` and
 * `ProfileProvider.tsx` and is never named in the public `sdk.d.ts` contract.
 *
 * Write-surface fence: only the three READ fields are mirrored here; the
 * provider's `setProfile` is never stored, so the public surface stays
 * read + subscribe, one getter away from a write path.
 *
 * Array immutability: the store owns its own `profiles` array. `set()` copies
 * the caller's array in (never aliasing the provider's React-state array) and
 * the getter returns a frozen copy out, so a plugin cannot mutate host state
 * through the read-only surface (`sdk.profileScope.profiles.push(...)` is a
 * no-op-or-throw, and never reaches the stored snapshot).
 */

/** Immutable view of the host's management-profile scope. */
export interface ProfileScopeSnapshot {
  readonly profile: string;
  readonly currentProfile: string;
  readonly profiles: readonly string[];
}

type Subscriber = () => void;

// Internal mutable snapshot. Seeded with a safe empty default so a read before
// `ProfileProvider` mounts never throws or returns undefined. The empty default
// is indistinguishable from a genuine "no profiles / own-profile" state, so a
// consumer that needs the live value must subscribe; the bootstrap call plus
// change notifications make subscribing alone sufficient (no polling, no
// re-read-after-paint choreography).
let _snapshot: { profile: string; currentProfile: string; profiles: string[] } = {
  profile: "",
  currentProfile: "",
  profiles: [],
};

const _subscribers: Set<Subscriber> = new Set();

/** Element-wise array compare (length + each index): add / remove / reorder all differ. */
function _profilesEqual(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export const profileBridge = {
  /** Current snapshot. `profiles` is a fresh frozen copy (caller cannot mutate the store). */
  get(): ProfileScopeSnapshot {
    return {
      profile: _snapshot.profile,
      currentProfile: _snapshot.currentProfile,
      profiles: Object.freeze([..._snapshot.profiles]),
    };
  },

  /**
   * Replace the snapshot and notify, but only when a field actually changed
   * (no spurious callback storms). All three fields are equal-weight: a
   * `currentProfile`-only delta notifies. `profile` / `currentProfile` compare
   * by string equality; `profiles` compares element-wise.
   */
  set(value: ProfileScopeSnapshot): void {
    const changed =
      value.profile !== _snapshot.profile ||
      value.currentProfile !== _snapshot.currentProfile ||
      !_profilesEqual(value.profiles, _snapshot.profiles);
    if (!changed) return;

    // Ordering invariant: replace the snapshot BEFORE notifying, so a callback
    // that re-reads `get()` observes the new value. Copy the array in so the
    // provider's React-state array is never aliased or mutable via the SDK.
    _snapshot = {
      profile: value.profile,
      currentProfile: value.currentProfile,
      profiles: [...value.profiles],
    };

    // Iterate a snapshot copy of the subscriber set so a callback that
    // subscribes / unsubscribes mid-notification affects only later flips,
    // never the in-flight one. Isolate each callback so one thrower cannot
    // starve siblings or escape into the provider's effect.
    for (const cb of [..._subscribers]) {
      try {
        cb();
      } catch (err) {
        console.error("[hermes] profileScope subscriber threw:", err);
      }
    }
  },

  /**
   * Register a zero-arg callback; returns an idempotent unsubscribe thunk.
   *
   * Bootstrap delivery: the callback fires once, synchronously, at subscribe
   * time. Plugin bundles load asynchronously (see `usePlugins.ts`), so a
   * consumer can subscribe AFTER the provider's first `set()`; without the
   * bootstrap call it would read nothing until the next switcher flip. Firing
   * immediately makes "subscribe and read the getters in the callback" a
   * complete consumption pattern with no missable window between a `get()`
   * and a `subscribe()`. Same isolation as a change dispatch: a throwing
   * callback is logged, never escapes, and stays registered.
   */
  subscribe(cb: Subscriber): () => void {
    _subscribers.add(cb);
    try {
      cb();
    } catch (err) {
      console.error("[hermes] profileScope subscriber threw:", err);
    }
    return () => {
      _subscribers.delete(cb);
    };
  },
};
