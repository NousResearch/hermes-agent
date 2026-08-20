/**
 * Type-level contract guard for the plugin SDK `profileScope` surface.
 *
 * Regression for the exact bug this surface exists to prevent: shipping a
 * shape a plain-JS plugin cannot consume. The motivating non-React consumer
 * probes `typeof sdk.profileScope === "object"` and calls `.subscribe(cb)`, so
 * the field MUST be an OBJECT with `profile` / `currentProfile` / `profiles`
 * plus a `subscribe(cb: () => void): () => void` method, never a bare hook.
 *
 * This file has no runtime (it is not picked up by the Vitest `*.test.ts`
 * glob). It fails `npm run typecheck` (tsc) if someone "simplifies"
 * `profileScope` back to a function, softens `subscribe`, or widens a field.
 */
import type { HermesPluginSDK, ProfileScopeValue } from "./sdk";

// Compile-time exact-equality assertion helpers.
type Equal<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends <T>() => T extends Y ? 1 : 2
    ? true
    : false;
type Expect<T extends true> = T;

// The field is optional (`profileScope?:`); assert against the non-optional shape.
type Scope = NonNullable<HermesPluginSDK["profileScope"]>;

/**
 * Each element is `Expect<true>`; any failing assertion collapses to
 * `Expect<false>`, which is a type error. Exported so it is never flagged as
 * an unused declaration.
 */
export type ProfileScopeContractAssertions = [
  // 1. The exposed shape is exactly ProfileScopeValue (object + subscribe).
  Expect<Equal<Scope, ProfileScopeValue>>,
  // 2. It is an object, NOT callable: a bare hook `() => ProfileScopeValue`
  //    would have a call signature; assert Scope has none.
  Expect<Equal<Scope extends (...args: never[]) => unknown ? true : false, false>>,
  // 3. Full subscribe signature: zero-arg `() => void` callback in, unsubscribe
  //    `() => void` out. Blocks softening to a snapshot-arg callback or a
  //    `void` return.
  Expect<Equal<Scope["subscribe"], (cb: () => void) => () => void>>,
  // 4. Fields are correctly typed; `profiles` is `readonly string[]` (contents
  //    immutable), not `string[]`.
  Expect<Equal<Scope["profile"], string>>,
  Expect<Equal<Scope["currentProfile"], string>>,
  Expect<Equal<Scope["profiles"], readonly string[]>>,
];
