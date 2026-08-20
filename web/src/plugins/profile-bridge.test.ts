import { describe, it, expect, vi } from "vitest";
import type { ProfileScopeSnapshot } from "./profile-bridge";

// Each test gets a fresh module instance so module-level store state (the
// snapshot and the subscriber set) never leaks across cases. The seed-default
// case in particular needs a pristine store.
async function freshBridge() {
  vi.resetModules();
  return (await import("./profile-bridge")).profileBridge;
}

describe("profileBridge", () => {
  it("get() returns the seeded empty default before any set()", async () => {
    const bridge = await freshBridge();
    expect(bridge.get()).toEqual({
      profile: "",
      currentProfile: "",
      profiles: [],
    });
  });

  it("set() replaces the snapshot BEFORE notifying (ordering invariant)", async () => {
    const bridge = await freshBridge();
    const seen: ProfileScopeSnapshot[] = [];
    bridge.subscribe(() => {
      // A zero-arg callback that re-reads get() must observe the NEW value.
      seen.push(bridge.get());
    });
    // seen[0] is the bootstrap delivery of the seed; the change lands after it.
    bridge.set({
      profile: "alice",
      currentProfile: "default",
      profiles: ["default", "alice"],
    });
    expect(seen).toHaveLength(2);
    expect(seen[0].profile).toBe("");
    expect(seen[1].profile).toBe("alice");
    expect(seen[1].currentProfile).toBe("default");
    expect([...seen[1].profiles]).toEqual(["default", "alice"]);
  });

  it("subscribe() after set() immediately delivers the current state (bootstrap)", async () => {
    // Plugin bundles load asynchronously, so subscribing AFTER the provider's
    // first set() is the normal case for a plain-JS consumer. The bootstrap
    // call must hand it the live value; nothing waits for the next flip.
    const bridge = await freshBridge();
    bridge.set({
      profile: "alice",
      currentProfile: "default",
      profiles: ["default", "alice"],
    });

    const seen: ProfileScopeSnapshot[] = [];
    bridge.subscribe(() => {
      seen.push(bridge.get());
    });

    expect(seen).toHaveLength(1);
    expect(seen[0].profile).toBe("alice");
    expect(seen[0].currentProfile).toBe("default");
    expect([...seen[0].profiles]).toEqual(["default", "alice"]);
  });

  it("a throwing bootstrap callback neither escapes subscribe() nor loses registration", async () => {
    const bridge = await freshBridge();
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const cb = vi.fn(() => {
      throw new Error("boom at bootstrap");
    });

    expect(() => bridge.subscribe(cb)).not.toThrow();
    expect(cb).toHaveBeenCalledTimes(1); // bootstrap attempt happened
    expect(errSpy).toHaveBeenCalledTimes(1);

    // Still registered: the next real change reaches it.
    bridge.set({ profile: "alice", currentProfile: "d", profiles: [] });
    expect(cb).toHaveBeenCalledTimes(2);
    errSpy.mockRestore();
  });

  it("notifies on a currentProfile-only change (fields are equal-weight)", async () => {
    const bridge = await freshBridge();
    bridge.set({ profile: "", currentProfile: "default", profiles: [] });
    const cb = vi.fn();
    bridge.subscribe(cb);
    cb.mockClear(); // drop the bootstrap call; this test counts change dispatches
    bridge.set({ profile: "", currentProfile: "gibson", profiles: [] });
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it("notifies on profiles add / remove / reorder (element-wise compare)", async () => {
    const bridge = await freshBridge();
    bridge.set({ profile: "", currentProfile: "d", profiles: ["a", "b"] });
    const cb = vi.fn();
    bridge.subscribe(cb);
    cb.mockClear(); // drop the bootstrap call; this test counts change dispatches
    bridge.set({ profile: "", currentProfile: "d", profiles: ["a", "b", "c"] }); // add
    bridge.set({ profile: "", currentProfile: "d", profiles: ["a", "b"] }); // remove
    bridge.set({ profile: "", currentProfile: "d", profiles: ["b", "a"] }); // reorder
    expect(cb).toHaveBeenCalledTimes(3);
  });

  it("does NOT notify when all three fields are unchanged", async () => {
    const bridge = await freshBridge();
    bridge.set({ profile: "x", currentProfile: "d", profiles: ["a"] });
    const cb = vi.fn();
    bridge.subscribe(cb);
    cb.mockClear(); // drop the bootstrap call; this test counts change dispatches
    bridge.set({ profile: "x", currentProfile: "d", profiles: ["a"] }); // identical
    expect(cb).not.toHaveBeenCalled();
  });

  it("keeps subscribe/unsubscribe idempotent and isolates the in-flight dispatch", async () => {
    const bridge = await freshBridge();
    const order: string[] = [];
    let unsubB: () => void = () => {};
    // `a` unsubscribes `b` mid-notification; because notify iterates a
    // [...subs] copy captured before dispatch, `b` still runs THIS flip.
    bridge.subscribe(() => {
      order.push("a");
      unsubB();
    });
    unsubB = bridge.subscribe(() => {
      order.push("b");
    });
    order.length = 0; // drop the two bootstrap calls; this test tracks dispatches

    bridge.set({ profile: "1", currentProfile: "d", profiles: [] });
    expect(order).toEqual(["a", "b"]);

    // Next flip: b is gone.
    order.length = 0;
    bridge.set({ profile: "2", currentProfile: "d", profiles: [] });
    expect(order).toEqual(["a"]);

    // Double-unsubscribe is a no-op (Set delete is idempotent).
    expect(() => {
      unsubB();
      unsubB();
    }).not.toThrow();
  });

  it("isolates a throwing subscriber so siblings still run", async () => {
    const bridge = await freshBridge();
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const calls: string[] = [];
    bridge.subscribe(() => {
      calls.push("first");
      throw new Error("boom");
    });
    bridge.subscribe(() => {
      calls.push("second");
    });
    calls.length = 0; // drop bootstrap calls; this test tracks the dispatch
    errSpy.mockClear(); // (the first subscriber already threw once at bootstrap)

    expect(() =>
      bridge.set({ profile: "z", currentProfile: "d", profiles: [] }),
    ).not.toThrow();
    expect(calls).toEqual(["first", "second"]);
    expect(errSpy).toHaveBeenCalledTimes(1);
    errSpy.mockRestore();
  });

  it("returns a frozen profiles copy that cannot mutate the store", async () => {
    const bridge = await freshBridge();
    bridge.set({ profile: "", currentProfile: "d", profiles: ["a", "b"] });

    const got = bridge.get().profiles;
    expect(Object.isFrozen(got)).toBe(true);
    expect(() => {
      (got as string[]).push("c");
    }).toThrow();
    // The internal snapshot is untouched by the attempted mutation.
    expect([...bridge.get().profiles]).toEqual(["a", "b"]);

    // Copy-in: mutating the caller's array after set() must not affect the store.
    const input = ["a", "b"];
    bridge.set({ profile: "p", currentProfile: "d", profiles: input });
    input.push("MUTATED");
    expect([...bridge.get().profiles]).toEqual(["a", "b"]);
  });
});
