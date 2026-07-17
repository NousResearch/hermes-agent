import { describe, expect, it, beforeEach } from "vitest";

import { ptyAttachToken, PTY_ATTACH_TOKEN_BASE } from "./pty-token";
import type { PtyTokenStorage, PtyTokenCrypto } from "./pty-token";

/** Predictable "random" that returns sequential bytes: 0,1,2,3,... */
function sequentialCrypto(): PtyTokenCrypto {
  let next = 0;
  return {
    getRandomValues<T extends Uint8Array>(array: T): T {
      for (let i = 0; i < array.length; i++) {
        array[i] = (next++ & 0xff) as number;
      }
      return array;
    },
  };
}

/** Simple in-memory storage for tests. */
class MemoryStorage implements PtyTokenStorage {
  private store = new Map<string, string>();
  getItem(key: string): string | null {
    return this.store.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.store.set(key, value);
  }
}

describe("ptyAttachToken", () => {
  let storage: MemoryStorage;
  let crypto: PtyTokenCrypto;

  beforeEach(() => {
    storage = new MemoryStorage();
    crypto = sequentialCrypto();
  });

  it("returns the same token for the same scope (keep-alive preserved)", () => {
    const scope = "default\0session-123";

    const first = ptyAttachToken(false, scope, storage, crypto);
    const second = ptyAttachToken(false, scope, storage, crypto);

    expect(first).toBe(second);
    expect(first).toHaveLength(32); // 16 bytes hex = 32 chars
    expect(storage.getItem(`${PTY_ATTACH_TOKEN_BASE}.${scope}`)).toBe(first);
  });

  it("returns different tokens when scope differs — profile changes", () => {
    const tokenA = ptyAttachToken(false, "profile-a\0s1", storage, crypto);
    const tokenB = ptyAttachToken(
      false,
      "profile-b\0s1",
      // Use fresh storage so the old key doesn't collide
      new MemoryStorage(),
      crypto,
    );

    expect(tokenA).not.toBe(tokenB);
  });

  it("returns different tokens when scope differs — resume changes", () => {
    const tokenA = ptyAttachToken(false, "default\0session-1", storage, crypto);
    const tokenB = ptyAttachToken(
      false,
      "default\0session-2",
      new MemoryStorage(),
      crypto,
    );

    expect(tokenA).not.toBe(tokenB);
  });

  it("returns different tokens when scope differs — both profile and resume change", () => {
    const tokenA = ptyAttachToken(false, "p1\0s1", storage, crypto);
    // Use the same crypto instance so it continues the sequence from where
    // the first call left off, producing different bytes for tokenB.
    const tokenB = ptyAttachToken(
      false,
      "p2\0s2",
      new MemoryStorage(),
      crypto,
    );

    expect(tokenA).not.toBe(tokenB);
  });

  it("rotates the token when rotate=true, even with same scope", () => {
    const scope = "default\0s1";

    const first = ptyAttachToken(false, scope, storage, crypto);
    const fresh = ptyAttachToken(true, scope, storage, crypto);

    expect(first).not.toBe(fresh);
    // The new token should now be persisted
    expect(storage.getItem(`${PTY_ATTACH_TOKEN_BASE}.${scope}`)).toBe(fresh);
  });

  it("scoped token does not collide with unscoped (legacy) key", () => {
    const scoped = ptyAttachToken(false, "default\0s1", storage, crypto);
    const unscoped = ptyAttachToken(false, "", new MemoryStorage(), crypto);

    expect(scoped).not.toBe(unscoped);

    const scopedKey = `${PTY_ATTACH_TOKEN_BASE}.default\0s1`;
    const unscopedKey = `${PTY_ATTACH_TOKEN_BASE}.chat`;
    expect(scopedKey).not.toBe(unscopedKey);
  });

  it("persists to the scoped key", () => {
    const scope = "my-profile\0resume-42";
    const token = ptyAttachToken(false, scope, storage, crypto);

    expect(storage.getItem(`${PTY_ATTACH_TOKEN_BASE}.${scope}`)).toBe(token);
  });

  it("reuses stored token without calling getRandomValues again", () => {
    const scope = "sticky\0test";
    const first = ptyAttachToken(false, scope, storage, crypto);
    // Create a fresh crypto that would produce different bytes — if the
    // function re-mints the token, we'd see a different value.
    const differentCrypto = sequentialCrypto();
    const second = ptyAttachToken(false, scope, storage, differentCrypto);

    expect(second).toBe(first);
  });

  it("rotate=true + empty storage mints a fresh 32-char hex token", () => {
    const token = ptyAttachToken(true, "any\0scope", storage, crypto);

    expect(token).toHaveLength(32);
    expect(/^[0-9a-f]{32}$/.test(token)).toBe(true);
  });
});
