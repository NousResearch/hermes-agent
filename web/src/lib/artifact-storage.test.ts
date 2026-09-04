import { describe, expect, it } from "vitest";

import {
  LEGACY_PROFILE_ARTIFACT_STORAGE_PREFIX,
  MAX_PERSISTED_ARTIFACT_BYTES,
  artifactStorageBytes,
  forgetArtifactProfileScope,
  getArtifactStorageKey,
  isArtifactPinned,
  makeArtifactId,
  pruneArtifactProfileIdentities,
  readStoredArtifacts,
  setArtifactPinned,
  clearStoredArtifacts,
  type StoredArtifact,
} from "./artifact-storage";

function memoryStorage(): Storage {
  const values = new Map<string, string>();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value); },
    removeItem: (key) => { values.delete(key); },
    clear: () => values.clear(),
    key: (index) => Array.from(values.keys())[index] ?? null,
    get length() { return values.size; },
  };
}

const artifact: StoredArtifact = {
  id: makeArtifactId("session-1", "html", "html", "Demo app", "<html>demo</html>"),
  sessionId: "session-1",
  kind: "html",
  language: "html",
  title: "Demo app",
  code: "<html>demo</html>",
  createdAt: 1700000000000,
};

describe("persistent native chat artifacts", () => {
  it("pins and unpins artifacts in browser storage", () => {
    const storage = memoryStorage();
    expect(isArtifactPinned(storage, artifact.id)).toBe(false);
    expect(setArtifactPinned(storage, artifact, true)).toBe(true);
    expect(isArtifactPinned(storage, artifact.id)).toBe(true);
    expect(setArtifactPinned(storage, artifact, false)).toBe(true);
    expect(isArtifactPinned(storage, artifact.id)).toBe(false);
  });

  it("rejects artifacts larger than the persistence budget", () => {
    const storage = memoryStorage();
    const oversized = { ...artifact, code: "x".repeat(MAX_PERSISTED_ARTIFACT_BYTES + 1) };
    expect(setArtifactPinned(storage, oversized, true)).toBe(false);
    expect(isArtifactPinned(storage, oversized.id)).toBe(false);
  });

  it("recovers from malformed storage without throwing", () => {
    const storage = memoryStorage();
    storage.setItem("hermes.native-chat.artifacts.v1", "not json");
    expect(isArtifactPinned(storage, artifact.id)).toBe(false);
  });

  it("does not collide when only the middle of two artifacts differs", () => {
    const prefix = "a".repeat(80);
    const suffix = "b".repeat(80);
    const first = `${prefix}x${suffix}`;
    const second = `${prefix}y${suffix}`;

    expect(makeArtifactId("session-1", "code", "ts", "demo", first))
      .not.toBe(makeArtifactId("session-1", "code", "ts", "demo", second));
  });

  it("rebinds legacy fingerprint IDs to the current content ID", () => {
    const storage = memoryStorage();
    const legacyArtifact = { ...artifact, id: "session-1|html|Demo app|42|prefix" };
    storage.setItem("hermes.native-chat.artifacts.v1", JSON.stringify([legacyArtifact]));

    const migrated = readStoredArtifacts(storage, "");
    expect(migrated[0]?.id).toBe(
      makeArtifactId(artifact.sessionId, artifact.kind, artifact.language, artifact.title, artifact.code),
    );
    expect(migrated[0]?.id).not.toBe(legacyArtifact.id);
  });

  it("rejects out-of-range timestamps and enforces limits during legacy migration", () => {
    const storage = memoryStorage();
    const invalidTimestamp = { ...artifact, id: "invalid-date", createdAt: 8_640_000_000_000_001 };
    const oversized = { ...artifact, id: "oversized", code: "x".repeat(MAX_PERSISTED_ARTIFACT_BYTES + 1) };
    storage.setItem("hermes.native-chat.artifacts.v1", JSON.stringify([invalidTimestamp, oversized, artifact]));

    expect(readStoredArtifacts(storage, "")).toEqual([artifact]);
  });

  it("keeps migrated records within the profile total budget", () => {
    const storage = memoryStorage();
    const legacy = Array.from({ length: 11 }, (_, index) => ({
      ...artifact,
      id: `legacy-${index}`,
      sessionId: `legacy-session-${index}`,
      code: "x".repeat(190_000),
      createdAt: index,
    }));
    storage.setItem("hermes.native-chat.artifacts.v1", JSON.stringify(legacy));

    const migrated = readStoredArtifacts(storage, "");
    expect(artifactStorageBytes(migrated)).toBeLessThanOrEqual(2_000_000);
    expect(migrated.length).toBeLessThan(legacy.length);
  });

  it("migrates profile-scoped v2 records into an opaque v3 key", () => {
    const storage = memoryStorage();
    const legacyKey = `${LEGACY_PROFILE_ARTIFACT_STORAGE_PREFIX}writer`;
    storage.setItem(legacyKey, JSON.stringify([artifact]));

    expect(readStoredArtifacts(storage, "writer")).toEqual([artifact]);
    expect(storage.getItem(legacyKey)).toBeNull();
    expect(storage.getItem(getArtifactStorageKey("writer", storage))).toContain("Demo app");
  });

  it("rotates a profile identity when its profile scope is forgotten", () => {
    const storage = memoryStorage();
    const oldKey = getArtifactStorageKey("writer", storage);
    expect(setArtifactPinned(storage, artifact, true, "writer")).toBe(true);
    expect(forgetArtifactProfileScope(storage, "writer")).toBe(true);

    expect(storage.getItem(oldKey)).toBeNull();
    expect(getArtifactStorageKey("writer", storage)).not.toBe(oldKey);
    expect(readStoredArtifacts(storage, "writer")).toEqual([]);
  });

  it("treats empty and explicit default profile selectors as one scope", () => {
    const storage = memoryStorage();
    expect(setArtifactPinned(storage, artifact, true, "")).toBe(true);
    expect(isArtifactPinned(storage, artifact.id, "default")).toBe(true);
    expect(getArtifactStorageKey("", storage)).toBe(getArtifactStorageKey("default", storage));
  });

  it("clears v1, v2, and v3 records for an explicit clear", () => {
    const storage = memoryStorage();
    storage.setItem("hermes.native-chat.artifacts.v1", JSON.stringify([artifact]));
    storage.setItem(`${LEGACY_PROFILE_ARTIFACT_STORAGE_PREFIX}writer`, JSON.stringify([artifact]));
    setArtifactPinned(storage, artifact, true, "writer");

    expect(clearStoredArtifacts(storage, "writer").ok).toBe(true);
    expect(storage.getItem("hermes.native-chat.artifacts.v1")).not.toBeNull();
    expect(storage.getItem(`${LEGACY_PROFILE_ARTIFACT_STORAGE_PREFIX}writer`)).toBeNull();
    expect(readStoredArtifacts(storage, "writer")).toEqual([]);

    expect(clearStoredArtifacts(storage, "").ok).toBe(true);
    expect(storage.getItem("hermes.native-chat.artifacts.v1")).toBeNull();
  });

  it("preserves a pin written by another tab between read and write", () => {
    const storage = memoryStorage();
    const other = {
      ...artifact,
      id: makeArtifactId("other-session", "code", "ts", "Other", "const other = true;"),
      sessionId: "other-session",
      kind: "code" as const,
      language: "ts",
      title: "Other",
      code: "const other = true;",
    };
    const target = { ...artifact, id: makeArtifactId("writer-session", "code", "ts", "Writer", "const writer = true;"), sessionId: "writer-session", kind: "code" as const, language: "ts", title: "Writer", code: "const writer = true;" };
    const key = getArtifactStorageKey("writer", storage);
    const originalGetItem = storage.getItem;
    const originalSetItem = storage.setItem;
    let targetReads = 0;
    storage.getItem = (candidateKey) => {
      const value = originalGetItem(candidateKey);
      if (candidateKey === key && targetReads++ === 0) {
        originalSetItem(key, JSON.stringify([other]));
      }
      return value;
    };

    expect(setArtifactPinned(storage, target, true, "writer")).toBe(true);
    storage.getItem = originalGetItem;
    expect(readStoredArtifacts(storage, "writer").map((entry) => entry.id)).toEqual([other.id, target.id]);
  });

  it("prunes a deleted profile identity so a recreated name starts empty", () => {
    const storage = memoryStorage();
    const oldKey = getArtifactStorageKey("writer", storage);
    expect(setArtifactPinned(storage, artifact, true, "writer")).toBe(true);

    expect(pruneArtifactProfileIdentities(storage, ["default"])).toBe(true);
    expect(storage.getItem(oldKey)).toBeNull();
    expect(getArtifactStorageKey("writer", storage)).not.toBe(oldKey);
    expect(readStoredArtifacts(storage, "writer")).toEqual([]);
  });
});
