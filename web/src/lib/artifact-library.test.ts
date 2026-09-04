import { describe, expect, it } from "vitest";

import {
  MAX_PERSISTED_ARTIFACT_STORAGE_BYTES,
  artifactStorageBytes,
  clearStoredArtifacts,
  LEGACY_ARTIFACT_STORAGE_KEY,
  getArtifactStorageKey,
  makeArtifactId,
  readStoredArtifacts,
  setArtifactPinned,
  setArtifactPinnedWithResult,
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
  id: makeArtifactId("default-session", "code", "ts", "demo", "const demo = true;"),
  sessionId: "default-session",
  kind: "code",
  language: "ts",
  title: "demo",
  code: "const demo = true;",
  createdAt: 1700000000000,
};

describe("profile-scoped artifact library storage", () => {
  it("migrates legacy artifacts into the default scope and isolates named profiles", () => {
    const storage = memoryStorage();
    storage.setItem(LEGACY_ARTIFACT_STORAGE_KEY, JSON.stringify([artifact]));

    expect(readStoredArtifacts(storage, "")).toEqual([artifact]);
    expect(readStoredArtifacts(storage, "writer")).toEqual([]);
    expect(storage.getItem(LEGACY_ARTIFACT_STORAGE_KEY)).toBeNull();

    const writerArtifact = {
      ...artifact,
      id: makeArtifactId("writer-session", artifact.kind, artifact.language, artifact.title, artifact.code),
      sessionId: "writer-session",
    };
    expect(setArtifactPinned(storage, writerArtifact, true, "writer")).toBe(true);
    expect(readStoredArtifacts(storage, "")).toEqual([artifact]);
    expect(readStoredArtifacts(storage, "writer")).toEqual([writerArtifact]);
    expect(storage.getItem(getArtifactStorageKey("writer"))).toContain("writer-session");
  });

  it("rejects a profile total beyond the storage budget and clears it explicitly", () => {
    const storage = memoryStorage();
    const bulky = (index: number): StoredArtifact => ({
      ...artifact,
      id: `bulk-${index}`,
      sessionId: `bulk-session-${index}`,
      code: "x".repeat(190_000),
      createdAt: index,
    });

    for (let index = 0; index < 10; index += 1) {
      expect(setArtifactPinned(storage, bulky(index), true, "writer")).toBe(true);
    }
    const current = readStoredArtifacts(storage, "writer");
    expect(artifactStorageBytes(current)).toBeLessThanOrEqual(MAX_PERSISTED_ARTIFACT_STORAGE_BYTES);

    const rejected = setArtifactPinnedWithResult(storage, bulky(10), true, "writer");
    expect(rejected).toMatchObject({ ok: false, reason: "storage-limit" });
    expect(readStoredArtifacts(storage, "writer")).toHaveLength(10);

    expect(clearStoredArtifacts(storage, "writer")).toMatchObject({ ok: true, artifacts: [] });
    expect(readStoredArtifacts(storage, "writer")).toEqual([]);
  });

  it("reports an empty profile as using zero bytes", () => {
    expect(artifactStorageBytes([])).toBe(0);
  });
});
