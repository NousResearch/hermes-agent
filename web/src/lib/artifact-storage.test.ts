import { describe, expect, it } from "vitest";

import {
  MAX_PERSISTED_ARTIFACT_BYTES,
  isArtifactPinned,
  setArtifactPinned,
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
  id: "session-1|html|Demo app|42|prefix",
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
});
