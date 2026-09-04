import { describe, expect, it } from "vitest";

import type { StoredArtifact } from "./artifact-storage";
import { filterStoredArtifacts, sortStoredArtifacts } from "./artifact-library";

const artifacts: StoredArtifact[] = [
  {
    id: "one",
    sessionId: "writer-session",
    kind: "code",
    language: "typescript",
    title: "Writer helper",
    code: "export function writerHelper() {}",
    createdAt: 100,
  },
  {
    id: "two",
    sessionId: "writer-session",
    kind: "html",
    language: "html",
    title: "Landing page",
    code: "<html><body>writer</body></html>",
    createdAt: 300,
  },
  {
    id: "three",
    sessionId: "reader-session",
    kind: "code",
    language: "python",
    title: "Reader script",
    code: "print('reader')",
    createdAt: 200,
  },
];

describe("artifact library helpers", () => {
  it("filters by query, kind, and source session without mutating records", () => {
    const result = filterStoredArtifacts(artifacts, {
      query: "writer",
      kind: "code",
      sessionId: "writer-session",
    });

    expect(result.map((artifact) => artifact.id)).toEqual(["one"]);
    expect(artifacts.map((artifact) => artifact.id)).toEqual(["one", "two", "three"]);
  });

  it("sorts newest artifacts first without mutating the input", () => {
    const result = sortStoredArtifacts(artifacts);
    expect(result.map((artifact) => artifact.id)).toEqual(["two", "three", "one"]);
    expect(artifacts.map((artifact) => artifact.id)).toEqual(["one", "two", "three"]);
  });
});
