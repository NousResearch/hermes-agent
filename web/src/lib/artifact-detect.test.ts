import { describe, expect, it } from "vitest";

import { artifactDownloadName, detectArtifact } from "./artifact-detect";

describe("native chat artifact detection", () => {
  it("promotes a substantial HTML document and derives its title", () => {
    const html = `<!doctype html><html><head><title>Demo app</title></head><body><main>${"content ".repeat(30)}</main></body></html>`;
    expect(detectArtifact("html", html)).toMatchObject({ kind: "html", title: "Demo app" });
  });

  it("promotes a long source fence but leaves short snippets inline", () => {
    const source = Array.from({ length: 50 }, (_, index) => `export const line${index} = ${index};`).join("\n");
    expect(detectArtifact("typescript", source)).toMatchObject({ kind: "code", language: "typescript" });
    expect(detectArtifact("typescript", "const answer = 42;")).toBeNull();
  });

  it("does not promote prose or a tiny HTML fragment", () => {
    expect(detectArtifact("text", "This is a very long paragraph that should remain prose.")).toBeNull();
    expect(detectArtifact("html", "<p>Hello</p>")).toBeNull();
  });

  it("creates safe download names without duplicating an extension", () => {
    expect(artifactDownloadName("html", "html", "Demo app")).toBe("Demo-app.html");
    expect(artifactDownloadName("code", "typescript", "app.ts")).toBe("app.ts");
  });
});
