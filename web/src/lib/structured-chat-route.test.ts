import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");

describe("structured chat dashboard route", () => {
  it("sends /chat/structured to the persistent PTY host", () => {
    expect(appSource).toContain("shouldRedirectStructuredToChat");
    expect(appSource).toContain("chatLocationFromStructured");
    expect(appSource).toContain('normalizedPath === "/chat/structured"');
  });
});
