import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");

describe("structured chat dashboard route", () => {
  it("registers /chat/structured independently from the persistent PTY host", () => {
    expect(appSource).toContain('const StructuredChatPage = lazy(() => import("@/pages/StructuredChatPage"));');
    expect(appSource).toContain('"/chat/structured": StructuredChatPage');
    expect(appSource).toContain('const isChatSurface = isChatRoute || normalizedPath === "/chat/structured";');
  });
});
