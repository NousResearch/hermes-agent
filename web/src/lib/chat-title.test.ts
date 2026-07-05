import { describe, expect, it } from "vitest";

import { normalizeSessionTitle, titleFromSessionInfoPayload } from "./chat-title";

describe("normalizeSessionTitle", () => {
  it("trims non-empty session titles", () => {
    expect(normalizeSessionTitle("  Rename the dashboard  ")).toBe(
      "Rename the dashboard",
    );
  });

  it("treats blank and non-string values as no title", () => {
    expect(normalizeSessionTitle("   ")).toBeNull();
    expect(normalizeSessionTitle(null)).toBeNull();
    expect(normalizeSessionTitle(42)).toBeNull();
  });
});

describe("titleFromSessionInfoPayload", () => {
  it("returns undefined when the payload has no title field", () => {
    expect(titleFromSessionInfoPayload({ model: "test/model" })).toBeUndefined();
    expect(titleFromSessionInfoPayload(null)).toBeUndefined();
  });

  it("returns null when the title field is present but empty", () => {
    expect(titleFromSessionInfoPayload({ title: "" })).toBeNull();
    expect(titleFromSessionInfoPayload({ title: "   " })).toBeNull();
  });

  it("returns the normalized title when present", () => {
    expect(titleFromSessionInfoPayload({ title: "  Live session title " })).toBe(
      "Live session title",
    );
  });
});
