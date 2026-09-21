import { describe, expect, it } from "vitest";
import { editDraft, emptyDraft } from "./draft";

describe("creation draft", () => {
  it("invalidates a reviewed plan and request when an input changes", () => {
    const draft = { ...emptyDraft(), spaceId: "space-a", outputNodeIds: ["node-a"], plan: { planId: "p" } as never, requestId: "request-a" };
    const changed = editDraft(draft, { overrides: { prompt: { prompt: "new" } } });
    expect(changed.spaceId).toBe("space-a");
    expect(changed.plan).toBeNull();
    expect(changed.requestId).toBeNull();
  });
  it("keeps original draft immutable", () => {
    const original = emptyDraft();
    const changed = editDraft(original, { spaceId: "space-a" });
    expect(original.spaceId).toBeNull();
    expect(changed.spaceId).toBe("space-a");
  });
});
