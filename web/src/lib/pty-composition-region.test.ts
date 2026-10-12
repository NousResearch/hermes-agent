import { describe, expect, it, vi } from "vitest";

import { createCompositionRegionTracker } from "./pty-composition-region";

function tracker(initial = "") {
  const textarea = { value: initial };
  const onEnd = vi.fn();
  const onCommit = vi.fn();
  return { textarea, onEnd, onCommit, t: createCompositionRegionTracker(textarea, { onEnd, onCommit }) };
}

describe("createCompositionRegionTracker", () => {
  // Device trace (Galaxy Z Fold, SwiftKey): "The" typed as three plain
  // commits, then at the space bar compositionstart/compositionupdate("The")/
  // compositionend("The") over the committed word, then the space.
  it("does not hand the forwarder a composition re-opened over committed text", () => {
    const { textarea, onEnd, onCommit, t } = tracker("The");
    t.onStart();
    t.onUpdate("The");
    expect(t.isReopened()).toBe(true);
    t.onEnd("The");
    textarea.value = "The ";
    expect(onEnd).toHaveBeenCalledOnce();
    expect(onCommit).not.toHaveBeenCalled();
  });

  it("hands the forwarder text a fresh composition added", () => {
    const { textarea, onCommit, t } = tracker("The ");
    t.onStart();
    textarea.value = "The ä";
    t.onUpdate("ä");
    expect(t.isReopened()).toBe(false);
    t.onEnd("ä");
    expect(onCommit).toHaveBeenCalledExactlyOnceWith("ä");
  });

  it("ignores a cancelled or empty composition and one that added nothing", () => {
    const { onCommit, t } = tracker("abc");
    t.onStart();
    t.onEnd("");
    t.onStart();
    t.onEnd(null);
    t.onStart();
    t.onEnd("abc");
    expect(onCommit).not.toHaveBeenCalled();
  });

  it("judges each composition on its own start state", () => {
    const { textarea, onCommit, t } = tracker("The");
    t.onStart();
    t.onUpdate("The");
    t.onEnd("The");
    textarea.value = "The ";
    t.onStart();
    textarea.value = "The w";
    t.onUpdate("w");
    textarea.value = "The world";
    t.onUpdate("world");
    t.onEnd("world");
    expect(onCommit).toHaveBeenCalledExactlyOnceWith("world");
  });
});
