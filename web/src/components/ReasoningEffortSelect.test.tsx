// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getReasoningEffort: vi.fn(),
  setReasoningEffort: vi.fn(),
}));
vi.mock("@/lib/api", () => ({ api: apiMocks }));

import { ReasoningEffortSelect } from "./ReasoningEffortSelect";

let container: HTMLDivElement;
let root: Root;
async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}
async function rerender(ui: ReactNode) {
  await act(async () => root.render(ui));
}
function select() { return container.querySelector("select") as HTMLSelectElement; }
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}
const data = (main_raw: string, delegation_raw = "") => ({ main_raw, delegation_raw });
const success = (scope: string, raw: string) => ({ ok: true, scope, raw });
const props = (profile = "alpha", onSaved = vi.fn()) => ({ scope: "main" as const, profile, refreshKey: 0, onSaved });

beforeEach(() => {
  (globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true;
  apiMocks.getReasoningEffort.mockReset();
  apiMocks.setReasoningEffort.mockReset();
});
afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ReasoningEffortSelect", () => {
  it("loads a saved nonempty effort into its controlled select", async () => {
    apiMocks.getReasoningEffort.mockResolvedValue(data("high"));
    await render(<ReasoningEffortSelect {...props()} />);
    expect(select().value).toBe("high");
    expect(select().getAttribute("value")).toBeNull();
    expect(select().disabled).toBe(false);
  });

  it("rolls back the optimistic choice after a failed save", async () => {
    apiMocks.getReasoningEffort.mockResolvedValue(data("medium"));
    apiMocks.setReasoningEffort.mockRejectedValue(new Error("offline"));
    await render(<ReasoningEffortSelect {...props()} />);
    await act(async () => {
      select().value = "high";
      select().dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(select().value).toBe("medium");
    expect(container.querySelector('[role="alert"]')?.textContent).toBe("Failed to save reasoning effort");
    expect(apiMocks.setReasoningEffort).toHaveBeenCalledWith("main", "high", "alpha");
  });

  it("loads the new profile and ignores a stale save completion", async () => {
    const save = deferred<ReturnType<typeof success>>();
    apiMocks.getReasoningEffort.mockImplementation((profile: string) => Promise.resolve(profile === "alpha" ? data("low") : data("xhigh")));
    apiMocks.setReasoningEffort.mockReturnValue(save.promise);
    const onSaved = vi.fn();
    await render(<ReasoningEffortSelect {...props("alpha", onSaved)} />);
    await act(async () => {
      select().value = "high";
      select().dispatchEvent(new Event("change", { bubbles: true }));
    });
    await rerender(<ReasoningEffortSelect {...props("beta", onSaved)} />);
    expect(select().value).toBe("xhigh");
    await act(async () => save.resolve(success("main", "high")));
    expect(select().value).toBe("xhigh");
    expect(onSaved).not.toHaveBeenCalled();
  });
});
