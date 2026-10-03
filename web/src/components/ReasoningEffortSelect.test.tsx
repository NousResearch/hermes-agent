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
const data = (main_raw: string, delegation_raw = "", extra: Record<string, unknown> = {}) => ({
  main_raw, delegation_raw, main_effective: main_raw || "medium", main_source: "global",
  main_model: "provider/model", ...extra,
});
const success = (scope: string, raw: string) => ({ ok: true, scope, raw });
const props = (profile = "alpha", onSaved = vi.fn()) => ({ scope: "main" as const, profile, refreshKey: 0, onSaved });

beforeEach(() => {
  (globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true;
  apiMocks.getReasoningEffort.mockReset();
  apiMocks.setReasoningEffort.mockReset();
  apiMocks.setReasoningEffort.mockImplementation((_scope: string, effort: string) => Promise.resolve(success("main", effort)));
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
    expect(apiMocks.setReasoningEffort).toHaveBeenCalledWith("main", "high", "alpha", "global", undefined);
  });

  it("shows effective override, edits through a draft, and uses the unchanged main model", async () => {
    apiMocks.getReasoningEffort.mockResolvedValue(data("__custom__", "", {
      main_custom: "budgeted", main_effective: "budgeted", main_source: "model_override",
    }));
    apiMocks.setReasoningEffort.mockResolvedValue({ ...data("high", "", { main_effective: "high", main_source: "model_override" }), ...success("main", "high") });
    const onSaved = vi.fn();
    await render(<ReasoningEffortSelect {...props("alpha", onSaved)} />);
    expect(container.textContent).toContain("Effective for provider/model: budgeted (model override)");
    expect(container.textContent).toContain("Configured custom value: budgeted");
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Edit model override")!.click());
    expect(apiMocks.setReasoningEffort).not.toHaveBeenCalled();
    const draftSelect = container.querySelector('[aria-label="Model override effort"]') as HTMLSelectElement;
    expect(draftSelect.value).toBe("none");
    expect(container.textContent).toContain("Configured custom value: budgeted");
    await act(async () => {
      draftSelect.value = "high";
      draftSelect.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Save")!.click());
    expect(apiMocks.setReasoningEffort).toHaveBeenLastCalledWith("main", "high", "alpha", "model", "provider/model");
    expect(onSaved).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain("Global reasoning default: high");
  });

  it("Cancel discards edit without calling the API", async () => {
    apiMocks.getReasoningEffort.mockResolvedValue(data("high", "", { main_effective: "high", main_source: "model_override" }));
    await render(<ReasoningEffortSelect {...props()} />);
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Edit model override")!.click());
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Cancel")!.click());
    expect(apiMocks.setReasoningEffort).not.toHaveBeenCalled();
    expect(container.querySelector('[aria-label="Model override effort"]')).toBeNull();
  });

  it("uses global without changing its displayed value optimistically", async () => {
    const save = deferred<any>();
    apiMocks.getReasoningEffort.mockResolvedValue(data("medium", "", { main_effective: "high", main_source: "model_override" }));
    apiMocks.setReasoningEffort.mockReturnValue(save.promise);
    await render(<ReasoningEffortSelect {...props()} />);
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Use global setting")!.click());
    expect(apiMocks.setReasoningEffort).toHaveBeenCalledWith("main", "", "alpha", "model", "provider/model");
    expect(container.textContent).toContain("Global reasoning default: medium");
    await act(async () => save.resolve({ ...data("medium", "", { main_effective: "medium", main_source: "global" }), ...success("main", "") }));
    expect(container.textContent).toContain("Global reasoning default: medium");
  });

  it("submits Disabled using the none payload", async () => {
    apiMocks.getReasoningEffort.mockResolvedValue(data("high"));
    await render(<ReasoningEffortSelect {...props()} />);
    expect([...select().options].filter((option) => option.value === "none")).toHaveLength(1);
    expect([...select().options].some((option) => option.value === "disabled")).toBe(false);
    expect([...select().options].find((option) => option.value === "none")?.textContent).toBe("Disabled");
    await act(async () => {
      select().value = "none";
      select().dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(apiMocks.setReasoningEffort).toHaveBeenCalledWith("main", "none", "alpha", "global", undefined);
  });


  it("renders and saves a custom reasoning value without treating it as disabled", async () => {
    apiMocks.getReasoningEffort.mockResolvedValue(data("__custom__", "", {
      main_custom: "vendor-special", main_source: "global",
    }));
    await render(<ReasoningEffortSelect {...props()} />);
    expect(select().value).toBe("__custom__");
    expect(container.textContent).toContain("Configured custom value: vendor-special");
    expect(container.textContent).toContain("Global reasoning default: vendor-special");
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

  it("ignores a save completion after the profile changes", async () => {
    const save = deferred<ReturnType<typeof success>>();
    apiMocks.getReasoningEffort.mockImplementation((profile: string) => Promise.resolve(data(profile === "alpha" ? "low" : "high")));
    apiMocks.setReasoningEffort.mockReturnValue(save.promise);
    const onSaved = vi.fn();
    await render(<ReasoningEffortSelect {...props("alpha", onSaved)} />);
    await act(async () => {
      select().value = "medium";
      select().dispatchEvent(new Event("change", { bubbles: true }));
    });
    await rerender(<ReasoningEffortSelect {...props("beta", onSaved)} />);
    await act(async () => save.resolve(success("main", "medium")));
    expect(select().value).toBe("high");
    expect(onSaved).not.toHaveBeenCalled();
  });
});
