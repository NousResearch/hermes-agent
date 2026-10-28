// @vitest-environment jsdom
import { act, StrictMode, useLayoutEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useEphemeralReveal } from "./useEphemeralReveal";

let root: Root;
let current: ReturnType<typeof useEphemeralReveal>;
let hidden = false;

function Harness({ scope }: { scope?: string }) {
  const state = useEphemeralReveal(scope);
  useLayoutEffect(() => { current = state; });
  return <span>{Object.values(state.revealed).join(",")}</span>;
}

function mount(scope?: string) {
  act(() => root.render(<StrictMode><Harness scope={scope} /></StrictMode>));
}

async function reveal(key = "KEY", value = "test-secret") {
  await act(async () => { await current.reveal(key, async () => value); });
}

function visibility(value: boolean) {
  hidden = value;
  act(() => document.dispatchEvent(new Event("visibilitychange")));
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  hidden = false;
  vi.spyOn(document, "hidden", "get").mockImplementation(() => hidden);
  root = createRoot(document.createElement("div"));
  mount();
});

afterEach(() => {
  act(() => root.unmount());
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("useEphemeralReveal", () => {
  it("expires each key 30 seconds after reveal, restarting only the re-revealed key", async () => {
    await reveal();
    expect(current.revealed).toEqual({ KEY: "test-secret" });
    act(() => vi.advanceTimersByTime(20_000));
    await reveal("OTHER", "other-secret");
    await reveal("KEY", "replacement");
    expect(vi.getTimerCount()).toBe(2);
    act(() => vi.advanceTimersByTime(10_000));
    expect(current.revealed).toEqual({ KEY: "replacement", OTHER: "other-secret" });
    act(() => vi.advanceTimersByTime(19_999));
    expect(current.revealed.KEY).toBe("replacement");
    act(() => vi.advanceTimersByTime(1));
    expect(current.revealed).toEqual({});
    expect(vi.getTimerCount()).toBe(0);
    await reveal();
    act(() => vi.advanceTimersByTime(10_000));
    await reveal("OTHER");
    act(() => vi.advanceTimersByTime(20_000));
    expect(current.revealed).toEqual({ OTHER: "test-secret" });
  });

  it("cancels the timer when a key is hidden", async () => {
    await reveal();
    act(() => current.hide("KEY"));
    expect(current.revealed).toEqual({});
    expect(vi.getTimerCount()).toBe(0);
  });

  it("clears all values and timers only when the document becomes hidden", async () => {
    await reveal();
    await reveal("OTHER");
    visibility(false);
    expect(Object.keys(current.revealed)).toHaveLength(2);
    visibility(true);
    expect(current.revealed).toEqual({});
    expect(vi.getTimerCount()).toBe(0);
    visibility(false);
    expect(current.revealed).toEqual({});
  });

  it("clears values and timers on window blur while the document stays visible", async () => {
    await reveal();
    await reveal("OTHER");
    act(() => { window.dispatchEvent(new Event("blur")); });
    expect(document.hidden).toBe(false);
    expect(current.revealed).toEqual({});
    expect(vi.getTimerCount()).toBe(0);
  });

  it("clears values and timers on scope change", async () => {
    await reveal();
    mount("another-profile");
    expect(current.revealed).toEqual({});
    expect(vi.getTimerCount()).toBe(0);
    await reveal("OTHER");
    expect(current.revealed.OTHER).toBe("test-secret");
  });

  it("cancels timers when the owning route or section unmounts", async () => {
    await reveal();
    await reveal("OTHER");
    act(() => root.render(null));
    expect(vi.getTimerCount()).toBe(0);
    mount();
    expect(current.revealed).toEqual({});
  });

  it.each(["hidden", "blur", "scope", "unmount", "hide"])("discards a pending response after %s", async reason => {
    let resolve!: (value: string) => void;
    const response = new Promise<string>(done => { resolve = done; });
    let pending!: Promise<void>;
    act(() => { pending = current.reveal("KEY", () => response); });
    if (reason === "hidden") { visibility(true); visibility(false); }
    if (reason === "blur") act(() => { window.dispatchEvent(new Event("blur")); });
    if (reason === "scope") mount("another-profile");
    if (reason === "unmount") act(() => root.render(null));
    if (reason === "hide") act(() => current.hide("KEY"));
    await act(async () => { resolve("late-secret"); await pending; });
    expect(current.revealed).toEqual({});
    expect(vi.getTimerCount()).toBe(0);
  });
});
