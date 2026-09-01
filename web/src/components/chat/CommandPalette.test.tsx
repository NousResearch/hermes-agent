// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

import { CommandPalette } from "./CommandPalette";

describe("CommandPalette", () => {
  let root: Root;
  let host: HTMLDivElement;
  const props = () => ({
    onClose: vi.fn(),
    onFocusComposer: vi.fn(),
    onNewChat: vi.fn(),
    onToggleSessions: vi.fn(),
    onInsertPrompt: vi.fn(),
    queuedCount: 0,
    onClearQueue: vi.fn(),
  });

  beforeEach(() => {
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
  });

  it("filters commands and runs the selected action", async () => {
    const callbacks = props();
    await act(async () => root.render(createElement(CommandPalette, callbacks)));
    const search = host.querySelector<HTMLInputElement>("input[aria-label='Search commands']")!;
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      setter?.call(search, "new");
      search.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(host.querySelector("button[aria-label='New chat']")).toBeTruthy();
    expect(host.querySelector("button[aria-label='Focus composer']")).toBeNull();
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='New chat']")?.click());
    expect(callbacks.onNewChat).toHaveBeenCalledTimes(1);
    expect(callbacks.onClose).toHaveBeenCalledTimes(1);
  });

  it("shows and clears the queue command only when prompts are queued", async () => {
    const callbacks = props();
    callbacks.queuedCount = 2;
    await act(async () => root.render(createElement(CommandPalette, callbacks)));
    expect(host.textContent).toContain("2 queued prompts");
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Clear prompt queue']")?.click());
    expect(callbacks.onClearQueue).toHaveBeenCalledTimes(1);
  });

  it("offers automation slash-command drafts", async () => {
    const callbacks = props();
    await act(async () => root.render(createElement(CommandPalette, callbacks)));
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Draft heartbeat command']")?.click());
    expect(callbacks.onInsertPrompt).toHaveBeenCalledWith("/heartbeat every 10m ");
    expect(callbacks.onClose).toHaveBeenCalledTimes(1);
  });
});
