// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const clipboard = vi.hoisted(() => ({
  copyTextToClipboard: vi.fn(),
}));

vi.mock("@/lib/clipboard", () => clipboard);

import { MessageActions } from "./MessageActions";

describe("MessageActions", () => {
  let root: Root;
  let host: HTMLDivElement;

  beforeEach(() => {
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
    clipboard.copyTextToClipboard.mockReset();
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
  });

  it("copies the exact assistant message and exposes success feedback", async () => {
    clipboard.copyTextToClipboard.mockResolvedValue(true);
    await act(async () => root.render(createElement(MessageActions, {
      message: "## Answer\n\nKeep this exact text.",
      messageRole: "assistant",
      onUseAsPrompt: vi.fn(),
    })));

    const copyButton = host.querySelector<HTMLButtonElement>("button[aria-label='Copy assistant message']");
    expect(copyButton).toBeTruthy();
    await act(async () => copyButton?.click());

    expect(clipboard.copyTextToClipboard).toHaveBeenCalledWith("## Answer\n\nKeep this exact text.");
    expect(copyButton?.textContent).toContain("Copied");
    expect(host.querySelector("[role='status']")?.textContent).toContain("Copied");
  });

  it("passes the exact user message to the draft callback and announces it", async () => {
    const onUseAsPrompt = vi.fn();
    await act(async () => root.render(createElement(MessageActions, {
      message: "Keep this as a follow-up prompt",
      messageRole: "user",
      onUseAsPrompt,
    })));

    const useAsPrompt = host.querySelector<HTMLButtonElement>("button[aria-label='Use user message as prompt']");
    expect(useAsPrompt).toBeTruthy();
    await act(async () => useAsPrompt?.click());

    expect(onUseAsPrompt).toHaveBeenCalledWith("Keep this as a follow-up prompt");
    expect(host.querySelector("[role='status']")?.textContent).toContain("Draft filled");
  });

  it("announces when clipboard copying is unavailable", async () => {
    clipboard.copyTextToClipboard.mockResolvedValue(false);
    await act(async () => root.render(createElement(MessageActions, {
      message: "A message that cannot be copied",
      messageRole: "assistant",
      onUseAsPrompt: vi.fn(),
    })));

    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Copy assistant message']")?.click());

    expect(host.querySelector("[role='status']")?.textContent).toContain("Copy failed");
  });

  it("speaks assistant messages through the optional callback", async () => {
    const onSpeak = vi.fn().mockResolvedValue(undefined);
    await act(async () => root.render(createElement(MessageActions, {
      message: "Read this aloud",
      messageRole: "assistant",
      onUseAsPrompt: vi.fn(),
      onSpeak,
    })));

    const speak = host.querySelector<HTMLButtonElement>("button[aria-label='Speak assistant message']");
    expect(speak).toBeTruthy();
    await act(async () => speak?.click());
    expect(onSpeak).toHaveBeenCalledWith("Read this aloud");
    expect(host.querySelector("[role='status']")?.textContent).toContain("Spoken");
  });

  it("delegates edit and run-again actions to the parent", async () => {
    const onEdit = vi.fn();
    const onRegenerate = vi.fn();
    await act(async () => root.render(createElement("div", null,
      createElement(MessageActions, { message: "edit me", messageRole: "user", onUseAsPrompt: vi.fn(), onEdit }),
      createElement(MessageActions, { message: "run me", messageRole: "assistant", onUseAsPrompt: vi.fn(), onRegenerate }),
    )));

    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Edit user message']")?.click());
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Run assistant message again']")?.click());
    expect(onEdit).toHaveBeenCalledWith("edit me");
    expect(onRegenerate).toHaveBeenCalledTimes(1);
  });
});
