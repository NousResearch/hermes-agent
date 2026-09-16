// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApprovalCard } from "./ApprovalCard";

describe("ApprovalCard", () => {
  let root: Root;
  let host: HTMLDivElement;
  beforeEach(() => { host = document.createElement("div"); document.body.appendChild(host); root = createRoot(host); });
  afterEach(() => { vi.useRealTimers(); act(() => root.unmount()); host.remove(); });

  it("renders untrusted command text as text and submits the selected choice", async () => {
    const onRespond = vi.fn().mockResolvedValue(undefined);
    await act(async () => root.render(createElement(ApprovalCard, {
      request: { request_id: "req-1", command: "<img src=x onerror=alert(1)>", choices: ["once", "deny"] }, onRespond,
    })));
    expect(host.querySelector("img")).toBeNull();
    expect(host.textContent).toContain("<img src=x onerror=alert(1)>");
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.click());
    expect(onRespond).toHaveBeenCalledWith("once");
  });

  it("keeps expired requests visible but disables their choices", async () => {
    const onRespond = vi.fn().mockResolvedValue(undefined);
    await act(async () => root.render(createElement(ApprovalCard, {
      request: { request_id: "req-expired", description: "Expired", choices: ["once", "always"], expires_at: (Date.now() - 1000) / 1000 }, onRespond,
    })));
    expect(host.textContent).toContain("expired");
    expect(host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.disabled).toBe(true);
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.click());
    expect(onRespond).not.toHaveBeenCalled();
  });

  it("falls back to deny for unsupported choices", async () => {
    const onRespond = vi.fn().mockResolvedValue(undefined);
    await act(async () => root.render(createElement(ApprovalCard, {
      request: { request_id: "req-unknown", choices: ["allow"] }, onRespond,
    })));
    expect(host.querySelector("button[data-choice='allow']")).toBeNull();
    expect(host.querySelector("button[data-choice='deny']")).not.toBeNull();
  });

  it("fails closed for explicit empty choices and malformed capability flags", async () => {
    const onRespond = vi.fn().mockResolvedValue(undefined);
    await act(async () => root.render(createElement(ApprovalCard, {
      request: { request_id: "req-malformed", choices: [], allow_session: "yes", allow_permanent: 1 } as never, onRespond,
    })));
    expect(host.querySelector("button[data-choice='once']")).toBeNull();
    expect(host.querySelector("button[data-choice='session']")).toBeNull();
    expect(host.querySelector("button[data-choice='always']")).toBeNull();
    expect(host.querySelector("button[data-choice='deny']")).not.toBeNull();
  });

  it("disables choices while submitting and exposes rejected responses", async () => {
    let reject!: (reason: Error) => void;
    const onRespond = vi.fn(() => new Promise<void>((_, r) => { reject = r; }));
    await act(async () => root.render(createElement(ApprovalCard, {
      request: { request_id: "req-1", description: "Run it", choices: ["once"] }, onRespond,
    })));
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.click());
    expect(host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.disabled).toBe(true);
    await act(async () => reject(new Error("network down")));
    expect(host.textContent).toContain("network down");
    expect(host.querySelector("[role='alert']")).toBeTruthy();
  });

  it("transitions live from active to expired when timers advance beyond expires_at", async () => {
    vi.useFakeTimers();
    try {
      const nowMs = 1_700_000_000_000;
      vi.setSystemTime(nowMs);
      const onRespond = vi.fn().mockResolvedValue(undefined);
      const request = {
        request_id: "req-live-expired",
        description: "Dangerous operation",
        choices: ["once", "deny"],
        expires_at: (nowMs + 5_000) / 1000,
      };

      await act(async () => {
        root.render(createElement(ApprovalCard, { request, onRespond }));
      });

      expect(host.textContent).toContain("Approval required");
      expect(host.textContent).not.toContain("Approval expired");
      const choiceButton = host.querySelector<HTMLButtonElement>("button[data-choice='once']");
      expect(choiceButton).not.toBeNull();
      expect(choiceButton?.disabled).toBe(false);

      await act(async () => {
        vi.advanceTimersByTime(5_001);
      });

      expect(host.textContent).toContain("Approval expired");
      expect(choiceButton?.disabled).toBe(true);

      await act(async () => {
        choiceButton?.click();
      });
      expect(onRespond).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });
});
