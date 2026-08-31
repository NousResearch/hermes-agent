// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApprovalCard } from "./ApprovalCard";

describe("ApprovalCard", () => {
  let root: Root;
  let host: HTMLDivElement;
  beforeEach(() => { host = document.createElement("div"); document.body.appendChild(host); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); host.remove(); });

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
});
