// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ClarificationCard } from "./ClarificationCard";

describe("ClarificationCard", () => {
  let root: Root;
  let host: HTMLDivElement;
  beforeEach(() => { host = document.createElement("div"); document.body.appendChild(host); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); host.remove(); });

  it("submits a single choice and keeps the answer control accessible", async () => {
    const onRespond = vi.fn().mockResolvedValue(undefined);
    await act(async () => root.render(createElement(ClarificationCard, {
      request: { request_id: "req-1", question: "Which?", choices: ["A"] }, onRespond,
    })));
    expect(host.querySelector("[role='dialog']")?.getAttribute("aria-label")).toBe("Clarification required");
    expect(host.querySelector("input[aria-label='Clarification answer']")).toBeTruthy();
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='A']")?.click());
    expect(onRespond).toHaveBeenCalledWith("A");
  });

  it("renders batch questions and sends each answer with its question id", async () => {
    const onRespond = vi.fn().mockResolvedValue(undefined);
    await act(async () => root.render(createElement(ClarificationCard, {
      request: { request_id: "req-1", questions: [{ qid: "q0", question: "Color?", choices: ["red"] }, { qid: "q1", question: "Name?" }] }, onRespond,
    })));
    expect(host.textContent).toContain("Color?");
    expect(host.textContent).toContain("Name?");
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='q0:red']")?.click());
    const inputs = host.querySelectorAll<HTMLInputElement>("input");
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      setter?.call(inputs[1], "Hermes");
      inputs[1].dispatchEvent(new Event("input", { bubbles: true }));
      inputs[1].dispatchEvent(new Event("change", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
    });
    expect(onRespond).toHaveBeenNthCalledWith(1, "red", "q0");
    expect(onRespond).toHaveBeenNthCalledWith(2, "Hermes", "q1");
  });
});
