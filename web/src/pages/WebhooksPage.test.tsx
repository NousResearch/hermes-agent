// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import WebhooksPage from "./WebhooksPage";
import { I18nProvider } from "@/i18n";
import { PageHeaderProvider } from "@/contexts/PageHeaderProvider";

const mocks = vi.hoisted(() => ({ getWebhooks: vi.fn(), createWebhook: vi.fn(), copy: vi.fn() }));
vi.mock("@/lib/api", () => ({ api: mocks }));
vi.mock("@/lib/clipboard", () => ({ copyTextToClipboard: mocks.copy }));
let container: HTMLDivElement;
let root: Root;
const secret = "synthetic-route-credential";
const url = "http://localhost:8644/webhooks/synthetic-hook";
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function button(text: string, within: Element = document.body) {
  const match = Array.from(within.querySelectorAll("button")).find((el) => el.textContent?.trim() === text);
  if (!match) throw new Error(`Button not rendered: ${text}`);
  return match;
}
function click(el: Element) {
  el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.getWebhooks.mockResolvedValue({ enabled: true, subscriptions: [] });
  mocks.createWebhook.mockImplementation(async () => {
    mocks.getWebhooks.mockResolvedValue({ enabled: true, subscriptions: [
      { name: "synthetic-hook", url, deliver: "log", deliver_only: false, enabled: true, events: [] },
    ] });
    return { url, secret };
  });
  mocks.copy.mockResolvedValue(true);
  vi.stubGlobal("matchMedia", () => ({ matches: false, media: "", addEventListener() {}, removeEventListener() {} }));
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.unstubAllGlobals();
});

async function createSubscription() {
  await act(async () => root.render(
    <I18nProvider><MemoryRouter initialEntries={["/webhooks"]}>
      <PageHeaderProvider pluginTabs={[]}><WebhooksPage /></PageHeaderProvider>
    </MemoryRouter></I18nProvider>,
  ));
  await act(async () => click(button("New subscription")));
  const input = container.querySelector<HTMLInputElement>("#webhook-name");
  if (!input) throw new Error("Create form not rendered");
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "synthetic-hook");
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await act(async () => click(button("Create")));
  expect(mocks.createWebhook).toHaveBeenCalledWith(expect.objectContaining({ name: "synthetic-hook" }));
  const dialog = container.querySelector('[role="dialog"]');
  if (!dialog) throw new Error("Created credentials not rendered");
  return dialog;
}

describe("Webhook creation credentials", () => {
  it("copies the actual URL, secret and complete Bearer header from the created page", async () => {
    const dialog = await createSubscription();
    const copies = Array.from(dialog.querySelectorAll('button[aria-label="Copy"]'));
    expect(copies).toHaveLength(3);
    for (const copy of copies) await act(async () => click(copy));
    expect(mocks.copy.mock.calls.map(([value]) => value)).toEqual([
      url, secret, ["Authorization:", "Bearer", secret].join(" "),
    ]);
    expect(dialog.textContent).toContain("Bearer auth header");
    expect(container.querySelector("main")?.textContent?.replace(dialog.textContent ?? "", "")).not.toContain(secret);
  });

  it.each(["Done", "Close", "Escape", "backdrop"])("clears one-time credentials after %s and reopening", async (method) => {
    const dialog = await createSubscription();
    expect(dialog.textContent).toContain(secret);
    await act(async () => {
      if (method === "Escape") document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
      else if (method === "backdrop") click(dialog);
      else if (method === "Close") {
        const close = dialog.querySelector('button[aria-label="Close"]');
        if (!close) throw new Error("Close control not rendered");
        click(close);
      } else click(button("Done", dialog));
    });
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(container.textContent).not.toContain(secret);
    await act(async () => click(button("New subscription")));
    expect(container.querySelector("#webhook-name")).not.toBeNull();
    expect(container.textContent).not.toContain(secret);
    expect(container.textContent).not.toContain("Bearer auth header");
  });
});
