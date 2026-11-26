// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import WebhooksPage from "./WebhooksPage";

const api = vi.hoisted(() => ({
  getWebhooks: vi.fn(),
  createWebhook: vi.fn(),
}));
const header = vi.hoisted(() => ({ end: null as ReactNode }));
vi.mock("@/lib/api", () => ({ api }));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setEnd: (node: ReactNode) => { header.end = node; } }),
}));

let root: Root;
let headerRoot: Root;
let container: HTMLDivElement;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function button(label: RegExp): HTMLButtonElement {
  const found = [...document.querySelectorAll<HTMLButtonElement>("button")].find((b) =>
    label.test(b.textContent?.trim() ?? ""),
  );
  if (!found) throw new Error(`Missing button ${label}`);
  return found;
}

async function click(el: Element) {
  await act(async () => { (el as HTMLElement).click(); });
}

async function openForm(name: string) {
  await act(async () => { headerRoot.render(<>{header.end}</>); });
  await click(button(/^new subscription$/i));
  const input = document.getElementById("webhook-name") as HTMLInputElement;
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
    setter.call(input, name);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

beforeEach(async () => {
  api.getWebhooks.mockResolvedValue({ enabled: true, base_url: "http://h", subscriptions: [] });
  api.createWebhook.mockResolvedValue({ url: "http://h/webhooks/x", secret: "s" });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  headerRoot = createRoot(document.body.appendChild(document.createElement("div")));
  await act(async () => { root.render(<WebhooksPage />); });
});

afterEach(async () => {
  await act(async () => { root.unmount(); headerRoot.unmount(); });
  document.body.replaceChildren();
  vi.clearAllMocks();
});

it("pins a new route to generic by default", async () => {
  await openForm("plain-hook");
  await click(button(/^create$/i));
  expect(api.createWebhook).toHaveBeenCalledWith(
    expect.objectContaining({ name: "plain-hook", signature: "generic" }),
  );
});

it("sends the picked signature scheme", async () => {
  await openForm("gh-hook");
  const combo = document.getElementById("webhook-signature")?.closest('[role="combobox"]')
    ?? document.querySelector('#webhook-signature [role="combobox"]');
  if (!combo) throw new Error("Missing signature select");
  await click(combo);
  const option = [...document.querySelectorAll('[role="option"]')].find((o) => o.textContent?.trim() === "GitHub");
  if (!option) throw new Error("Missing GitHub option");
  await click(option);
  await click(button(/^create$/i));
  expect(api.createWebhook).toHaveBeenCalledWith(
    expect.objectContaining({ name: "gh-hook", signature: "github" }),
  );
});
