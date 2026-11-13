// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import ProfilesPage from "./ProfilesPage";

const header = vi.hoisted(() => ({ setEnd: vi.fn() }));
vi.mock("@/contexts/usePageHeader", () => ({ usePageHeader: () => header }));
vi.mock("@/contexts/useProfileScope", () => ({ useProfileScope: () => ({ setProfile: vi.fn() }) }));

let root: Root;
let headerRoot: Root;
let container: HTMLDivElement;
let headerContainer: HTMLDivElement;
let reply: Record<string, unknown>;
let createRequests: number;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function button(label: string, scope: ParentNode = document) {
  const found = Array.from(scope.querySelectorAll("button")).find(el => el.textContent === label);
  if (!found) throw new Error(`Missing button: ${label}`);
  return found;
}

async function click(el: Element) {
  await act(async () => { el.dispatchEvent(new MouseEvent("click", { bubbles: true })); });
}

async function create(pickModel = false) {
  await act(async () => { root.render(<MemoryRouter><ProfilesPage /></MemoryRouter>); });
  await click(button("Create", headerContainer));
  const input = document.querySelector<HTMLInputElement>("#profile-name")!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "warning-bot");
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  if (pickModel) {
    await click(document.querySelector('#profile-model [role="combobox"]')!);
    await click(document.querySelector('[role="option"]:last-child')!);
  }
  await click(button("Create", document.querySelector('[role="dialog"]')!));
}

beforeEach(() => {
  createRequests = 0;
  reply = { ok: true, name: "warning-bot", path: "/profiles/warning-bot" };
  container = document.createElement("div");
  headerContainer = document.createElement("div");
  document.body.append(container, headerContainer);
  root = createRoot(container);
  headerRoot = createRoot(headerContainer);
  header.setEnd.mockImplementation((node: ReactNode) => headerRoot.render(node));
  vi.stubGlobal("matchMedia", () => ({ matches: true }));
  Element.prototype.scrollIntoView = () => undefined;
  vi.stubGlobal("fetch", vi.fn(async (url: string, options?: RequestInit) => {
    if (url === "/api/profiles" && options?.method === "POST") {
      createRequests += 1;
      return new Response(JSON.stringify(reply), { status: Number(reply.status ?? 200) });
    }
    if (url === "/api/profiles") return new Response(JSON.stringify({ profiles: [] }));
    if (url === "/api/profiles/active") return new Response(JSON.stringify({ active: "default", current: "default" }));
    if (url.startsWith("/api/model/options")) return new Response(JSON.stringify({ providers: [{ slug: "test", name: "Test", models: ["model"] }] }));
    throw new Error(`Unexpected request: ${url}`);
  }));
});

afterEach(async () => {
  await act(async () => { root.unmount(); headerRoot.unmount(); });
  container.remove();
  headerContainer.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it("renders every creation warning with success after closing the modal", async () => {
  vi.useFakeTimers();
  reply.warnings = ["Subscription logins stay with default.", "A plugin could not be copied."];
  await create();
  expect(createRequests).toBe(1);
  expect(document.querySelector('[role="dialog"]')).toBeNull();
  const toast = document.querySelector('[role="status"]')!;
  expect(toast.textContent).toContain("warning-bot");
  for (const warning of reply.warnings as string[]) {
    expect(Array.from(toast.querySelectorAll("p")).some(p => p.textContent === warning)).toBe(true);
  }
  await act(async () => { vi.advanceTimersByTime(5000); });
  expect(document.querySelector('[role="status"]')).toBe(toast);
  await click(document.querySelector('[aria-label="Dismiss"]')!);
  expect(document.querySelector('[role="status"]')).toBeNull();
  vi.useRealTimers();
});

it("clears the previous receipt when starting another creation", async () => {
  reply.warnings = ["Old creation warning"];
  await create();
  expect(document.querySelector('[role="status"]')!.textContent).toContain("Old creation warning");
  reply = { ok: true, name: "warning-bot" };
  await create();
  expect(createRequests).toBe(2);
  expect(document.body.textContent).not.toContain("Old creation warning");
  expect(document.querySelector('[role="status"]')!.className).toContain("text-success");
  expect(document.querySelector('[aria-label="Dismiss"]')).toBeNull();
});

it("keeps the previous receipt when a later creation fails", async () => {
  reply.warnings = ["Old creation warning"];
  await create();
  reply = { status: 500, detail: "boom" };
  await create();
  expect(createRequests).toBe(2);
  expect(document.querySelector('[role="status"]')!.textContent).toContain("Old creation warning");
});

it.each([{ warnings: undefined }, { warnings: [] }])("preserves success for warning-free and older replies (%j)", async ({ warnings }) => {
  vi.useFakeTimers();
  if (warnings !== undefined) reply.warnings = warnings;
  await create();
  expect(createRequests).toBe(1);
  const toast = document.querySelector('[role="status"]')!;
  expect(toast.textContent).toContain("warning-bot");
  expect(toast.className).toContain("text-success");
  await act(async () => { vi.advanceTimersByTime(5000); });
  await act(async () => { vi.advanceTimersByTime(250); });
  expect(document.querySelector('[role="status"]')).toBeNull();
  expect(document.querySelector('[role="dialog"]')).toBeNull();
});

it.each([{ warnings: [] }, { warnings: ["Subscription logins stay with default.", "A plugin could not be copied."] }])("keeps model-save failure visible alongside success and warnings %j", async ({ warnings }) => {
  vi.useFakeTimers();
  reply.warnings = warnings;
  reply.model_set = false;
  await create(true);
  const toast = document.querySelector('[role="status"]')!;
  expect(toast.textContent).toContain("warning-bot");
  expect(toast.textContent).toContain("model could not be saved");
  for (const warning of reply.warnings as string[]) expect(toast.textContent).toContain(warning);
  expect(toast.className).toContain("text-destructive");
  await act(async () => { vi.advanceTimersByTime(5000); });
  expect(document.querySelector('[role="status"]')).toBe(toast);
  expect(toast.querySelectorAll("p")).toHaveLength(warnings.length + 2);
  await click(document.querySelector('[aria-label="Dismiss"]')!);
  expect(document.querySelector('[role="status"]')).toBeNull();
  expect(document.querySelector('[role="dialog"]')).toBeNull();
});
