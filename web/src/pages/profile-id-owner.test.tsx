// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import ProfileBuilderPage from "./ProfileBuilderPage";
import ProfilesPage from "./ProfilesPage";

// Both dashboard name checks take their grammar from the shared PROFILE_ID_ARG_RE owner
// (state-24, #155595). Under a swapped owner a private copy keeps accepting `work` and fails here.
vi.mock("@hermes/shared/profile-id", () => ({ PROFILE_ID_ARG_RE: /^zz\.[a-z]+$/ }));
const header = vi.hoisted(() => ({ setEnd: vi.fn() }));
vi.mock("@/contexts/usePageHeader", () => ({ usePageHeader: () => header }));
vi.mock("@/contexts/useProfileScope", () => ({ useProfileScope: () => ({ setProfile: vi.fn() }) }));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root;
let headerRoot: Root;
let container: HTMLDivElement;
let headerContainer: HTMLDivElement;

async function type(input: HTMLInputElement, value: string) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

beforeEach(() => {
  container = document.createElement("div");
  headerContainer = document.createElement("div");
  document.body.append(container, headerContainer);
  root = createRoot(container);
  headerRoot = createRoot(headerContainer);
  header.setEnd.mockImplementation((node: ReactNode) => headerRoot.render(node));
  vi.stubGlobal("matchMedia", () => ({ matches: true }));
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url === "/api/profiles") return new Response(JSON.stringify({ profiles: [] }));
    if (url === "/api/profiles/active") return new Response(JSON.stringify({ active: "default", current: "default" }));
    if (url.startsWith("/api/model/options")) return new Response(JSON.stringify({ providers: [] }));
    throw new Error(`Unexpected request: ${url}`);
  }));
});

afterEach(async () => {
  await act(async () => { root.unmount(); headerRoot.unmount(); });
  container.remove();
  headerContainer.remove();
  vi.unstubAllGlobals();
});

it.each([["work", "true"], ["zz.custom", "false"]])("ProfilesPage marks %s aria-invalid=%s", async (name, invalid) => {
  await act(async () => { root.render(<MemoryRouter><ProfilesPage /></MemoryRouter>); });
  const create = Array.from(document.querySelectorAll("button")).find((b) => b.textContent === "Create");
  await act(async () => { create!.dispatchEvent(new MouseEvent("click", { bubbles: true })); });
  const input = document.querySelector<HTMLInputElement>("#profile-name")!;
  await type(input, name);
  expect(input.getAttribute("aria-invalid")).toBe(invalid);
});

it.each([["work", true], ["zz.custom", false]])("ProfileBuilderPage shows the name rule for %s: %s", async (name, shown) => {
  await act(async () => { root.render(<MemoryRouter><ProfileBuilderPage /></MemoryRouter>); });
  await type(document.querySelector<HTMLInputElement>("#pb-name")!, name);
  expect(container.textContent!.includes("must start with a letter or digit")).toBe(shown);
});
