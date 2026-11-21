// @vitest-environment jsdom
// A dashboard user with no older Hermes server sees the System page as before: the same requests on
// load (plus the one plan read) and no "move to a single Hermes" card, whether the server has the
// backend-migration routes or not.

import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";

import { I18nProvider } from "@/i18n";
import SystemPage from "@/pages/SystemPage";

const PAGE_REQUESTS = [
  "GET /api/status",
  "GET /api/system/stats",
  "GET /api/memory",
  "GET /api/credentials/pool",
  "GET /api/ops/checkpoints",
  "GET /api/ops/hooks",
  "GET /api/curator",
  "GET /api/portal",
  "GET /api/hermes/update/check",
  "GET /api/gateway/migrate/plan",
];
const PLAN = "GET /api/gateway/migrate/backends/plan";

const NOTHING_TO_MOVE = {
  exit_code: 0,
  plan: { nothing_to_do: true, blockers: [], unsupported: [] },
  run: { state: "none", step: "", steps: [], message: "", old_address: "", new_address: "", started_at: 0, finished_at: null },
};

let root: Root | null = null;

afterEach(async () => {
  await act(async () => root?.unmount());
  root = null;
  document.body.innerHTML = "";
  vi.unstubAllGlobals();
});

async function loadPage(planAnswer: { status: number; body: unknown }): Promise<string[]> {
  const requests: string[] = [];
  vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input, init) => {
    const request = `${init?.method ?? "GET"} ${String(input).replace(/\?.*/, "")}`;
    requests.push(request);
    const answer = request === PLAN ? planAnswer : { status: 404, body: {} };
    return new Response(JSON.stringify(answer.body), { status: answer.status, headers: { "Content-Type": "application/json" } });
  }));
  const el = document.createElement("div");
  document.body.append(el);
  root = createRoot(el);
  await act(async () => root!.render(<MemoryRouter><I18nProvider><SystemPage /></I18nProvider></MemoryRouter>));
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
  return requests;
}

describe("System page without an older Hermes server", () => {
  it.each([
    ["a server without the routes (404)", { status: 404, body: { detail: "Not Found" } }],
    ["nothing to move", { status: 200, body: NOTHING_TO_MOVE }],
  ])("shows no card and loads the page's own data as before: %s", async (_name, planAnswer) => {
    const requests = await loadPage(planAnswer);
    expect(requests.filter((r) => r !== PLAN).sort()).toEqual([...PAGE_REQUESTS].sort());
    expect(requests.filter((r) => r === PLAN).length).toBeLessThanOrEqual(1);
    expect(document.querySelector('[data-testid="backend-migration-card"]')).toBeNull();
  });
});
