// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeAll, expect, it, vi } from "vitest";
import { exposePluginSDK, getPluginComponent } from "./registry";

const tasks = [
  { id: "final", title: "Release review", status: "review", current_attention: {
    kind: "awaiting-test", run_id: 1, badge: { label: "Final PR", tone: "success" }
  } },
  { id: "test", title: "Testing review", status: "review", current_attention: { kind: "awaiting-test", run_id: 2 } },
  { id: "answer", title: "Question review", status: "review", current_attention: { kind: "awaiting-answer", run_id: 3 } },
  { id: "failure", title: "Failed run", status: "ready", current_attention: { kind: "run-failed", run_id: 4 } },
  { id: "stale", title: "Already running", status: "running", current_attention: {
    kind: "awaiting-test", run_id: 5, badge: { label: "Old approval", tone: "success" }
  } },
  { id: "escaped", title: "Other producer", status: "review", current_attention: {
    kind: "awaiting-test", run_id: 6, badge: { label: "<img src=x onerror=alert(1)>", tone: "info" }
  } },
];

beforeAll(async () => {
  exposePluginSDK();
  const sdk = window.__HERMES_PLUGIN_SDK__!;
  sdk.useI18n = () => ({ t: { kanban: null } } as never);
  sdk.fetchJSON = vi.fn(async (url: string) => {
    const path = url.split("?")[0];
    if (path.endsWith("/board")) return {
      columns: ["review", "ready", "running"].map(name => ({ name, tasks: tasks.filter(t => t.status === name) })),
      assignees: [], tenants: [], latest_event_id: 0,
    };
    if (path.endsWith("/boards")) return { boards: [], current: "" };
    if (path.includes("/tasks/")) return { task: tasks.find(t => path.endsWith("/" + t.id)), runs: [], comments: [], events: [], attachments: [] };
    return {};
  }) as typeof sdk.fetchJSON;
  sdk.buildWsUrl = () => new Promise(() => {});
  // @ts-expect-error The served plugin is a plain JavaScript IIFE, not a typed module.
  await import("../../../plugins/kanban/dashboard/dist/index.js");
});

let root: Root | undefined;
let container: HTMLDivElement;
afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

it("renders the served plugin's colored card badges and the same single badge in detail", async () => {
  const Page = getPluginComponent("kanban")!;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => root!.render(<Page />));
  const card = container.querySelector('[data-task-id="final"]')!;
  const badge = card.querySelector(".hermes-kanban-task-attention--success")!;
  expect(badge.textContent).toBe("Final PR");
  expect(card.querySelectorAll(".hermes-kanban-task-attention span")).toHaveLength(1);
  expect(card.textContent).not.toContain("Awaiting your testing");
  for (const [id, tone, label] of [
    ["test", "info", "Awaiting your testing"], ["answer", "warning", "Needs your answer"],
    ["failure", "danger", "Run failed"], ["escaped", "info", "<img src=x onerror=alert(1)>"],
  ]) {
    expect(container.querySelector(`[data-task-id="${id}"] .hermes-kanban-task-attention--${tone}`)?.textContent).toBe(label);
  }
  expect(container.textContent).not.toContain("Old approval");
  expect(container.querySelector("img")).toBeNull();
  await act(async () => card.dispatchEvent(new MouseEvent("click", { bubbles: true })));
  const drawer = container.querySelector(".hermes-kanban-drawer-body")!;
  expect(drawer.querySelectorAll(".hermes-kanban-task-attention span")).toHaveLength(1);
  expect(drawer.querySelector(".hermes-kanban-task-attention--success")?.textContent).toBe("Final PR");
  expect(drawer.textContent).not.toContain("Awaiting your testing");
});
