// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";

const task = {
  id: "blocked-example", title: "Investigate failure", status: "blocked",
  assignee: "worker", workspace_kind: "scratch", priority: 0,
  block_kind: "needs_input", block_recurrences: 3, consecutive_failures: 2,
  last_failure_error: "<script>fixture failure</script>",
};
let root: Root;
let container: HTMLDivElement;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

async function openBoard(value: Record<string, unknown> = task) {
  let Page: React.ComponentType;
  const fetchJSON = vi.fn(async (url: string) => {
    const path = new URL(url, "http://localhost").pathname;
    if (path.endsWith("/board")) return { columns: [{ name: value.status, tasks: [value] }], assignees: ["worker"], tenants: [] };
    if (path.endsWith("/boards")) return { boards: [] };
    if (path.endsWith("/tasks/blocked-example")) return { task: value, comments: [], events: [], runs: [] };
    if (path.endsWith("/profiles")) return { profiles: [] };
    if (path.endsWith("/tasks")) return { tasks: [value] };
    return {};
  });
  Object.assign(window, {
    __HERMES_PLUGIN_SDK__: {
      React, hooks: React,
      components: { Card: "div", CardContent: "div", Badge: "span", Button: "button", Input: "input", Label: "label", Select: "select", SelectOption: "option" },
      utils: { cn: (...parts: string[]) => parts.filter(Boolean).join(" "), timeAgo: () => "" },
      fetchJSON, buildWsUrl: () => Promise.resolve("ws://localhost/fixture"),
    },
    __HERMES_PLUGINS__: { register: (_name: string, component: React.ComponentType) => { Page = component; } },
  });
  vi.stubGlobal("WebSocket", class { close() {} });
  vi.resetModules();
  // Execute the shipped IIFE through its real SDK registration, without source extraction.
  // @ts-expect-error The no-build dashboard bundle has no TypeScript declaration.
  await import("../../../plugins/kanban/dashboard/dist/index.js");
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => { root.render(React.createElement(Page!)); });
  const card = container.querySelector<HTMLElement>('[data-task-id="blocked-example"]');
  expect(card).not.toBeNull();
  return card!;
}

async function openDrawer(card: HTMLElement) {
  await act(async () => { card.dispatchEvent(new MouseEvent("click", { bubbles: true })); });
  const drawer = container.querySelector(".hermes-kanban-drawer-body");
  expect(drawer).not.toBeNull();
  return drawer!;
}

function row(label: string) {
  return Array.from(container.querySelectorAll(".hermes-kanban-meta-row"))
    .find(el => el.querySelector(".hermes-kanban-meta-label")?.textContent === label)
    ?.querySelector(".hermes-kanban-meta-value")?.textContent;
}

afterEach(async () => {
  await act(async () => { root?.unmount(); });
  container?.remove();
  vi.unstubAllGlobals();
  localStorage.clear();
  Reflect.deleteProperty(window, "__HERMES_PLUGIN_SDK__");
  Reflect.deleteProperty(window, "__HERMES_PLUGINS__");
});

it("does not label a retained kind as a current block after recovery", async () => {
  const card = await openBoard({ ...task, status: "ready", block_recurrences: 0, consecutive_failures: 0 });
  expect(card.textContent).not.toContain("needs_input");
  await openDrawer(card);
  expect(row("Status")).toBe("ready");
  expect(row("Block kind")).toBeUndefined();
  expect(row("Block recurrences")).toBeUndefined();
  expect(row("Consecutive failures")).toBeUndefined();
  // Retained failure text is historical, not a current block reason.
  expect(row("Last failure")).toBe(task.last_failure_error);
});

it("shows the current block kind on the card and existing failure fields in its drawer", async () => {
  const card = await openBoard();
  expect(card.textContent).toContain("needs_input");
  await openDrawer(card);
  expect(row("Block kind")).toBe(task.block_kind);
  expect(row("Block recurrences")).toBe("3");
  expect(row("Consecutive failures")).toBe("2");
  expect(row("Last failure")).toBe(task.last_failure_error);
  expect(container.querySelector("script")).toBeNull();
});

it("keeps legacy tasks without block metadata usable", async () => {
  const legacy: Record<string, unknown> = { ...task };
  for (const key of ["block_kind", "block_recurrences", "consecutive_failures", "last_failure_error"]) {
    delete legacy[key];
  }
  const card = await openBoard(legacy);
  await openDrawer(card);
  expect(row("Status")).toBe("blocked");
  expect(row("Block recurrences")).toBeUndefined();
  expect(row("Last failure")).toBeUndefined();
});
