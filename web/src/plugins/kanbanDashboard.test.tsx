// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  authedFetch: vi.fn(),
  buildWsAuthParam: vi.fn(),
  buildWsUrl: vi.fn(),
  fetchJSON: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: {},
  authedFetch: apiMocks.authedFetch,
  buildWsAuthParam: apiMocks.buildWsAuthParam,
  buildWsUrl: apiMocks.buildWsUrl,
  fetchJSON: apiMocks.fetchJSON,
}));

const emptyBoard = {
  assignees: [],
  columns: ["triage", "todo", "ready", "running", "blocked", "review", "done"].map(
    name => ({ name, tasks: [] }),
  ),
  latest_event_id: 0,
};

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

class FakeWebSocket {
  close() {}
}

async function waitFor(condition: () => boolean, timeoutMs = 5000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeoutMs) throw new Error("condition never became true");
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 20));
    });
  }
}

function click(element: Element | null) {
  if (!element) throw new Error("element not rendered");
  element.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
}

function setTextareaValue(element: HTMLTextAreaElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
  setter?.call(element, value);
  element.dispatchEvent(new Event("input", { bubbles: true }));
}

beforeEach(async () => {
  vi.resetAllMocks();
  localStorage.clear();
  apiMocks.buildWsUrl.mockResolvedValue("ws://localhost/api/plugins/kanban/events");
  apiMocks.authedFetch.mockResolvedValue({
    ok: false,
    status: 413,
    text: async () => JSON.stringify({ detail: "image too large" }),
  });

  let boardLoads = 0;
  apiMocks.fetchJSON.mockImplementation(async (url: string, init?: RequestInit) => {
    if (url.includes("/tasks") && init?.method === "POST") {
      return { task: { id: "t_created" } };
    }
    if (url.includes("/config")) return { render_markdown: true };
    if (url.includes("/boards")) {
      return { boards: [{ slug: "default", name: "Default" }], current: "default" };
    }
    if (url.includes("/board")) {
      boardLoads += 1;
      return { ...emptyBoard, latest_event_id: boardLoads };
    }
    throw new Error(`unexpected request: ${url}`);
  });

  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal("ResizeObserver", class { disconnect() {} observe() {} unobserve() {} });

  const { exposePluginSDK, getPluginComponent } = await import("./registry");
  exposePluginSDK();
  // The kanban dashboard intentionally ships as a plain IIFE plugin bundle.
  // @ts-expect-error -- no module declarations are generated for plugin bundles.
  await import("../../../plugins/kanban/dashboard/dist/index.js");
  const KanbanPage = getPluginComponent("kanban");
  if (!KanbanPage) throw new Error("kanban plugin did not register");

  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<KanbanPage />));
  await waitFor(() => Boolean(container.querySelector('[title="Create task in this column"]')));
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("kanban dashboard clipboard uploads", () => {
  it("keeps a post-create upload failure visible after the board reload", async () => {
    await act(async () => click(container.querySelector('[title="Create task in this column"]')));

    const textareas = Array.from(container.querySelectorAll("textarea"));
    expect(textareas).toHaveLength(2);
    await act(async () => setTextareaValue(textareas[0], "Task with screenshot"));

    const image = new File([new Uint8Array([1, 2, 3])], "shot.png", { type: "image/png" });
    const paste = new Event("paste", { bubbles: true, cancelable: true });
    Object.defineProperty(paste, "clipboardData", {
      value: { items: [], files: { 0: image, length: 1 } },
    });
    await act(async () => textareas[1].dispatchEvent(paste));

    const form = container.querySelector("form");
    if (!form) throw new Error("create form not rendered");
    await act(async () => form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));

    await waitFor(() => !container.querySelector("form"));
    await waitFor(() => apiMocks.fetchJSON.mock.calls.filter(
      ([url]) => String(url).includes("/board") && !String(url).includes("/boards"),
    ).length >= 2);

    const taskPosts = apiMocks.fetchJSON.mock.calls.filter(
      ([url, init]) => String(url).includes("/tasks") && init?.method === "POST",
    );
    expect(taskPosts).toHaveLength(1);
    expect(apiMocks.authedFetch).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain(
      "Task created, but attachment upload failed: image too large",
    );
  });
});