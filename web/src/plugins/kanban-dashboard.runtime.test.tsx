// @vitest-environment jsdom
import React, { act, useContext, type ComponentType, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { BrowserRouter, useNavigate } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ProfileProvider } from "@/contexts/ProfileProvider";
import { ProfileContext } from "@/contexts/profile-context";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

type Task = {
  id: string;
  title: string;
  status: string;
  workspace_kind: string;
  workspace_path: string;
  skills: string[];
};

type BoardFixture = { slug: string; tasks: Task[] };

const fetchJSON = vi.fn();
let KanbanPage: ComponentType<Record<string, never>>;

function element(tag: string) {
  return function Element({ children, ...props }: Record<string, unknown>) {
    return React.createElement(tag, props, children as ReactNode);
  };
}

function Select({ children, onValueChange, ...props }: Record<string, unknown>) {
  return (
    <select
      {...props}
      onChange={(event) =>
        (onValueChange as ((value: string) => void) | undefined)?.(event.currentTarget.value)
      }
    >
      {children as ReactNode}
    </select>
  );
}

function Checkbox({ checked, onCheckedChange, ...props }: Record<string, unknown>) {
  return (
    <input
      {...props}
      type="checkbox"
      checked={Boolean(checked)}
      onChange={(event) =>
        (onCheckedChange as ((value: boolean) => void) | undefined)?.(
          event.currentTarget.checked,
        )
      }
    />
  );
}

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor() { FakeWebSocket.instances.push(this); }
  emit(data: unknown) { this.onmessage?.({ data: JSON.stringify(data) }); }
  close() {}
}

Object.assign(window, {
  __HERMES_PLUGIN_SDK__: {
    React,
    hooks: {
      useState: React.useState,
      useEffect: React.useEffect,
      useCallback: React.useCallback,
      useMemo: React.useMemo,
      useRef: React.useRef,
      useContext: React.useContext,
      createContext: React.createContext,
      useNavigate,
    },
    components: {
      Card: element("section"),
      CardContent: element("div"),
      Badge: element("span"),
      Button: element("button"),
      Input: element("input"),
      Label: element("label"),
      Select,
      SelectOption: element("option"),
      Checkbox,
      ConfirmDialog: () => null,
    },
    fetchJSON,
    authedFetch: vi.fn(),
    buildWsUrl: vi.fn(async () => "ws://localhost/events"),
    utils: {
      cn: (...values: unknown[]) => values.filter(Boolean).join(" "),
      timeAgo: () => "now",
    },
    useI18n: () => ({ t: { kanban: null }, locale: "en" }),
  },
  __HERMES_PLUGINS__: {
    register: (name: string, component: ComponentType<Record<string, never>>) => {
      if (name === "kanban") KanbanPage = component;
    },
    registerSlot: () => undefined,
  },
});
vi.stubGlobal("WebSocket", FakeWebSocket);

// This executes the shipped, hand-authored IIFE and captures its registration.
// @ts-expect-error TS7016 -- the shipped IIFE intentionally has no TypeScript declarations.
await import("../../../plugins/kanban/dashboard/dist/index.js");

const defaultTask = task("t_default", "Default task");
const opsTask = task("t_ops", "Ops task");

function task(id: string, title: string, status = "ready"): Task {
  return {
    id,
    title,
    status,
    workspace_kind: "scratch",
    workspace_path: "",
    skills: [],
  };
}

function detail(
  value: Task,
  overrides: Partial<{
    links: { parents: string[]; children: string[] };
    child_results: Task[];
  }> = {},
) {
  return {
    task: value,
    comments: [],
    events: [],
    attachments: [],
    links: { parents: [], children: [] },
    child_results: [],
    ...overrides,
  };
}

function board(value: BoardFixture) {
  const statuses = ["triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done", "archived"];
  return {
    columns: statuses.map((status) => ({
      name: status,
      tasks: value.tasks.filter((item) => item.status === status),
    })),
    tenants: [],
    assignees: [],
    latest_event_id: 0,
  };
}

function installApi({
  boards = [
    { slug: "default", tasks: [defaultTask] },
    { slug: "ops", tasks: [opsTask] },
  ],
  locate = new Map([
    ["t_default", { board: "default", task: defaultTask }],
    ["t_ops", { board: "ops", task: opsTask }],
  ]),
}: {
  boards?: BoardFixture[];
  locate?: Map<string, unknown>;
} = {}) {
  fetchJSON.mockImplementation(async (rawUrl: string) => {
    const url = new URL(rawUrl, window.location.origin);
    const selectedBoard = url.searchParams.get("board") || "default";
    if (url.pathname.endsWith("/config")) return { render_markdown: false };
    if (url.pathname.endsWith("/boards")) {
      return {
        current: "default",
        boards: boards.map((entry) => ({
          slug: entry.slug,
          name: entry.slug,
          total: entry.tasks.length,
        })),
      };
    }
    if (url.pathname.endsWith("/board")) {
      const fixture = boards.find((entry) => entry.slug === selectedBoard);
      if (!fixture) throw new Error("404: board not found");
      return board(fixture);
    }
    if (url.pathname.includes("/tasks/locate/")) {
      const id = decodeURIComponent(url.pathname.split("/").at(-1) || "");
      const result = locate.get(id);
      if (result instanceof Error) throw result;
      if (!result) throw new Error('404: {"detail":"task not found"}');
      return result;
    }
    if (url.pathname.includes("/tasks/") && !url.pathname.includes("/home-")) {
      const id = decodeURIComponent(url.pathname.split("/").at(-1) || "");
      const item = boards
        .find((entry) => entry.slug === selectedBoard)
        ?.tasks.find((candidate) => candidate.id === id);
      if (!item) throw new Error('404: {"detail":"task not found"}');
      return detail(item);
    }
    if (url.pathname.endsWith("/home-channels")) return { home_channels: [] };
    if (url.pathname.endsWith("/orchestration")) return { enabled: false };
    if (url.pathname.endsWith("/profiles")) return { profiles: [] };
    return {};
  });
}

let container: HTMLDivElement;
let root: Root;

async function renderKanban(path: string) {
  window.history.replaceState({}, "", path);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(
    <BrowserRouter>
      <KanbanPage />
    </BrowserRouter>,
  ));
}

function ProfileSwitcher() {
  const { setProfile } = useContext(ProfileContext);
  return <button onClick={() => setProfile("research")}>Use research profile</button>;
}

async function renderProductionShell(path: string) {
  window.history.replaceState({}, "", path);
  window.__HERMES_PLUGIN_SDK__!.basePath = "/dashboard";
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(
    <BrowserRouter basename="/dashboard">
      <ProfileProvider>
        <KanbanPage />
        <ProfileSwitcher />
      </ProfileProvider>
    </BrowserRouter>,
  ));
}

async function click(target: Element) {
  await act(async () => {
    target.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

async function navigate(path: string) {
  await act(async () => {
    window.history.pushState({}, "", path);
    window.dispatchEvent(new PopStateEvent("popstate"));
  });
}

async function selectBoard(slug: string) {
  const select = Array.from(container.querySelectorAll("select")).find((candidate) =>
    Array.from(candidate.options).some((option) => option.value === slug),
  );
  if (!select) throw new Error(`missing board switcher for ${slug}`);
  await act(async () => {
    select.value = slug;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function waitForText(text: string) {
  await vi.waitFor(() => expect(container.textContent).toContain(text));
}

function card(id: string) {
  const target = container.querySelector(`[data-task-id="${id}"]`);
  if (!target) throw new Error(`missing card ${id}`);
  return target;
}

beforeEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  FakeWebSocket.instances = [];
  window.__HERMES_PLUGIN_SDK__!.basePath = "";
  window.localStorage.clear();
  document.body.innerHTML = "";
  installApi();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("Kanban browser deep links", () => {
  it("lets a canonical board+task URL override localStorage and survive refresh", async () => {
    window.localStorage.setItem("hermes.kanban.selectedBoard", "default");
    await renderKanban("/kanban?profile=work&board=ops&task=t_ops");

    await waitForText("Ops task");
    expect(container.querySelector(".hermes-kanban-drawer")).not.toBeNull();
    expect(window.location.search).toBe("?profile=work&board=ops&task=t_ops");
    expect(window.localStorage.getItem("hermes.kanban.selectedBoard")).toBe("ops");
    expect(
      fetchJSON.mock.calls.some(([url]) => String(url).includes("/tasks/locate/")),
    ).toBe(false);

    await act(async () => root.unmount());
    root = createRoot(container);
    await act(async () => root.render(
      <BrowserRouter>
        <KanbanPage />
      </BrowserRouter>,
    ));
    await waitForText("Ops task");
  });

  it("locates a unique cross-board task, switches boards, and opens it", async () => {
    window.localStorage.setItem("hermes.kanban.selectedBoard", "default");
    await renderKanban("/kanban?profile=work&task=t_ops");

    await waitForText("Ops task");
    expect(container.querySelector(".hermes-kanban-drawer")).not.toBeNull();
    expect(window.localStorage.getItem("hermes.kanban.selectedBoard")).toBe("ops");
    expect(window.location.search).toBe("?profile=work&board=ops&task=t_ops");
    expect(fetchJSON.mock.calls.filter(
      ([url]) => String(url).endsWith("/tasks/locate/t_ops"),
    )).toHaveLength(1);
    expect(container.textContent).not.toContain("outside the current filters");
  });

  it("keeps router state synchronized through the production profile shell and basename", async () => {
    await renderProductionShell(
      "/dashboard/kanban?profile=work&view=compact&board=ops",
    );
    await waitForText("Ops task");
    const push = vi.spyOn(window.history, "pushState");

    await click(card("t_ops"));
    await vi.waitFor(() => expect(container.querySelector(".hermes-kanban-drawer")).not.toBeNull());
    expect(window.location.pathname).toBe("/dashboard/kanban");
    expect(window.location.search).toBe(
      "?profile=work&view=compact&board=ops&task=t_ops",
    );
    expect(push).toHaveBeenCalledTimes(1);

    const switcher = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Use research profile",
    );
    if (!switcher) throw new Error("missing profile switcher");
    await click(switcher);
    await vi.waitFor(() => expect(window.location.search).toContain("profile=research"));

    expect(window.location.pathname).toBe("/dashboard/kanban");
    expect(window.location.search).toContain("view=compact");
    expect(window.location.search).toContain("board=ops");
    expect(window.location.search).toContain("task=t_ops");
    expect(container.querySelector(".hermes-kanban-drawer")).not.toBeNull();
  });

  it("preserves a task-only target while ignoring a stale stored board", async () => {
    let resolveLookup!: (value: unknown) => void;
    const pendingLookup = new Promise((resolve) => { resolveLookup = resolve; });
    window.localStorage.setItem("hermes.kanban.selectedBoard", "deleted");
    const implementation = fetchJSON.getMockImplementation();
    fetchJSON.mockImplementation((rawUrl: string, init?: RequestInit) => {
      if (rawUrl.endsWith("/tasks/locate/t_ops")) return pendingLookup;
      return implementation?.(rawUrl, init);
    });

    await renderKanban("/kanban?profile=work&task=t_ops");
    await vi.waitFor(() => expect(fetchJSON.mock.calls.some(
      ([url]) => String(url).endsWith("/tasks/locate/t_ops"),
    )).toBe(true));
    expect(window.location.search).toBe("?profile=work&task=t_ops");

    await act(async () => resolveLookup({ board: "ops", task: opsTask }));
    await waitForText("Ops task");
    expect(window.location.search).toBe("?profile=work&board=ops&task=t_ops");
  });

  it("canonicalizes legacy task_id and preserves the bare-route localStorage fallback", async () => {
    window.localStorage.setItem("hermes.kanban.selectedBoard", "ops");
    await renderKanban("/kanban?profile=work&task_id=t_ops");

    await waitForText("Ops task");
    await vi.waitFor(() => expect(container.querySelector(".hermes-kanban-drawer")).not.toBeNull());
    expect(window.location.search).toBe("?profile=work&board=ops&task=t_ops");

    await act(async () => root.unmount());
    window.history.replaceState({}, "", "/kanban?profile=work");
    root = createRoot(container);
    await act(async () => root.render(
      <BrowserRouter>
        <KanbanPage />
      </BrowserRouter>,
    ));
    await waitForText("Ops task");
    expect(container.querySelector(".hermes-kanban-drawer")).toBeNull();
    expect(window.location.search).toBe("?profile=work&board=ops");
  });

  it.each([
    ["missing", new Map<string, unknown>(), "not found or is archived"],
    [
      "ambiguous",
      new Map<string, unknown>([
        ["t_problem", new Error('409: {"detail":"task id is ambiguous across active boards"}')],
      ]),
      "ambiguous across active boards",
    ],
  ])("cleans up a %s task link without guessing a board", async (_kind, locate, notice) => {
    installApi({ locate });
    await renderKanban("/kanban?profile=work&task=t_problem");

    await waitForText(notice);
    expect(container.querySelector(".hermes-kanban-drawer")).toBeNull();
    expect(window.location.search).toBe("?profile=work&board=default");
  });

  it("rejects an ambiguous task-only link even when one match is visible locally", async () => {
    installApi({
      locate: new Map<string, unknown>([
        ["t_default", new Error('409: {"detail":"task id is ambiguous across active boards"}')],
      ]),
    });
    await renderKanban("/kanban?task=t_default");

    await waitForText("ambiguous across active boards");
    expect(container.querySelector(".hermes-kanban-drawer")).toBeNull();
    expect(
      fetchJSON.mock.calls.some(([url]) => String(url).endsWith("/tasks/locate/t_default")),
    ).toBe(true);
  });

  it("falls back visibly from an invalid board and removes its task target", async () => {
    await renderKanban("/kanban?profile=work&board=missing&task=t_lost");

    await waitForText("Board missing was not found or is archived");
    await waitForText("Default task");
    expect(container.querySelector(".hermes-kanban-drawer")).toBeNull();
    expect(window.location.search).toBe("?profile=work&board=default");
    expect(window.localStorage.getItem("hermes.kanban.selectedBoard")).not.toBe("missing");
  });

  it.each([
    ["invalid board", "/kanban?board=missing&task=t_lost", "Board missing was not found"],
    ["missing task", "/kanban?board=ops&task=t_lost", "was not found or is archived"],
    ["archived task", "/kanban?board=default&task=t_archived", "is archived"],
    ["ambiguous task", "/kanban?task=t_problem", "ambiguous across active boards"],
  ])("does not write history while traversing a %s destination", async (_kind, path, notice) => {
    const archived = task("t_archived", "Archived task", "archived");
    installApi({
      boards: [
        { slug: "default", tasks: [defaultTask, archived] },
        { slug: "ops", tasks: [opsTask] },
      ],
      locate: new Map<string, unknown>([
        ["t_problem", new Error('409: {"detail":"task id is ambiguous across active boards"}')],
      ]),
    });
    window.history.replaceState({}, "", "/kanban?board=default");
    window.history.pushState({}, "", path);
    window.history.pushState({}, "", "/kanban?board=ops");
    await renderKanban("/kanban?board=ops");
    await waitForText("Ops task");
    const push = vi.spyOn(window.history, "pushState");
    const replace = vi.spyOn(window.history, "replaceState");

    const backEvent = new Promise((resolve) =>
      window.addEventListener("popstate", resolve, { once: true }),
    );
    window.history.back();
    await act(async () => backEvent);
    await waitForText(notice);

    expect(push).not.toHaveBeenCalled();
    expect(replace).not.toHaveBeenCalled();
    expect(window.location.search).toBe(new URL(path, window.location.origin).search);

    const forwardEvent = new Promise((resolve) =>
      window.addEventListener("popstate", resolve, { once: true }),
    );
    window.history.forward();
    await act(async () => forwardEvent);
    await waitForText("Ops task");
    expect(push).not.toHaveBeenCalled();
    expect(replace).not.toHaveBeenCalled();
  });

  it("opens a board-qualified active task outside the visible columns via detail", async () => {
    const implementation = fetchJSON.getMockImplementation();
    fetchJSON.mockImplementation((rawUrl: string, init?: RequestInit) => {
      const url = new URL(rawUrl, window.location.origin);
      if (url.pathname.endsWith("/board") && url.searchParams.get("board") === "ops") {
        return Promise.resolve(board({ slug: "ops", tasks: [] }));
      }
      return implementation?.(rawUrl, init);
    });

    await renderKanban("/kanban?board=ops&task=t_ops");

    await waitForText("outside the current filters");
    await waitForText("Ops task");
    expect(container.querySelector(".hermes-kanban-drawer")).not.toBeNull();
    expect(
      fetchJSON.mock.calls.some(([url]) =>
        String(url).includes("/tasks/t_ops?board=ops"),
      ),
    ).toBe(true);
    expect(
      fetchJSON.mock.calls.some(([url]) => String(url).includes("/tasks/locate/")),
    ).toBe(false);
  });

  it("rejects a board-qualified archived task with a visible notice", async () => {
    const archived = task("t_archived", "Archived task", "archived");
    installApi({ boards: [{ slug: "default", tasks: [archived] }] });

    await renderKanban("/kanban?board=default&task=t_archived");

    await waitForText("Card t_archived is archived");
    expect(container.querySelector(".hermes-kanban-drawer")).toBeNull();
    expect(window.location.search).toBe("?board=default");
  });

  it.each([
    ["transport", new Error("Failed to fetch")],
    ["server", new Error('500: {"detail":"temporary failure"}')],
    ["corrupt response", { unexpected: true }],
  ])("keeps a %s lookup failure retryable without changing the URL", async (_kind, failure) => {
    installApi({ locate: new Map([["t_ops", failure]]) });
    await renderKanban("/kanban?profile=work&task=t_ops");

    await waitForText("Could not load card t_ops");
    expect(container.querySelector(".hermes-kanban-drawer")).toBeNull();
    expect(window.location.search).toBe("?profile=work&task=t_ops");
    expect(container.textContent).toContain("retry");
  });

  it("keeps a transient board-qualified detail failure retryable", async () => {
    const implementation = fetchJSON.getMockImplementation();
    fetchJSON.mockImplementation((rawUrl: string, init?: RequestInit) => {
      const url = new URL(rawUrl, window.location.origin);
      if (url.pathname.endsWith("/board") && url.searchParams.get("board") === "ops") {
        return Promise.resolve(board({ slug: "ops", tasks: [] }));
      }
      if (rawUrl.includes("/tasks/t_ops?board=ops")) {
        return Promise.reject(new Error('500: {"detail":"temporary failure"}'));
      }
      return implementation?.(rawUrl, init);
    });
    await renderKanban("/kanban?board=ops&task=t_ops");

    await waitForText("Could not load card t_ops");
    expect(container.querySelector(".hermes-kanban-drawer")).toBeNull();
    expect(window.location.search).toBe("?board=ops&task=t_ops");
  });

  it("deduplicates a pending locator across WebSocket board refreshes", async () => {
    let resolveLookup!: (value: unknown) => void;
    const pendingLookup = new Promise((resolve) => { resolveLookup = resolve; });
    const implementation = fetchJSON.getMockImplementation();
    fetchJSON.mockImplementation((rawUrl: string, init?: RequestInit) => {
      if (rawUrl.endsWith("/tasks/locate/t_ops")) return pendingLookup;
      return implementation?.(rawUrl, init);
    });
    await renderKanban("/kanban?task=t_ops");
    await vi.waitFor(() => expect(fetchJSON.mock.calls.filter(
      ([url]) => String(url).endsWith("/tasks/locate/t_ops"),
    )).toHaveLength(1));
    await vi.waitFor(() => expect(FakeWebSocket.instances.length).toBeGreaterThan(0));
    const push = vi.spyOn(window.history, "pushState");
    const replace = vi.spyOn(window.history, "replaceState");

    await act(async () => {
      FakeWebSocket.instances.at(-1)?.emit({
        cursor: 1,
        events: [{ id: 1, task_id: "t_default" }],
      });
      await new Promise((resolve) => setTimeout(resolve, 300));
    });
    expect(fetchJSON.mock.calls.filter(
      ([url]) => String(url).endsWith("/tasks/locate/t_ops"),
    )).toHaveLength(1);
    expect(window.location.search).toBe("?task=t_ops");
    expect(container.querySelector(".hermes-kanban-drawer")).toBeNull();
    expect(container.textContent).not.toContain("Could not load card");
    expect(window.localStorage.getItem("hermes.kanban.selectedBoard")).toBe("default");
    expect(push).not.toHaveBeenCalled();
    expect(replace).not.toHaveBeenCalled();

    await act(async () => resolveLookup({ board: "ops", task: opsTask }));
    await waitForText("Ops task");
    expect(fetchJSON.mock.calls.filter(
      ([url]) => String(url).endsWith("/tasks/locate/t_ops"),
    )).toHaveLength(1);
    expect(container.querySelector(".hermes-kanban-drawer")).not.toBeNull();
    expect(container.textContent).not.toContain("outside the current filters");
    expect(container.textContent).not.toContain("Could not load card");
    expect(window.localStorage.getItem("hermes.kanban.selectedBoard")).toBe("ops");
    expect(window.location.search).toBe("?board=ops&task=t_ops");
    expect(push).not.toHaveBeenCalled();
    expect(replace).toHaveBeenCalledTimes(1);
  });

  it("uses pushState for open/close and performs zero writes across Back/Back/Forward", async () => {
    await renderKanban("/kanban?board=ops");
    await waitForText("Ops task");
    const push = vi.spyOn(window.history, "pushState");
    const replace = vi.spyOn(window.history, "replaceState");

    await click(card("t_ops"));
    await vi.waitFor(() => expect(container.querySelector(".hermes-kanban-drawer")).not.toBeNull());
    expect(push).toHaveBeenCalledTimes(1);
    expect(replace).not.toHaveBeenCalled();
    expect(window.location.search).toBe("?board=ops&task=t_ops");

    const close = container.querySelector<HTMLButtonElement>(".hermes-kanban-drawer-close");
    if (!close) throw new Error("missing close button");
    await click(close);
    expect(push).toHaveBeenCalledTimes(2);
    expect(window.location.search).toBe("?board=ops");
    push.mockClear();
    replace.mockClear();

    const firstBack = new Promise((resolve) =>
      window.addEventListener("popstate", resolve, { once: true }),
    );
    window.history.back();
    await act(async () => firstBack);
    await vi.waitFor(() => expect(container.querySelector(".hermes-kanban-drawer")).not.toBeNull());
    expect(window.location.search).toBe("?board=ops&task=t_ops");

    const secondBack = new Promise((resolve) =>
      window.addEventListener("popstate", resolve, { once: true }),
    );
    window.history.back();
    await act(async () => secondBack);
    await vi.waitFor(() => expect(container.querySelector(".hermes-kanban-drawer")).toBeNull());
    expect(window.location.search).toBe("?board=ops");

    const forward = new Promise((resolve) =>
      window.addEventListener("popstate", resolve, { once: true }),
    );
    window.history.forward();
    await act(async () => forward);
    await vi.waitFor(() => expect(container.querySelector(".hermes-kanban-drawer")).not.toBeNull());
    expect(window.location.search).toBe("?board=ops&task=t_ops");
    expect(push).not.toHaveBeenCalled();
    expect(replace).not.toHaveBeenCalled();
  });

  it.each(["shade", "Escape"])("pushes board-only history when closing by %s", async (method) => {
    await renderKanban("/kanban?board=ops&task=t_ops");
    await waitForText("Ops task");
    const push = vi.spyOn(window.history, "pushState");

    if (method === "shade") {
      const shade = container.querySelector(".hermes-kanban-drawer-shade");
      if (!shade) throw new Error("missing drawer shade");
      await click(shade);
    } else {
      await act(async () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" })));
    }

    await vi.waitFor(() => expect(container.querySelector(".hermes-kanban-drawer")).toBeNull());
    expect(push).toHaveBeenCalledTimes(1);
    expect(window.location.search).toBe("?board=ops");
  });

  it("pushes a board switch, clears the drawer, and persists the destination", async () => {
    await renderKanban("/kanban?board=ops&task=t_ops");
    await waitForText("Ops task");
    const push = vi.spyOn(window.history, "pushState");

    await selectBoard("default");

    await waitForText("Default task");
    expect(container.querySelector(".hermes-kanban-drawer")).toBeNull();
    expect(push).toHaveBeenCalledTimes(1);
    expect(window.location.search).toBe("?board=default");
    expect(window.localStorage.getItem("hermes.kanban.selectedBoard")).toBe("default");
  });

  it("suppresses a stale lookup after a newer popstate navigation", async () => {
    let resolveLookup!: (value: unknown) => void;
    const pendingLookup = new Promise((resolve) => { resolveLookup = resolve; });
    const implementation = fetchJSON.getMockImplementation();
    fetchJSON.mockImplementation((rawUrl: string, init?: RequestInit) => {
      if (rawUrl.endsWith("/tasks/locate/t_ops")) return pendingLookup;
      return implementation?.(rawUrl, init);
    });
    window.localStorage.setItem("hermes.kanban.selectedBoard", "default");
    await renderKanban("/kanban?task=t_ops");
    await vi.waitFor(() =>
      expect(fetchJSON.mock.calls.some(([url]) => String(url).endsWith("/tasks/locate/t_ops"))).toBe(true),
    );

    await act(async () => {
      window.history.pushState({}, "", "/kanban?board=default&task=t_default");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    await waitForText("Default task");
    await act(async () => resolveLookup({ board: "ops", task: opsTask }));
    expect(window.location.search).toBe("?board=default&task=t_default");
    expect(container.textContent).not.toContain("Ops task");
  });

  it("suppresses a stale qualified detail after navigating to a newer task", async () => {
    let resolveDetail!: (value: unknown) => void;
    const pendingDetail = new Promise((resolve) => { resolveDetail = resolve; });
    const implementation = fetchJSON.getMockImplementation();
    fetchJSON.mockImplementation((rawUrl: string, init?: RequestInit) => {
      if (rawUrl.includes("/tasks/t_filtered?board=ops")) return pendingDetail;
      const url = new URL(rawUrl, window.location.origin);
      if (url.pathname.endsWith("/board") && url.searchParams.get("board") === "ops") {
        return Promise.resolve(board({ slug: "ops", tasks: [opsTask] }));
      }
      return implementation?.(rawUrl, init);
    });
    await renderKanban("/kanban?board=ops&task=t_filtered");
    await vi.waitFor(() =>
      expect(fetchJSON.mock.calls.some(([url]) => String(url).includes("/tasks/t_filtered"))).toBe(true),
    );

    await navigate("/kanban?board=ops&task=t_ops");
    await waitForText("Ops task");
    await act(async () => resolveDetail(detail(task("t_filtered", "Stale filtered task"))));

    expect(window.location.search).toBe("?board=ops&task=t_ops");
    expect(container.textContent).not.toContain("Stale filtered task");
  });

  it("lets only the newest A response win in an A-B-A board race", async () => {
    const oldA = task("t_old_a", "Old A response");
    const newA = task("t_new_a", "Newest A response");
    let resolveOldA!: (value: unknown) => void;
    let resolveNewA!: (value: unknown) => void;
    const oldAPromise = new Promise((resolve) => { resolveOldA = resolve; });
    const newAPromise = new Promise((resolve) => { resolveNewA = resolve; });
    let defaultRequests = 0;
    const implementation = fetchJSON.getMockImplementation();
    fetchJSON.mockImplementation((rawUrl: string, init?: RequestInit) => {
      const url = new URL(rawUrl, window.location.origin);
      if (url.pathname.endsWith("/board") && url.searchParams.get("board") === "default") {
        defaultRequests += 1;
        return defaultRequests === 1 ? oldAPromise : newAPromise;
      }
      return implementation?.(rawUrl, init);
    });
    await renderKanban("/kanban?board=ops");
    await waitForText("Ops task");

    await navigate("/kanban?board=default");
    await vi.waitFor(() => expect(defaultRequests).toBe(1));
    await navigate("/kanban?board=ops");
    await waitForText("Ops task");
    await navigate("/kanban?board=default");
    await vi.waitFor(() => expect(defaultRequests).toBe(2));
    await act(async () => resolveNewA(board({ slug: "default", tasks: [newA] })));
    await waitForText("Newest A response");
    await act(async () => resolveOldA(board({ slug: "default", tasks: [oldA] })));

    expect(container.textContent).toContain("Newest A response");
    expect(container.textContent).not.toContain("Old A response");
  });

  it("follows task links reached through browser Back and Forward", async () => {
    window.localStorage.setItem("hermes.kanban.selectedBoard", "default");
    window.history.replaceState({}, "", "/kanban?board=default&task=t_default");
    window.history.pushState({}, "", "/kanban?board=ops&task=t_ops");
    await renderKanban("/kanban?board=ops&task=t_ops");
    await waitForText("Ops task");

    const backEvent = new Promise((resolve) =>
      window.addEventListener("popstate", resolve, { once: true }),
    );
    window.history.back();
    await act(async () => backEvent);
    await waitForText("Default task");
    expect(window.location.search).toBe("?board=default&task=t_default");

    const forwardEvent = new Promise((resolve) =>
      window.addEventListener("popstate", resolve, { once: true }),
    );
    window.history.forward();
    await act(async () => forwardEvent);
    await waitForText("Ops task");
    expect(window.location.search).toBe("?board=ops&task=t_ops");
  });

  it("copies the actual drawer identity with profile context and exposes fallback failure", async () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    await renderKanban("/kanban?profile=work&board=ops&task=t_ops&search=ignored");
    await waitForText("Ops task");
    // Deliberately desynchronize the address bar without notifying the plugin.
    // Copy Link must use the mounted drawer's props, not these stale URL ids.
    window.history.replaceState({}, "", "/kanban?profile=work&board=default&task=t_wrong");
    const copy = Array.from(container.querySelectorAll("button")).find((button) =>
      button.textContent?.includes("Copy link"),
    );
    if (!copy) throw new Error("missing copy link button");
    await click(copy);
    expect(writeText).toHaveBeenCalledWith(
      `${window.location.origin}/kanban?profile=work&board=ops&task=t_ops`,
    );
    await waitForText("Copied");

    writeText.mockRejectedValueOnce(new Error("clipboard denied"));
    const execCommand = vi.fn(() => true);
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: execCommand,
    });
    await click(copy);
    await vi.waitFor(() => expect(execCommand).toHaveBeenCalledWith("copy"));

    writeText.mockRejectedValueOnce(new Error("clipboard denied again"));
    execCommand.mockReturnValueOnce(false);
    await click(copy);
    await waitForText("Copy command was not accepted");
  });

  it("opens a related child with one task-to-task history entry", async () => {
    const parent = task("t_parent", "Parent task");
    const child = task("t_child", "Child task");
    let resolveChild!: (value: unknown) => void;
    const pendingChild = new Promise((resolve) => { resolveChild = resolve; });
    installApi({ boards: [{ slug: "ops", tasks: [parent, child] }] });
    const implementation = fetchJSON.getMockImplementation();
    fetchJSON.mockImplementation((rawUrl: string, init?: RequestInit) => {
      if (rawUrl.includes("/tasks/t_parent?board=ops")) {
        return Promise.resolve(detail(parent, {
          links: { parents: [], children: [child.id] },
          child_results: [child],
        }));
      }
      if (rawUrl.includes("/tasks/t_child?board=ops")) return pendingChild;
      return implementation?.(rawUrl, init);
    });
    await renderKanban("/kanban?board=ops&task=t_parent");
    await waitForText("Parent task");
    await waitForText("Child Results");
    const push = vi.spyOn(window.history, "pushState");
    const open = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Open",
    );
    if (!open) throw new Error("missing related-task open button");

    await click(open);
    const drawer = container.querySelector(".hermes-kanban-drawer");
    if (!drawer) throw new Error("missing task drawer");
    expect(drawer.textContent).not.toContain("Parent task");
    await act(async () => resolveChild(detail(child)));
    await vi.waitFor(() => expect(drawer.textContent).toContain("Child task"));
    expect(push).toHaveBeenCalledTimes(1);
    expect(window.location.search).toBe("?board=ops&task=t_child");
  });
});
