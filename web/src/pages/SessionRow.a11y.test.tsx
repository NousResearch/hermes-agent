// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Regression tests for #96918: iOS WebKit does not reliably synthesize click
// events from taps on non-interactive elements, so a bare `<div onClick>`
// session row header and a bare `<th onClick>` sortable header are dead on
// iOS Safari/Brave while working on desktop Chromium. The fix gives the row
// header role="button" + tabIndex + aria-expanded + Enter/Space keydown, and
// makes SortHeader focusable/keyboard-operable with aria-sort state.

const apiMocks = vi.hoisted(() => ({
  getSessionMessages: vi.fn(async () => ({
    session_id: "s1",
    messages: [{ role: "user", content: "hello", timestamp: 1 }],
  })),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, api: { ...actual.api, ...apiMocks } };
});

vi.mock("@/i18n", () => ({
  useI18n: () => ({
    t: {
      common: {
        live: "Live",
        msgs: "msgs",
        tools: "tools",
        of: "of",
        page: "Page",
        collapse: "Collapse",
        expand: "Expand",
      },
      sessions: {
        untitledSession: "Untitled session",
        noMessages: "No messages",
        deleteSession: "Delete session",
        selectSession: "Select session",
        resumeInChat: "Resume in Chat",
        roles: { user: "User", assistant: "Assistant", system: "System", tool: "Tool" },
      },
      analytics: {},
    },
  }),
}));

vi.mock("@/components/Markdown", () => ({
  Markdown: ({ content }: { content: string }) => <>{content}</>,
}));

vi.mock("@/plugins", () => ({
  PluginSlot: () => null,
}));

import { SessionRow } from "./SessionsPage";
import { SortHeader } from "./AnalyticsPage";
import type { SessionInfo } from "@/lib/api";

let container: HTMLDivElement;
let root: Root;

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

beforeEach(() => {
  apiMocks.getSessionMessages.mockClear();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

const baseSession: SessionInfo = {
  id: "s1",
  source: "cli",
  model: "org/model-x",
  title: "Test session",
  started_at: 0,
  ended_at: null,
  last_active: 0,
  is_active: false,
  message_count: 3,
  tool_call_count: 0,
  input_tokens: 0,
  output_tokens: 0,
  preview: "hello world",
};

function makeRowProps(overrides: Partial<Parameters<typeof SessionRow>[0]> = {}) {
  return {
    session: baseSession,
    isExpanded: false,
    isSelected: false,
    onToggle: vi.fn(),
    onSelectClick: vi.fn(),
    onDelete: vi.fn(),
    onExport: vi.fn(),
    onRename: vi.fn(async () => {}),
    resumeInChatEnabled: false,
    ...overrides,
  };
}

function keydown(el: Element, key: string, target?: Element) {
  const event = new KeyboardEvent("keydown", { key, bubbles: true });
  (target ?? el).dispatchEvent(event);
}

describe("SessionRow a11y (#96918)", () => {
  it("exposes the header as an accessible toggle control", async () => {
    await render(
      <MemoryRouter>
        <SessionRow {...makeRowProps()} />
      </MemoryRouter>,
    );
    const header = container.querySelector('div[role="button"]');
    expect(header).not.toBeNull();
    expect(header!.getAttribute("tabindex")).toBe("0");
    expect(header!.getAttribute("aria-expanded")).toBe("false");
  });

  it("toggles via Enter and Space keydown on the focused header", async () => {
    const onToggle = vi.fn();
    await render(
      <MemoryRouter>
        <SessionRow {...makeRowProps({ onToggle })} />
      </MemoryRouter>,
    );
    const header = container.querySelector('div[role="button"]')!;
    await act(async () => {
      keydown(header, "Enter");
    });
    await act(async () => {
      keydown(header, " ");
    });
    expect(onToggle).toHaveBeenCalledTimes(2);
    await act(async () => {
      keydown(header, "a");
    });
    expect(onToggle).toHaveBeenCalledTimes(2);
  });

  it("ignores keydown originating from a nested focused control", async () => {
    const onToggle = vi.fn();
    await render(
      <MemoryRouter>
        <SessionRow {...makeRowProps({ onToggle })} />
      </MemoryRouter>,
    );
    const header = container.querySelector('div[role="button"]')!;
    const nested = header.querySelector("span")!;
    await act(async () => {
      keydown(header, "Enter", nested);
    });
    expect(onToggle).not.toHaveBeenCalled();
  });

  it("lazy-loads messages when expanded", async () => {
    await render(
      <MemoryRouter>
        <SessionRow {...makeRowProps({ isExpanded: true })} />
      </MemoryRouter>,
    );
    await act(async () => {});
    expect(apiMocks.getSessionMessages).toHaveBeenCalledWith("s1");
    expect(container.textContent).toContain("hello");
  });
});

describe("SortHeader a11y (#96918)", () => {
  function renderHeader(props: {
    col?: string;
    sortKey?: string;
    sortDir?: "asc" | "desc";
    toggle?: (key: string) => void;
  }) {
    const toggle =
      props.toggle ?? vi.fn();
    const ui = (
      <table>
        <thead>
          <tr>
            <SortHeader
              label="Messages"
              col={props.col ?? "messages"}
              sortKey={props.sortKey ?? "messages"}
              sortDir={props.sortDir ?? "asc"}
              toggle={toggle}
            />
          </tr>
        </thead>
      </table>
    );
    return { toggle, promise: render(ui) };
  }

  it("is focusable and reports aria-sort ascending when active asc", async () => {
    const { promise } = renderHeader({ sortDir: "asc" });
    await promise;
    const th = container.querySelector("th")!;
    expect(th.getAttribute("tabindex")).toBe("0");
    expect(th.getAttribute("aria-sort")).toBe("ascending");
  });

  it("reports aria-sort descending when active desc", async () => {
    const { promise } = renderHeader({ sortDir: "desc" });
    await promise;
    const th = container.querySelector("th")!;
    expect(th.getAttribute("aria-sort")).toBe("descending");
  });

  it("reports aria-sort none when the column is not the active sort key", async () => {
    const { promise } = renderHeader({ col: "messages", sortKey: "tokens" });
    await promise;
    const th = container.querySelector("th")!;
    expect(th.getAttribute("aria-sort")).toBe("none");
  });

  it("toggles via Enter and Space keydown on the th", async () => {
    const toggle = vi.fn();
    const { promise } = renderHeader({ toggle });
    await promise;
    const th = container.querySelector("th")!;
    await act(async () => {
      keydown(th, "Enter");
    });
    await act(async () => {
      keydown(th, " ");
    });
    expect(toggle).toHaveBeenCalledWith("messages");
    expect(toggle).toHaveBeenCalledTimes(2);
    await act(async () => {
      keydown(th, "Tab");
    });
    expect(toggle).toHaveBeenCalledTimes(2);
  });

  it("still toggles on click", async () => {
    const toggle = vi.fn();
    const { promise } = renderHeader({ toggle });
    await promise;
    const th = container.querySelector("th")!;
    await act(async () => {
      th.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(toggle).toHaveBeenCalledWith("messages");
  });
});