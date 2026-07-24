/** @vitest-environment jsdom */

import { act, useState, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "@/i18n";
import { api, type SessionInfo } from "@/lib/api";
import {
  SessionRow,
  SessionStatsSummary,
  SessionsViewTabs,
} from "./SessionsPage";

const SESSION: SessionInfo = {
  id: "session-1",
  source: "android-dashboard",
  model: "provider/a-very-long-mobile-model-name",
  title: "A session title that remains readable on a narrow phone",
  started_at: 1_700_000_000,
  ended_at: 1_700_000_100,
  last_active: 1_700_000_100,
  is_active: false,
  message_count: 12,
  tool_call_count: 3,
  input_tokens: 100,
  output_tokens: 50,
  preview: "Preview",
};

function SessionRowHarness({ onExport }: { onExport: (id: string) => void }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <MemoryRouter>
      <I18nProvider>
        <SessionRow
          isExpanded={expanded}
          isSelected={false}
          onDelete={() => {}}
          onExport={onExport}
          onRename={async () => {}}
          onSelectClick={() => {}}
          onToggle={() => setExpanded((value) => !value)}
          resumeInChatEnabled={false}
          session={SESSION}
        />
      </I18nProvider>
    </MemoryRouter>
  );
}

describe("SessionsPage mobile structure", () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
    vi.restoreAllMocks();
  });

  function render(node: ReactNode) {
    act(() => root.render(node));
  }

  it("uses a deterministic 2x2 phone grid and separates platform counters", () => {
    render(
      <SessionStatsSummary
        stats={{
          total: 40,
          active_store: 8,
          archived: 12,
          messages: 900,
          by_source: { "a-very-long-platform-name": 17 },
        }}
      />,
    );

    const primary = host.querySelector("dl");
    const platforms = host.querySelector<HTMLElement>(
      "[aria-label='Sessions by platform']",
    );

    expect(primary?.children).toHaveLength(4);
    expect(primary?.className).toContain("grid-cols-2");
    expect(primary?.className).toContain("sm:flex");
    expect(platforms).not.toBeNull();
    expect(primary?.contains(platforms)).toBe(false);
    expect(platforms?.className).toContain("border-t");
    expect(platforms?.className).toContain("sm:border-l");
  });

  it("renders full-width, touch-sized tabs with radio semantics", () => {
    const onChange = vi.fn();
    render(
      <I18nProvider>
        <SessionsViewTabs value="overview" onChange={onChange} />
      </I18nProvider>,
    );

    const group = host.querySelector<HTMLElement>("[role='radiogroup']");
    const tabs = Array.from(
      host.querySelectorAll<HTMLButtonElement>("[role='radio']"),
    );

    expect(group?.className).toContain("w-full");
    expect(group?.className).toContain("[&_button]:min-h-11");
    expect(group?.className).toContain("[&_button]:flex-1");
    expect(tabs).toHaveLength(2);
    expect(tabs[0].getAttribute("aria-checked")).toBe("true");
    expect(tabs[1].getAttribute("aria-checked")).toBe("false");

    act(() => tabs[1].click());
    expect(onChange).toHaveBeenCalledWith("list");
  });

  it("uses a native expansion button with synchronized ARIA state", async () => {
    vi.spyOn(api, "getSessionMessages").mockResolvedValue({
      session_id: SESSION.id,
      messages: [{ role: "user", content: "A message" }],
    });
    render(<SessionRowHarness onExport={() => {}} />);

    const collapsed = host.querySelector<HTMLButtonElement>(
      "button[aria-controls]",
    );
    expect(collapsed).toBeInstanceOf(HTMLButtonElement);
    expect(collapsed?.type).toBe("button");
    expect(collapsed?.tabIndex).toBe(0);
    expect(collapsed?.getAttribute("aria-expanded")).toBe("false");
    collapsed?.focus();
    expect(document.activeElement).toBe(collapsed);

    await act(async () => {
      collapsed?.click();
      await Promise.resolve();
    });

    const expanded = host.querySelector<HTMLButtonElement>(
      "button[aria-controls]",
    );
    const panelId = expanded?.getAttribute("aria-controls");
    expect(expanded?.getAttribute("aria-expanded")).toBe("true");
    expect(panelId).toBeTruthy();
    expect(document.getElementById(panelId ?? "")?.getAttribute("role")).toBe(
      "region",
    );
    const messageList = host.querySelector<HTMLElement>(
      "[data-session-message-list]",
    );
    expect(messageList?.className).toContain("max-h-[55dvh]");
    expect(messageList?.className).toContain("lg:max-h-[600px]");
  });

  it("keeps session actions independent from expansion", () => {
    const onExport = vi.fn();
    render(<SessionRowHarness onExport={onExport} />);

    const expand = host.querySelector<HTMLButtonElement>(
      "button[aria-controls]",
    );
    const exportButton = host.querySelector<HTMLButtonElement>(
      "button[aria-label='Export session']",
    );

    act(() => exportButton?.click());
    expect(onExport).toHaveBeenCalledWith(SESSION.id);
    expect(expand?.getAttribute("aria-expanded")).toBe("false");
  });
});
