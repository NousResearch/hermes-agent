/** @vitest-environment jsdom */

import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OperationalActionGroup } from "@/components/OperationalActionGroup";
import { PageHeaderProvider } from "@/contexts/PageHeaderProvider";
import { I18nProvider } from "@/i18n";
import { api, type ManagedFileEntry } from "@/lib/api";
import FilesPage, { MobileFilesList } from "./FilesPage";
import { LogOutput, LogsFilterBar } from "./LogsPage";

const DIRECTORY: ManagedFileEntry = {
  is_directory: true,
  mime_type: null,
  mtime: 1_700_000_000,
  name: "a-very-long-mobile-directory-name-that-must-wrap",
  path: "/managed/a-very-long-mobile-directory-name-that-must-wrap",
  size: null,
};

const FILE: ManagedFileEntry = {
  is_directory: false,
  mime_type: "text/plain",
  mtime: 1_700_000_100,
  name: "operational-report.txt",
  path: "/managed/operational-report.txt",
  size: 2048,
};

describe("Operational page mobile foundations", () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
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

  async function flushEffects() {
    await act(async () => {
      await Promise.resolve();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it("keeps the desktop file table while selecting the mobile list below lg", async () => {
    vi.spyOn(api, "listFiles").mockResolvedValue({
      can_change_path: true,
      entries: [FILE],
      locked_root: "/managed",
      parent: null,
      path: "/managed",
      root: "/managed",
    });

    render(
      <MemoryRouter initialEntries={["/files"]}>
        <I18nProvider>
          <PageHeaderProvider pluginTabs={[]}>
            <FilesPage />
          </PageHeaderProvider>
        </I18nProvider>
      </MemoryRouter>,
    );
    await flushEffects();

    const mobile = host.querySelector<HTMLElement>("[data-testid='files-mobile-list']");
    const desktop = host.querySelector<HTMLElement>("[data-testid='files-desktop-table']");

    expect(api.listFiles).toHaveBeenCalled();
    expect(api.listFiles).toHaveBeenLastCalledWith("/managed");
    expect(mobile?.className).toContain("lg:hidden");
    expect(desktop?.className).toContain("hidden");
    expect(desktop?.className).toContain("lg:block");
    expect(mobile?.textContent).toContain(FILE.name);
    expect(desktop?.textContent).toContain(FILE.name);
  });

  it("renders the phone file list with complete metadata and unchanged actions", () => {
    const onDelete = vi.fn();
    const onDownload = vi.fn();
    const onOpen = vi.fn();
    const onOpenParent = vi.fn();

    render(
      <MobileFilesList
        entries={[DIRECTORY, FILE]}
        parent="/managed"
        onDelete={onDelete}
        onDownload={onDownload}
        onOpen={onOpen}
        onOpenParent={onOpenParent}
      />,
    );

    const list = host.querySelector<HTMLElement>("[data-testid='files-mobile-list']");
    expect(list?.className).toContain("lg:hidden");
    expect(list?.textContent).toContain(DIRECTORY.name);
    expect(list?.textContent).toContain("Directory");
    expect(list?.textContent).toContain("2.0 KB");
    expect(list?.textContent).toContain("Modified");

    const buttons = Array.from(host.querySelectorAll("button"));
    expect(buttons.every((button) => button.className.includes("min-h-11") || button.closest("[aria-label^='Actions for']"))).toBe(true);

    act(() => buttons.find((button) => button.textContent === "Parent directory")?.click());
    expect(onOpenParent).toHaveBeenCalledTimes(1);
    act(() => buttons.find((button) => button.textContent?.includes("Open"))?.click());
    expect(onOpen).toHaveBeenCalledWith(DIRECTORY);
    act(() => buttons.find((button) => button.textContent?.includes("Download"))?.click());
    expect(onDownload).toHaveBeenCalledWith(FILE);
    act(() => buttons.filter((button) => button.textContent?.includes("Delete"))[1]?.click());
    expect(onDelete).toHaveBeenCalledWith(FILE);
  });

  it("uses a shared semantic action group with phone and desktop geometry", () => {
    const onClick = vi.fn();
    render(
      <OperationalActionGroup separated aria-label="Operational actions">
        <button type="button" onClick={onClick}>Run</button>
        <button type="button">Delete</button>
      </OperationalActionGroup>,
    );

    const group = host.querySelector<HTMLElement>("[aria-label='Operational actions']");
    expect(group?.className).toContain("grid-cols-2");
    expect(group?.className).toContain("[&_button]:min-h-11");
    expect(group?.className).toContain("sm:flex");
    expect(group?.className).toContain("lg:[&_button]:min-h-0");
    expect(group?.className).toContain("border-t");

    act(() => host.querySelector<HTMLButtonElement>("button")?.click());
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("keeps log filters touch-sized and log formatting inside an internal scroller", () => {
    const onFileChange = vi.fn();
    render(
      <>
        <LogsFilterBar
          component="all"
          file="agent"
          level="ALL"
          lineCount={100}
          labels={{
            component: "Component",
            file: "File",
            level: "Level",
            lines: "Lines",
            title: "Log filters",
          }}
          onComponentChange={() => {}}
          onFileChange={onFileChange}
          onLevelChange={() => {}}
          onLineCountChange={() => {}}
        />
        <LogOutput
          emptyLabel="No log lines"
          lines={["2026-07-23  keep   exact spacing"]}
          loading={false}
        />
      </>,
    );

    const toolbar = host.querySelector<HTMLElement>("[role='toolbar']");
    const segment = toolbar?.querySelector<HTMLElement>("[role='radiogroup']");
    const output = host.querySelector<HTMLElement>("[data-testid='log-output']");
    const line = output?.querySelector("div");

    expect(toolbar?.getAttribute("aria-label")).toBe("Log filters");
    expect(segment?.className).toContain("[&_button]:min-h-11");
    expect(output?.className).toContain("h-[45dvh]");
    expect(output?.className).toContain("lg:h-auto");
    expect(output?.className).toContain("lg:min-h-[400px]");
    expect(output?.className).toContain("lg:max-h-[calc(100vh-220px)]");
    expect(output?.className).not.toContain("sm:min-h-[400px]");
    expect(output?.className).toContain("overflow-auto");
    expect(line?.className).toContain("whitespace-pre");
    expect(line?.textContent).toBe("2026-07-23  keep   exact spacing");

    const gateway = Array.from(toolbar?.querySelectorAll("button") ?? []).find(
      (button) => button.textContent === "GATEWAY",
    );
    act(() => gateway?.click());
    expect(onFileChange).toHaveBeenCalledWith("gateway");
  });
});
