// @vitest-environment jsdom
// #93263: an unparseable config.yaml must be visible in the dashboard, not only on a terminal.

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ReactNode } from "react";

import { I18nProvider } from "@/i18n";
import { ConfigParseErrorBanner } from "./ConfigParseErrorBanner";
import type { StatusResponse } from "@/lib/api";

let container: HTMLDivElement;
let root: Root;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<I18nProvider>{ui}</I18nProvider>));
}

async function rerender(ui: ReactNode) {
  await act(async () => root.render(<I18nProvider>{ui}</I18nProvider>));
}

function statusWith(error: string | null | undefined): StatusResponse {
  return { config_parse_error: error } as StatusResponse;
}

function banner(): HTMLElement | null {
  return container.querySelector('[data-testid="config-parse-error-banner"]');
}

beforeEach(() => {
  sessionStorage.clear();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ConfigParseErrorBanner", () => {
  it("names the failing line and the repair command", async () => {
    await render(
      <ConfigParseErrorBanner
        status={statusWith("has a formatting error at line 4")}
      />,
    );
    expect(banner()?.textContent).toContain(
      "config.yaml has a formatting error at line 4",
    );
    expect(banner()?.textContent).toContain("hermes config edit");
  });

  it("renders nothing while the config parses or on older backends", async () => {
    await render(<ConfigParseErrorBanner status={null} />);
    expect(banner()).toBeNull();
    await rerender(<ConfigParseErrorBanner status={statusWith(null)} />);
    expect(banner()).toBeNull();
    await rerender(<ConfigParseErrorBanner status={statusWith(undefined)} />);
    expect(banner()).toBeNull();
  });

  it("stays dismissed for the same error and re-surfaces for a new one", async () => {
    await render(
      <ConfigParseErrorBanner
        status={statusWith("has a formatting error at line 4")}
      />,
    );
    await act(async () => {
      banner()?.querySelector("button")?.click();
    });
    expect(banner()).toBeNull();

    await rerender(
      <ConfigParseErrorBanner
        status={statusWith("has a formatting error at line 9")}
      />,
    );
    expect(banner()?.textContent).toContain("line 9");
  });
});
