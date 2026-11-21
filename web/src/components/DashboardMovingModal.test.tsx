// @vitest-environment jsdom
// The Hermes Cloud "dashboard is moving" modal: shown only when the backend names a web app,
// dismissed only by its own actions, and "Later" lasts the browser session.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ReactNode } from "react";

import { I18nProvider } from "@/i18n";
import { DashboardMovingModal } from "./DashboardMovingModal";
import type { StatusResponse } from "@/lib/api";

const WEBAPP = "https://otter--app.agents.nousresearch.com";

let container: HTMLDivElement;
let root: Root;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<I18nProvider>{ui}</I18nProvider>));
}

const modal = () =>
  document.querySelector<HTMLElement>('[data-testid="dashboard-moving-modal"]');
const button = (label: string) =>
  Array.from(modal()?.querySelectorAll("button") ?? []).find(
    (el) => el.textContent === label,
  );
const cloud = { webapp_url: WEBAPP } as StatusResponse;

beforeEach(() => {
  sessionStorage.clear();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.restoreAllMocks();
});

describe("DashboardMovingModal", () => {
  it("stays hidden on a local dashboard and before status loads", async () => {
    await render(<DashboardMovingModal status={null} />);
    expect(modal()).toBeNull();
    await act(async () =>
      root.render(
        <I18nProvider>
          <DashboardMovingModal status={{} as StatusResponse} />
        </I18nProvider>,
      ),
    );
    expect(modal()).toBeNull();
  });

  it("opens as a modal dialog with the retirement date on Hermes Cloud", async () => {
    await render(<DashboardMovingModal status={cloud} />);
    expect(modal()?.getAttribute("aria-modal")).toBe("true");
    expect(modal()?.textContent).toContain("Your dashboard is moving");
    expect(modal()?.textContent).toContain(
      "This dashboard moves to the Hermes web app on January 15.",
    );
    expect(document.activeElement).toBe(button("Open in web app"));
  });

  it("is not dismissed by Escape or a backdrop click", async () => {
    await render(<DashboardMovingModal status={cloud} />);
    await act(async () => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
      modal()?.click();
    });
    expect(modal()).not.toBeNull();
  });

  it("opens the web app and links its settings", async () => {
    const open = vi.spyOn(window, "open").mockReturnValue(null);
    await render(<DashboardMovingModal status={cloud} />);
    await act(async () => button("Open in web app")?.click());
    expect(open).toHaveBeenCalledWith(WEBAPP, "_self");
    expect(modal()?.querySelector("a")?.getAttribute("href")).toBe(
      `${WEBAPP}/#/settings`,
    );
  });

  it("Later hides it for the rest of the browser session", async () => {
    await render(<DashboardMovingModal status={cloud} />);
    await act(async () => button("Later")?.click());
    expect(modal()).toBeNull();

    await act(async () => root.unmount());
    await render(<DashboardMovingModal status={cloud} />);
    expect(modal()).toBeNull();
  });
});
