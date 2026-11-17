// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

import { I18nProvider } from "@/i18n";
import { ProfileContext } from "@/contexts/profile-context";
import { ApiHint } from "./ApiHint";

let container: HTMLDivElement;
let root: Root;

async function render(pathname: string, profile: string) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  const scope = { profile, currentProfile: "default", profiles: [], setProfile: () => {} };
  await act(async () =>
    root.render(
      <I18nProvider>
        <ProfileContext.Provider value={scope}>
          <ApiHint pathname={pathname} />
        </ProfileContext.Provider>
      </I18nProvider>,
    ),
  );
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ApiHint", () => {
  it("names the route, the profile's CLI command and a docs link for a panel", async () => {
    await render("/cron", "coder");
    const footer = container.querySelector("footer");
    expect(footer?.textContent).toContain("GET /api/cron/jobs");
    expect(footer?.textContent).toContain("hermes -p coder cron list");
    expect(footer?.querySelector("a")?.getAttribute("href")).toBe(
      "https://hermes-agent.nousresearch.com/docs/user-guide/features/cron",
    );
  });

  it("renders nothing on a page with no single API/CLI equivalent", async () => {
    await render("/chat", "");
    expect(container.querySelector("footer")).toBeNull();
  });
});
