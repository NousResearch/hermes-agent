// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "@/lib/api";
import { AutomationBlueprints } from "./AutomationBlueprints";

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  api: { getAutomationBlueprints: vi.fn() },
}));

describe("AutomationBlueprints", () => {
  let root: Root;
  let container: HTMLDivElement;

  afterEach(() => {
    root?.unmount();
    container?.remove();
    vi.restoreAllMocks();
  });

  it("renders a URL slot as a URL input", async () => {
    vi.mocked(api.getAutomationBlueprints).mockResolvedValue({
      blueprints: [{
        appUrl: "hermes://blueprint/watch-page", category: "general", command: "/blueprint watch-page",
        description: "Watch a page.", fields: [{ default: null, help: "Page URL", label: "Page URL",
          name: "url", optional: false, options: [], strict: true, type: "url" }], key: "watch-page",
        tags: [], title: "Watch page",
      }],
    });
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);

    await act(async () => {
      root.render(<AutomationBlueprints profile="default" />);
      await Promise.resolve();
    });
    const setup = Array.from(container.querySelectorAll("button")).find(button => button.textContent === "Set up");
    expect(setup).toBeTruthy();
    await act(async () => setup?.dispatchEvent(new MouseEvent("click", { bubbles: true })));

    const input = container.querySelector('input[placeholder="Page URL"]');
    expect(input?.getAttribute("type")).toBe("url");
  });
});
