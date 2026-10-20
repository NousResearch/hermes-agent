// @vitest-environment jsdom
// The plugin sidebar badge (kanban pause, #141451): a chip while the endpoint
// reports an active state, a dot when the rail is collapsed, nothing otherwise.

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PluginNavBadge } from "./PluginNavBadge";

const fetchJSON = vi.fn();

vi.mock("@/lib/api", () => ({
  fetchJSON: (...args: unknown[]) => fetchJSON(...args),
}));

const badge = {
  endpoint: "/api/plugins/kanban/paused",
  path: "paused",
  labelKey: "kanban.pausedBadge",
};

let container: HTMLDivElement;
let root: Root;

async function render(ui: React.ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

beforeEach(() => {
  fetchJSON.mockReset();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("PluginNavBadge", () => {
  it("reads the pause state off the endpoint and shows the reason in the tooltip", async () => {
    fetchJSON.mockResolvedValue({
      count: 1,
      paused: [{ board: "default", name: "Default", reason: "reloading the provider" }],
    });

    await render(<PluginNavBadge badge={badge} collapsed={false} label="PAUSED" />);
    await act(async () => {});

    const chip = container.querySelector('[data-testid="plugin-nav-badge"]');
    expect(chip?.textContent).toBe("PAUSED");
    expect(chip?.getAttribute("title")).toBe("Default — reloading the provider");
    expect(fetchJSON).toHaveBeenCalledWith("/api/plugins/kanban/paused");
  });

  it("counts held boards so the chip never implies the one being viewed", async () => {
    fetchJSON.mockResolvedValue({
      count: 3,
      paused: [
        { board: "default", name: "Default", reason: null },
        { board: "sumaris", name: "Sumaris", reason: null },
        { board: "gantt", name: "Gantt", reason: null },
      ],
    });

    await render(
      <PluginNavBadge
        badge={{ ...badge, countPath: "count", labelCountKey: "kanban.pausedBadgeCount" }}
        collapsed={false}
        label="PAUSED"
        labelCount="{count} PAUSED"
      />,
    );
    await act(async () => {});

    expect(container.querySelector('[data-testid="plugin-nav-badge"]')?.textContent).toBe("3 PAUSED");
  });

  it("collapses to a red dot so the rail still signals a pause", async () => {
    fetchJSON.mockResolvedValue({ count: 1, paused: [{ board: "default", reason: null }] });

    await render(<PluginNavBadge badge={badge} collapsed label="PAUSED" />);
    await act(async () => {});

    expect(container.querySelector('[data-testid="plugin-nav-badge-dot"]')).not.toBeNull();
    expect(container.querySelector('[data-testid="plugin-nav-badge"]')).toBeNull();
  });

  it("renders nothing while the board runs", async () => {
    fetchJSON.mockResolvedValue({ pause: null });

    await render(<PluginNavBadge badge={badge} collapsed={false} label="PAUSED" />);
    await act(async () => {});

    expect(container.textContent).toBe("");
  });

  it("stays silent when the endpoint fails — a badge is decoration", async () => {
    fetchJSON.mockRejectedValue(new Error("offline"));

    await render(<PluginNavBadge badge={badge} collapsed={false} label="PAUSED" />);
    await act(async () => {});

    expect(container.textContent).toBe("");
  });
});
