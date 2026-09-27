// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FleetActivityResponse } from "@/lib/api";

const apiMocks = vi.hoisted(() => ({
  getFleetActivity: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: apiMocks,
}));
vi.mock("@nous-research/ui/ui/components/card", () => ({
  Card: ({ children, className }: { children?: unknown; className?: string }) => (
    <div className={className}>{children as never}</div>
  ),
  CardContent: ({ children }: { children?: unknown }) => <div>{children as never}</div>,
}));
vi.mock("@nous-research/ui/ui/components/typography/h2", () => ({
  H2: ({ children }: { children?: unknown }) => <h2>{children as never}</h2>,
}));

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

async function waitFor(cond: () => boolean, timeoutMs = 3000) {
  const start = Date.now();
  while (!cond()) {
    if (Date.now() - start > timeoutMs) throw new Error("waitFor: condition never became true");
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
    });
  }
}

async function renderPage() {
  const { default: FleetActivityPage } = await import("./FleetActivityPage");
  await act(async () => {
    root.render(<FleetActivityPage />);
  });
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  apiMocks.getFleetActivity.mockReset();
});

afterEach(() => {
  act(() => {
    root.unmount();
  });
  container.remove();
  vi.useRealTimers();
  vi.resetModules();
});

describe("FleetActivityPage", () => {
  it("renders running Kanban tasks and live gateway sessions from the aggregation endpoint", async () => {
    const response: FleetActivityResponse = {
      kanban_tasks: [
        {
          board: "default",
          board_name: "Default",
          task_id: "t_abc123",
          title: "Ship the fleet panel",
          profile: "anika",
          started_at: Date.now() / 1000 - 90,
          elapsed_seconds: 90,
          last_heartbeat_at: Date.now() / 1000 - 5,
          heartbeat_age_seconds: 5,
        },
      ],
      gateway_sessions: [
        {
          profile: "default",
          session_key: "telegram:12345",
          platform: "telegram",
          display_name: "Raaj",
          chat_type: "dm",
          started_at: Date.now() / 1000 - 30,
          elapsed_seconds: 30,
        },
      ],
      count: 2,
    };
    apiMocks.getFleetActivity.mockResolvedValue(response);

    await renderPage();
    await waitFor(() => container.textContent?.includes("Ship the fleet panel") ?? false);

    expect(container.textContent).toContain("Ship the fleet panel");
    expect(container.textContent).toContain("Raaj");
    expect(container.textContent).toContain("telegram");
    expect(apiMocks.getFleetActivity).toHaveBeenCalled();
  });

  it("shows empty-state copy when nothing is running", async () => {
    apiMocks.getFleetActivity.mockResolvedValue({
      kanban_tasks: [],
      gateway_sessions: [],
      count: 0,
    } satisfies FleetActivityResponse);

    await renderPage();
    await waitFor(() => container.textContent?.includes("No Kanban tasks running") ?? false);

    expect(container.textContent).toContain("No Kanban tasks running right now.");
    expect(container.textContent).toContain("No gateway sessions mid-turn right now.");
  });

  it("surfaces a load error without crashing", async () => {
    apiMocks.getFleetActivity.mockRejectedValue(new Error("network down"));

    await renderPage();
    await waitFor(() => container.textContent?.includes("network down") ?? false);

    expect(container.textContent).toContain("Could not load fleet activity");
  });
});
