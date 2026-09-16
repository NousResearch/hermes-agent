// @vitest-environment jsdom
import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const approvalMocks = vi.hoisted(() => ({
  useRemoteApprovals: vi.fn(),
}));
const headerMocks = vi.hoisted(() => ({
  setAfterTitle: vi.fn(),
  setEnd: vi.fn(),
}));

vi.mock("@/contexts/RemoteApprovals", () => approvalMocks);
vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({ profile: "work", currentProfile: "default" }),
}));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => headerMocks,
}));

import PendingApprovalsPage from "./PendingApprovalsPage";

let root: Root;
let container: HTMLDivElement;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const approval = {
  request_id: "req-work",
  command: "rm -rf /tmp/work",
  description: "delete work files",
  choices: ["once", "deny"] as ("once" | "deny")[],
  profile: "work",
  session_id: "runtime-work",
  session_key: "stored-work",
  stored_session_id: "stored-work",
  source: "desktop",
  title: "Work session",
  status: "waiting_approval" as const,
  created_at: Math.floor(Date.now() / 1000),
  expires_at: Math.floor(Date.now() / 1000) + 300,
};

function LocationProbe() {
  const location = useLocation();
  return createElement("output", { "data-location": `${location.pathname}${location.search}` });
}

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

beforeEach(() => {
  headerMocks.setAfterTitle.mockReset();
  headerMocks.setEnd.mockReset();
  approvalMocks.useRemoteApprovals.mockReturnValue({
    approvals: [approval],
    error: null,
    lastUpdatedAt: Date.now(),
    loading: false,
    refresh: vi.fn().mockResolvedValue(undefined),
    refreshing: false,
    respond: vi.fn().mockResolvedValue(undefined),
  });
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("PendingApprovalsPage", () => {
  it("renders a profile-scoped approval, uses only backend choices, and opens its durable session", async () => {
    await render(
      <MemoryRouter initialEntries={["/approvals"]}>
        <PendingApprovalsPage />
        <LocationProbe />
      </MemoryRouter>,
    );

    expect(container.textContent).toContain("Work session");
    expect(container.textContent).toContain("Desktop");
    expect(container.textContent).toContain("Waiting for approval");
    expect(container.querySelector("button[data-choice='always']")).toBeNull();

    await act(async () => container.querySelector<HTMLButtonElement>("button[data-choice='once']")?.click());
    expect(approvalMocks.useRemoteApprovals.mock.results[0].value.respond).toHaveBeenCalledWith(approval, "once");

    await act(async () => container.querySelector<HTMLButtonElement>("button[data-action='open-session']")?.click());
    expect(container.querySelector("output[data-location]")?.getAttribute("data-location"))
      .toBe("/chat?resume=stored-work&profile=work");
  });

  it("renders the empty state with user-visible message and server-owned explanation when there are no approvals", async () => {
    approvalMocks.useRemoteApprovals.mockReturnValue({
      approvals: [],
      error: null,
      lastUpdatedAt: Date.now(),
      loading: false,
      refresh: vi.fn().mockResolvedValue(undefined),
      refreshing: false,
      respond: vi.fn().mockResolvedValue(undefined),
    });

    await render(
      <MemoryRouter initialEntries={["/approvals"]}>
        <PendingApprovalsPage />
      </MemoryRouter>,
    );

    expect(container.textContent).toContain("No pending approvals");
    expect(container.textContent).toContain(
      "There are no server-owned approval requests waiting in this profile.",
    );
  });

  it("renders the error branch with heading, role=alert, and exact error text", async () => {
    const errorText = "Gateway connection refused";
    approvalMocks.useRemoteApprovals.mockReturnValue({
      approvals: [],
      error: errorText,
      lastUpdatedAt: null,
      loading: false,
      refresh: vi.fn().mockResolvedValue(undefined),
      refreshing: false,
      respond: vi.fn().mockResolvedValue(undefined),
    });

    await render(
      <MemoryRouter initialEntries={["/approvals"]}>
        <PendingApprovalsPage />
      </MemoryRouter>,
    );

    expect(container.textContent).toContain("Could not load pending approvals");
    expect(container.textContent).toContain(errorText);

    const heading = Array.from(container.querySelectorAll("p")).find(
      (p) => p.textContent === "Could not load pending approvals",
    );
    expect(heading).not.toBeNull();

    const alertElement = container.querySelector("[role='alert']");
    expect(alertElement).not.toBeNull();
    expect(alertElement?.textContent).toContain(errorText);
  });
});
