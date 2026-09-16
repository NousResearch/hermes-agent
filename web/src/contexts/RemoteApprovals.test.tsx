// @vitest-environment jsdom
import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const gatewayMocks = vi.hoisted(() => ({
  connect: vi.fn(),
  request: vi.fn(),
  close: vi.fn(),
}));

vi.mock("@/lib/gatewayClient", () => ({
  GatewayClient: class {
    connect = gatewayMocks.connect;
    request = gatewayMocks.request;
    close = gatewayMocks.close;
  },
}));
vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({ profile: "work" }),
}));

import type { PendingApprovalsResponse } from "@/lib/remote-approvals";
import {
  RemoteApprovalsProvider,
  useRemoteApprovals,
} from "./RemoteApprovals";

let root: Root;
let container: HTMLDivElement;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function Consumer() {
  const { approvals, loading, error, respond, refresh } = useRemoteApprovals();
  const approval = approvals[0];
  return createElement(
    "div",
    null,
    createElement("output", { "data-loading": String(loading) }, error || approval?.title || "empty"),
    approval && createElement(
      "button",
      { type: "button", onClick: () => void respond(approval, "once") },
      "approve",
    ),
    createElement("button", { type: "button", "data-refresh": "true", onClick: () => void refresh() }, "refresh"),
  );
}

function WrongScopeConsumer() {
  const { approvals, respond } = useRemoteApprovals();
  const approval = approvals[0];
  return approval && createElement(
    "button",
    { type: "button", onClick: () => void respond({ ...approval, profile: "other" }, "once").catch(() => undefined) },
    "approve-other",
  );
}


async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

const approval = {
  request_id: "req-work",
  command: "rm -rf /tmp/work",
  description: "delete work files",
  choices: ["once", "deny"],
  profile: "work",
  session_id: "runtime-work",
  session_key: "stored-work",
  stored_session_id: "stored-work",
  source: "desktop",
  title: "Work session",
  status: "waiting_approval",
};

beforeEach(() => {
  gatewayMocks.connect.mockReset().mockResolvedValue(undefined);
  gatewayMocks.request.mockReset().mockImplementation(async (method: string) =>
    method === "approval.respond" ? { resolved: 1 } : { approvals: [approval], profile: "work" },
  );
  gatewayMocks.close.mockReset();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("RemoteApprovalsProvider", () => {
  it("loads only the selected profile and responds through the existing gateway approval RPC", async () => {
    await render(createElement(RemoteApprovalsProvider, null, createElement(Consumer)));

    await vi.waitFor(() => expect(container.textContent).toContain("Work session"));
    expect(gatewayMocks.request).toHaveBeenCalledWith("approval.pending.all", { profile: "work" });

    await act(async () => container.querySelector<HTMLButtonElement>("button")?.click());
    expect(gatewayMocks.request).toHaveBeenCalledWith("approval.respond", {
      choice: "once",
      request_id: "req-work",
      session_id: "runtime-work",
      profile: "work",
    });
  });

  it("does not trust a backend response profile to widen the selected scope", async () => {
    gatewayMocks.request.mockResolvedValue({
      approvals: [{ ...approval, profile: "other", title: "Other session" }],
      profile: "other",
    });

    await render(createElement(RemoteApprovalsProvider, null, createElement(Consumer)));

    await vi.waitFor(() => expect(container.textContent).toContain("empty"));
    expect(container.textContent).not.toContain("Other session");
  });

  it("rejects a stale approval object from another profile before sending a response", async () => {
    await render(createElement(RemoteApprovalsProvider, null, createElement(WrongScopeConsumer)));

    await vi.waitFor(() => expect(container.textContent).toContain("approve-other"));
    await act(async () => container.querySelector<HTMLButtonElement>("button")?.click());
    expect(gatewayMocks.request.mock.calls.some(([method]) => method === "approval.respond")).toBe(false);
  });

  it("does not let an older refresh resurrect a cleared approval", async () => {
    let resolveFirst!: (value: PendingApprovalsResponse) => void;
    const first = new Promise<PendingApprovalsResponse>((resolve) => { resolveFirst = resolve; });
    gatewayMocks.request.mockReset();
    gatewayMocks.request.mockImplementationOnce(() => first).mockResolvedValueOnce({ approvals: [] });

    await render(createElement(RemoteApprovalsProvider, null, createElement(Consumer)));
    await vi.waitFor(() => expect(gatewayMocks.request).toHaveBeenCalledTimes(1));
    await act(async () => container.querySelector<HTMLButtonElement>("[data-refresh='true']")?.click());
    await vi.waitFor(() => expect(gatewayMocks.request).toHaveBeenCalledTimes(2));
    await act(async () => resolveFirst({ approvals: [approval] }));
    expect(container.textContent).toContain("empty");
    expect(container.textContent).not.toContain("Work session");
  });
});
