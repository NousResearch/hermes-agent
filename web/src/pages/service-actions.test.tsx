// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState, type ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, expect, it, vi } from "vitest";
import { SystemActionsProvider } from "@/contexts/SystemActions";
import { PageHeaderContext } from "@/contexts/page-header-context";
import { setManagementProfile } from "@/lib/api";
import ChannelsPage from "./ChannelsPage";
import WebhooksPage from "./WebhooksPage";
import SystemPage from "./SystemPage";

function Shell({ children }: { children: ReactNode }) {
  const [end, setEnd] = useState<ReactNode>(null);
  return <MemoryRouter><PageHeaderContext.Provider value={{ setEnd, setAfterTitle: () => {}, setTitle: () => {} }}>
    <SystemActionsProvider>{end}{children}</SystemActionsProvider>
  </PageHeaderContext.Provider></MemoryRouter>;
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); setManagementProfile(""); });

it.each(["channels", "cancel", "webhooks", "system-restart", "system-update"])(
  "requires typed intent at the live page before sending a service action: %s", async (mode) => {
    const requests: { url: string; body: Record<string, unknown> }[] = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      const path = String(url).split("?")[0];
      let data: unknown = null;
      if (path === "/api/gateway/restart" || path === "/api/hermes/update") {
        requests.push({ url: String(url), body: JSON.parse(String(init?.body ?? "{}")) });
        data = { ok: true, name: path.includes("restart") ? "gateway-restart" : "hermes-update", pid: 123 };
      } else if (path.includes("/actions/")) data = { running: false, exit_code: 0, lines: [] };
      else if (path === "/api/messaging/platforms") data = { platforms: [], env_path: "fixture.env" };
      else if (path === "/api/webhooks") data = { enabled: false, subscriptions: [] };
      else if (path === "/api/webhooks/enable") data = { ok: true, restart_started: false };
      else if (path === "/api/status") data = { gateway_running: true, can_update_hermes: true, active_sessions: 0 };
      else if (path === "/api/hermes/update/check") data = { can_apply: true, update_available: true, behind: 1, update_command: "hermes update", commits: [] };
      else if (path.includes("credential")) data = { providers: [] };
      else if (path.includes("checkpoints")) data = { sessions: [], total_bytes: 0 };
      else if (path.includes("hooks")) data = { hooks: [] };
      return new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } });
    });
    vi.stubGlobal("fetch", fetchMock);
    setManagementProfile("coder");
    render(<Shell>{mode.startsWith("system") ? <SystemPage /> : mode === "webhooks" ? <WebhooksPage /> : <ChannelsPage />}</Shell>);
    if (mode === "webhooks") fireEvent.click(await screen.findByRole("button", { name: "Enable webhooks" }));
    const action = mode === "system-update" ? "UPDATE" : "RESTART";
    const name = mode === "system-update" ? /Update now/i : mode === "system-restart" ? /^Restart$/i : /^Restart gateway$/i;
    fireEvent.click(await screen.findByRole("button", { name }));
    const input = await screen.findByRole("textbox", { name: `Type ${action} to confirm.` });
    expect(requests).toHaveLength(0);
    if (mode === "cancel") {
      fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Cancel" }));
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
      expect(requests).toHaveLength(0);
      return;
    }
    fireEvent.change(input, { target: { value: action.toLowerCase() } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(requests).toHaveLength(0);
    fireEvent.change(input, { target: { value: action } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0].body.confirmation).toBe(action);
    expect(requests[0].body.idempotency_key).toMatch(/^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$/);
    if (action === "RESTART") expect(requests[0].url).toContain("profile=coder");
  }
);
