import { describe, expect, it, vi } from "vitest";
import { completeMcpDashboardOAuth } from "./mcp-dashboard-oauth";

describe("completeMcpDashboardOAuth", () => {
  it("opens the authorization URL in the dashboard browser and polls to approval", async () => {
    const authWindow = { location: { href: "" }, opener: {} } as unknown as Window;
    const open = vi.fn().mockReturnValue(authWindow);
    const start = vi.fn().mockResolvedValue({
      flow_id: "flow-1",
      server_name: "reports",
      status: "authorization_required",
      authorization_url: "https://idp.example/authorize",
      error: null,
    });
    const status = vi
      .fn()
      .mockResolvedValueOnce({
        flow_id: "flow-1",
        server_name: "reports",
        status: "authorization_required",
        authorization_url: "https://idp.example/authorize",
        error: null,
        tools: [],
      })
      .mockResolvedValueOnce({
        flow_id: "flow-1",
        server_name: "reports",
        status: "approved",
        authorization_url: "https://idp.example/authorize",
        error: null,
        tools: [{ name: "list_reports", description: "List reports" }],
      });

    const result = await completeMcpDashboardOAuth({
      serverName: "reports",
      start,
      status,
      open,
      sleep: async () => {},
    });

    expect(open).toHaveBeenCalledWith(
      "about:blank",
      "_blank",
    );
    expect(authWindow.opener).toBeNull();
    expect(authWindow.location.href).toBe("https://idp.example/authorize");
    expect(status).toHaveBeenCalledTimes(2);
    expect(result.status).toBe("approved");
  });

  it("surfaces a terminal OAuth error", async () => {
    const close = vi.fn();
    await expect(
      completeMcpDashboardOAuth({
        serverName: "reports",
        start: async () => ({
          flow_id: "flow-2",
          server_name: "reports",
          status: "error",
          authorization_url: null,
          error: "registration denied",
        }),
        status: vi.fn(),
        open: vi.fn().mockReturnValue({ location: { href: "" }, close }),
        sleep: async () => {},
      }),
    ).rejects.toThrow("registration denied");
    expect(close).toHaveBeenCalledOnce();
  });

  it("fails before starting when the browser blocks the popup", async () => {
    const start = vi.fn();
    await expect(
      completeMcpDashboardOAuth({
        serverName: "reports",
        start,
        status: vi.fn(),
        open: vi.fn().mockReturnValue(null),
      }),
    ).rejects.toThrow("popup was blocked");
    expect(start).not.toHaveBeenCalled();
  });

  it("fails when the authorization window closes before approval", async () => {
    const authWindow = { location: { href: "" }, opener: {}, closed: false } as unknown as Window;
    const status = vi.fn().mockImplementation(async () => {
      Object.defineProperty(authWindow, "closed", { value: true });
      return {
        flow_id: "flow-closed",
        server_name: "reports",
        status: "authorization_required",
        authorization_url: "https://idp.example/authorize",
        error: null,
      };
    });

    await expect(
      completeMcpDashboardOAuth({
        serverName: "reports",
        start: async () => ({
          flow_id: "flow-closed",
          server_name: "reports",
          status: "authorization_required",
          authorization_url: "https://idp.example/authorize",
          error: null,
        }),
        status,
        open: vi.fn().mockReturnValue(authWindow),
        sleep: async () => {},
      }),
    ).rejects.toThrow("authorization window was closed");
  });

  it("retries a transient status failure", async () => {
    const authWindow = { location: { href: "" }, opener: {}, closed: false } as unknown as Window;
    const status = vi
      .fn()
      .mockRejectedValueOnce(new Error("temporary network failure"))
      .mockResolvedValueOnce({
        flow_id: "flow-retry",
        server_name: "reports",
        status: "approved",
        authorization_url: "https://idp.example/authorize",
        error: null,
        tools: [],
      });

    const result = await completeMcpDashboardOAuth({
      serverName: "reports",
      start: async () => ({
        flow_id: "flow-retry",
        server_name: "reports",
        status: "authorization_required",
        authorization_url: "https://idp.example/authorize",
        error: null,
      }),
      status,
      open: vi.fn().mockReturnValue(authWindow),
      sleep: async () => {},
    });

    expect(result.status).toBe("approved");
    expect(status).toHaveBeenCalledTimes(2);
  });
});
