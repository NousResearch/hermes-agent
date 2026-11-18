import { describe, expect, it } from "vitest";

import viteConfig from "../../vite.config";

describe("web/vite.config.ts dev server", () => {
  const server = viteConfig.server ?? {};

  it("proxies the dev page's sockets with their own Host and Origin", () => {
    // The backend admits an upgrade whose Origin names the Host it was sent
    // to. Vite keeps the browser's Host (localhost:5173) unless changeOrigin
    // is set, so the dev page's own Origin matches it whether HERMES_DASHBOARD_URL
    // is the backend or a port forward to it. Rewriting Origin to the target
    // breaks the forward: the backend never sees the forward's port as its own.
    const api = server.proxy?.["/api"];

    if (!api || typeof api === "string") {
      throw new Error("/api proxy is not configured");
    }

    expect(api.ws).toBe(true);
    expect(api.changeOrigin).toBeFalsy();
    expect(api.rewriteWsOrigin).toBeFalsy();
    expect(api.configure).toBeUndefined();
  });

  it("does not let other local pages read the injected token", () => {
    expect(server.cors).toBe(false);
  });
});
