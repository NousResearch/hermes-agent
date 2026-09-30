import { describe, expect, it } from "vitest";

import viteConfig from "../../vite.config";
import { presentOwnPageAsBackend } from "./dev-proxy-origin";

type Listener = (
  proxyReq: { setHeader(name: string, value: string): unknown },
  req: { headers: Record<string, string | undefined> },
) => void;

function forwardedOrigin(
  listener: Listener,
  headers: Record<string, string | undefined>,
): string | undefined {
  const sent: Record<string, string> = { origin: headers.origin ?? "" };

  listener({ setHeader: (name, value) => (sent[name] = value) }, { headers });

  return sent.origin || undefined;
}

describe("presentOwnPageAsBackend", () => {
  const rewrite = presentOwnPageAsBackend("http://127.0.0.1:9119");

  it("presents the dev page's own upgrade with the backend origin", () => {
    expect(
      forwardedOrigin(rewrite, {
        host: "localhost:5173",
        origin: "http://localhost:5173",
      }),
    ).toBe("http://127.0.0.1:9119");
    expect(
      forwardedOrigin(rewrite, {
        host: "127.0.0.1:5173",
        origin: "http://127.0.0.1:5173",
      }),
    ).toBe("http://127.0.0.1:9119");
  });

  it("leaves another page's Origin for the backend to refuse", () => {
    for (const origin of [
      "http://localhost:3000",
      "http://127.0.0.1:5173", // same port, different host: not this page
      "https://evil.example",
      "null",
    ]) {
      expect(
        forwardedOrigin(rewrite, { host: "localhost:5173", origin }),
      ).toBe(origin);
    }
  });

  it("adds no Origin to a client that sent none", () => {
    expect(
      forwardedOrigin(rewrite, { host: "localhost:5173" }),
    ).toBeUndefined();
  });

  it("uses the configured backend's origin", () => {
    expect(
      forwardedOrigin(presentOwnPageAsBackend("http://127.0.0.1:9200/"), {
        host: "localhost:5173",
        origin: "http://localhost:5173",
      }),
    ).toBe("http://127.0.0.1:9200");
  });
});

describe("web/vite.config.ts dev server", () => {
  const server = viteConfig.server ?? {};

  it("wires the rewrite into the /api WebSocket proxy", () => {
    const api = server.proxy?.["/api"];

    if (!api || typeof api === "string") {
      throw new Error("/api proxy is not configured");
    }

    expect(api.ws).toBe(true);
    expect(api.rewriteWsOrigin).toBeFalsy();

    const listeners: Record<string, Listener> = {};
    const proxy = {
      on: (event: string, listener: Listener) => {
        listeners[event] = listener;
      },
    };

    api.configure?.(proxy as never, api);

    const target = new URL(String(api.target)).origin;

    expect(
      forwardedOrigin(listeners.proxyReqWs, {
        host: "localhost:5173",
        origin: "http://localhost:5173",
      }),
    ).toBe(target);
    expect(
      forwardedOrigin(listeners.proxyReqWs, {
        host: "localhost:5173",
        origin: "http://localhost:3000",
      }),
    ).toBe("http://localhost:3000");
  });

  it("does not let other local pages read the injected token", () => {
    expect(server.cors).toBe(false);
  });
});
