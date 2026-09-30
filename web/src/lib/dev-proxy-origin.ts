/**
 * `npm run dev` serves the dashboard from Vite and proxies `/api` (WebSockets
 * included) to the backend. The browser stamps Vite's origin on those
 * upgrades, and the backend's WebSocket gate trusts only its own origin and
 * the desktop dev renderer it was spawned for — no fixed dev-server port. So
 * the proxy presents upgrades from Vite's OWN page with the backend's origin.
 *
 * Only Vite's own page: a same-origin browser request sends an Origin whose
 * host is the Host it targets. Any other page that reaches the proxy keeps its
 * Origin and the backend refuses it, so the proxy never launders a foreign
 * page into a trusted one (Vite's built-in `rewriteWsOrigin` rewrites every
 * upgrade).
 */

interface UpgradeRequest {
  headers: { host?: string; origin?: string | string[] };
}

interface ProxyRequest {
  setHeader(name: string, value: string): unknown;
}

export function presentOwnPageAsBackend(
  backendUrl: string,
): (proxyReq: ProxyRequest, req: UpgradeRequest) => void {
  const backendOrigin = new URL(backendUrl).origin;

  return (proxyReq, req) => {
    const { host, origin } = req.headers;

    if (!host || typeof origin !== "string") {
      return;
    }

    let originHost: string;

    try {
      originHost = new URL(origin).host;
    } catch {
      return; // "null" and other opaque origins are not Vite's page
    }

    if (originHost === host.toLowerCase()) {
      proxyReq.setHeader("origin", backendOrigin);
    }
  };
}
