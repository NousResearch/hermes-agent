import { defineConfig, type Plugin } from "vite";
import type { IncomingMessage, ServerResponse } from "node:http";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

const BACKEND = process.env.HERMES_DASHBOARD_URL ?? "http://127.0.0.1:9119";

/**
 * In production the Python `hermes dashboard` server injects a one-shot
 * session token into `index.html` (see `hermes_cli/web_server.py`). The
 * Vite dev server serves its own `index.html`, so unless we forward that
 * token, every protected `/api/*` call 401s.
 *
 * This plugin fetches the running dashboard's `index.html` on each dev page
 * load, scrapes the `window.__HERMES_SESSION_TOKEN__` assignment, and
 * re-injects it into the dev HTML. No-op in production builds.
 */
function isLoopbackAddress(address: string | undefined): boolean {
  if (!address) return false;

  const normalized = address.replace(/^\[|\]$/g, "").toLowerCase();
  if (normalized === "localhost" || normalized === "::1") return true;
  if (normalized.startsWith("127.")) return true;
  if (normalized.startsWith("::ffff:")) {
    return isLoopbackAddress(normalized.slice("::ffff:".length));
  }

  return false;
}

function acceptsHtml(req: IncomingMessage): boolean {
  const accept = req.headers.accept;
  const values = Array.isArray(accept) ? accept : [accept ?? ""];
  return values.some((value) => value.includes("text/html"));
}

function isHtmlDocumentRoute(req: IncomingMessage): boolean {
  if (req.method !== "GET" && req.method !== "HEAD") return false;

  const pathname = new URL(req.url ?? "/", "http://vite.invalid").pathname;
  const leaf = pathname.split("/").pop() ?? "";
  return (
    acceptsHtml(req) ||
    pathname === "/" ||
    pathname.endsWith(".html") ||
    !leaf.includes(".")
  );
}

function isProtectedDashboardRoute(req: IncomingMessage): boolean {
  const pathname = new URL(req.url ?? "/", "http://vite.invalid").pathname;
  return (
    pathname === "/api" ||
    pathname.startsWith("/api/") ||
    pathname === "/dashboard-plugins" ||
    pathname.startsWith("/dashboard-plugins/") ||
    isHtmlDocumentRoute(req)
  );
}

function blockRemoteDashboardAccess(
  req: IncomingMessage,
  res: ServerResponse,
  next: () => void,
): void {
  if (
    isLoopbackAddress(req.socket.remoteAddress) ||
    !isProtectedDashboardRoute(req)
  ) {
    next();
    return;
  }

  res.statusCode = 403;
  res.setHeader("content-type", "text/plain; charset=utf-8");
  res.end(
    "Hermes dashboard dev access is limited to loopback clients. " +
      "Run the production dashboard or connect through localhost to use protected APIs.",
  );
}

function hermesDevToken(): Plugin {
  const TOKEN_RE = /window\.__HERMES_SESSION_TOKEN__\s*=\s*"([^"]+)"/;
  const EMBEDDED_RE =
    /window\.__HERMES_DASHBOARD_EMBEDDED_CHAT__\s*=\s*(true|false)/;
  const LEGACY_TUI_RE =
    /window\.__HERMES_DASHBOARD_TUI__\s*=\s*(true|false)/;

  return {
    name: "hermes:dev-session-token",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use(blockRemoteDashboardAccess);
    },
    async transformIndexHtml() {
      try {
        const res = await fetch(BACKEND, { headers: { accept: "text/html" } });
        const html = await res.text();
        const match = html.match(TOKEN_RE);
        if (!match) {
          console.warn(
            `[hermes] Could not find session token in ${BACKEND} — ` +
              `is \`hermes dashboard\` running? /api calls will 401.`,
          );
          return;
        }
        const embeddedMatch = html.match(EMBEDDED_RE);
        const legacyMatch = html.match(LEGACY_TUI_RE);
        const embeddedJs = embeddedMatch
          ? embeddedMatch[1]
          : legacyMatch
            ? legacyMatch[1]
            : "false";
        return [
          {
            tag: "script",
            injectTo: "head",
            children:
              `window.__HERMES_SESSION_TOKEN__="${match[1]}";` +
              `window.__HERMES_DASHBOARD_EMBEDDED_CHAT__=${embeddedJs};`,
          },
        ];
      } catch (err) {
        console.warn(
          `[hermes] Dashboard at ${BACKEND} unreachable — ` +
            `start it with \`hermes dashboard\` or set HERMES_DASHBOARD_URL. ` +
            `(${(err as Error).message})`,
        );
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), hermesDevToken()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    // When @nous-research/ui is symlinked via `file:../../design-language`,
    // Node's module resolution would pick up shared deps from
    // design-language/node_modules/*, giving us two copies + breaking
    // hooks (useRef-of-null), webgl contexts, etc. Force everything that
    // exists in BOTH places to use the dashboard's copy.
    //
    // Don't list packages here that only exist in the DS (nanostores,
    // @nanostores/react) — Vite dedupe errors out when it can't find
    // them at the project root.
    dedupe: [
      "react",
      "react-dom",
      "@react-three/fiber",
      "@observablehq/plot",
      "three",
      "leva",
      "gsap",
    ],
  },
  build: {
    outDir: "../hermes_cli/web_dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": {
        target: BACKEND,
        ws: true,
      },
      // Same host as `hermes dashboard` must serve these; Vite has no
      // dashboard-plugins/* files, so without this, plugin scripts 404
      // or receive index.html in dev.
      "/dashboard-plugins": BACKEND,
    },
  },
});
