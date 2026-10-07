import { defineConfig, type Plugin } from "vite";
import babel from "@rolldown/plugin-babel";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";

/** React Compiler preset scoped to modules that can actually contain
 *  components/hooks (JSX syntax or a react-ish import). The preset's default
 *  code filter matches any PascalCase/use* declaration — effectively every TS
 *  module — which made the babel pass parse the whole codebase. */
function compilerPreset() {
  const preset = reactCompilerPreset();
  preset.rolldown.filter.code = /\/>|<\/|from\s*['"][^'"]*react/;
  return preset;
}
import tailwindcss from "@tailwindcss/vite";
import path from "path";
import { fileURLToPath } from "node:url";

const configDir: string = fileURLToPath(new URL(".", import.meta.url));

const BACKEND = process.env.RABBIT_DASHBOARD_URL ?? "http://127.0.0.1:9119";

/**
 * In production the Python `rabbit dashboard` server injects a one-shot
 * session token into `index.html` (see `rabbit_cli/web_server.py`). The
 * Vite dev server serves its own `index.html`, so unless we forward that
 * token, every protected `/api/*` call 401s.
 *
 * This plugin fetches the running dashboard's `index.html` on each dev page
 * load and forwards its runtime bootstrap values into the dev HTML. No-op in
 * production builds.
 */
function rabbitDevToken(): Plugin {
  const TOKEN_RE = /window\.__RABBIT_SESSION_TOKEN__\s*=\s*"([^"]+)"/;
  const EMBEDDED_RE =
    /window\.__RABBIT_DASHBOARD_EMBEDDED_CHAT__\s*=\s*(true|false)/;
  const INITIAL_PROFILE_RE =
    /window\.__RABBIT_INITIAL_PROFILE__\s*=\s*("(?:\\.|[^"\\])*")/;

  return {
    name: "rabbit:dev-session-token",
    apply: "serve",
    async transformIndexHtml() {
      try {
        const res = await fetch(BACKEND, { headers: { accept: "text/html" } });
        const html = await res.text();
        const match = html.match(TOKEN_RE);
        if (!match) {
          console.warn(
            `[rabbit] Could not find session token in ${BACKEND} — ` +
              `is \`rabbit dashboard\` running? /api calls will 401.`,
          );
          return;
        }
        const embeddedMatch = html.match(EMBEDDED_RE);
        const embeddedJs = embeddedMatch ? embeddedMatch[1] : "true";
        const initialProfileMatch = html.match(INITIAL_PROFILE_RE);
        const initialProfileJs = initialProfileMatch?.[1] ?? '""';
        return [
          {
            tag: "script",
            injectTo: "head",
            children:
              `window.__RABBIT_SESSION_TOKEN__="${match[1]}";` +
              `window.__RABBIT_DASHBOARD_EMBEDDED_CHAT__=${embeddedJs};` +
              `window.__RABBIT_INITIAL_PROFILE__=${initialProfileJs};`,
          },
        ];
      } catch (err) {
        console.warn(
          `[rabbit] Dashboard at ${BACKEND} unreachable — ` +
            `start it with \`rabbit dashboard\` or set RABBIT_DASHBOARD_URL. ` +
            `(${(err as Error).message})`,
        );
      }
    },
  };
}

export default defineConfig({
  plugins: [
    react(),
    babel({ presets: [compilerPreset()] }),
    tailwindcss(),
    rabbitDevToken(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(configDir, "./src"),
      "@rabbit/shared": path.resolve(configDir, "../apps/shared/src"),
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
    outDir: "../rabbit_cli/web_dist",
    emptyOutDir: true,
    // Shell stays a bit over Vite's 500 kB default after vendor splits;
    // page/xterm chunks load on demand. Keep a modest ceiling so a true
    // regression still warns.
    chunkSizeWarningLimit: 600,
    // Split heavy vendors so the first dashboard paint does not download
    // xterm/three/plot/etc. until a route actually needs them. Lazy page
    // imports in App.tsx create the route boundaries; these groups keep
    // shared node_modules out of every page chunk.
    rolldownOptions: {
      output: {
        codeSplitting: {
          minSize: 20_000,
          groups: [
            {
              name: "react-vendor",
              test: /node_modules[\\/](react|react-dom|scheduler|react-router|react-router)([\\/]|$)/,
            },
            {
              name: "xterm",
              test: /node_modules[\\/]@xterm[\\/]/,
            },
            {
              name: "three",
              test: /node_modules[\\/](three|@react-three)([\\/]|$)/,
            },
            {
              name: "plot",
              test: /node_modules[\\/]@observablehq[\\/]plot([\\/]|$)/,
            },
            {
              name: "motion",
              test: /node_modules[\\/](motion|framer-motion)([\\/]|$)/,
            },
            {
              name: "ui",
              test: /node_modules[\\/]@nous-research[\\/]ui([\\/]|$)/,
            },
            {
              name: "vendor",
              test: /node_modules[\\/]/,
            },
          ],
        },
      },
    },
  },
  server: {
    proxy: {
      "/api": {
        target: BACKEND,
        ws: true,
      },
      // Same host as `rabbit dashboard` must serve these; Vite has no
      // dashboard-plugins/* files, so without this, plugin scripts 404
      // or receive index.html in dev.
      "/dashboard-plugins": BACKEND,
    },
  },
});
