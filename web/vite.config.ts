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
import { execFileSync } from "node:child_process";
import path from "path";

const BACKEND = process.env.HERMES_DASHBOARD_URL ?? "http://127.0.0.1:9119";

export const BUILD_PROVENANCE_FILE = "build-provenance.json";

interface BuildProvenance {
  schemaVersion: 1;
  commitSha: string | null;
  branch: string | null;
  dirty: boolean | null;
  builtAt: string;
  invocation: string;
}

function gitOutput(repoRoot: string, args: string[]): string | null {
  try {
    return execFileSync("git", ["-C", repoRoot, ...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 2_000,
    }).trim();
  } catch {
    return null;
  }
}

function buildTimestamp(): string {
  // Reproducible package builders conventionally provide SOURCE_DATE_EPOCH.
  // Honor it when present instead of injecting wall-clock entropy into their
  // outputs; ordinary dashboard builds still record their actual build time.
  const sourceDateEpoch = process.env.SOURCE_DATE_EPOCH;
  if (sourceDateEpoch && /^\d+$/.test(sourceDateEpoch)) {
    const timestamp = new Date(Number(sourceDateEpoch) * 1_000);
    if (!Number.isNaN(timestamp.getTime())) return timestamp.toISOString();
  }
  return new Date().toISOString();
}

function buildInvocation(): string {
  const viteArgv = process.argv
    .slice(1)
    .map((arg, index) => (index === 0 ? path.basename(arg) : arg))
    .join(" ");
  const npmEvent = process.env.npm_lifecycle_event;
  if (npmEvent) return `npm run ${npmEvent}${viteArgv ? ` -> ${viteArgv}` : ""}`;
  return viteArgv || "vite build";
}

function packagedDirtyState(): boolean | null {
  const value = process.env.BUILD_SOURCE_DIRTY?.trim().toLowerCase();
  if (["1", "true", "yes", "dirty"].includes(value ?? "")) return true;
  if (["0", "false", "no", "clean"].includes(value ?? "")) return false;
  return null;
}

function collectBuildProvenance(repoRoot: string): BuildProvenance {
  const insideWorkTree =
    gitOutput(repoRoot, ["rev-parse", "--is-inside-work-tree"]) === "true";
  const commitSha = insideWorkTree
    ? gitOutput(repoRoot, ["rev-parse", "HEAD"])
    : (process.env.HERMES_GIT_SHA ??
      process.env.HERMES_REVISION ??
      process.env.GITHUB_SHA ??
      null);
  const branch = insideWorkTree
    ? (gitOutput(repoRoot, ["symbolic-ref", "--short", "-q", "HEAD"]) ?? "detached")
    : (process.env.BUILD_SOURCE_BRANCH ??
      process.env.GITHUB_HEAD_REF ??
      (process.env.GITHUB_REF_TYPE === "branch" ? process.env.GITHUB_REF_NAME : null) ??
      null);
  const status = insideWorkTree
    ? gitOutput(repoRoot, ["status", "--porcelain=v1", "--untracked-files=normal"])
    : null;

  return {
    schemaVersion: 1,
    commitSha: commitSha || null,
    branch: branch || null,
    dirty:
      insideWorkTree && status !== null
        ? status.length > 0
        : packagedDirtyState(),
    builtAt: buildTimestamp(),
    invocation: buildInvocation(),
  };
}

/** Emit source identity from Vite itself so every build path is stamped. */
export function hermesBuildProvenance(
  repoRoot = path.resolve(__dirname, ".."),
): Plugin {
  return {
    name: "hermes:build-provenance",
    apply: "build",
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: BUILD_PROVENANCE_FILE,
        source: `${JSON.stringify(collectBuildProvenance(repoRoot), null, 2)}\n`,
      });
    },
  };
}

/**
 * In production the Python `hermes dashboard` server injects a one-shot
 * session token into `index.html` (see `hermes_cli/web_server.py`). The
 * Vite dev server serves its own `index.html`, so unless we forward that
 * token, every protected `/api/*` call 401s.
 *
 * This plugin fetches the running dashboard's `index.html` on each dev page
 * load and forwards its runtime bootstrap values into the dev HTML. No-op in
 * production builds.
 */
function hermesDevToken(): Plugin {
  const TOKEN_RE = /window\.__HERMES_SESSION_TOKEN__\s*=\s*"([^"]+)"/;
  const EMBEDDED_RE =
    /window\.__HERMES_DASHBOARD_EMBEDDED_CHAT__\s*=\s*(true|false)/;
  const INITIAL_PROFILE_RE =
    /window\.__HERMES_INITIAL_PROFILE__\s*=\s*("(?:\\.|[^"\\])*")/;

  return {
    name: "hermes:dev-session-token",
    apply: "serve",
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
        const embeddedJs = embeddedMatch ? embeddedMatch[1] : "true";
        const initialProfileMatch = html.match(INITIAL_PROFILE_RE);
        const initialProfileJs = initialProfileMatch?.[1] ?? '""';
        return [
          {
            tag: "script",
            injectTo: "head",
            children:
              `window.__HERMES_SESSION_TOKEN__="${match[1]}";` +
              `window.__HERMES_DASHBOARD_EMBEDDED_CHAT__=${embeddedJs};` +
              `window.__HERMES_INITIAL_PROFILE__=${initialProfileJs};`,
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
  plugins: [
    react(),
    babel({ presets: [compilerPreset()] }),
    tailwindcss(),
    hermesDevToken(),
    hermesBuildProvenance(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@hermes/shared": path.resolve(__dirname, "../apps/shared/src"),
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
      // Same host as `hermes dashboard` must serve these; Vite has no
      // dashboard-plugins/* files, so without this, plugin scripts 404
      // or receive index.html in dev.
      "/dashboard-plugins": BACKEND,
    },
  },
});
