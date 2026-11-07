// Build-time helper for `experimental.renderBuiltUrl` in vite.config.ts.
//
// Behind a path-prefix reverse proxy (``X-Forwarded-Prefix: /hermes``) the
// backend rewrites index.html and the built CSS, and injects
// ``window.__HERMES_BASE_PATH__`` (see ``mount_spa`` in
// hermes_cli/web_server_dashboard.py). JS bundles are served as built, so any
// URL Vite bakes into JS (the preload lists of lazy route chunks and their
// CSS, imported asset URLs) would point at the host root and 404 under a
// prefix. Resolving those URLs against the runtime base path keeps lazy routes
// under the mount. An empty base path yields the same root-absolute URLs as a
// default build.
//
// Only JS-hosted URLs are changed. HTML and CSS keep Vite's default
// root-absolute form, which the backend already prefixes per request.
// ``globalThis`` (not ``window``) so the expression is also safe in a worker.

export type BuiltUrlHostType = "js" | "css" | "html";

export function renderDashboardBuiltUrl(
  filename: string,
  hostType: BuiltUrlHostType,
): { runtime: string } | undefined {
  if (hostType !== "js") return undefined;
  return {
    runtime: `(globalThis.__HERMES_BASE_PATH__||"")+${JSON.stringify(`/${filename}`)}`,
  };
}
