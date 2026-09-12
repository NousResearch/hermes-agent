import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * Contracts on the static PWA assets in `web/public/` and the tags in
 * `web/index.html`.
 *
 * These files are copied verbatim into the built bundle, so nothing else type-
 * checks or lints them — a wrong path here costs installability silently, with
 * no build error and no runtime exception. The server-side half (the URLs
 * resolving, and surviving a reverse-proxy prefix) lives in
 * `tests/hermes_cli/test_web_server_pwa.py`.
 */

const publicDir = fileURLToPath(new URL("../../public/", import.meta.url));
const indexHtml = readFileSync(
  fileURLToPath(new URL("../../index.html", import.meta.url)),
  "utf-8",
);
const manifest = JSON.parse(
  readFileSync(`${publicDir}manifest.webmanifest`, "utf-8"),
) as {
  display: string;
  theme_color: string;
  start_url: string;
  scope: string;
  icons: { src: string; sizes: string; purpose?: string }[];
  shortcuts?: { url: string; icons?: { src: string }[] }[];
};

/** Every URL the manifest declares, from whichever field carries it. */
function manifestUrls(): string[] {
  return [
    manifest.start_url,
    manifest.scope,
    ...manifest.icons.map((icon) => icon.src),
    ...(manifest.shortcuts ?? []).flatMap((shortcut) => [
      shortcut.url,
      ...(shortcut.icons ?? []).map((icon) => icon.src),
    ]),
  ];
}

describe("manifest", () => {
  it("declares only relative URLs so a proxy prefix resolves them", () => {
    // A browser resolves these against the manifest's own URL. Root-absolute
    // ("/icons/…") would point outside a /hermes mount and 404 there, which
    // downgrades the install to a bookmark with no icon and no standalone
    // window. Relative URLs need no rebuild per deployment.
    for (const url of manifestUrls()) {
      expect(url, url).not.toMatch(/^\//);
      expect(url, url).not.toMatch(/^[a-z]+:/i);
    }
  });

  it("points every icon at a file that ships in the bundle", () => {
    for (const icon of manifest.icons) {
      expect(existsSync(`${publicDir}${icon.src}`), icon.src).toBe(true);
    }
  });

  it("meets the browser bar for an installable app", () => {
    // Chrome requires a standalone display mode plus a 192px and a 512px icon
    // before it offers installation at all.
    expect(manifest.display).toBe("standalone");
    const sizes = new Set(manifest.icons.map((icon) => icon.sizes));
    expect(sizes.has("192x192")).toBe(true);
    expect(sizes.has("512x512")).toBe(true);
  });

  it("ships a maskable icon so Android does not letterbox it", () => {
    const maskable = manifest.icons.filter((icon) =>
      (icon.purpose ?? "").split(/\s+/).includes("maskable"),
    );
    expect(maskable.length).toBeGreaterThan(0);
  });
});

describe("index.html", () => {
  it("links the manifest and an apple-touch-icon that both exist", () => {
    // iOS reads the touch icon from the HTML, never from the manifest, so the
    // home-screen icon is blank without this tag.
    for (const rel of ["manifest", "apple-touch-icon"]) {
      const match = indexHtml.match(
        new RegExp(`<link rel="${rel}" href="(/[^"]+)"`),
      );
      expect(match, `missing rel="${rel}"`).toBeTruthy();
      expect(existsSync(`${publicDir}${match![1].slice(1)}`)).toBe(true);
    }
  });

  it("declares a theme colour matching the manifest", () => {
    // A mismatch shows as a colour flash between the splash screen and the app.
    const match = indexHtml.match(/<meta name="theme-color" content="([^"]+)"/);
    expect(match).toBeTruthy();
    expect(match![1]).toBe(manifest.theme_color);
  });
});

describe("service worker", () => {
  const sw = readFileSync(`${publicDir}sw.js`, "utf-8");

  /** Parse a top-level `const NAME = [ "…", … ];` array out of the worker. */
  function ruleList(name: string): string[] {
    const match = sw.match(new RegExp(`const ${name} = \\[([^\\]]*)\\]`));
    expect(match, `${name} not found in sw.js`).toBeTruthy();
    return [...match![1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  }

  it("never caches the API surface", () => {
    // /api carries live agent state, secrets and the session token. Anything
    // cached there would be written to CacheStorage on disk and could be
    // replayed to the page after the backend revoked it.
    expect(ruleList("NEVER_CACHE")).toContain("/api/");
  });

  it("never caches the auth flow or operator plugin scripts", () => {
    const never = ruleList("NEVER_CACHE");
    expect(never).toContain("/auth/");
    expect(never).toContain("/dashboard-plugins/");
  });

  it("only serves cache-first from content-hashed asset paths", () => {
    // A cache-first hit is returned without ever consulting the network, so it
    // is only correct where the URL itself changes with the content. Vite
    // hashes filenames under /assets; nothing else in the bundle is immutable.
    expect(ruleList("IMMUTABLE_PREFIXES")).toEqual(["/assets/"]);
  });

  it("keeps the offline notice out of the never-cache set", () => {
    // It is precached on install; a never-cache rule would make the offline
    // fallback itself require the network.
    for (const prefix of ruleList("NEVER_CACHE")) {
      expect("/offline.html".startsWith(prefix)).toBe(false);
    }
  });
});
