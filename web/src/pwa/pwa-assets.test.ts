import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const WEB_ROOT = fileURLToPath(new URL("../../", import.meta.url));

function readWebFile(relativePath: string): string {
  return readFileSync(new URL(relativePath, `file://${WEB_ROOT}/`), "utf8");
}

describe("Android PWA manifest", () => {
  const manifest = JSON.parse(
    readWebFile("public/manifest.webmanifest"),
  ) as Record<string, unknown>;

  it("uses the Hermes dashboard install properties", () => {
    expect(manifest).toMatchObject({
      name: "Hermes Agent",
      short_name: "Hermes",
      start_url: "./sessions",
      scope: "./",
      display: "standalone",
      background_color: "#041c1c",
      theme_color: "#041c1c",
      prefer_related_applications: false,
    });
    expect(manifest.description).toEqual(expect.any(String));
  });

  it("references only the approved Android icon locations", () => {
    expect(manifest.icons).toEqual([
      {
        src: "icons/hermes-192.png",
        sizes: "192x192",
        type: "image/png",
        purpose: "any",
      },
      {
        src: "icons/hermes-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "any",
      },
    ]);
  });
});

describe("Android PWA document metadata", () => {
  const html = readWebFile("index.html");

  it("links the manifest and Android standalone metadata", () => {
    expect(html).toContain('<link rel="manifest" href="manifest.webmanifest" />');
    expect(html).toContain('<meta name="theme-color" content="#041c1c" />');
    expect(html).toContain('<meta name="mobile-web-app-capable" content="yes" />');
    expect(html).toContain('<meta name="application-name" content="Hermes" />');
    expect(html).toContain("viewport-fit=cover");
  });

  it("does not add Apple-specific metadata", () => {
    expect(html.toLowerCase()).not.toContain("apple-mobile-web-app");
    expect(html.toLowerCase()).not.toContain("apple-touch-icon");
  });
});
