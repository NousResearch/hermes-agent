/**
 * Copy @nous-research/ui static assets into public/ (cross-platform).
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, "..");
const uiDist = path.join(root, "node_modules", "@nous-research", "ui", "dist");

const pairs = [
  ["fonts", "fonts"],
  ["assets", "ds-assets"],
];

for (const [srcName, destName] of pairs) {
  const src = path.join(uiDist, srcName);
  const dest = path.join(root, "public", destName);
  fs.rmSync(dest, { recursive: true, force: true });
  if (!fs.existsSync(src)) {
    console.warn(`[sync-web-assets] skip ${destName}: missing ${src}`);
    continue;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.cpSync(src, dest, { recursive: true });
}
