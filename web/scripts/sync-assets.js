import { copyFileSync, existsSync, mkdirSync, rmSync, readdirSync, statSync } from "fs";
import { join } from "path";
import { fileURLToPath } from "url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const root = join(__dirname, "..");

const uiDist = join(root, "node_modules", "@nous-research", "ui", "dist");
const fontSrc = join(uiDist, "fonts");
const assetSrc = join(uiDist, "assets");
const fontDst = join(root, "public", "fonts");
const assetDst = join(root, "public", "ds-assets");

function copyDirRecursive(src, dst) {
  if (!existsSync(src)) {
    console.error(`Source directory not found: ${src}`);
    process.exit(1);
  }
  if (existsSync(dst)) rmSync(dst, { recursive: true, force: true });
  mkdirSync(dst, { recursive: true });
  for (const entry of readdirSync(src)) {
    const srcPath = join(src, entry);
    const dstPath = join(dst, entry);
    if (statSync(srcPath).isDirectory()) {
      copyDirRecursive(srcPath, dstPath);
    } else {
      copyFileSync(srcPath, dstPath);
    }
  }
}

copyDirRecursive(fontSrc, fontDst);
copyDirRecursive(assetSrc, assetDst);
console.log("Assets synced: fonts + ds-assets");
