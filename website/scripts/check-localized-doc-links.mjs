#!/usr/bin/env node
// Deterministic probe: walks every non-default locale's built output and fails
// if it finds the `/docs/<locale>/docs/` substring anywhere in emitted
// HTML/JS/JSON/CSS. That substring is the smoking gun for hard-coded `/docs/`
// markdown links combined with Docusaurus localizing baseUrl per locale:
// the site is served at `/docs/<locale>/` for non-default locales, and a link
// target like `/docs/user-guide/foo` gets prefixed to `/docs/<locale>/docs/...`.
//
// Run after `npm run build`. Exits 1 with a list of offending files if any
// occurrences are found. Exits 0 silently on a clean tree.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, dirname, resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const websiteDir = resolve(scriptDir, "..");
const buildDir = join(websiteDir, "build");

const NON_DEFAULT_LOCALES = ["zh-Hans", "ko"];
const SCANNABLE_EXTENSIONS = new Set([".html", ".js", ".json", ".css", ".txt"]);

function listFilesRecursive(dir) {
  const out = [];
  let entries;
  try {
    entries = readdirSync(dir);
  } catch (err) {
    if (err.code === "ENOENT") return out;
    throw err;
  }
  for (const entry of entries) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      out.push(...listFilesRecursive(full));
    } else if (st.isFile()) {
      const dot = entry.lastIndexOf(".");
      const ext = dot === -1 ? "" : entry.slice(dot);
      if (SCANNABLE_EXTENSIONS.has(ext)) out.push(full);
    }
  }
  return out;
}

const offenders = [];
let scannedAtLeastOneLocale = false;

for (const locale of NON_DEFAULT_LOCALES) {
  const localeRoot = join(buildDir, locale);
  let st;
  try {
    st = statSync(localeRoot);
  } catch (err) {
    if (err.code === "ENOENT") {
      console.warn(
        `check-localized-doc-links: skipping ${locale} — ${relative(websiteDir, localeRoot)} does not exist (was the site built?)`,
      );
      continue;
    }
    throw err;
  }
  if (!st.isDirectory()) continue;
  scannedAtLeastOneLocale = true;

  const needle = `/docs/${locale}/docs/`;
  for (const file of listFilesRecursive(localeRoot)) {
    const content = readFileSync(file, "utf8");
    if (content.includes(needle)) {
      offenders.push({ file: relative(websiteDir, file), needle });
    }
  }
}

if (!scannedAtLeastOneLocale) {
  console.error(
    `check-localized-doc-links: no localized build output found under ${relative(process.cwd(), buildDir)}. Run \`npm run build\` first.`,
  );
  process.exit(1);
}

if (offenders.length > 0) {
  console.error(
    `check-localized-doc-links: found ${offenders.length} file(s) emitting localized double-prefixed /docs/<locale>/docs/ links:`,
  );
  for (const { file, needle } of offenders) {
    console.error(`  ${file}  (contains "${needle}")`);
  }
  console.error(
    "\nThis usually means a markdown file under website/docs/ contains a hard-coded `/docs/...` link. Rewrite those to root-relative form (e.g. `/user-guide/foo` instead of `/docs/user-guide/foo`).",
  );
  process.exit(1);
}

console.log("check-localized-doc-links: no /docs/<locale>/docs/ links found in generated output.");
