#!/usr/bin/env node
// Runs website/scripts/extract-skills.py before docusaurus build/start so
// that website/src/data/skills.json (imported by src/pages/skills/index.tsx)
// exists without contributors needing to remember to run the Python script
// manually. CI workflows still run the extraction explicitly, which is a
// no-op duplicate but matches their historical behaviour.
//
// If python3 or its deps (pyyaml) aren't available on the local machine, we
// fall back to writing an empty skills.json so `npm run build` still
// succeeds — the Skills Hub page just shows an empty state. CI always has
// the deps installed, so production deploys get real data.

import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const websiteDir = resolve(scriptDir, "..");
const extractScript = join(scriptDir, "extract-skills.py");
const outputFile = join(websiteDir, "src", "data", "skills.json");

function writeEmptyFallback(reason) {
  mkdirSync(dirname(outputFile), { recursive: true });
  writeFileSync(outputFile, "[]\n");
  console.warn(
    `[prebuild] extract-skills.py skipped (${reason}); wrote empty skills.json. ` +
      `Install python3 + pyyaml locally for a populated Skills Hub page.`,
  );
}

if (!existsSync(extractScript)) {
  writeEmptyFallback("extract script missing");
  process.exit(0);
}

const result = spawnSync("python3", [extractScript], {
  stdio: "inherit",
  cwd: websiteDir,
});

if (result.error && result.error.code === "ENOENT") {
  writeEmptyFallback("python3 not found");
  process.exit(0);
}

if (result.status !== 0) {
  writeEmptyFallback(`extract-skills.py exited with status ${result.status}`);
  process.exit(0);
}
