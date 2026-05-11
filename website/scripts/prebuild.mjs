#!/usr/bin/env node
// Runs website/scripts/extract-skills.py and generate-llms-txt.py before
// docusaurus build/start so that:
//   - website/src/data/skills.json (imported by src/pages/skills/index.tsx)
//   - website/static/llms.txt (agent-friendly short docs index)
//   - website/static/llms-full.txt (full docs concat for LLM context)
// all exist without contributors remembering to run Python scripts manually.
// CI workflows still run the extraction explicitly, which is a no-op duplicate
// but matches their historical behaviour.
//
// If python3 or its deps (pyyaml) aren't available on the local machine, we
// fall back to writing an empty skills.json so `npm run build` still
// succeeds — the Skills Hub page just shows an empty state, and llms.txt
// generation is skipped. CI always has the deps installed, so production
// deploys get real data.

import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const websiteDir = resolve(scriptDir, "..");
const extractScript = join(scriptDir, "extract-skills.py");
const llmsScript = join(scriptDir, "generate-llms-txt.py");
const outputFile = join(websiteDir, "src", "data", "skills.json");

function writeEmptyFallback(reason) {
  mkdirSync(dirname(outputFile), { recursive: true });
  writeFileSync(outputFile, "[]\n");
  console.warn(
    `[prebuild] extract-skills.py skipped (${reason}); wrote empty skills.json. ` +
      `Install python3 + pyyaml locally for a populated Skills Hub page.`,
  );
}

function runPython(script, label) {
  if (!existsSync(script)) {
    console.warn(`[prebuild] ${label} skipped (script missing)`);
    return false;
  }
  const r = spawnSync("python3", [script], { stdio: "inherit", cwd: websiteDir });
  if (r.error && r.error.code === "ENOENT") {
    console.warn(`[prebuild] ${label} skipped (python3 not found)`);
    return false;
  }
  if (r.status !== 0) {
    console.warn(`[prebuild] ${label} exited with status ${r.status}`);
    return false;
  }
  return true;
}

// 1) skills.json — required for the Skills Hub page.
if (!existsSync(extractScript)) {
  writeEmptyFallback("extract script missing");
} else {
  const r = spawnSync("python3", [extractScript], {
    stdio: "inherit",
    cwd: websiteDir,
  });
  if (r.error && r.error.code === "ENOENT") {
    writeEmptyFallback("python3 not found");
  } else if (r.status !== 0) {
    writeEmptyFallback(`extract-skills.py exited with status ${r.status}`);
  }
}

// 2) llms.txt + llms-full.txt — agent-friendly docs entrypoints. Non-fatal.
runPython(llmsScript, "generate-llms-txt.py");
