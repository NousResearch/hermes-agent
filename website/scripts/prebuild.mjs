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
const repoDir = resolve(websiteDir, "..");

function pythonCandidates() {
  const names = process.platform === "win32" ? ["python.exe"] : ["python"];
  return [
    ...names.map((name) => join(repoDir, ".venv", process.platform === "win32" ? "Scripts" : "bin", name)),
    ...names.map((name) => join(repoDir, "venv", process.platform === "win32" ? "Scripts" : "bin", name)),
    "python3",
    "python",
  ];
}

function runPythonScript(script, label) {
  if (!existsSync(script)) {
    return { ok: false, reason: `${label} missing` };
  }
  for (const python of pythonCandidates()) {
    if (python.includes("/bin/") || python.includes("\\Scripts\\")) {
      if (!existsSync(python)) continue;
    }
    const r = spawnSync(python, [script], { stdio: "inherit", cwd: websiteDir });
    if (r.error && r.error.code === "ENOENT") continue;
    if (r.status === 0) return { ok: true, python };
    return { ok: false, reason: `${label} exited with status ${r.status}` };
  }
  return { ok: false, reason: "python not found" };
}

function writeEmptyFallback(reason) {
  mkdirSync(dirname(outputFile), { recursive: true });
  writeFileSync(outputFile, "[]\n");
  console.warn(
    `[prebuild] extract-skills.py skipped (${reason}); wrote empty skills.json. ` +
      `Install python3 + pyyaml locally for a populated Skills Hub page.`,
  );
}

function runPython(script, label) {
  const result = runPythonScript(script, label);
  if (!result.ok) {
    console.warn(`[prebuild] ${label} skipped (${result.reason})`);
  }
  return result.ok;
}

// 1) skills.json — required for the Skills Hub page.
const extractResult = runPythonScript(extractScript, "extract-skills.py");
if (!extractResult.ok) {
  writeEmptyFallback(extractResult.reason);
}

// 2) llms.txt + llms-full.txt — agent-friendly docs entrypoints. Non-fatal.
runPython(llmsScript, "generate-llms-txt.py");
