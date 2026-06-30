#!/usr/bin/env node
// Runs website/scripts/extract-skills.py and generate-llms-txt.py before
// docusaurus build/start so that:
//   - website/static/api/skills.json (lazy-fetched by src/pages/skills/index.tsx)
//   - website/static/api/skills-meta.json (sidecar metadata for the Skills Hub)
//   - website/static/llms.txt (agent-friendly short docs index)
//   - website/static/llms-full.txt (full docs concat for LLM context)
// all exist without contributors remembering to run Python scripts manually.
// CI workflows still run the extraction explicitly, which is a no-op duplicate
// but matches their historical behaviour.
//
// We also try to pull a fresh copy of skills-index.json (the unified
// multi-source catalog) from the live docs site if it's not already on disk.
// That way local `npm run build` doesn't have to wait on
// scripts/build_skills_index.py crawling every skill source — which takes
// several minutes and burns GitHub API quota — but still gets the same
// 2000+ external skills the deployed site has.
//
// If python3 or its deps (pyyaml) aren't available on the local machine, we
// fall back to writing an empty skills.json so `npm run build` still
// succeeds — the Skills Hub page just shows an empty state, and llms.txt
// generation is skipped. CI always has the deps installed, so production
// deploys get real data.

import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync, existsSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const websiteDir = resolve(scriptDir, "..");
const extractScript = join(scriptDir, "extract-skills.py");
const llmsScript = join(scriptDir, "generate-llms-txt.py");
const cronBlueprintsScript = join(scriptDir, "extract-automation-blueprints.py");
// Prefer the hermes-agent venv (it has pyyaml + the agent packages these
// scripts import); honor an explicit $PYTHON override; fall back to bare
// python3 last (which on many machines lacks pyyaml -> empty Skills Hub).
const venvPython = resolve(scriptDir, "../../venv/bin/python");
const PYTHON = process.env.PYTHON ?? (existsSync(venvPython) ? venvPython : "python3");
const outputFile = join(websiteDir, "static", "api", "skills.json");
const unifiedIndexFile = join(websiteDir, "static", "api", "skills-index.json");
const UNIFIED_INDEX_URL =
  "https://hermes-agent.nousresearch.com/docs/api/skills-index.json";
const UNIFIED_INDEX_MAX_AGE_MS = 24 * 60 * 60 * 1000; // 24h

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
  const r = spawnSync(PYTHON, [script], { stdio: "inherit", cwd: websiteDir });
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

async function ensureUnifiedIndex() {
  // If we have a recent copy on disk, trust it.
  if (existsSync(unifiedIndexFile)) {
    try {
      const age = Date.now() - statSync(unifiedIndexFile).mtimeMs;
      if (age < UNIFIED_INDEX_MAX_AGE_MS) {
        return true;
      }
      console.log(
        `[prebuild] skills-index.json is ${(age / 3600000).toFixed(1)}h old; ` +
          `refreshing from ${UNIFIED_INDEX_URL}`,
      );
    } catch {
      // fall through to re-fetch
    }
  }

  try {
    const resp = await fetch(UNIFIED_INDEX_URL, {
      headers: { accept: "application/json" },
    });
    if (!resp.ok) {
      console.warn(
        `[prebuild] skills-index.json fetch returned HTTP ${resp.status}; ` +
          `using local copy if any`,
      );
      return existsSync(unifiedIndexFile);
    }
    const text = await resp.text();
    // Sanity check: must be valid JSON with a skills array
    try {
      const parsed = JSON.parse(text);
      if (!parsed || !Array.isArray(parsed.skills)) {
        console.warn(
          "[prebuild] skills-index.json from live site has no skills array; ignoring",
        );
        return existsSync(unifiedIndexFile);
      }
    } catch (e) {
      console.warn(`[prebuild] skills-index.json from live site is not valid JSON: ${e}`);
      return existsSync(unifiedIndexFile);
    }
    mkdirSync(dirname(unifiedIndexFile), { recursive: true });
    writeFileSync(unifiedIndexFile, text);
    console.log(
      `[prebuild] downloaded skills-index.json from ${UNIFIED_INDEX_URL} ` +
        `(${(text.length / 1024).toFixed(0)} KB)`,
    );
    return true;
  } catch (e) {
    console.warn(`[prebuild] skills-index.json fetch failed: ${e}`);
    return existsSync(unifiedIndexFile);
  }
}

// 0) Pull unified index if we don't have a fresh one.
await ensureUnifiedIndex();

// 1) skills.json — required for the Skills Hub page.
if (!existsSync(extractScript)) {
  writeEmptyFallback("extract script missing");
} else {
  const r = spawnSync(PYTHON, [extractScript], {
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

// 3) automation-blueprints-index.json — Automation Blueprints catalog page. Non-fatal; the page
//    renders an empty state if the generator can't run.
runPython(cronBlueprintsScript, "extract-automation-blueprints.py");
