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
// If Python or its deps (pyyaml) aren't available on the local machine, we
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
const repoRoot = resolve(websiteDir, "..");
const extractScript = join(scriptDir, "extract-skills.py");
const llmsScript = join(scriptDir, "generate-llms-txt.py");
const outputFile = join(websiteDir, "src", "data", "skills.json");
const pythonCandidates = [
  process.env.PYTHON,
  join(repoRoot, "venv", "bin", "python"),
  join(repoRoot, ".venv", "bin", "python"),
  "python3",
].filter(Boolean);

function runWithPython(script) {
  let lastResult = null;
  let lastPython = pythonCandidates.join(", ");
  for (const python of pythonCandidates) {
    if (python.includes("/") && !existsSync(python)) {
      continue;
    }
    const result = spawnSync(python, [script], {
      stdio: "inherit",
      cwd: websiteDir,
    });
    if (result.error && result.error.code === "ENOENT") {
      lastResult = result;
      continue;
    }
    if (result.status === 0) {
      return { python, result };
    }
    lastPython = python;
    lastResult = result;
  }
  return { python: lastPython, result: lastResult };
}

function writeEmptyFallback(reason) {
  mkdirSync(dirname(outputFile), { recursive: true });
  writeFileSync(outputFile, "[]\n");
  console.warn(
    `[prebuild] extract-skills.py skipped (${reason}); wrote empty skills.json. ` +
      `Install Python + pyyaml locally for a populated Skills Hub page.`,
  );
}

function runPython(script, label) {
  if (!existsSync(script)) {
    console.warn(`[prebuild] ${label} skipped (script missing)`);
    return false;
  }
  const { python, result: r } = runWithPython(script);
  if (!r || (r.error && r.error.code === "ENOENT")) {
    console.warn(`[prebuild] ${label} skipped (Python not found)`);
    return false;
  }
  if (r.status !== 0) {
    console.warn(`[prebuild] ${label} exited with status ${r.status} using ${python}`);
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
  const { python, result: r } = runWithPython(extractScript);
  if (!r || (r.error && r.error.code === "ENOENT")) {
    writeEmptyFallback("Python not found");
  } else if (r.status !== 0) {
    writeEmptyFallback(`extract-skills.py exited with status ${r.status} using ${python}`);
  }
}

// 2) llms.txt + llms-full.txt — agent-friendly docs entrypoints. Non-fatal.
runPython(llmsScript, "generate-llms-txt.py");
