#!/usr/bin/env node

import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(webRoot, "..");
const standalone = existsSync(resolve(webRoot, "package-lock.json"));

function run(command, args, cwd, capture = false) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: capture ? "utf8" : undefined,
    stdio: capture ? "pipe" : "inherit",
  });
  if (result.error) throw result.error;
  return result;
}

function runBuild(root) {
  const mappedWebRoot = resolve(root, "web");
  const dependencyRoot = standalone ? mappedWebRoot : root;
  const steps = [
    [resolve(dependencyRoot, "node_modules", "typescript", "bin", "tsc"), "-b"],
    [resolve(dependencyRoot, "node_modules", "vite", "bin", "vite.js"), "build"],
  ];

  for (const [entrypoint, ...args] of steps) {
    const result = run(process.execPath, [entrypoint, ...args], mappedWebRoot);
    if (result.status !== 0) return result.status ?? 1;
  }
  return 0;
}

function createSubstMapping(target) {
  for (let code = "Z".charCodeAt(0); code >= "D".charCodeAt(0); code -= 1) {
    const drive = `${String.fromCharCode(code)}:`;
    if (existsSync(`${drive}\\`)) continue;
    const result = run("subst.exe", [drive, target], repoRoot, true);
    if (result.status === 0) return drive;
  }
  throw new Error(
    "Unable to allocate a temporary drive for the web build. Free a drive letter and retry.",
  );
}

let status;
if (process.platform === "win32" && /[#?]/.test(repoRoot)) {
  const drive = createSubstMapping(repoRoot);
  console.log(
    `[web] Vite-incompatible install path detected; building through temporary ${drive} mapping.`,
  );
  try {
    status = runBuild(`${drive}\\`);
  } finally {
    const cleanup = run("subst.exe", [drive, "/D"], repoRoot, true);
    if (cleanup.status !== 0) {
      const detail = cleanup.stderr?.trim();
      throw new Error(
        `Failed to remove temporary ${drive} mapping${detail ? `: ${detail}` : "."}`,
      );
    }
  }
} else {
  status = runBuild(repoRoot);
}

process.exit(status);
