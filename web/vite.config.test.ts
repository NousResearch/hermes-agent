import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { build } from "vite";

import {
  BUILD_PROVENANCE_FILE,
  hermesBuildProvenance,
} from "./vite.config";

type Provenance = {
  schemaVersion: number;
  commitSha: string | null;
  branch: string | null;
  dirty: boolean | null;
  builtAt: string;
  invocation: string;
};

const temporaryPaths: string[] = [];

function git(cwd: string, ...args: string[]): string {
  return execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
}

async function makeRepository(): Promise<string> {
  const repo = await mkdtemp(path.join(tmpdir(), "hermes-web-provenance-repo-"));
  temporaryPaths.push(repo);
  await mkdir(path.join(repo, "web"));
  await writeFile(path.join(repo, "web", "index.html"), "<main>Hermes</main>\n");
  git(repo, "init", "--quiet");
  git(repo, "add", ".");
  git(
    repo,
    "-c",
    "user.name=Hermes Test",
    "-c",
    "user.email=hermes-test@example.invalid",
    "commit",
    "--quiet",
    "-m",
    "fixture",
  );
  return repo;
}

async function buildFixture(repo: string): Promise<Provenance> {
  const outDir = await mkdtemp(path.join(tmpdir(), "hermes-web-provenance-dist-"));
  temporaryPaths.push(outDir);
  await build({
    configFile: false,
    root: path.join(repo, "web"),
    logLevel: "silent",
    plugins: [hermesBuildProvenance(repo)],
    build: { outDir, emptyOutDir: true },
  });
  return JSON.parse(
    await readFile(path.join(outDir, BUILD_PROVENANCE_FILE), "utf8"),
  ) as Provenance;
}

afterEach(async () => {
  vi.unstubAllEnvs();
  await Promise.all(
    temporaryPaths.splice(0).map((temporaryPath) =>
      rm(temporaryPath, { recursive: true, force: true }),
    ),
  );
});

describe("hermesBuildProvenance", () => {
  it("emits the committed source identity into an external bundle", async () => {
    const repo = await makeRepository();

    const provenance = await buildFixture(repo);

    expect(provenance.schemaVersion).toBe(1);
    expect(provenance.commitSha).toBe(git(repo, "rev-parse", "HEAD"));
    expect(provenance.branch).toBe(git(repo, "branch", "--show-current"));
    expect(provenance.dirty).toBe(false);
    expect(Number.isNaN(Date.parse(provenance.builtAt))).toBe(false);
    expect(provenance.invocation.length).toBeGreaterThan(0);
  });

  it("preserves the dirty-tree fact after the build completes", async () => {
    const repo = await makeRepository();
    await writeFile(path.join(repo, "web", "index.html"), "<main>Dirty Hermes</main>\n");

    const provenance = await buildFixture(repo);

    expect(provenance.commitSha).toBe(git(repo, "rev-parse", "HEAD"));
    expect(provenance.dirty).toBe(true);
  });

  it("uses packaged source metadata when the build has no git directory", async () => {
    const source = await mkdtemp(path.join(tmpdir(), "hermes-web-provenance-source-"));
    temporaryPaths.push(source);
    await mkdir(path.join(source, "web"));
    await writeFile(path.join(source, "web", "index.html"), "<main>Packaged Hermes</main>\n");
    const sourceRevision = "a".repeat(40);
    vi.stubEnv("HERMES_GIT_SHA", sourceRevision);
    vi.stubEnv("BUILD_SOURCE_BRANCH", "packaging-branch");
    vi.stubEnv("BUILD_SOURCE_DIRTY", "false");

    const provenance = await buildFixture(source);

    expect(provenance.commitSha).toBe(sourceRevision);
    expect(provenance.branch).toBe("packaging-branch");
    expect(provenance.dirty).toBe(false);
  });

  it("does not mislabel a packaged release tag as a branch", async () => {
    const source = await mkdtemp(path.join(tmpdir(), "hermes-web-provenance-tag-"));
    temporaryPaths.push(source);
    await mkdir(path.join(source, "web"));
    await writeFile(path.join(source, "web", "index.html"), "<main>Tagged Hermes</main>\n");
    vi.stubEnv("HERMES_GIT_SHA", "b".repeat(40));
    vi.stubEnv("BUILD_SOURCE_BRANCH", "");
    vi.stubEnv("GITHUB_REF_NAME", "v1.2.3");
    vi.stubEnv("BUILD_SOURCE_DIRTY", "false");

    const provenance = await buildFixture(source);

    expect(provenance.branch).toBeNull();
  });
});