import { describe, it, expect } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const packageJsonPath = path.join(__dirname, "../../package.json");

describe("TUI build script", () => {
  it("should not contain chmod in the build script (Windows compatibility)", () => {
    const raw = fs.readFileSync(packageJsonPath, "utf-8");
    const pkg = JSON.parse(raw);
    const buildScript = pkg.scripts?.build ?? "";

    expect(buildScript).not.toContain("chmod");
  });

  it("should still compile and bundle the entry point", () => {
    const raw = fs.readFileSync(packageJsonPath, "utf-8");
    const pkg = JSON.parse(raw);
    const buildScript = pkg.scripts?.build ?? "";

    expect(buildScript).toContain("tsc -p tsconfig.build.json");
    expect(buildScript).toContain("npm run build:compile");
  });
});
