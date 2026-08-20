import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

// Regression guard: every dashboard copy action must go through
// copyTextToClipboard (web/src/lib/clipboard.ts), which falls back to a
// selection-based copy when the async Clipboard API is unavailable —
// e.g. self-hosted dashboards served over plain HTTP on a LAN, where
// window.isSecureContext is false and navigator.clipboard is undefined.
//
// Clipboard READS (paste paths) have no equivalent legacy fallback and are
// exempt; they must feature-detect and fail soft instead.
//
// Ported from paperclipai/paperclip#10875.

const SRC_ROOT = join(__dirname, "..");

// The helper itself is the single allowed writer.
const ALLOWLIST = new Set(["lib/clipboard.ts"]);

function collectSourceFiles(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      collectSourceFiles(full, out);
    } else if (/\.(ts|tsx)$/.test(entry) && !/\.(test|spec)\.(ts|tsx)$/.test(entry)) {
      out.push(full);
    }
  }
  return out;
}

describe("clipboard write regression guard", () => {
  it("web/src has no direct navigator.clipboard write calls outside lib/clipboard.ts", () => {
    const offenders: string[] = [];
    for (const file of collectSourceFiles(SRC_ROOT)) {
      const rel = relative(SRC_ROOT, file).replace(/\\/g, "/");
      if (ALLOWLIST.has(rel)) continue;
      const source = readFileSync(file, "utf8");
      // Match writeText / write on navigator.clipboard (incl. optional
      // chaining); reads (read / readText) are intentionally not matched.
      const pattern = /navigator\.clipboard\??\s*\.\s*write(Text)?\s*[(.]/g;
      let match: RegExpExecArray | null;
      while ((match = pattern.exec(source)) !== null) {
        const line = source.slice(0, match.index).split("\n").length;
        offenders.push(`${rel}:${line}`);
      }
    }
    expect(
      offenders,
      `Direct navigator.clipboard writes found — use copyTextToClipboard from "@/lib/clipboard" instead:\n${offenders.join("\n")}`,
    ).toEqual([]);
  });
});
