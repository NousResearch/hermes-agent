import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const source = readFileSync(new URL("./JarvisMemoryPage.tsx", import.meta.url), "utf8");
const css = readFileSync(new URL("./jarvis-memory-vault.css", import.meta.url), "utf8");

describe("JarvisMemoryPage live data integration", () => {
  it("keeps the approved Memory Vault visual shell", () => {
    expect(source).toContain("Memory Vault");
    expect(source).toContain("MemoryField");
    expect(source).toContain("Obsidian Memory Vault");
    expect(source).toContain("Knowledge constellation");
    expect(css).toContain(".memory-stage");
    expect(css).toContain(".vault-core");
  });

  it("wires the Memory page to Jarvis overview memory data", () => {
    expect(source).toContain("api.getJarvisOverview()");
    expect(source).toContain("buildLiveClusters(overview?.memory_vault)");
    expect(source).toContain("memoryVault?.note_count");
    expect(source).toContain("memoryVault?.decision_count");
    expect(source).toContain("memoryVault?.product_note_count");
    expect(source).toContain("memoryVault?.recent_notes");
    expect(source).toContain("memory?.quick_links");
    expect(source).toContain("selectedNote?.href");
  });

  it("removes static prototype vault counts from visible metrics", () => {
    expect(source).not.toContain('value="12,842"');
    expect(source).not.toContain('value="184"');
    expect(source).not.toContain('value="7"');
    expect(source).not.toContain('value="100%"');
    expect(source).not.toContain("12,842 indexed notes");
    expect(source).not.toContain("obsidian://open");
  });
});
