import { readFileSync } from "node:fs";
import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { isCodexProvider } from "@/lib/reasoning-effort";

vi.mock("@nous-research/ui/ui/components/select", () => ({
  Select: ({ children }: { children?: ReactNode }) => <div>{children}</div>,
  SelectOption: ({
    children,
    value,
  }: {
    children?: ReactNode;
    value: string;
  }) => <span data-value={value}>{children}</span>,
}));

vi.mock("@/lib/api", () => ({
  api: { getConfig: vi.fn() },
}));

import { ReasoningPicker } from "./ReasoningPicker";

function renderForProvider(provider: string): string {
  return renderToStaticMarkup(
    <ReasoningPicker
      currentModel="gpt-5.6-sol"
      showMode={isCodexProvider(provider)}
    />,
  );
}

describe("ReasoningPicker reasoning mode visibility", () => {
  it("shows standard and pro for openai-codex", () => {
    const markup = renderForProvider("openai-codex");

    expect(markup).toContain(">mode<");
    expect(markup).toContain("Standard");
    expect(markup).toContain("Pro");
  });

  it("hides the mode control for non-Codex providers", () => {
    const markup = renderForProvider("openai");

    expect(markup).not.toContain(">mode<");
    expect(markup).not.toContain("Standard");
    expect(markup).not.toContain("Pro");
  });
});

describe("ChatSidebar reasoning mode wiring", () => {
  it("uses the effective REST provider to control mode visibility", () => {
    const source = readFileSync(
      new URL("./ChatSidebar.tsx", import.meta.url),
      "utf8",
    );

    expect(source).toMatch(
      /setEffectiveProvider\(String\(r\?\.provider \?\? ""\)\)/,
    );
    expect(source).toContain(
      "showMode={isCodexProvider(effectiveProvider)}",
    );
  });
});
