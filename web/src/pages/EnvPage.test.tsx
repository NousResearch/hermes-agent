// @vitest-environment jsdom
// Covers the 1Password-managed locked-row branch of EnvVarRow: a key whose
// info.managed_by === "onepassword" must render as read-only (no edit/save/
// clear/reveal controls) with the "Managed via 1Password" explainer, matching
// the desktop app's KeyField locked state (apps/desktop's
// credential-key-ui.tsx). See hermes_cli/config.py save_env_value(), which
// refuses to write a plaintext .env copy for these keys server-side.

import { describe, it, expect, afterEach } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ReactNode } from "react";

import { I18nProvider } from "@/i18n";
import { EnvVarRow } from "./EnvPage";
import type { EnvVarInfo } from "@/lib/api";

let container: HTMLDivElement;
let root: Root;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<I18nProvider>{ui}</I18nProvider>));
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

const managedInfo: EnvVarInfo = {
  is_set: true,
  redacted_value: "sk-...abcd",
  description: "Anthropic API key",
  url: null,
  category: "provider",
  is_password: true,
  tools: [],
  advanced: false,
  managed_by: "onepassword",
};

describe("EnvVarRow — 1Password-managed key", () => {
  it("renders locked, read-only, with no edit/save/clear/reveal controls", async () => {
    await render(
      <EnvVarRow
        varKey="ANTHROPIC_API_KEY"
        info={managedInfo}
        edits={{}}
        setEdits={() => {}}
        revealed={{}}
        saving={null}
        onSave={() => {}}
        onClear={() => {}}
        onReveal={() => {}}
        onCancelEdit={() => {}}
      />,
    );

    expect(container.textContent).toContain("sk-...abcd");
    expect(container.textContent).toContain("Managed via 1Password");
    expect(container.querySelectorAll("button").length).toBe(0);
  });
});
