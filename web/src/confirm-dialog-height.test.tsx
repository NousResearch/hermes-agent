// @vitest-environment jsdom
// The design system's ConfirmDialog (@nous-research/ui) centres a Radix
// AlertDialog with no height cap, and Radix locks page scroll while it is
// open: a long description (a catalog entry's install summary) pushed Cancel
// and Install past the bottom of the window with no way to reach them.
// index.css caps every `[role="alertdialog"]` at the viewport and scrolls it
// inside. jsdom cannot compute layout, so these tests pin both halves of that
// contract: the rule is there, and the element carrying the role is the panel
// that holds the description and both actions.

import { ConfirmDialog } from "@nous-research/ui/ui/components/confirm-dialog";
import { readFileSync } from "node:fs";
import path from "node:path";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

let container: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

// Vitest serves `.css` imports empty (CSS processing is off in tests), so read
// the stylesheet next to this file from disk.
function alertDialogRule(): string {
  const testPath = expect.getState().testPath;
  if (!testPath) throw new Error("vitest did not report the test file path");
  const indexCss = readFileSync(path.join(path.dirname(testPath), "index.css"), "utf8");
  return /\[role="alertdialog"\]\s*\{([^}]*)\}/.exec(indexCss)?.[1] ?? "";
}

describe("confirm dialogs fit the viewport", () => {
  it("caps every alert dialog at the viewport height and scrolls inside it", () => {
    const rule = alertDialogRule();

    expect(rule).toMatch(/max-height:\s*calc\(100dvh - 2rem\);/);
    expect(rule).toMatch(/overflow-y:\s*auto;/);
  });

  it("puts the role on the panel that holds the description and both actions", async () => {
    const description = "registers tool(s): " + Array.from({ length: 400 }, (_, i) => `tool_${i}`).join(", ");
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);

    await act(async () =>
      root?.render(
        <ConfirmDialog
          confirmLabel="Install"
          description={description}
          onCancel={() => {}}
          onConfirm={() => {}}
          open
          title="Install this plugin?"
        />,
      ),
    );

    const panels = document.querySelectorAll('[role="alertdialog"]');
    expect(panels).toHaveLength(1);
    const panel = panels[0];
    expect(panel.textContent).toContain("tool_399");
    expect(Array.from(panel.querySelectorAll("button"), (button) => button.textContent)).toEqual([
      "Cancel",
      "Install",
    ]);
  });
});
