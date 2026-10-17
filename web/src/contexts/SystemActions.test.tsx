// @vitest-environment jsdom
// The update toast names what a committed update still owes for every committed outcome: a
// partial run (exit 1 after record_user_action) shows the producer's own instruction, and only
// this action's receipt may name debt (the status route attaches the latest receipt otherwise).

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

const apiMocks = vi.hoisted(() => ({
  getActionStatus: vi.fn(),
  getStatus: vi.fn(),
  restartGateway: vi.fn(),
  updateHermes: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ api: apiMocks }));

import { I18nProvider } from "@/i18n";
import { SystemActionsProvider } from "./SystemActions";
import { useSystemActions } from "./useSystemActions";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

const INSTRUCTION =
  "Your local changes are parked in stash@{0}; run `git stash pop` in ~/.hermes/hermes-agent.";

let container: HTMLDivElement;
let root: Root;

function Probe() {
  const { runAction } = useSystemActions();
  return <button onClick={() => void runAction("update")} />;
}

async function runUpdate(receiptActionId: string): Promise<string> {
  apiMocks.updateHermes.mockResolvedValue({
    action_id: "c".repeat(32),
    name: "hermes-update",
    ok: true,
    pid: 1,
  });
  apiMocks.getActionStatus.mockResolvedValue({
    exit_code: 1,
    lines: [],
    name: "hermes-update",
    pid: null,
    running: false,
    receipt: {
      action_id: receiptActionId,
      outcome: "partial",
      followups: [],
      user_action: { step: "local_changes", reason: INSTRUCTION },
    },
  });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () =>
    root.render(
      <I18nProvider>
        <SystemActionsProvider>
          <Probe />
        </SystemActionsProvider>
      </I18nProvider>,
    ),
  );
  await act(async () => container.querySelector("button")!.click());
  for (let i = 0; i < 20 && !document.body.textContent?.includes("exit 1"); i++) {
    await act(async () => new Promise((resolve) => setTimeout(resolve, 10)));
  }
  return document.body.textContent ?? "";
}

beforeEach(() => {
  for (const fn of Object.values(apiMocks)) fn.mockReset();
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  document.body.innerHTML = "";
});

describe("SystemActionsProvider update toast", () => {
  it("names a partial update's owed user action verbatim", async () => {
    const text = await runUpdate("c".repeat(32));
    expect(text).toContain("exit 1");
    expect(text).toContain(`local_changes: ${INSTRUCTION}`);
    // A rerun does not restore a parked stash.
    expect(text).not.toContain("re-run `hermes update`");
  });

  it("never names another run's debt", async () => {
    const text = await runUpdate("d".repeat(32));
    expect(text).toContain("exit 1");
    expect(text).not.toContain("local_changes");
  });
});
