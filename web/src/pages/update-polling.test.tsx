// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import SystemPage from "./SystemPage";
import { SystemActionsProvider } from "@/contexts/SystemActions";
import { useSystemActions } from "@/contexts/useSystemActions";

const mocks = vi.hoisted(() => ({
  status: vi.fn(), update: vi.fn(), toast: vi.fn(),
}));
vi.mock("@/lib/api", () => ({ api: {
  getActionStatus: mocks.status, updateHermes: mocks.update,
  checkHermesUpdate: vi.fn(async () => ({update_available: true, can_apply: true})),
  ...Object.fromEntries(["getStatus", "getSystemStats", "getMemory", "getCredentialPool",
    "getCheckpoints", "getHooks", "getCurator", "getPortal", "getGatewayMigratePlan"].map(
    name => [name, vi.fn(async () => { throw new Error("not needed"); })])),
}}));
vi.mock("@/i18n", () => ({useI18n: () => ({t: {status: {
  actionFinished: "Action finished", actionFailed: "Action failed",
}}})}));
vi.mock("react-router", () => ({Link: ({children}: {children: ReactNode}) => <span>{children}</span>}));
vi.mock("@nous-research/ui/ui/components/toast", () => ({
  Toast: ({toast}: {toast: {message: string} | null}) => <span>{toast?.message}</span>,
}));
vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({toast: null, showToast: mocks.toast}),
}));
vi.mock("@nous-research/ui/ui/components/confirm-dialog", () => ({
  ConfirmDialog: ({open, onConfirm, confirmLabel}: {open: boolean; onConfirm: () => void; confirmLabel: string}) =>
    open ? <button onClick={onConfirm}>{confirmLabel}</button> : null,
}));
vi.mock("@/components/HermesConsoleModal", () => ({HermesConsoleModal: () => null}));
vi.mock("@/components/DeleteConfirmDialog", () => ({DeleteConfirmDialog: () => null}));

function Consumer() {
  const {runAction, isBusy} = useSystemActions();
  return <button onClick={() => void runAction("update")}>{isBusy ? "busy" : "start"}</button>;
}
let root: Root;
let container: HTMLDivElement;
const id = "e".repeat(32);
const pending = {name: "hermes-update", action_id: id, state: "pending", running: false,
  exit_code: null, pid: null, lines: []};
async function click(label: string) {
  const button = Array.from(container.querySelectorAll("button")).find(b => b.textContent?.trim() === label);
  expect(button, label).toBeDefined();
  await act(async () => button!.click());
}
async function advance(ms: number) { await act(async () => vi.advanceTimersByTimeAsync(ms)); }
beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.useFakeTimers(); vi.clearAllMocks();
  mocks.update.mockResolvedValue({ok: true, name: "hermes-update", action_id: id});
  mocks.status.mockResolvedValue(pending);
  container = document.createElement("div"); document.body.append(container); root = createRoot(container);
});
afterEach(async () => {
  try {
    await act(async () => root.unmount());
  } finally {
    container.remove();
    vi.restoreAllMocks();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  }
});

it.each(["provider", "page"])("%s releases its polling timer on unmount", async consumer => {
  const setTimer = vi.spyOn(globalThis, "setTimeout");
  const clearTimer = vi.spyOn(globalThis, "clearTimeout");
  const unrelatedTick = vi.fn();
  const unrelatedTimer = setInterval(unrelatedTick, 100);
  try {
    await act(async () => root.render(consumer === "provider"
      ? <SystemActionsProvider><Consumer /></SystemActionsProvider> : <SystemPage />));
    await click(consumer === "provider" ? "start" : "Update now");
    if (consumer === "page") await click("Update now");
    expect(mocks.status).toHaveBeenCalledTimes(1);
    await advance(1600);
    expect(mocks.status).toHaveBeenCalledTimes(2);

    // Observe the timer that actually reschedules polling, not GSAP/jsdom's
    // independent animation interval (which can outlive either component).
    const pollDelay = consumer === "provider" ? 1500 : 1200;
    const pollingTimers = setTimer.mock.calls.flatMap(([, delay], index) =>
      delay === pollDelay ? [setTimer.mock.results[index].value] : []);
    expect(pollingTimers).toHaveLength(2);
    await act(async () => root.unmount());
    expect(clearTimer).toHaveBeenCalledWith(pollingTimers[1]);

    const ticksBeforeUnmount = unrelatedTick.mock.calls.length;
    await advance(5000);
    expect(mocks.status).toHaveBeenCalledTimes(2);
    expect(unrelatedTick.mock.calls.length).toBeGreaterThan(ticksBeforeUnmount);
  } finally {
    clearInterval(unrelatedTimer);
  }
});

it.each(["provider", "page"])("%s bounds abandoned unknown identity without accepting identity-free success", async consumer => {
  mocks.status.mockResolvedValue({...pending, state: "finished", action_id: undefined, exit_code: 0});
  await act(async () => root.render(consumer === "provider"
    ? <SystemActionsProvider><Consumer /></SystemActionsProvider> : <SystemPage />));
  await click(consumer === "provider" ? "start" : "Update now");
  if (consumer === "page") await click("Update now");
  expect(container.textContent).not.toContain("Action finished");
  expect(container.textContent).not.toContain("done");
  mocks.status.mockResolvedValue({...pending, state: "abandoned", action_id: undefined});
  await advance(1600);
  expect(container.textContent).toContain("Update outcome unknown");
  await advance(1600);
  expect(mocks.status).toHaveBeenCalledTimes(2);
});

it.each(["provider", "page"])("%s ignores stale terminal identities and retries unavailable status", async consumer => {
  mocks.status.mockResolvedValue({...pending, action_id: "a".repeat(32), state: "finished", exit_code: 0});
  await act(async () => root.render(consumer === "provider"
    ? <SystemActionsProvider><Consumer /></SystemActionsProvider> : <SystemPage />));
  await click(consumer === "provider" ? "start" : "Update now");
  if (consumer === "page") await click("Update now");
  expect(container.textContent).not.toContain("Action finished");
  expect(container.textContent).not.toContain("done");
  mocks.status.mockRejectedValueOnce(new Error("dashboard restart gap"))
    .mockResolvedValueOnce({...pending, state: "unknown", action_id: undefined})
    .mockResolvedValue({...pending, state: "finished", exit_code: 0});
  await advance(1600);
  await advance(1600);
  await advance(1600);
  expect(mocks.status).toHaveBeenCalledTimes(4);
  expect(container.textContent).toContain(consumer === "provider" ? "Action finished" : "done");
});

it.each(["provider", "page"])("%s stops abandoned polling without reporting update failure", async consumer => {
  mocks.status.mockResolvedValue({...pending, state: "abandoned"});
  await act(async () => root.render(consumer === "provider"
    ? <SystemActionsProvider><Consumer /></SystemActionsProvider> : <SystemPage />));
  await click(consumer === "provider" ? "start" : "Update now");
  if (consumer === "page") await click("Update now");
  expect(container.textContent).toContain("Update outcome unknown");
  expect(container.textContent).not.toContain("Action failed");
  expect(container.textContent).not.toContain("exit null");
  await advance(1600);
  expect(mocks.status).toHaveBeenCalledTimes(1);
});

it.each(["provider", "page"])("%s keeps polling pending after registry loss until current terminal", async consumer => {
  await act(async () => root.render(consumer === "provider"
    ? <SystemActionsProvider><Consumer /></SystemActionsProvider> : <SystemPage />));
  await click(consumer === "provider" ? "start" : "Update now");
  if (consumer === "page") await click("Update now");
  expect(mocks.status).toHaveBeenCalledTimes(1);
  expect(container.textContent).not.toContain("Action failed");
  expect(container.textContent).not.toContain("exit null");
  mocks.status.mockResolvedValue({...pending, state: "finished", exit_code: 0});
  await advance(1600);
  expect(mocks.status).toHaveBeenCalledTimes(2);
  expect(container.textContent).toContain(consumer === "provider" ? "Action finished" : "done");
  await advance(1600);
  expect(mocks.status).toHaveBeenCalledTimes(2);
});
