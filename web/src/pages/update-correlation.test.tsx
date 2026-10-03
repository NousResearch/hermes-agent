// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import SystemPage from "./SystemPage";
import { SystemActionsProvider } from "@/contexts/SystemActions";
import { useSystemActions } from "@/contexts/useSystemActions";

vi.mock("@/i18n", () => ({useI18n: () => ({t: {status: {
  actionFinished: "Action finished", actionFailed: "Action failed",
}}})}));
vi.mock("react-router", () => ({Link: ({children}: {children: ReactNode}) => <span>{children}</span>}));
vi.mock("@nous-research/ui/ui/components/toast", () => ({
  Toast: ({toast}: {toast: {message: string} | null}) => <span>{toast?.message}</span>,
}));
const showToast = vi.hoisted(() => vi.fn());
vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({toast: null, showToast}),
}));
vi.mock("@nous-research/ui/ui/components/confirm-dialog", () => ({
  ConfirmDialog: ({open, onConfirm, confirmLabel}: {open: boolean; onConfirm: () => void; confirmLabel: string}) =>
    open ? <button onClick={onConfirm}>{confirmLabel}</button> : null,
}));
vi.mock("@/components/HermesConsoleModal", () => ({HermesConsoleModal: () => null}));
vi.mock("@/components/DeleteConfirmDialog", () => ({DeleteConfirmDialog: () => null}));

const expectedId = "a".repeat(32);
const newerId = "b".repeat(32);
const newer = {name: "hermes-update", action_id: newerId, state: "finished", running: false,
  exit_code: 7, pid: 12345, lines: ["newer attempt failed"]};
const recovered = {...newer, action_id: expectedId, state: "finished", exit_code: 0,
  pid: null, lines: [], receipt: {update_id: expectedId, outcome: "success"}};
let exactStatus: unknown;
let newerStatus: unknown;
let root: Root;
let container: HTMLDivElement;
const fetchMock = vi.fn();
function Consumer() {
  const {runAction, isBusy} = useSystemActions();
  return <button onClick={() => void runAction("update")}>{isBusy ? "busy" : "start"}</button>;
}
function statusUrls(): URL[] {
  return fetchMock.mock.calls.map(([input]) => new URL(String(input), "http://localhost"))
    .filter(url => url.pathname === "/api/actions/hermes-update/status");
}
async function click(label: string) {
  const button = Array.from(container.querySelectorAll("button")).find(b => b.textContent?.trim() === label);
  expect(button, label).toBeDefined();
  await act(async () => button!.click());
}
async function start(consumer: string) {
  await act(async () => root.render(consumer === "provider"
    ? <SystemActionsProvider><Consumer /></SystemActionsProvider> : <SystemPage />));
  await click(consumer === "provider" ? "start" : "Update now");
  if (consumer === "page") await click("Update now");
}
async function advance(ms: number) { await act(async () => vi.advanceTimersByTimeAsync(ms)); }
beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.useFakeTimers(); vi.clearAllMocks();
  exactStatus = recovered;
  newerStatus = newer;
  fetchMock.mockImplementation(async (input: string, init?: RequestInit) => {
    const url = new URL(String(input), "http://localhost");
    let body: unknown;
    if (url.pathname === "/api/hermes/update" && init?.method === "POST") {
      body = {ok: true, name: "hermes-update", action_id: expectedId};
    } else if (url.pathname === "/api/hermes/update/check") {
      body = {update_available: true, can_apply: true};
    } else if (url.pathname === "/api/actions/hermes-update/status") {
      body = url.searchParams.get("action_id") === expectedId ? exactStatus : newerStatus;
    } else {
      body = {providers: [], builtin_files: {}, sessions: [], hooks: [],
        profiles: [], blockers: [], already_multiplexed: true};
    }
    return new Response(JSON.stringify(body), {status: 200, headers: {"Content-Type": "application/json"}});
  });
  vi.stubGlobal("fetch", fetchMock);
  container = document.createElement("div"); document.body.append(container); root = createRoot(container);
});
afterEach(async () => {
  try { await act(async () => root.unmount()); }
  finally { container.remove(); vi.restoreAllMocks(); vi.useRealTimers(); vi.unstubAllGlobals(); }
});

it.each(["provider", "page"])("%s requests the admitted attempt through the real API client and recovers archived success", async consumer => {
  await start(consumer);
  expect(statusUrls()).toHaveLength(1);
  expect(statusUrls()[0].searchParams.get("action_id")).toBe(expectedId);
  expect(container.textContent).toContain(consumer === "provider" ? "Action finished" : "done");
  expect(container.textContent).not.toContain("Action failed");
  await advance(5000);
  expect(statusUrls()).toHaveLength(1);
});

it.each([
  ["provider", "finished"], ["provider", "abandoned"],
  ["page", "finished"], ["page", "abandoned"],
])("%s stops safely when A has no terminal archive and B is %s", async (consumer, newerState) => {
  newerStatus = {...newer, state: newerState, exit_code: newerState === "finished" ? 7 : null};
  exactStatus = {...newer, action_id: expectedId, state: "superseded", exit_code: null, pid: null, lines: []};
  await start(consumer);
  expect(statusUrls()[0].searchParams.get("action_id")).toBe(expectedId);
  expect(container.textContent).toContain("Update outcome unknown");
  expect(container.textContent).not.toContain("Action finished");
  expect(container.textContent).not.toContain("Action failed");
  expect(container.textContent).not.toContain("exit 7");
  expect(container.textContent).not.toContain("newer attempt failed");
  if (consumer === "provider") expect(container.querySelector("button")?.textContent).toBe("start");
  await advance(5000);
  expect(statusUrls()).toHaveLength(1);
});
