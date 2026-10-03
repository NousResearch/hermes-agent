// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ getManagementProfile: vi.fn(() => "alpha"), setDelegationRouting: vi.fn(), getMoaModels: vi.fn(), pickerProps: null as null | { onApply: (value: { provider: string; model: string }) => Promise<unknown> } }));
vi.mock("@/lib/api", () => ({ api: mocks, getManagementProfile: mocks.getManagementProfile }));
vi.mock("@/components/ReasoningEffortSelect", () => ({ ReasoningEffortSelect: () => null }));
vi.mock("@/components/ModelReloadConfirm", () => ({ ModelReloadConfirm: () => null }));
vi.mock("@/components/ModelPickerDialog", () => ({ ModelPickerDialog: (props: typeof mocks.pickerProps) => { mocks.pickerProps = props; return <button onClick={() => void props?.onApply({ provider: "new-provider", model: "new/model" })}>Choose model</button>; } }));
vi.mock("@/components/ConfirmDialog", () => ({ ConfirmDialog: ({ open, title, onConfirm, onCancel }: { open: boolean; title: string; onConfirm: () => void; onCancel: () => void }) => open ? <div role="dialog" aria-label={title}><button onClick={onCancel}>Cancel</button><button onClick={onConfirm}>Confirm</button></div> : null }));

import { ModelSettingsPanel } from "./ModelsPage";

let container: HTMLDivElement;
let root: Root;
async function render(ui: ReactNode) { container = document.createElement("div"); document.body.append(container); root = createRoot(container); await act(async () => root.render(ui)); }
async function rerender(ui: ReactNode) { await act(async () => root.render(ui)); }
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>((res) => { resolve = res; }); return { promise, resolve }; }
const aux = { main: { provider: "main-provider", model: "main/model" }, delegation: { provider: "custom", model: "custom/model" }, tasks: [] } as never;
const saved = vi.fn();
const props = () => ({ aux, refreshKey: 0, onSaved: saved });
const buttonWithText = (text: string) => [...container.querySelectorAll<HTMLButtonElement>("button")].find((button) => button.textContent?.includes(text));
const chooseDelegation = async () => act(async () => { buttonWithText("Change model")!.click(); });
const clickDialog = async (text: string) => act(async () => { [...document.body.querySelectorAll<HTMLButtonElement>("button")].find((button) => button.textContent === text)!.click(); });

beforeEach(() => { (globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true; mocks.getManagementProfile.mockReturnValue("alpha"); mocks.setDelegationRouting.mockReset().mockResolvedValue({ ok: true }); mocks.getMoaModels.mockReset().mockResolvedValue({ reference_models: [], aggregator: { provider: "", model: "" }, presets: {} }); mocks.pickerProps = null; saved.mockReset(); });
afterEach(async () => { await act(async () => root?.unmount()); container?.remove(); document.body.innerHTML = ""; });

describe("ModelSettingsPanel routing confirmation", () => {
  it("prompts for provider change and confirms with clear enabled and reset disabled", async () => {
    mocks.setDelegationRouting.mockResolvedValueOnce({ routing_confirmation_required: true }); await render(<ModelSettingsPanel {...props()} />); await chooseDelegation(); await act(async () => { await mocks.pickerProps!.onApply({ provider: "new-provider", model: "new/model" }); });
    expect(document.body.querySelector('[aria-label="Clear custom delegation routing?"]')).not.toBeNull(); expect(mocks.setDelegationRouting).toHaveBeenCalledTimes(1); await clickDialog("Confirm");
    expect(mocks.setDelegationRouting).toHaveBeenLastCalledWith({ provider: "new-provider", model: "new/model", reset_routing: false, confirm_clear_routing: true, profile: "alpha" }); expect(saved).toHaveBeenCalledTimes(1);
  });
  it("cancels provider change without a second write", async () => {
    mocks.setDelegationRouting.mockResolvedValueOnce({ routing_confirmation_required: true }); await render(<ModelSettingsPanel {...props()} />); await chooseDelegation(); await act(async () => { await mocks.pickerProps!.onApply({ provider: "new-provider", model: "new/model" }); }); await clickDialog("Cancel"); expect(mocks.setDelegationRouting).toHaveBeenCalledTimes(1); expect(saved).not.toHaveBeenCalled();
  });
  it("confirms reset with reset enabled and cancel does not write", async () => {
    await render(<ModelSettingsPanel {...props()} />); await act(async () => buttonWithText("Reset to parent routing")!.click()); await clickDialog("Confirm"); expect(mocks.setDelegationRouting).toHaveBeenCalledWith({ provider: "", model: "", reset_routing: true, confirm_clear_routing: true, profile: "alpha" }); expect(saved).toHaveBeenCalledTimes(1);
    mocks.setDelegationRouting.mockClear(); saved.mockClear(); await act(async () => buttonWithText("Reset to parent routing")!.click()); await clickDialog("Cancel"); expect(mocks.setDelegationRouting).not.toHaveBeenCalled(); expect(saved).not.toHaveBeenCalled();
  });
  it("invalidates a pending confirmation on profile switch", async () => {
    mocks.setDelegationRouting.mockResolvedValueOnce({ routing_confirmation_required: true }); await render(<ModelSettingsPanel {...props()} />); await chooseDelegation(); await act(async () => { await mocks.pickerProps!.onApply({ provider: "new-provider", model: "new/model" }); }); mocks.getManagementProfile.mockReturnValue("beta"); await rerender(<ModelSettingsPanel {...props()} />); expect(document.body.querySelector('[role="dialog"]')).toBeNull(); expect(mocks.setDelegationRouting).toHaveBeenCalledTimes(1);
  });
  it("ignores stale routing request completion after profile switch", async () => {
    const request = deferred<{ routing_confirmation_required: boolean }>(); mocks.setDelegationRouting.mockReturnValueOnce(request.promise); await render(<ModelSettingsPanel {...props()} />); await chooseDelegation(); let applying!: Promise<unknown>; await act(async () => { applying = mocks.pickerProps!.onApply({ provider: "new-provider", model: "new/model" }); }); mocks.getManagementProfile.mockReturnValue("beta"); await rerender(<ModelSettingsPanel {...props()} />); await act(async () => { request.resolve({ routing_confirmation_required: true }); await applying; }); expect(document.body.querySelector('[role="dialog"]')).toBeNull(); expect(saved).not.toHaveBeenCalled(); expect(mocks.setDelegationRouting).toHaveBeenCalledTimes(1);
  });
  it("renders rejection as visible error", async () => {
    mocks.setDelegationRouting.mockResolvedValueOnce({ routing_confirmation_required: true }); await render(<ModelSettingsPanel {...props()} />); await chooseDelegation(); await act(async () => { await mocks.pickerProps!.onApply({ provider: "new-provider", model: "new/model" }); }); mocks.setDelegationRouting.mockRejectedValueOnce(new Error("routing failed")); await clickDialog("Confirm"); expect(container.querySelector('[role="alert"]')?.textContent).toContain("routing failed");
  });
});
