// @vitest-environment jsdom
import { act, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { api, setManagementProfile } from "../lib/api";
import { I18nProvider } from "./context";
import { useI18n } from "./useI18n";

let root: Root;
let container: HTMLDivElement;
let current: ReturnType<typeof useI18n>;
function Probe() {
  const value = useI18n();
  useEffect(() => {
    current = value;
  }, [value]);
  return <span>{value.locale}</span>;
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  localStorage.clear();
  setManagementProfile("");
  container = document.createElement("div");
  root = createRoot(container);
  vi.spyOn(api, "getConfigRevision").mockResolvedValue({
    path: "/profile/config.yaml",
    mtime_ns: 1,
    size: 1,
  });
});
afterEach(async () => {
  await act(async () => root.unmount());
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  setManagementProfile("");
});
it("serializes profile-scoped saves and preserves the last displayed locale on failure", async () => {
  vi.spyOn(api, "getConfig").mockResolvedValue({ display: { language: "en" } });
  const first = deferred<{ ok: boolean }>();
  const save = vi
    .spyOn(api, "saveConfig")
    .mockReturnValueOnce(first.promise)
    .mockRejectedValueOnce(new Error("offline"));
  await act(async () =>
    root.render(
      <I18nProvider>
        <Probe />
      </I18nProvider>,
    ),
  );
  let firstSave!: Promise<void>;
  let secondSave!: Promise<void>;
  await act(async () => {
    firstSave = current.setLocale("zh");
  });
  expect(container.textContent).toBe("en");
  setManagementProfile("second");
  await act(async () => {
    secondSave = current.setLocale("de");
    void secondSave.catch(() => {});
  });
  expect(save).toHaveBeenCalledTimes(1);
  await act(async () => {
    first.resolve({ ok: true });
    await firstSave;
    await expect(secondSave).rejects.toThrow("offline");
  });
  expect(save.mock.calls).toEqual([
    [{ display: { language: "zh" } }, ""],
    [{ display: { language: "de" } }, "second"],
  ]);
  expect(container.textContent).toBe("en");
  expect(document.documentElement.lang).toBe("en");
});
it("retries a revision whose read overlapped a language save instead of acknowledging unapplied data", async () => {
  const stale = deferred<Record<string, unknown>>();
  const read = vi
    .spyOn(api, "getConfig")
    .mockReturnValueOnce(stale.promise)
    .mockResolvedValue({ display: { language: "fr" } });
  vi.spyOn(api, "saveConfig").mockResolvedValue({ ok: true });
  await act(async () =>
    root.render(
      <I18nProvider>
        <Probe />
      </I18nProvider>,
    ),
  );
  await act(async () => {
    await current.setLocale("zh");
  });
  expect(container.textContent).toBe("zh");
  await act(async () => {
    stale.resolve({ display: { language: "en" } });
  });
  expect(container.textContent).toBe("zh");
  await act(async () => {
    window.dispatchEvent(new Event("focus"));
  });
  expect(read).toHaveBeenCalledTimes(2);
  expect(container.textContent).toBe("fr");
  expect(localStorage.getItem("hermes-locale")).toBe("fr");
});
