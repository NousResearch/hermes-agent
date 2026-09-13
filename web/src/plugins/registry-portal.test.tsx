// @vitest-environment jsdom
import { act, createElement, createContext, useContext } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { exposePluginSDK } from "./registry";

it("lets plugin overlays leave their container while retaining React context", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  exposePluginSDK();
  const sdk = window.__HERMES_PLUGIN_SDK__!;
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  const Context = createContext("outside");
  function Overlay() {
    return createElement("div", { id: "plugin-overlay" }, useContext(Context));
  }
  try {
    await act(async () => {
      root.render(createElement(Context.Provider, { value: "plugin context" },
        sdk.createPortal(createElement(Overlay), document.body)));
    });
    const overlay = document.getElementById("plugin-overlay")!;
    expect(overlay.parentElement).toBe(document.body);
    expect(container.contains(overlay)).toBe(false);
    expect(overlay.textContent).toBe("plugin context");
  } finally {
    await act(async () => root.unmount());
    container.remove();
    vi.unstubAllGlobals();
    delete window.__HERMES_PLUGIN_SDK__;
    delete window.__HERMES_PLUGINS__;
  }
  expect(document.getElementById("plugin-overlay")).toBeNull();
});
