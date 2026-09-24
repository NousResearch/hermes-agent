// @vitest-environment jsdom
import { expect, it } from "vitest";
import { exposePluginSDK } from "./registry";
import { getManagementProfile, setManagementProfile } from "@/lib/api";

it("exposes the actual host management-profile getter, never URL inference", () => {
  const profile = getManagementProfile();
  const sdk = window.__HERMES_PLUGIN_SDK__;
  const registry = window.__HERMES_PLUGINS__;
  const url = window.location.href;
  try {
    window.history.replaceState(null, "", "/config?profile=misleading");
    exposePluginSDK();
    const getter = window.__HERMES_PLUGIN_SDK__!.api.getManagementProfile;
    expect(getter).toBe(getManagementProfile);
    setManagementProfile("selected-work"); expect(getter()).toBe("selected-work");
    setManagementProfile("other"); expect(getter()).toBe("other");
    setManagementProfile(""); expect(getter()).toBe("");
  } finally {
    setManagementProfile(profile);
    window.__HERMES_PLUGIN_SDK__ = sdk; window.__HERMES_PLUGINS__ = registry;
    window.history.replaceState(null, "", url);
  }
});
