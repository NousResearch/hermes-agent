// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { useLayoutEffect } from "react";
import type { ComponentType } from "react";
import { PluginPage } from "./PluginPage";
import { exposePluginSDK } from "./registry";

(
  globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

const Demo: ComponentType = () => <div>plugin-loaded</div>;

type Register = (name: string, component: ComponentType) => void;

function registerPlugin(name: string, component: ComponentType) {
  exposePluginSDK();
  (
    window as unknown as { __HERMES_PLUGINS__: { register: Register } }
  ).__HERMES_PLUGINS__.register(name, component);
}

function GapInjector({ name }: { name: string }) {
  useLayoutEffect(() => {
    registerPlugin(name, Demo);
  }, [name]);
  return null;
}

describe("PluginPage", () => {
  afterEach(() => {
    cleanup();
  });

  it("surfaces a plugin registered in the render->subscribe gap", () => {
    render(
      <>
        <GapInjector name="regtest-gap" />
        <PluginPage name="regtest-gap" />
      </>,
    );

    expect(screen.getByText("plugin-loaded")).toBeTruthy();
    expect(screen.queryByText("Loading...")).toBeNull();
  });
});
