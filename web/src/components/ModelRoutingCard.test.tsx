// @vitest-environment jsdom
//
// Regression coverage for the dead 'Add route' button on the fallback-chain
// editor (partner feedback ⑤). The editor rows used to be re-derived from the
// config value on every render, and the serializer drops all-blank rows — so
// the blank row 'Add route' appended was immediately parsed away and the button
// looked like it did nothing. These tests drive the real component and assert a
// row appears and can be filled.

import { act, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", () => ({
  api: { getModelOptions: vi.fn() },
}));

import { I18nProvider } from "@/i18n";
import { ROUTING_KEYS } from "@/lib/model-routing";
import { setNestedValue } from "@/lib/nested";
import { ModelRoutingCard } from "./ModelRoutingCard";

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

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

function providerInputs(): HTMLInputElement[] {
  return [
    ...container.querySelectorAll<HTMLInputElement>('input[aria-label^="Provider "]'),
  ];
}

function addRouteButton(): HTMLButtonElement | null {
  return (
    [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (b) => b.textContent?.trim() === "Add route",
    ) ?? null
  );
}

async function click(el: Element | null) {
  if (!el) throw new Error("element not rendered");
  await act(async () => {
    el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  });
}

async function typeInto(input: HTMLInputElement, value: string) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(
      input,
      value,
    );
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

describe("ModelRoutingCard — Add route", () => {
  it("appends an editable row to the subagent fallback chain", async () => {
    const onChange = vi.fn();
    await render(
      <ModelRoutingCard
        config={{}}
        schema={{ [ROUTING_KEYS.subagentFallback]: {} }}
        onChange={onChange}
      />,
    );

    expect(providerInputs()).toHaveLength(0);

    await click(addRouteButton());
    expect(providerInputs()).toHaveLength(1);

    await click(addRouteButton());
    expect(providerInputs()).toHaveLength(2);

    // The appended row is a real, fillable input — typing reaches onChange.
    // onChange only ever sees the serialized value (blank rows dropped).
    await typeInto(providerInputs()[0], "openrouter");
    expect(onChange).toHaveBeenLastCalledWith(ROUTING_KEYS.subagentFallback, [
      { provider: "openrouter", model: "" },
    ]);
  });

  it("appends an editable row to the main-agent fallback chain", async () => {
    const onChange = vi.fn();
    await render(
      <ModelRoutingCard
        config={{}}
        schema={{ fallback_providers: {} }}
        onChange={onChange}
      />,
    );

    expect(providerInputs()).toHaveLength(0);
    await click(addRouteButton());
    expect(providerInputs()).toHaveLength(1);
    expect(onChange).toHaveBeenLastCalledWith(ROUTING_KEYS.mainFallback, []);
  });

  it("re-derives the editor when config changes externally", async () => {
    const onChange = vi.fn();
    const schema = { [ROUTING_KEYS.subagentFallback]: {} };
    await render(
      <ModelRoutingCard config={{}} schema={schema} onChange={onChange} />,
    );
    await click(addRouteButton());
    expect(providerInputs()).toHaveLength(1);

    // Simulate a form reset/import: the host replaces the config value.
    await act(async () => {
      root.render(
        <I18nProvider>
          <ModelRoutingCard
            config={{
              delegation: { fallback_providers: [{ provider: "anthropic", model: "c/d" }] },
            }}
            schema={schema}
            onChange={onChange}
          />
        </I18nProvider>,
      );
    });
    expect(providerInputs()).toHaveLength(1);
    expect(providerInputs()[0].value).toBe("anthropic");
  });

  it("keeps the added row when the host applies onChange to its state", async () => {
    // Mirrors ConfigPage/ModelsPage: onChange writes into page state, which is
    // fed straight back as the `config` prop. The old derive-from-config editor
    // lost the blank row on that round-trip.
    function Host({ schema }: { schema: Record<string, unknown> }) {
      const [config, setConfig] = useState<Record<string, unknown>>({});
      return (
        <ModelRoutingCard
          config={config}
          schema={schema}
          onChange={(key, value) =>
            setConfig((prev) => setNestedValue(prev, key, value))
          }
        />
      );
    }

    await render(<Host schema={{ [ROUTING_KEYS.subagentFallback]: {} }} />);
    expect(providerInputs()).toHaveLength(0);
    await click(addRouteButton());
    expect(providerInputs()).toHaveLength(1);
    await click(addRouteButton());
    expect(providerInputs()).toHaveLength(2);
  });
});
