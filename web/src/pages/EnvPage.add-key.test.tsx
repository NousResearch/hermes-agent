// @vitest-environment jsdom
// "Add a custom key" with a name the catalog already knows opened an editor the user could not
// see: a messaging-platform credential (channel_managed) is listed in no section of this page, a
// provider key's row sits in its collapsed provider group, and an unset tool or gateway key is
// under "Show more" once its section has a key set. The name field cleared and nothing appeared,
// so a token typed here never reached .env.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { EnvVarInfo } from "@/lib/api";
import EnvPage from "./EnvPage";

const getEnvVars = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api", () => ({ api: { getEnvVars } }));
vi.mock("@/contexts/usePageHeader", () => ({ usePageHeader: () => ({ setAfterTitle: () => undefined }) }));
vi.mock("@/components/OAuthProvidersCard", () => ({ OAuthProvidersCard: () => null }));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));

let root: Root;
let container: HTMLDivElement;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function row(overrides: Partial<EnvVarInfo>): EnvVarInfo {
  return {
    is_set: false, redacted_value: null, description: "", url: null, category: "custom",
    is_password: true, tools: [], advanced: false, ...overrides,
  };
}

const VARS: Record<string, EnvVarInfo> = {
  TELEGRAM_BOT_TOKEN: row({ category: "messaging", channel_managed: true, description: "Telegram bot token" }),
  OPENAI_API_KEY: row({ category: "provider", description: "OpenAI API key" }),
  MY_TOOL_TOKEN: row({ is_set: true, redacted_value: "abcd...wxyz", custom: true }),
};

function customCard(): HTMLElement {
  const card = container.querySelector<HTMLElement>("#section-custom");
  if (!card) throw new Error("Missing the Custom Keys card");
  return card;
}

function nameInput(): HTMLInputElement {
  const input = customCard().querySelector<HTMLInputElement>('input[aria-label="Variable name"]');
  if (!input) throw new Error("Missing the custom key name field");
  return input;
}

function addButton(): HTMLButtonElement {
  const button = Array.from(customCard().querySelectorAll("button")).find((b) => b.textContent === "Add");
  if (!button) throw new Error("Missing the Add button");
  return button;
}

function valueEditors(): HTMLInputElement[] {
  return Array.from(customCard().querySelectorAll<HTMLInputElement>('input[placeholder="Enter value..."]'));
}

async function typeName(value: string) {
  const input = nameInput();
  const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  await act(async () => {
    setValue?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

async function add() {
  await act(async () => {
    addButton().click();
    nameInput().dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  });
}

beforeEach(async () => {
  getEnvVars.mockResolvedValue(structuredClone(VARS));
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root.render(<MemoryRouter><EnvPage /></MemoryRouter>);
  });
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
});

it("points a messaging-platform credential to the Channels page instead of swallowing it", async () => {
  await typeName("telegram_bot_token");

  expect(customCard().textContent).toContain("TELEGRAM_BOT_TOKEN");
  expect(customCard().textContent).toContain("Channels page");
  expect(customCard().querySelector('a[href="/channels"]')).not.toBeNull();
  expect(addButton().disabled).toBe(true);

  await add();

  expect(nameInput().value).toBe("telegram_bot_token");
  expect(valueEditors()).toHaveLength(0);
});

it("opens a value field next to the form for a known key whose own row is folded away", async () => {
  await typeName("OPENAI_API_KEY");
  await add();

  expect(nameInput().value).toBe("");
  const labels = Array.from(customCard().querySelectorAll("label"), (label) => label.textContent);
  expect(labels).toContain("OPENAI_API_KEY");
  expect(valueEditors()).toHaveLength(1);
  expect(customCard().textContent).toContain("OpenAI API key");
});

it("still opens an unset row for a name the catalog does not know", async () => {
  await typeName("MY_NEW_SERVICE_KEY");
  await add();

  const labels = Array.from(customCard().querySelectorAll("label"), (label) => label.textContent);
  expect(labels).toContain("MY_NEW_SERVICE_KEY");
  expect(valueEditors()).toHaveLength(1);
});
