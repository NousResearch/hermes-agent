// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getWebhooks: vi.fn(),
  createWebhook: vi.fn(),
}));
vi.mock("@/lib/api", () => ({ api: apiMocks }));

let root: Root | undefined;
let container: HTMLDivElement | undefined;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function clickButton(label: string) {
  const button = Array.from(document.querySelectorAll("button")).find(
    (element) => element.textContent?.trim() === label,
  );
  if (!button) throw new Error(`Button not found: ${label}`);
  button.click();
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  apiMocks.getWebhooks.mockReset();
  apiMocks.createWebhook.mockReset();
});

it("keeps a create warning in the one-time secret dialog until it closes", async () => {
  const warning = "Delivery target unavailable; subscription was saved.";
  apiMocks.getWebhooks.mockResolvedValue({ enabled: true, subscriptions: [] });
  apiMocks.createWebhook.mockResolvedValue({
    url: "https://example.test/hooks/demo",
    secret: "one-time-secret",
    warning,
  });

  const [{ default: WebhooksPage }, { I18nProvider }, { PageHeaderProvider }] =
    await Promise.all([
      import("./WebhooksPage"),
      import("@/i18n"),
      import("@/contexts/PageHeaderProvider"),
    ]);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root!.render(
      <I18nProvider>
        <MemoryRouter>
          <PageHeaderProvider pluginTabs={[]}>
            <WebhooksPage />
          </PageHeaderProvider>
        </MemoryRouter>
      </I18nProvider>,
    );
  });

  await act(async () => clickButton("New subscription"));
  const name = document.querySelector<HTMLInputElement>("#webhook-name");
  if (!name) throw new Error("Name input not rendered");
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(name, "demo");
    name.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await act(async () => clickButton("Create"));

  const dialog = document.querySelector('[role="dialog"]');
  expect(apiMocks.createWebhook).toHaveBeenCalledOnce();
  expect(dialog?.textContent).toContain("one-time-secret");
  expect(dialog?.querySelector('[role="alert"]')?.textContent).toBe(warning);
  expect(document.querySelector('[role="alert"]')).toBe(dialog?.querySelector('[role="alert"]'));

  await act(async () => clickButton("Done"));
  expect(document.querySelector('[role="dialog"]')).toBeNull();
  await act(async () => clickButton("New subscription"));
  expect(document.querySelector('[role="alert"]')).toBeNull();
});
