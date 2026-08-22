// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { DashboardEmbedRequest } from "@/lib/dashboard-embed";
import { useProfileScope } from "@/contexts/useProfileScope";

const chatMock = vi.hoisted(() => ({ props: null as Record<string, unknown> | null }));

vi.mock("@/pages/ChatPage", () => {
  function MockChatPage(props: Record<string, unknown>) {
    chatMock.props = props;
    const { profile } = useProfileScope();
    return <div data-testid="chat-page" data-profile={profile} data-props={JSON.stringify(props)} />;
  }
  return { default: MockChatPage };
});

import { EmbeddedChatHost } from "./EmbeddedChatHost";

const request: DashboardEmbedRequest = {
  authBridge: false,
  embedId: "console-wolf",
  parentOrigin: "https://console.runi.services",
  profile: "wolf",
};

const mounted: Array<{ root: ReturnType<typeof createRoot>; node: HTMLDivElement }> = [];

async function render(requestValue: DashboardEmbedRequest) {
  const node = document.createElement("div");
  document.body.append(node);
  const root = createRoot(node);
  mounted.push({ root, node });
  await act(async () => root.render(<EmbeddedChatHost request={requestValue} />));
  return node;
}

afterEach(async () => {
  for (const { root, node } of mounted.splice(0)) {
    await act(async () => root.unmount());
    node.remove();
  }
  vi.restoreAllMocks();
});

describe("EmbeddedChatHost", () => {
  it("renders the stripped terminal and announces readiness", async () => {
    const postMessage = vi.fn();
    Object.defineProperty(window, "parent", {
      configurable: true,
      value: { postMessage },
    });

    const node = await render(request);
    expect(node.querySelector('[data-testid="chat-page"]')?.getAttribute("data-props"))
      .toContain('"embedded":true');
    expect(node.querySelector('[data-testid="chat-page"]')?.getAttribute("data-profile"))
      .toBe("wolf");
    await act(async () => {
      (chatMock.props?.onPtyStateChange as ((state: string) => void))("open");
    });
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ event: "ready", embedId: "console-wolf" }),
      "https://console.runi.services",
    );

    await act(async () => {
      (chatMock.props?.onPtyStateChange as ((state: string) => void))("reconnecting");
    });
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ event: "reconnecting", embedId: "console-wolf" }),
      "https://console.runi.services",
    );
  });

  it("uses the OAuth bridge only to notify the opener and close", async () => {
    const postMessage = vi.fn();
    const close = vi.spyOn(window, "close").mockImplementation(() => {});
    Object.defineProperty(window, "opener", {
      configurable: true,
      value: { postMessage },
    });

    const node = await render({ ...request, authBridge: true });
    expect(node.querySelector('[data-testid="chat-page"]')).toBeNull();
    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ event: "authenticated" }),
      "https://console.runi.services",
    );
    expect(close).toHaveBeenCalled();
  });
});
