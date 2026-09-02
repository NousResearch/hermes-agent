// @vitest-environment jsdom
// OAuthLoginModal expiry behaviour: when the sign-in window lapses locally,
// the modal must surface the backend poller's actionable message when it has
// one, keep polling when the backend still considers the session pending
// (clock skew), and fall back to guidance naming the common cause instead of
// the old bare "Session expired".

import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { I18nProvider } from "@/i18n";

const apiMocks = vi.hoisted(() => ({
  cancelOAuthSession: vi.fn(async () => ({ ok: true })),
  pollOAuthSession: vi.fn(),
  startOAuthLogin: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: {
    cancelOAuthSession: apiMocks.cancelOAuthSession,
    pollOAuthSession: apiMocks.pollOAuthSession,
    startOAuthLogin: apiMocks.startOAuthLogin,
  },
}));

vi.mock("@/lib/clipboard", () => ({
  copyTextToClipboard: vi.fn(async () => true),
}));

import { OAuthLoginModal } from "./OAuthLoginModal";

let container: HTMLDivElement;
let root: Root;

const provider = {
  cli_command: "hermes login nous",
  disconnectable: true,
  docs_url: "https://example.com/nous",
  flow: "device_code" as const,
  id: "nous",
  name: "Nous Portal",
  status: { logged_in: false },
};

function deviceStart(expiresIn: number) {
  return {
    expires_in: expiresIn,
    flow: "device_code",
    poll_interval: 5,
    session_id: "sess-1",
    user_code: "ABCD-EFGH",
    verification_url: "https://portal.example/device",
  };
}

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<I18nProvider>{ui}</I18nProvider>));
}

/** Fast-forward the countdown (1s ticks) until it lapses. */
async function exhaustCountdown(seconds: number) {
  for (let i = 0; i <= seconds; i++) {
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
  }
  await act(async () => {});
}

beforeEach(() => {
  (globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true;
  vi.useFakeTimers();
  apiMocks.startOAuthLogin.mockReset();
  apiMocks.pollOAuthSession.mockReset();
  apiMocks.cancelOAuthSession.mockReset().mockResolvedValue({ ok: true });
  // The steady state during the countdown: still pending.
  apiMocks.pollOAuthSession.mockResolvedValue({
    session_id: "sess-1",
    status: "pending",
  });
});

afterEach(() => {
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
  root?.unmount();
  container?.remove();
});

describe("OAuthLoginModal local expiry", () => {
  it("surfaces the backend error_message when the session lapsed", async () => {
    apiMocks.startOAuthLogin.mockResolvedValue(deviceStart(3));
    await render(
      <OAuthLoginModal
        provider={provider}
        onClose={() => {}}
        onSuccess={() => {}}
        onError={() => {}}
      />,
    );

    // The final poll flips to the backend's enriched message.
    apiMocks.pollOAuthSession.mockResolvedValue({
      session_id: "sess-1",
      status: "error",
      error_message:
        "Timed out waiting for device authorization. Portal sign-in is required before the device code can be approved.",
    });
    await exhaustCountdown(5);
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await act(async () => {});

    expect(container.textContent).toContain(
      "Portal sign-in is required",
    );
  });

  it("keeps polling when the backend still reports pending (clock skew)", async () => {
    apiMocks.startOAuthLogin.mockResolvedValue(deviceStart(3));
    await render(
      <OAuthLoginModal
        provider={provider}
        onClose={() => {}}
        onSuccess={() => {}}
        onError={() => {}}
      />,
    );

    await exhaustCountdown(5);
    await act(async () => {});

    // No error box: the modal is still in the polling phase.
    expect(container.textContent).toContain("ABCD-EFGH");
    expect(apiMocks.pollOAuthSession.mock.calls.length).toBeGreaterThan(0);
  });

  it("falls back to guidance naming the stalled-tab cause (pkce flow, no poll)", async () => {
    apiMocks.startOAuthLogin.mockResolvedValue({
      auth_url: "https://portal.example/auth",
      expires_in: 3,
      flow: "pkce",
      session_id: "sess-1",
    });
    await render(
      <OAuthLoginModal
        provider={provider}
        onClose={() => {}}
        onSuccess={() => {}}
        onError={() => {}}
      />,
    );

    await exhaustCountdown(6);
    await act(async () => {});

    expect(container.textContent).toContain("stalled in the opened tab");
    // PKCE has no poll endpoint call at expiry.
    expect(apiMocks.pollOAuthSession).not.toHaveBeenCalled();
  });

  it("a Retry after expiry starts fresh instead of insta-lapsing", async () => {
    apiMocks.startOAuthLogin.mockResolvedValueOnce(deviceStart(2));
    await render(
      <OAuthLoginModal
        provider={provider}
        onClose={() => {}}
        onSuccess={() => {}}
        onError={() => {}}
      />,
    );

    apiMocks.pollOAuthSession.mockResolvedValue({
      session_id: "sess-1",
      status: "error",
      error_message: "expired_token: code expired",
    });
    await exhaustCountdown(4);
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await act(async () => {});
    expect(container.textContent).toContain("expired_token");

    // Retry: new start response, fresh countdown.
    apiMocks.startOAuthLogin.mockResolvedValueOnce(deviceStart(60));
    apiMocks.pollOAuthSession.mockResolvedValue({
      session_id: "sess-1",
      status: "pending",
    });
    const retry = [...container.querySelectorAll("button")].find((b) =>
      b.textContent?.includes("Retry"),
    );
    expect(retry).toBeTruthy();
    await act(async () => retry!.click());
    await act(async () => {});

    // Well within the new window: still showing the code, no error.
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(container.textContent).toContain("ABCD-EFGH");
  });
});
