import { afterEach, describe, expect, it, vi } from "vitest";

import {
  restartGatewayFromChannelsPage,
  restartGatewayAfterTelegramOnboarding,
  restartGatewayFromSystemPage,
  restartGatewayFromWebhooksPage,
  updateHermesFromSystemPage,
} from "./service-mutation-callers";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("direct service mutation callers", () => {
  it.each([
    [
      "System page restart",
      restartGatewayFromSystemPage,
      "/api/gateway/restart",
      "RESTART",
    ],
    [
      "System page update",
      updateHermesFromSystemPage,
      "/api/hermes/update",
      "UPDATE",
    ],
    [
      "Webhooks page restart",
      restartGatewayFromWebhooksPage,
      "/api/gateway/restart",
      "RESTART",
    ],
    [
      "Channels page restart",
      restartGatewayFromChannelsPage,
      "/api/gateway/restart",
      "RESTART",
    ],
    [
      "Telegram onboarding fallback restart",
      restartGatewayAfterTelegramOnboarding,
      "/api/gateway/restart",
      "RESTART",
    ],
  ] as const)(
    "%s sends the required typed mutation contract",
    async (_name, caller, path, confirmation) => {
      vi.stubGlobal("window", {});
      vi.spyOn(crypto, "randomUUID").mockReturnValue(
        "00000000-0000-4000-8000-000000000001",
      );
      const fetchMock = vi.fn<typeof fetch>(
        async () =>
          new Response(JSON.stringify({ name: "action", ok: true, pid: 1 }), {
            headers: { "Content-Type": "application/json" },
            status: 200,
          }),
      );
      vi.stubGlobal("fetch", fetchMock);

      await caller();

      expect(fetchMock).toHaveBeenCalledWith(
        path,
        expect.objectContaining({
          body: JSON.stringify({
            confirmation,
            idempotency_key: "00000000-0000-4000-8000-000000000001",
          }),
          method: "POST",
        }),
      );
    },
  );
});
