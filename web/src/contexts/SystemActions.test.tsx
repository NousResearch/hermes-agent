// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const api = vi.hoisted(() => ({
  getActionStatus: vi.fn(),
  restartGateway: vi.fn(),
  updateHermes: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ api }));

import { SystemActionsProvider } from "./SystemActions";
import { useSystemActions } from "./useSystemActions";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function Probe() {
  const actions = useSystemActions();
  return (
    <button onClick={() => void actions.runAction("restart")}>
      restart
    </button>
  );
}

describe("SystemActionsProvider mutation contract", () => {
  it("routes the confirmed action through the guarded transport", async () => {
    api.restartGateway.mockResolvedValue({ name: "gateway-restart", ok: true });
    api.getActionStatus.mockResolvedValue({
      exit_code: 0,
      lines: [],
      name: "gateway-restart",
      running: false,
    });

    render(
      <SystemActionsProvider>
        <Probe />
      </SystemActionsProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "restart" }));

    await waitFor(() =>
      expect(api.restartGateway).toHaveBeenCalledWith(),
    );
  });
});
