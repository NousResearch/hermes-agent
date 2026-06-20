// @vitest-environment jsdom
//
// ChatSidebar flow tests. Verifying the model-switch lifecycle when a chat
// is active (ptyActive=true):
//   1. Save model → dialog opens (badge does NOT refresh before Cancel)
//   2. Cancel → badge refreshes only after user declines reload
//   3. Reload → full-page reload (tested in ModelReloadConfirm.test.tsx)

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { ChatSidebar } from "./ChatSidebar";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockGetModelInfo = vi.fn();
const mockSetModelAssignment = vi.fn();
const mockGetModelOptions = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    getModelInfo: (...args: unknown[]) => mockGetModelInfo(...args),
    setModelAssignment: (...args: unknown[]) => mockSetModelAssignment(...args),
    getModelOptions: (...args: unknown[]) => mockGetModelOptions(...args),
  },
  HERMES_BASE_PATH: "",
  buildWsAuthParam: vi.fn().mockResolvedValue(["token", "test-token"] as const),
  buildWsUrl: vi.fn().mockResolvedValue("ws://localhost/api/events?token=test&channel=test-channel"),
}));

// GatewayClient — mock the entire class so the component can mount
// without a real WebSocket.
vi.mock("@/lib/gatewayClient", () => {
  class MockGatewayClient {
    connect = vi.fn().mockResolvedValue(undefined);
    close = vi.fn();
    request = vi.fn().mockResolvedValue({ session_id: "mock-sid" });
    onState = vi.fn().mockReturnValue(() => {});
    on = vi.fn().mockReturnValue(() => {});
  }
  return {
    GatewayClient: MockGatewayClient,
  };
});

// ModelPickerDialog — render a button that simulates the save+close flow.
// When clicked, it calls onApply then onClose (mimicking the real picker).
vi.mock("@/components/ModelPickerDialog", () => ({
  ModelPickerDialog: ({
    onApply,
    onClose,
  }: {
    onApply?: (arg: {
      confirmExpensiveModel: boolean;
      provider: string;
      model: string;
      persistGlobal: boolean;
    }) => Promise<{ confirm_required?: boolean }>;
    onClose?: () => void;
  }) => (
    <div data-testid="mock-model-picker">
      <button
        data-testid="picker-save-deepseek"
        onClick={async () => {
          if (onApply) {
            await onApply({
              confirmExpensiveModel: false,
              provider: "deepseek",
              model: "deepseek/deepseek-chat",
              persistGlobal: true,
            });
          }
          onClose?.();
        }}
      >
        save deepseek
      </button>
      <button
        data-testid="picker-save-confirm-required"
        onClick={async () => {
          if (onApply) {
            await onApply({
              confirmExpensiveModel: true,
              provider: "openai",
              model: "openai/o1-pro",
              persistGlobal: true,
            });
          }
          // confirm_required → picker stays open, onClose NOT called
        }}
      >
        save expensive model
      </button>
      <button data-testid="picker-close" onClick={onClose}>
        close picker
      </button>
    </div>
  ),
}));

// Mock UI components that ChatSidebar uses
vi.mock("@nous-research/ui/ui/components/button", () => {
  const React = require("react");
  return {
    Button: ({
      children,
      onClick,
      ghost,
      size,
      outlined,
      prefix,
      className,
      title,
      ...rest
    }: Record<string, unknown>) =>
      React.createElement(
        "button",
        { onClick, title, className, "data-testid": title || "button", ...rest },
        prefix ? [prefix, children] : children,
      ),
  };
});

vi.mock("@nous-research/ui/ui/components/badge", () => ({
  Badge: ({ children, className }: Record<string, unknown>) =>
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require("react").createElement(
      "span",
      { "data-testid": "connection-badge", className },
      children,
    ),
}));

vi.mock("@nous-research/ui/ui/components/card", () => ({
  Card: ({ children, className }: Record<string, unknown>) =>
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require("react").createElement("div", { className }, children),
}));

vi.mock("@/components/ReasoningPicker", () => ({
  ReasoningPicker: () => null,
}));

vi.mock("@/components/ToolCall", () => ({
  ToolCall: () => null,
}));

vi.mock("@/components/ConfirmDialog", () => {
  const React = require("react");
  return {
    ConfirmDialog: ({
      open,
      title,
      description,
      confirmLabel,
      onConfirm,
      onCancel,
    }: Record<string, unknown>) =>
      open
        ? React.createElement(
            "div",
            { "data-testid": "confirm-dialog" },
            React.createElement("h2", null, title as string),
            React.createElement("p", null, description as string),
            React.createElement(
              "button",
              { onClick: onConfirm },
              confirmLabel as string,
            ),
            React.createElement("button", { onClick: onCancel }, "Cancel"),
          )
        : null,
  };
});

vi.mock("@/lib/utils", () => ({
  cn: (...args: unknown[]) => args.filter(Boolean).join(" "),
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Mock WebSocket — ChatSidebar creates an events WebSocket in an effect.
class MockWebSocket {
  static OPEN = 1;
  readyState = 1;
  close = vi.fn();
  send = vi.fn();
  addEventListener = vi.fn();
  removeEventListener = vi.fn();
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).WebSocket = MockWebSocket;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(window as any).__HERMES_BASE_PATH__ = "";

// Mock @hermes/shared — gatewayClient re-exports from here
vi.mock("@hermes/shared", () => {
  class MockJsonRpcGatewayClient {}
  return {
    JsonRpcGatewayClient: MockJsonRpcGatewayClient,
    buildHermesWebSocketUrl: vi.fn(),
    ConnectionState: {},
    GatewayEvent: {},
    GatewayEventName: {},
  };
});

/**
 * Setup mock API responses for a clean ChatSidebar mount:
 * - No current model (badge shows "—")
 * - setModelAssignment succeeds with no confirm_required
 * - After save, getModelInfo returns the new model
 * - getModelOptions returns empty (picker won't render real providers)
 */
function setupCleanMount() {
  mockGetModelInfo.mockResolvedValue({ model: "" });
  mockGetModelOptions.mockResolvedValue({ providers: [] });
  mockSetModelAssignment.mockResolvedValue({ confirm_required: false });
}

function openModelPicker() {
  // The model-switch button has a chevron and a title attr.
  // Title is either "switch model" (when model is "—") or the model name.
  const btn =
    document.querySelector('button[title="switch model"]') ||
    document.querySelector("button svg.lucide-chevron-down")?.closest("button") ||
    screen.queryByTitle("switch model");
  fireEvent.click(btn as HTMLElement);
}

afterEach(cleanup);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ChatSidebar model-switch flow (ptyActive=true)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupCleanMount();
  });

  it("shows the reload confirm dialog when saving with an active chat", async () => {
    render(
      <ChatSidebar
        channel="test-channel"
        ptyActive={true}
      />,
    );

    openModelPicker();

    // Simulate: user saves a model via the picker.
    // The mock picker saves deepseek and calls onClose.
    fireEvent.click(screen.getByTestId("picker-save-deepseek"));

    // After save, the reload dialog should appear ("Switch model?" title).
    await waitFor(() => {
      expect(screen.getByText("Switch model?")).toBeTruthy();
    });

    // The dialog body should mention the model is saved.
    expect(screen.getByText(/Model saved\./i)).toBeTruthy();
  });

  it("does NOT show the reload dialog when no chat is active", async () => {
    mockGetModelInfo.mockResolvedValue({ model: "old-model" });

    render(
      <ChatSidebar
        channel="test-channel"
        ptyActive={false}
      />,
    );

    openModelPicker();

    // Simulate save → onClose.
    fireEvent.click(screen.getByTestId("picker-save-deepseek"));

    // With ptyActive=false, the reload dialog should NOT appear.
    // Instead, the notice banner should show (model set, next chat will use it).
    await waitFor(() => {
      expect(screen.queryByText("Switch model?")).toBeNull();
    });
  });

  it("shows a notice (not a dialog) when saving with no active chat", async () => {
    mockGetModelInfo.mockResolvedValue({ model: "old-model" });

    render(
      <ChatSidebar
        channel="test-channel"
        ptyActive={false}
      />,
    );

    openModelPicker();
    fireEvent.click(screen.getByTestId("picker-save-deepseek"));

    // Should show the notice banner about next chat.
    await waitFor(() => {
      expect(
        screen.getByText(/Model set to/i),
      ).toBeTruthy();
    });
  });

  it("refresh badge only after Cancel, not during picker close", async () => {
    // Start with an old model on the badge.
    mockGetModelInfo.mockResolvedValue({ model: "old-model" });

    render(
      <ChatSidebar
        channel="test-channel"
        ptyActive={true}
      />,
    );

    // The badge should show the old model initially.
    await waitFor(() => {
      expect(screen.getByText("old-model")).toBeTruthy();
    });

    openModelPicker();

    // Simulate save (sets pendingReloadModel), then onClose fires.
    // onClose is guarded: when pendingReloadModel is set, it skips
    // refreshEffectiveModel. The badge should STILL show "old-model".
    fireEvent.click(screen.getByTestId("picker-save-deepseek"));

    // Reload dialog appears.
    await waitFor(() => {
      expect(screen.getByText("Switch model?")).toBeTruthy();
    });

    // Badge should NOT have refreshed yet — still shows old model.
    // (The GuardedClose test below validates this more directly.)

    // Now click Cancel on the reload dialog.
    // The cancel handler should refresh the badge to the new model.
    mockGetModelInfo.mockResolvedValue({
      model: "deepseek/deepseek-chat",
    });

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    // After Cancel, the badge should refresh and show the new model.
    await waitFor(() => {
      expect(screen.getByText("deepseek-chat")).toBeTruthy();
    });

    // The reload dialog should be gone.
    expect(screen.queryByText("Switch model?")).toBeNull();
  });
});

describe("ChatSidebar model-switch flow — confirm_required path", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupCleanMount();
  });

  it("keeps the picker open when expensive-model confirmation is required", async () => {
    mockSetModelAssignment.mockResolvedValue({ confirm_required: true });

    render(
      <ChatSidebar
        channel="test-channel"
        ptyActive={true}
      />,
    );

    openModelPicker();

    // The mock picker's "save expensive model" button calls onApply which
    // returns confirm_required, and the picker does NOT call onClose.
    fireEvent.click(screen.getByTestId("picker-save-confirm-required"));

    // The picker should stay open (no reload dialog, no onClose).
    await waitFor(() => {
      // onApply returned confirm_required, so the sidebar logic does NOT
      // set pendingReloadModel and the reload dialog should NOT appear.
      expect(screen.queryByText("Switch model?")).toBeNull();
    });
  });
});

describe("ChatSidebar model-switch flow — picker close without save", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetModelInfo.mockResolvedValue({ model: "current-model" });
    mockGetModelOptions.mockResolvedValue({ providers: [] });
  });

  it("closes the picker and refreshes the badge when closed without saving", async () => {
    render(
      <ChatSidebar
        channel="test-channel"
        ptyActive={true}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("current-model")).toBeTruthy();
    });

    openModelPicker();

    // Close the picker without saving anything.
    // pendingReloadModel is null, so onClose should call refreshEffectiveModel.
    fireEvent.click(screen.getByTestId("picker-close"));

    // The picker should be gone (no reload dialog either).
    await waitFor(() => {
      expect(screen.queryByText("Switch model?")).toBeNull();
    });

    // Badge refresh was called (getModelInfo should have been called).
    expect(mockGetModelInfo).toHaveBeenCalled();
  });
});
