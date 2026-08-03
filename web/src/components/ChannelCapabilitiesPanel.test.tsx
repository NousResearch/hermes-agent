// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ChannelCapabilitiesPanel } from "./ChannelCapabilitiesPanel";

const getChannelCapabilities = vi.fn();
const updateChannelCapabilities = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    getChannelCapabilities: (...args: unknown[]) => getChannelCapabilities(...args),
    updateChannelCapabilities: (...args: unknown[]) => updateChannelCapabilities(...args),
  },
}));

vi.mock("@/i18n", () => ({
  useI18n: () => ({
    t: {
      common: { loading: "Loading" },
      skills: {},
    },
  }),
}));

interface FrameProps {
  children?: ReactNode;
}

interface MockButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  outlined?: boolean;
  size?: string;
}

interface MockSwitchProps {
  "aria-label"?: string;
  checked: boolean;
  disabled?: boolean;
  onCheckedChange: (checked: boolean) => void;
}

vi.mock("@nous-research/ui/ui/components/card", () => ({
  Card: ({ children }: FrameProps) => <div>{children}</div>,
  CardContent: ({ children }: FrameProps) => <div>{children}</div>,
  CardHeader: ({ children }: FrameProps) => <div>{children}</div>,
  CardTitle: ({ children }: FrameProps) => <h2>{children}</h2>,
}));

vi.mock("@nous-research/ui/ui/components/button", () => ({
  Button: ({ outlined, size, ...props }: MockButtonProps) => {
    void outlined;
    void size;
    return <button {...props} />;
  },
}));

vi.mock("@nous-research/ui/ui/components/badge", () => ({
  Badge: ({ children }: FrameProps) => <span>{children}</span>,
}));

vi.mock("@nous-research/ui/ui/components/switch", () => ({
  Switch: ({ checked, disabled, onCheckedChange, ...props }: MockSwitchProps) => (
    <button
      {...props}
      aria-checked={checked}
      disabled={disabled}
      role="switch"
      onClick={() => onCheckedChange(!checked)}
    />
  ),
}));

vi.mock("lucide-react", () => ({
  AlertTriangle: () => null,
  Network: () => null,
  ShieldCheck: () => null,
}));

function channel(platform: string, label: string) {
  return {
    effective_toolsets: ["web"],
    explicit: true,
    implicit_toolsets: [],
    label,
    mcp: {
      available: [],
      effective: [],
      mode: "all" as const,
      selected: [],
    },
    platform,
    plugins_locked: false,
    toolsets: [
      {
        description: "Search the web",
        enabled: true,
        label: "Web",
        name: "web",
        tools: ["web_search"],
      },
    ],
  };
}

beforeEach(() => {
  getChannelCapabilities.mockImplementation((profile?: string) =>
    Promise.resolve(
      profile === "review"
        ? [channel("telegram", "Review Telegram")]
        : [channel("email", "Primary Email")],
    ),
  );
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("ChannelCapabilitiesPanel profile-scoped saves", () => {
  it("does not let an older profile save lock or update the active profile", async () => {
    let finishPrimary: (() => void) | undefined;
    let finishReview: (() => void) | undefined;
    updateChannelCapabilities.mockImplementation((platform: string) =>
      new Promise<{ channel: ReturnType<typeof channel> }>((resolve) => {
        const result = {
          channel:
            platform === "email"
              ? channel("email", "Primary Email")
              : channel("telegram", "Review Telegram"),
        };
        if (platform === "email") finishPrimary = () => resolve(result);
        else finishReview = () => resolve(result);
      }),
    );
    const onError = vi.fn();
    const onSaved = vi.fn();

    const view = render(
      <ChannelCapabilitiesPanel
        onError={onError}
        onSaved={onSaved}
        profile={undefined}
        query=""
      />,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Save abilities" }));
    await waitFor(() => expect(updateChannelCapabilities).toHaveBeenCalledTimes(1));

    view.rerender(
      <ChannelCapabilitiesPanel
        onError={onError}
        onSaved={onSaved}
        profile="review"
        query=""
      />,
    );
    await screen.findByRole("button", { name: "Review Telegram" });

    const reviewSave = screen.getByRole("button", { name: "Save abilities" });
    expect(reviewSave).toHaveProperty("disabled", false);
    fireEvent.click(reviewSave);
    await waitFor(() => expect(updateChannelCapabilities).toHaveBeenCalledTimes(2));

    finishPrimary?.();
    await waitFor(() => expect(reviewSave).toHaveProperty("disabled", true));
    expect(screen.getByRole("button", { name: "Review Telegram" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Primary Email" })).toBeNull();
    expect(onSaved).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();

    finishReview?.();
    await waitFor(() => expect(reviewSave).toHaveProperty("disabled", false));
    expect(onSaved).toHaveBeenCalledTimes(1);
  });
});
