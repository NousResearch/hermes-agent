// @vitest-environment jsdom
// Tests for the NS-656 memory-pressure banner: trigger selection,
// severity precedence, dismissal semantics, and escalation re-opening.

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ReactNode } from "react";

import { I18nProvider } from "@/i18n";
import { MemoryPressureBanner } from "./MemoryPressureBanner";
import type { StatusResponse, MemoryPressureStatus } from "@/lib/api";

let container: HTMLDivElement;
let root: Root;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<I18nProvider>{ui}</I18nProvider>));
}

async function rerender(ui: ReactNode) {
  await act(async () => root.render(<I18nProvider>{ui}</I18nProvider>));
}

beforeEach(() => {
  sessionStorage.clear();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

function statusWith(memory: MemoryPressureStatus | undefined): StatusResponse {
  return { memory } as StatusResponse;
}

function banner(): HTMLElement | null {
  return container.querySelector('[data-testid="memory-pressure-banner"]');
}

describe("MemoryPressureBanner", () => {
  it("renders nothing for null status or healthy memory", async () => {
    await render(<MemoryPressureBanner status={null} />);
    expect(banner()).toBeNull();
    await rerender(
      <MemoryPressureBanner status={statusWith({ pressure: "ok" })} />,
    );
    expect(banner()).toBeNull();
    // Older gateways: no memory block at all.
    await rerender(<MemoryPressureBanner status={statusWith(undefined)} />);
    expect(banner()).toBeNull();
  });

  it("renders nothing for unknown pressure (absence of evidence)", async () => {
    await render(
      <MemoryPressureBanner status={statusWith({ pressure: "unknown" })} />,
    );
    expect(banner()).toBeNull();
  });

  it("shows the elevated warning", async () => {
    await render(
      <MemoryPressureBanner status={statusWith({ pressure: "elevated" })} />,
    );
    expect(banner()?.textContent).toContain("running low on memory");
  });

  it("shows the OOM-restart notice even when current pressure is ok", async () => {
    await render(
      <MemoryPressureBanner
        status={statusWith({ pressure: "ok", last_boot_suspected_oom: true })}
      />,
    );
    expect(banner()?.textContent).toContain(
      "restarted unexpectedly, most likely because it ran out of memory",
    );
  });

  it("critical pressure outranks the OOM-restart notice", async () => {
    await render(
      <MemoryPressureBanner
        status={statusWith({
          pressure: "critical",
          last_boot_suspected_oom: true,
        })}
      />,
    );
    expect(banner()?.textContent).toContain("almost out of memory");
  });

  it("dismissal hides the banner and persists across re-renders", async () => {
    await render(
      <MemoryPressureBanner status={statusWith({ pressure: "elevated" })} />,
    );
    const dismiss = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    await act(async () => dismiss.click());
    expect(banner()).toBeNull();
    await rerender(
      <MemoryPressureBanner status={statusWith({ pressure: "elevated" })} />,
    );
    expect(banner()).toBeNull();
  });

  it("escalation to critical re-opens a dismissed banner", async () => {
    await render(
      <MemoryPressureBanner status={statusWith({ pressure: "elevated" })} />,
    );
    const dismiss = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    await act(async () => dismiss.click());
    expect(banner()).toBeNull();
    await rerender(
      <MemoryPressureBanner status={statusWith({ pressure: "critical" })} />,
    );
    expect(banner()?.textContent).toContain("almost out of memory");
  });

  it("a NEW OOM restart (different boot_id) re-opens a dismissed OOM notice", async () => {
    await render(
      <MemoryPressureBanner
        status={statusWith({
          pressure: "ok",
          last_boot_suspected_oom: true,
          boot_id: "2026-08-13T01:00:00+00:00",
        })}
      />,
    );
    const dismiss = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    await act(async () => dismiss.click());
    expect(banner()).toBeNull();
    // Same boot: stays dismissed.
    await rerender(
      <MemoryPressureBanner
        status={statusWith({
          pressure: "ok",
          last_boot_suspected_oom: true,
          boot_id: "2026-08-13T01:00:00+00:00",
        })}
      />,
    );
    expect(banner()).toBeNull();
    // The gateway OOM-loops and boots again: new incident, banner returns.
    await rerender(
      <MemoryPressureBanner
        status={statusWith({
          pressure: "ok",
          last_boot_suspected_oom: true,
          boot_id: "2026-08-13T02:00:00+00:00",
        })}
      />,
    );
    expect(banner()?.textContent).toContain("restarted unexpectedly");
  });

  it("recovery to ok resets a dismissed live-pressure banner for the next episode", async () => {
    await render(
      <MemoryPressureBanner status={statusWith({ pressure: "critical" })} />,
    );
    const dismiss = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    await act(async () => dismiss.click());
    expect(banner()).toBeNull();
    // Confirmed recovery clears live dismissals...
    await rerender(
      <MemoryPressureBanner status={statusWith({ pressure: "ok" })} />,
    );
    expect(banner()).toBeNull();
    // ...so the NEXT critical episode surfaces again.
    await rerender(
      <MemoryPressureBanner status={statusWith({ pressure: "critical" })} />,
    );
    expect(banner()?.textContent).toContain("almost out of memory");
  });

  it("unknown pressure (stale heartbeat) does NOT reset live dismissals", async () => {
    await render(
      <MemoryPressureBanner status={statusWith({ pressure: "elevated" })} />,
    );
    const dismiss = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    await act(async () => dismiss.click());
    await rerender(
      <MemoryPressureBanner status={statusWith({ pressure: "unknown" })} />,
    );
    expect(banner()).toBeNull();
    await rerender(
      <MemoryPressureBanner status={statusWith({ pressure: "elevated" })} />,
    );
    expect(banner()).toBeNull();
  });
});
