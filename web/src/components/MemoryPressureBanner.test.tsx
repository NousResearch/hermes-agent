// @vitest-environment jsdom
// Tests for the NS-656 memory-pressure banner: trigger selection,
// severity precedence, dismissal semantics, and escalation re-opening.

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ReactNode } from "react";

import { I18nProvider } from "@/i18n";
import { MemoryPressureBanner } from "./MemoryPressureBanner";
import type {
  StatusResponse,
  MemoryPressureStatus,
  DiskPressureStatus,
} from "@/lib/api";

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

function statusWithDisk(
  disk: DiskPressureStatus | undefined,
  memory?: MemoryPressureStatus,
): StatusResponse {
  return { memory, disk } as StatusResponse;
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

  it("a gateway reboot (boot_id change) re-opens a dismissed live-pressure banner", async () => {
    // Edge case: dismiss critical, gateway restarts, next poll is STILL
    // critical with no observed "ok" in between. The new boot is a new
    // incident — without boot-scoped keys it would stay hidden (and mask
    // the OOM notice too, since critical takes precedence).
    await render(
      <MemoryPressureBanner
        status={statusWith({
          pressure: "critical",
          boot_id: "2026-08-13T01:00:00+00:00",
        })}
      />,
    );
    const dismiss = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    await act(async () => dismiss.click());
    expect(banner()).toBeNull();
    // Same boot, still critical: stays dismissed.
    await rerender(
      <MemoryPressureBanner
        status={statusWith({
          pressure: "critical",
          boot_id: "2026-08-13T01:00:00+00:00",
        })}
      />,
    );
    expect(banner()).toBeNull();
    // Rebooted, still critical: new incident, banner returns.
    await rerender(
      <MemoryPressureBanner
        status={statusWith({
          pressure: "critical",
          last_boot_suspected_oom: true,
          boot_id: "2026-08-13T02:00:00+00:00",
        })}
      />,
    );
    expect(banner()?.textContent).toContain("almost out of memory");
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

  it("renders nothing for healthy or unknown disk", async () => {
    await render(
      <MemoryPressureBanner
        status={statusWithDisk({ pressure: "ok", free_mb: 5000 })}
      />,
    );
    expect(banner()).toBeNull();
    await rerender(
      <MemoryPressureBanner status={statusWithDisk({ pressure: "unknown" })} />,
    );
    expect(banner()).toBeNull();
  });

  it("shows the disk-critical warning with free-space detail", async () => {
    await render(
      <MemoryPressureBanner
        status={statusWithDisk({ pressure: "critical", free_mb: 120 })}
      />,
    );
    expect(banner()?.textContent).toContain("disk is almost full");
    expect(banner()?.textContent).toContain("(120 MB free)");
  });

  it("shows the disk-elevated warning", async () => {
    await render(
      <MemoryPressureBanner
        status={statusWithDisk({ pressure: "elevated", free_mb: 900 })}
      />,
    );
    expect(banner()?.textContent).toContain("disk is filling up");
  });

  it("disk critical outranks memory critical", async () => {
    // Imminent data loss beats imminent restart.
    await render(
      <MemoryPressureBanner
        status={statusWithDisk(
          { pressure: "critical", free_mb: 100 },
          { pressure: "critical" },
        )}
      />,
    );
    expect(banner()?.textContent).toContain("disk is almost full");
  });

  it("memory OOM notice outranks disk elevated", async () => {
    await render(
      <MemoryPressureBanner
        status={statusWithDisk(
          { pressure: "elevated", free_mb: 900 },
          { pressure: "ok", last_boot_suspected_oom: true },
        )}
      />,
    );
    expect(banner()?.textContent).toContain("restarted unexpectedly");
  });

  it("dismissing a disk warning does not mask a later memory warning", async () => {
    await render(
      <MemoryPressureBanner
        status={statusWithDisk({ pressure: "elevated", free_mb: 900 })}
      />,
    );
    const dismiss = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    await act(async () => dismiss.click());
    expect(banner()).toBeNull();
    await rerender(
      <MemoryPressureBanner
        status={statusWithDisk(
          { pressure: "elevated", free_mb: 900 },
          { pressure: "elevated" },
        )}
      />,
    );
    expect(banner()?.textContent).toContain("running low on memory");
  });

  it("disk escalation to critical re-opens a dismissed disk banner", async () => {
    await render(
      <MemoryPressureBanner
        status={statusWithDisk({ pressure: "elevated", free_mb: 900 })}
      />,
    );
    const dismiss = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    await act(async () => dismiss.click());
    expect(banner()).toBeNull();
    await rerender(
      <MemoryPressureBanner
        status={statusWithDisk({ pressure: "critical", free_mb: 150 })}
      />,
    );
    expect(banner()?.textContent).toContain("disk is almost full");
  });

  it("disk recovery to ok resets disk dismissals for the next episode", async () => {
    await render(
      <MemoryPressureBanner
        status={statusWithDisk({ pressure: "critical", free_mb: 150 })}
      />,
    );
    const dismiss = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    await act(async () => dismiss.click());
    // User frees space (or grows the disk)...
    await rerender(
      <MemoryPressureBanner
        status={statusWithDisk({ pressure: "ok", free_mb: 8000 })}
      />,
    );
    expect(banner()).toBeNull();
    // ...then the disk fills again: new episode surfaces.
    await rerender(
      <MemoryPressureBanner
        status={statusWithDisk({ pressure: "critical", free_mb: 150 })}
      />,
    );
    expect(banner()?.textContent).toContain("disk is almost full");
  });

  it("disk recovery does NOT reset memory dismissals (and vice versa)", async () => {
    // Dismiss a memory warning while disk is also elevated.
    await render(
      <MemoryPressureBanner
        status={statusWithDisk(
          { pressure: "elevated", free_mb: 900 },
          { pressure: "critical" },
        )}
      />,
    );
    const dismiss = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    // Trigger shown is memory critical (outranks disk elevated) — dismiss it.
    await act(async () => dismiss.click());
    // Disk warning is next in line and has its own key, so it surfaces...
    expect(banner()?.textContent).toContain("disk is filling up");
    const dismissDisk = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    await act(async () => dismissDisk.click());
    expect(banner()).toBeNull();
    // Disk recovers; memory still critical — its dismissal must survive.
    await rerender(
      <MemoryPressureBanner
        status={statusWithDisk(
          { pressure: "ok", free_mb: 8000 },
          { pressure: "critical" },
        )}
      />,
    );
    expect(banner()).toBeNull();
  });

  it("a gateway reboot (boot_id change) re-opens a dismissed disk banner", async () => {
    await render(
      <MemoryPressureBanner
        status={statusWithDisk(
          { pressure: "critical", free_mb: 150 },
          { pressure: "ok", boot_id: "2026-08-13T01:00:00+00:00" },
        )}
      />,
    );
    const dismiss = container.querySelector(
      '[data-testid="memory-pressure-banner"] button',
    ) as HTMLButtonElement;
    await act(async () => dismiss.click());
    expect(banner()).toBeNull();
    // Restart with the disk still full: new boot, warning returns.
    await rerender(
      <MemoryPressureBanner
        status={statusWithDisk(
          { pressure: "critical", free_mb: 150 },
          { pressure: "ok", boot_id: "2026-08-13T02:00:00+00:00" },
        )}
      />,
    );
    expect(banner()?.textContent).toContain("disk is almost full");
  });
});
