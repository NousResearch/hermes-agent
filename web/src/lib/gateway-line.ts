import type { StatusResponse } from "@/lib/api";
import type { useI18n } from "@/i18n";

export function gatewayLine(
  status: StatusResponse,
  t: ReturnType<typeof useI18n>["t"],
): { label: string; tone: string } {
  const g = t.app.gatewayStrip;
  const byState: Record<string, { label: string; tone: string }> = {
    running: { label: g.running, tone: "text-success" },
    starting: { label: g.starting, tone: "text-warning" },
    startup_failed: { label: g.failed, tone: "text-destructive" },
    stopped: { label: g.stopped, tone: "text-muted-foreground" },
  };
  if (status.gateway_state && byState[status.gateway_state]) {
    return byState[status.gateway_state];
  }
  return status.gateway_running
    ? { label: g.running, tone: "text-success" }
    : { label: g.off, tone: "text-muted-foreground" };
}
