import type { ActionNeeded, DashboardSnapshot } from "@hermes/dashboard-kit";
import { buildKnownDashboardSnapshots } from "./dashboard-signals";

export interface DailyBriefItem {
  id: string;
  title: string;
  summary: string;
  owner: string;
  severity: "low" | "normal" | "high" | "critical";
}

export interface CentralCommandData {
  actions: ActionNeeded[];
  brief: DailyBriefItem[];
  snapshots: DashboardSnapshot[];
  systemRead: string;
}

export function buildCentralCommandData(): CentralCommandData {
  const snapshots = buildKnownDashboardSnapshots();
  const actions = snapshots.flatMap((snapshot) => snapshot.actions ?? []);
  return {
    actions,
    brief: [
      {
        id: "command-dashboard-foundation",
        title: "Dashboard foundation is ready for controlled migrations.",
        summary: "V8-V11 added migration tracking, signal contracts, quality scoring, and agent governance.",
        owner: "Hermes",
        severity: "normal",
      },
      {
        id: "command-cost-capacity-gap",
        title: "Cost and capacity signals are still incomplete.",
        summary: "Projects expose health-like status before they expose normalized cost, capacity, and queue telemetry.",
        owner: "Operations",
        severity: "high",
      },
      {
        id: "command-static-adapter-risk",
        title: "Static adapters remain the main dashboard quality risk.",
        summary: "Khashi VC and Media Engine should stay stable until package-native migrations are scheduled.",
        owner: "Dashboard owners",
        severity: "normal",
      },
    ],
    snapshots,
    systemRead: "Hermes can now see dashboard readiness, missing signals, and migration priorities from one command layer.",
  };
}
