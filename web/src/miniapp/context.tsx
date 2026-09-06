import { createContext, useContext } from "react";

export type MiniAppTab = "status" | "skills" | "cron" | "sessions" | "users";

export interface ConfirmSpec {
  title: string;
  body: string;
  label: string;
  destructive: boolean;
  run: () => void | Promise<void>;
}

export interface MiniAppContextValue {
  tier: "admin" | "paired" | null;
  isAdmin: boolean;
  tab: MiniAppTab;
  goTab: (tab: MiniAppTab) => void;
  showToast: (message: string) => void;
  askConfirm: (spec: ConfirmSpec) => void;
  refreshStatus: () => void;
  /** Shared with StatusScreen so the Users tab's restart-needed banner can
   * trigger the SAME confirm sheet + action cross-tab (jump to Status,
   * then open the sheet) without StatusScreen needing to be mounted. */
  askRestartGateway: () => void;
  askUpdateHermes: () => void;
  /** Real backend truth from the last /api/status fetch — NOT restart-in-flight. */
  gwConnected: boolean;
  /** Client-only, true only during the optimistic post-restart-tap window.
   * Kept separate from gwConnected: a gateway that is simply stopped (never
   * running, no restart requested) must read "Stopped", not "Restarting…". */
  gwRestarting: boolean;
}

export const MiniAppContext = createContext<MiniAppContextValue | null>(null);

export function useMiniApp(): MiniAppContextValue {
  const ctx = useContext(MiniAppContext);
  if (!ctx) throw new Error("useMiniApp() called outside <MiniApp>");
  return ctx;
}
