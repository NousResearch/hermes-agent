import { createContext } from "react";
import type { ActionStatusResponse } from "@/lib/api";

export const SystemActionsContext = createContext<SystemActionsState | null>(
  null,
);

export type SystemAction = "restart" | "update";

export interface SystemActionsState {
  actionStatus: ActionStatusResponse | null;
  activeAction: SystemAction | null;
  dismissLog: () => void;
  isBusy: boolean;
  isRunning: boolean;
  pendingAction: SystemAction | null;
  runAction: (action: SystemAction) => Promise<void>;
}
