import { useContext } from "react";
import {
  SystemActionsContext,
  type SystemActionsState,
} from "./system-actions-context";

export function useSystemActions(): SystemActionsState {
  const ctx = useContext(SystemActionsContext);
  if (!ctx) {
    throw new Error(
      "useSystemActions must be used within a SystemActionsProvider",
    );
  }
  return ctx;
}
