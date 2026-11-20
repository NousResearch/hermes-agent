import { useEffect, useMemo, useRef } from "react";
import { useLocation, useNavigate } from "react-router";

import {
  createLearnHandoffController,
  hasLearnSeedState,
  type LearnHandoffController,
  type LearnSocketLike,
  readLearnSeedFromState,
  stripLearnSeedFromState,
} from "@/lib/learn-handoff";

export interface LearnHandoffBridge {
  readonly onSocketOpen: (socket: LearnSocketLike) => void;
  readonly onSocketClose: () => void;
  readonly reset: () => void;
}

/**
 * Consume one-shot `/learn` handoff payloads from React Router navigation state
 * (not URL query parameters) and deliver them once the chat PTY composer is ready.
 */
export function useLearnHandoff(isActive: boolean): LearnHandoffBridge {
  const location = useLocation();
  const navigate = useNavigate();
  const controllerRef = useRef<LearnHandoffController | null>(null);

  const getController = (): LearnHandoffController => {
    controllerRef.current ??= createLearnHandoffController();
    return controllerRef.current;
  };

  useEffect(() => {
    if (!isActive || !hasLearnSeedState(location.state)) return;
    const seed = readLearnSeedFromState(location.state);
    navigate(`${location.pathname}${location.search}`, {
      replace: true,
      state: stripLearnSeedFromState(location.state),
    });
    getController().enqueue(seed);
  }, [isActive, location.pathname, location.search, location.state, navigate]);

  useEffect(() => () => controllerRef.current?.reset(), []);

  return useMemo<LearnHandoffBridge>(
    () => ({
      onSocketOpen: (socket) => getController().onSocketOpen(socket),
      onSocketClose: () => controllerRef.current?.onSocketClose(),
      reset: () => controllerRef.current?.reset(),
    }),
    [],
  );
}
