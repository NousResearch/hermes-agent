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
  readonly onSocketClose: (socket: LearnSocketLike) => void;
  readonly onTransportTeardown: () => void;
  readonly onUserInput: () => void;
  readonly reset: () => void;
}

/**
 * Consume one-shot `/learn` handoff payloads from React Router navigation state
 * (not URL query parameters) and deliver them once a fresh chat PTY composer is ready.
 */
export function useLearnHandoff(isActive: boolean, scopeKey: string): LearnHandoffBridge {
  const location = useLocation();
  const navigate = useNavigate();
  const controllerRef = useRef<LearnHandoffController | null>(null);
  const prevScopeKeyRef = useRef(scopeKey);

  const getController = (): LearnHandoffController => {
    controllerRef.current ??= createLearnHandoffController();
    return controllerRef.current;
  };

  // Clear any unsent handoff when switching profile, channel, or resume target.
  // Ordered before location.state consumption so a simultaneous scope switch +
  // fresh navigation queues the new handoff cleanly.
  useEffect(() => {
    if (prevScopeKeyRef.current !== scopeKey) {
      prevScopeKeyRef.current = scopeKey;
      controllerRef.current?.reset();
    }
  }, [scopeKey]);

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
      onSocketClose: (socket) => controllerRef.current?.onSocketClose(socket),
      onTransportTeardown: () => controllerRef.current?.onTransportTeardown(),
      onUserInput: () => controllerRef.current?.onUserInput(),
      reset: () => controllerRef.current?.reset(),
    }),
    [],
  );
}
