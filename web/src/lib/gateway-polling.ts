/**
 * Configuration for polling a gateway status check.
 */
export interface PollGatewayConfig {
  /** Human-readable name for the target (used in time-out messages). */
  name: string;
  /** The desired boolean state to wait for (true = running, false = stopped). */
  desiredState: boolean;
  /** Maximum wall-clock time to keep polling, in milliseconds. */
  timeoutMs: number;
  /** Interval between polls, in milliseconds. */
  intervalMs: number;
}

/**
 * Poll a status-fn until it returns the desired boolean or the deadline
 * expires.  Accepts an optional AbortSignal so callers can cancel mid-poll
 * (component unmount, user navigation, etc.).
 *
 * Returns `true` when the desired state was confirmed, `false` if the
 * deadline passed or the signal was aborted.
 *
 * This is extracted from the ProfilesPage gateway-toggle handler so the
 * polling contract is independently testable.
 */
export function pollGatewayStatus(
  getStatus: (name: string) => Promise<boolean>,
  config: PollGatewayConfig,
  signal?: AbortSignal,
): Promise<boolean> {
  const deadline = Date.now() + config.timeoutMs;

  return new Promise<boolean>((resolve) => {
    let cancelled = false;

    if (signal) {
      const onAbort = () => {
        cancelled = true;
        resolve(false);
      };
      signal.addEventListener("abort", onAbort, { once: true });
    }

    const poll = () => {
      if (cancelled) return;

      if (Date.now() > deadline) {
        resolve(false);
        return;
      }

      getStatus(config.name)
        .then((running) => {
          if (cancelled) return;
          if (running === config.desiredState) {
            resolve(true);
          } else {
            setTimeout(poll, config.intervalMs);
          }
        })
        .catch(() => {
          if (!cancelled) setTimeout(poll, config.intervalMs);
        });
    };

    setTimeout(poll, config.intervalMs);
  });
}
