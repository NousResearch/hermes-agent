import { useEffect } from "react";
import { api } from "@/lib/api";

const POLL_MS = 3_000;

/**
 * Reload the Channels page when its list moved underneath it: a platform plugin the background
 * default install or another terminal added, a platform toggled elsewhere, the gateway coming up.
 * Desktop gets the same digest pushed as `platforms.changed`; the dashboard has no such push channel
 * for this page, so it polls the stat-cheap signature while the tab is visible (and once on return)
 * and runs `reload` only when it differs from `loaded`, the signature of the list on screen.
 * No poll before the first load returned a signature (an older backend sends none).
 */
export function usePlatformsSignaturePoll(loaded: string | null, reload: () => Promise<unknown>) {
  useEffect(() => {
    if (!loaded) return;
    let stopped = false;
    let inFlight = false;
    const check = async () => {
      if (stopped || inFlight || document.hidden) return;
      inFlight = true;
      try {
        const { signature } = await api.getMessagingPlatformsSignature();
        if (!stopped && signature !== loaded) await reload();
      } catch {
        // A dropped poll is retried on the next tick; the page keeps what it shows.
      } finally {
        inFlight = false;
      }
    };
    const id = window.setInterval(() => void check(), POLL_MS);
    const onVisible = () => void check();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      stopped = true;
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [loaded, reload]);
}
