import { RotateCcw } from "lucide-react";
import { useNavigate } from "react-router";

import {
  PTY_SESSION_ENDED_MESSAGE,
  PTY_START_FAILED_MESSAGE,
} from "@/lib/pty-close-copy";
import { Button } from "@nous-research/ui/ui/components/button";

/**
 * NS-504: the agent process exited (e.g. `/exit` or a new session).
 * Offer an in-place restart so the user never has to refresh the
 * whole page to get a working chat back.
 */
export function PtyEndedOverlay({
  endedReason,
  startFreshPty,
}: {
  endedReason: "start-failed" | "exited" | null;
  startFreshPty: () => void;
}) {
  const navigate = useNavigate();

  return (
    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 bg-black/60">
      <div className="max-w-[min(32rem,calc(100vw-3rem))] text-center text-sm tracking-wide text-white/80">
        {endedReason === "start-failed"
          ? PTY_START_FAILED_MESSAGE
          : PTY_SESSION_ENDED_MESSAGE}
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        <Button
          onClick={startFreshPty}
          prefix={<RotateCcw className="h-4 w-4" />}
          aria-label="Start a new chat session"
        >
          Start new session
        </Button>
        {endedReason === "exited" && (
          <Button
            outlined
            onClick={() => navigate("/logs")}
            aria-label="Open logs"
          >
            Open logs
          </Button>
        )}
      </div>
    </div>
  );
}
