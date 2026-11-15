import { useEffect, useRef, useState } from "react";

import { Button } from "@nous-research/ui/ui/components/button";
import { Copy } from "lucide-react";
import { cn } from "@/lib/utils";

import {
  COPY_STATE_RESET_MS,
  copyLastResponse,
  type CopyLastState,
} from "./chat-copy-last";

/** Button copy per outcome; "failed" is honest — nothing was copied. */
const LABELS: Record<CopyLastState, string> = {
  copied: "copied",
  failed: "copy failed",
  idle: "copy last response",
};

/**
 * Floating "copy last response" overlay on the chat terminal. The click reads the transcript over
 * HTTP and writes the browser clipboard (see chat-copy-last.ts) instead of typing `/copy` into the
 * PTY composer, which used to submit an unsent draft to the model (#149751). The label claims
 * "copied" only after that write resolved, and says "copy failed" when nothing reached the
 * clipboard.
 */
export function CopyLastButton({
  channel,
  sessionId,
  profile,
  color,
}: {
  channel: string;
  sessionId: string | null;
  profile?: string;
  color: string;
}) {
  const [state, setState] = useState<CopyLastState>("idle");
  const resetRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (resetRef.current) clearTimeout(resetRef.current);
    },
    [],
  );

  const onClick = () => {
    if (resetRef.current) clearTimeout(resetRef.current);
    void copyLastResponse({ channel, profile, sessionId }).then((text) => {
      setState(text ? "copied" : "failed");
      resetRef.current = setTimeout(() => setState("idle"), COPY_STATE_RESET_MS);
    });
  };

  return (
    <Button
      ghost
      onClick={onClick}
      title="Copy last assistant response as raw markdown"
      aria-label="Copy last assistant response"
      className={cn(
        "absolute z-10",
        "normal-case tracking-normal font-normal",
        "rounded border border-current/30",
        "bg-black/20",
        "opacity-70 hover:opacity-100 hover:border-current/60",
        "transition-opacity duration-150",
        "bottom-2 right-2 px-2 py-1 text-xs sm:bottom-3 sm:right-3 sm:px-2.5 sm:py-1.5",
        "lg:bottom-4 lg:right-4",
      )}
      style={{ color }}
    >
      <span className="inline-flex items-center gap-1.5">
        <Copy className="h-3 w-3 shrink-0" />
        <span className="hidden min-[400px]:inline tracking-wide">
          {LABELS[state]}
        </span>
      </span>
    </Button>
  );
}
