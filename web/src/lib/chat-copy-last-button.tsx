import { Button } from "@nous-research/ui/ui/components/button";
import { Copy } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Floating "copy last response" overlay on the chat terminal: sends `/copy` over
 * the PTY (see chat-copy-last.ts) and shows "copied" for the 1.5s reset window.
 */
export function CopyLastButton({
  onClick,
  copied,
  color,
}: {
  onClick: () => void;
  copied: boolean;
  color: string;
}) {
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
          {copied ? "copied" : "copy last response"}
        </span>
      </span>
    </Button>
  );
}
