import type { ComponentProps } from "react";
import type { LucideIcon } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { cn } from "@/lib/utils";

interface ChatToolbarButtonProps extends ComponentProps<typeof Button> {
  terminalForeground: string;
  icon: LucideIcon;
}

/** Quiet terminal chrome shares the terminal palette, not dashboard accents. */
export function ChatToolbarButton({
  terminalForeground,
  icon: Icon,
  children,
  className,
  style,
  ...props
}: ChatToolbarButtonProps) {
  return (
    <Button
      {...props}
      ghost
      className={cn(
        "inline-flex items-center justify-center gap-1.5 font-mono text-xs leading-4 normal-case tracking-normal font-normal",
        "rounded border border-current/30 bg-black/20",
        "opacity-70 hover:opacity-100 hover:border-current/60",
        "transition-opacity duration-150",
        "h-10 w-24 shrink-0 px-2 py-0 sm:h-8",
        className,
      )}
      style={{ ...style, color: terminalForeground }}
    >
      <Icon aria-hidden="true" className="size-4 shrink-0" strokeWidth={1.5} />
      <span>{children}</span>
    </Button>
  );
}
