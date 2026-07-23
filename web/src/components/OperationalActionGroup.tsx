import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

interface OperationalActionGroupProps extends ComponentProps<"div"> {
  separated?: boolean;
}

/**
 * Shared action-row geometry for dense operational cards.
 *
 * Phone actions use a two-column grid with comfortable touch targets; the
 * existing compact inline layout returns at the desktop shell breakpoint.
 */
export function OperationalActionGroup({
  className,
  separated = false,
  ...props
}: OperationalActionGroupProps) {
  return (
    <div
      role="group"
      className={cn(
        "grid w-full grid-cols-2 gap-2 [&_button]:min-h-11 [&_a]:min-h-11",
        "sm:ml-auto sm:flex sm:w-auto sm:shrink-0 sm:flex-wrap sm:items-center sm:gap-1 lg:[&_button]:min-h-0 lg:[&_a]:min-h-0",
        separated && "border-t border-border/60 pt-2 sm:border-t-0 sm:pt-0",
        className,
      )}
      {...props}
    />
  );
}
