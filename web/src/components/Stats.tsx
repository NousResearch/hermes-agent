import type { ComponentProps, ReactNode } from "react";

import { Typography } from "@nous-research/ui/ui/components/typography/index";
import { cn } from "@/lib/utils";

interface StatsProps extends ComponentProps<"div"> {
  items: {
    label: string | { key: string; node: ReactNode };
    value: string | { key: string; node: ReactNode };
  }[];
  flip?: boolean;
}

/**
 * Dashboard Stats adapter.
 *
 * Render the visual dot leader as a CSS border instead of repeated text so it
 * is decorative by construction and cannot be mistaken for readable content.
 */
export function Stats({ className, items, flip, ...props }: StatsProps) {
  return (
    <div className={cn("flex w-full flex-col gap-5", className)} {...props}>
      {items.map(({ label, value }) => {
        const valueText = (
          <Typography
            className="text-xs leading-[1.4] tracking-widest"
            expanded
          >
            {typeof value === "string" ? value : value.node}
          </Typography>
        );
        const labelText = (
          <Typography className="leading-none tracking-[0.2em] opacity-60" mono>
            {typeof label === "string" ? label : label.node}
          </Typography>
        );

        return (
          <div
            className="text-midground text-display grid grid-cols-[auto_1fr_auto] items-center gap-2.5"
            key={`${typeof label === "string" ? label : label.key}@@@${
              typeof value === "string" ? value : value.key
            }`}
          >
            {flip ? labelText : valueText}

            <span
              aria-hidden="true"
              className="min-w-0 border-b border-dotted border-current opacity-20"
            />

            {flip ? valueText : labelText}
          </div>
        );
      })}
    </div>
  );
}
