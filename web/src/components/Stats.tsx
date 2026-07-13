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
 * The shared design-system component renders its visual dot leader as text.
 * Mark that decorative separator hidden so accessibility tools do not treat
 * deliberately faint ornamentation as readable content.
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

            <Typography
              aria-hidden="true"
              className="min-w-0 overflow-hidden text-[13px] leading-[1.4] tracking-[0.4em] opacity-20"
              expanded
            >
              {"·".repeat(100)}
            </Typography>

            {flip ? valueText : labelText}
          </div>
        );
      })}
    </div>
  );
}
