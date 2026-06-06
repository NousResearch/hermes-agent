import { Badge } from "@nous-research/ui/ui/components/badge";
import { cn } from "@/lib/utils";
import type { CrewHealth } from "@/types/crew";

const HEALTH_CLASS: Record<CrewHealth, string> = {
  green: "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300",
  yellow: "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  red: "border-red-500/40 bg-red-500/10 text-red-600 dark:text-red-300",
  gray: "border-muted-foreground/30 bg-muted text-muted-foreground",
};

export interface CrewHealthBadgeProps {
  health: CrewHealth;
  reasons?: string[];
  className?: string;
}

export function CrewHealthBadge({ health, reasons = [], className }: CrewHealthBadgeProps) {
  const label = health.charAt(0).toUpperCase() + health.slice(1);
  return (
    <Badge
      className={cn("border text-xs capitalize", HEALTH_CLASS[health], className)}
      title={reasons.join(" • ") || label}
    >
      {label}
    </Badge>
  );
}
