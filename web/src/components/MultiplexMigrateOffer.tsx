import { Button } from "@nous-research/ui/ui/components/button";
import { CardContent } from "@nous-research/ui/ui/components/card";
import type { GatewayMigratePlan } from "@/lib/api";

/**
 * Gateway card footer offering `hermes gateway migrate --multiplex` when several profiles each run their
 * own gateway (with the plan's blockers). Renders nothing when there is nothing to offer.
 */
export function MultiplexMigrateOffer({ plan, onMigrate }: { plan: GatewayMigratePlan; onMigrate: () => void }) {
  if (plan.already_multiplexed || plan.profiles.length <= 1 || !(plan.eligible || plan.blockers.length > 0)) {
    return null;
  }
  return (
    <CardContent className="flex flex-col gap-2 border-t border-border py-4 text-sm">
      <div className="flex items-center justify-between gap-3">
        <span className="text-muted-foreground">
          Your profiles each run their own gateway. One multiplexed gateway serves every profile from a single process.
        </span>
        <Button
          size="sm"
          className="uppercase"
          onClick={onMigrate}
          disabled={!plan.eligible}
          title={plan.eligible ? undefined : "Fix the blockers below first"}
        >
          Migrate to a single multiplexed gateway
        </Button>
      </div>
      {plan.blockers.map((b) => (
        <div key={b} className="text-warning">• {b}</div>
      ))}
    </CardContent>
  );
}
