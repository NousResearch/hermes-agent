import { Button } from "@nous-research/ui/ui/components/button";
import { ArrowRight, Bot } from "lucide-react";
import { useCallback } from "react";

import { useProfileScope } from "@/contexts/useProfileScope";
import { cn } from "@/lib/utils";

interface PilotProjectBot {
  id: string;
  name: string;
  project: string;
  initials: string;
}

const PILOT_PROJECT_BOTS: readonly PilotProjectBot[] = [
  {
    id: "default",
    name: "Main CTO",
    project: "Chief of Staff",
    initials: "CTO",
  },
  {
    id: "mapautogen",
    name: "Map Autogen PM",
    project: "Map Autogen",
    initials: "MA",
  },
  {
    id: "microtwin",
    name: "MicroTwin PM",
    project: "MicroTwin",
    initials: "MT",
  },
  {
    id: "scenarioautogen",
    name: "Scenario Autocreation PM",
    project: "Scenario Autocreation",
    initials: "SA",
  },
  {
    id: "realtosim",
    name: "Real-to-Sim PM",
    project: "Real-to-Sim",
    initials: "RS",
  },
];

interface ProjectBotLauncherProps {
  className?: string;
  /** Close the mobile sheet after a Bot is selected. */
  onPicked?: () => void;
}

/** Open a persistent dashboard workspace for the main CTO or a pilot project Bot. */
export function ProjectBotLauncher({
  className,
  onPicked,
}: ProjectBotLauncherProps) {
  const { profile, currentProfile, profiles, setProfile } = useProfileScope();
  const selectedProfile = profile || currentProfile || "default";

  const launch = useCallback(
    (profileId: string) => {
      setProfile(profileId, { clearResume: true });
      onPicked?.();
    },
    [onPicked, setProfile],
  );

  return (
    <section
      aria-label="Bot workspaces"
      className={cn("flex min-w-0 flex-col gap-2 px-2", className)}
    >
      <div className="flex items-center gap-1.5 text-display text-xs tracking-wider text-text-tertiary">
        <Bot className="h-3.5 w-3.5" aria-hidden />
        <span>Bots</span>
        <span className="ml-auto rounded-full bg-primary/10 px-1.5 py-0.5 text-[0.625rem] text-primary">
          pilot
        </span>
      </div>

      <div className="grid grid-cols-1 gap-1.5">
        {PILOT_PROJECT_BOTS.map((bot) => {
          const available = profiles.includes(bot.id);
          const selected = selectedProfile === bot.id;

          return (
            <Button
              key={bot.id}
              outlined={!selected}
              size="sm"
              disabled={!available}
              aria-label={`Open ${bot.name}`}
              aria-current={selected ? "true" : undefined}
              onClick={() => launch(bot.id)}
              className={cn(
                "min-h-11 w-full min-w-0 justify-start gap-2 px-2 normal-case tracking-normal",
                selected && "border-primary/60 bg-primary/10",
              )}
            >
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-current/20 bg-background font-mono-ui text-[0.6875rem] font-semibold">
                {bot.initials}
              </span>
              <span className="flex min-w-0 flex-1 flex-col items-start leading-tight">
                <span className="w-full truncate text-left text-sm font-medium">
                  {bot.name}
                </span>
                <span className="w-full truncate text-left text-[0.6875rem] text-text-tertiary">
                  {available ? bot.project : "Not configured"}
                </span>
              </span>
              {available && <ArrowRight className="h-3.5 w-3.5 shrink-0" aria-hidden />}
            </Button>
          );
        })}
      </div>
    </section>
  );
}
