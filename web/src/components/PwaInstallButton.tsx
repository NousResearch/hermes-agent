import { useSyncExternalStore } from "react";
import { Download } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { installPromptStore } from "@/pwa/install-prompt";

export function PwaInstallButton() {
  const state = useSyncExternalStore(
    installPromptStore.subscribe,
    installPromptStore.getSnapshot,
    installPromptStore.getSnapshot,
  );

  if (!state.available) return null;

  return (
    <div className="px-3 pt-2.5">
      <Button
        outlined
        className="min-h-11 w-full justify-center"
        disabled={state.prompting}
        onClick={() => void installPromptStore.prompt()}
        prefix={<Download className="h-4 w-4" />}
        aria-label="Install Hermes"
      >
        Install Hermes
      </Button>
    </div>
  );
}
