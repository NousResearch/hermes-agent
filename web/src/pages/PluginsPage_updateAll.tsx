import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Label } from "@nous-research/ui/ui/components/label";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Switch } from "@nous-research/ui/ui/components/switch";
import type { Translations } from "@/i18n/types";
import { api, type AgentPluginsUpdateAllResponse, type PluginAutoUpdateState } from "@/lib/api";

type ShowToast = (msg: string, variant: "success" | "error") => void;

/** Counts for the "Update all" toast; a `needs_review` row changed nothing (its own Update walks the review). */
function updateAllCounts({ results }: AgentPluginsUpdateAllResponse) {
  const count = (status: string) => results.filter((r) => r.status === status).length;
  return { updated: count("updated"), needsReview: count("needs_review"), failed: count("failed") };
}

interface PluginUpdateAllButtonProps {
  disabled: boolean;
  loadHub: () => Promise<unknown>;
  showToast: ShowToast;
  t: Translations;
}

function PluginUpdateAllButton({ disabled, loadHub, showToast, t }: PluginUpdateAllButtonProps) {
  const [busy, setBusy] = useState(false);

  const onClick = async () => {
    setBusy(true);
    try {
      const { updated, needsReview, failed } = updateAllCounts(await api.updateAllAgentPlugins());
      showToast(t.pluginsPage.updateAllDone(updated, needsReview, failed), failed > 0 ? "error" : "success");
      await loadHub();
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Update failed", "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Button disabled={busy || disabled} ghost size="sm" onClick={() => void onClick()}>
      {busy ? <Spinner /> : <RefreshCw className="h-3 w-3" />}
      {busy ? t.pluginsPage.updatingAll : t.pluginsPage.updateAll}
    </Button>
  );
}

interface PluginAutoUpdateSwitchProps {
  showToast: ShowToast;
  t: Translations;
}

/** The profile's `plugins.auto_apply` switch; hidden until its state loads. */
function PluginAutoUpdateSwitch({ showToast, t }: PluginAutoUpdateSwitchProps) {
  const [state, setState] = useState<PluginAutoUpdateState | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.getPluginAutoUpdate().then(setState).catch(() => setState(null));
  }, []);

  const onChange = async (enabled: boolean) => {
    setBusy(true);
    try {
      setState(await api.setPluginAutoUpdate(enabled));
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Save failed", "error");
    } finally {
      setBusy(false);
    }
  };

  if (!state) return null;
  return (
    <div className="flex items-start gap-3">
      <Switch
        aria-label={t.pluginsPage.autoUpdate}
        checked={state.auto_apply}
        disabled={busy}
        onCheckedChange={(v) => void onChange(v)}
      />
      <div className="flex flex-col gap-1">
        <Label>{t.pluginsPage.autoUpdate}</Label>
        <p className="text-xs text-text-tertiary">{t.pluginsPage.autoUpdateDesc(state.check_hours)}</p>
      </div>
    </div>
  );
}

interface InstalledPluginsHeaderProps {
  loading: boolean;
  loadHub: () => Promise<unknown>;
  pluginCount: number;
  showToast: ShowToast;
  t: Translations;
}

/** The installed-plugins heading with the "Update all" action and the auto-update switch. */
export function InstalledPluginsHeader({ loading, loadHub, pluginCount, showToast, t }: InstalledPluginsHeaderProps) {
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-mondwest text-display text-xs tracking-[0.12em] text-text-secondary">
          {t.pluginsPage.pluginListHeading}
        </h3>

        <PluginUpdateAllButton disabled={loading || pluginCount === 0} {...{ loadHub, showToast, t }} />
      </div>

      <PluginAutoUpdateSwitch {...{ showToast, t }} />
    </>
  );
}
