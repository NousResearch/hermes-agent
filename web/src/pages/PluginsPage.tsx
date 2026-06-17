import { useCallback, useEffect, useState } from "react";
import { ExternalLink, RefreshCw, Trash2, Eye, EyeOff } from "lucide-react";
import type { Translations } from "@/i18n/types";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import type { HubAgentPluginRow, PluginsHubResponse } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { CommandBlock } from "@nous-research/ui/ui/components/command-block";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { ConfirmDialog } from "@nous-research/ui/ui/components/confirm-dialog";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useI18n } from "@/i18n";
import { PluginSlot } from "@/plugins";
import { cn } from "@/lib/utils";
import { usePageHeader } from "@/contexts/usePageHeader";

/** Select value for built-in memory (`config` uses empty string). Never use `""` — UI Select maps empty value to an empty label. */
const MEMORY_PROVIDER_BUILTIN = "__hermes_memory_builtin__";

export default function PluginsPage() {
  const [hub, setHub] = useState<PluginsHubResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [installId, setInstallId] = useState("");
  const [installForce, setInstallForce] = useState(false);
  const [installEnable, setInstallEnable] = useState(true);
  const [installBusy, setInstallBusy] = useState(false);
  const [rescanBusy, setRescanBusy] = useState(false);
  const [memorySel, setMemorySel] = useState(MEMORY_PROVIDER_BUILTIN);
  const [contextSel, setContextSel] = useState("compressor");
  const [providerBusy, setProviderBusy] = useState(false);
  const [rowBusy, setRowBusy] = useState<string | null>(null);

  const { toast, showToast } = useToast();
  const { t } = useI18n();
  const { setAfterTitle } = usePageHeader();

  const loadHub = useCallback(() => {
    return api
      .getPluginsHub()
      .then((h) => {
        setHub(h);
        const p = h.providers;
        setMemorySel(p.memory_provider ? p.memory_provider : MEMORY_PROVIDER_BUILTIN);
        setContextSel(p.context_engine || "compressor");
      })
      .catch(() => showToast(t.common.loading, "error"));
  }, [showToast, t.common.loading]);

  useEffect(() => {
    const id = window.setTimeout(() => {
      setLoading(true);
      void loadHub().finally(() => setLoading(false));
    }, 0);
    return () => window.clearTimeout(id);
  }, [loadHub]);

  const onRescan = useCallback(async () => {
    setRescanBusy(true);
    try {
      const rc = await api.rescanPlugins();
      showToast(
        `${t.pluginsPage.refreshDashboard} (${rc.count})`,
        "success",
      );
      await loadHub();
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Rescan failed", "error");
    } finally {
      setRescanBusy(false);
    }
  }, [loadHub, showToast, t.pluginsPage.refreshDashboard]);

  useEffect(() => {
    setAfterTitle(
      <Button
        ghost
        size="icon"
        className="shrink-0 text-muted-foreground hover:text-foreground"
        disabled={loading || rescanBusy}
        onClick={() => void onRescan()}
        aria-label={t.pluginsPage.refreshDashboard}
      >
        {rescanBusy ? <Spinner /> : <RefreshCw />}
      </Button>,
    );
    return () => setAfterTitle(null);
  }, [loading, rescanBusy, setAfterTitle, t.pluginsPage.refreshDashboard, onRescan]);

  const onInstall = async () => {
    const id = installId.trim();
    if (!id) {
      showToast(t.pluginsPage.installHint, "error");
      return;
    }
    setInstallBusy(true);
    try {
      const r = await api.installAgentPlugin({
        identifier: id,
        force: installForce,
        enable: installEnable,
      });
      showToast(`${r.plugin_name ?? id} installed`, "success");
      if ((r.warnings?.length ?? 0) > 0) showToast(r.warnings!.join(" "), "error");
      if ((r.missing_env?.length ?? 0) > 0)
        showToast(`${t.pluginsPage.missingEnvWarn} ${r.missing_env!.join(", ")}`, "error");
      setInstallId("");
      await loadHub();
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Install failed", "error");
    } finally {
      setInstallBusy(false);
    }
  };

  const onSaveProviders = async () => {
    setProviderBusy(true);
    try {
      await api.savePluginProviders({
        memory_provider:
          memorySel === MEMORY_PROVIDER_BUILTIN ? "" : memorySel,
        context_engine: contextSel,
      });
      showToast(t.pluginsPage.savedProviders, "success");
      await loadHub();
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Save failed", "error");
    } finally {
      setProviderBusy(false);
    }
  };

  const setRuntimeLoading = async (name: string, fn: () => Promise<unknown>) => {
    setRowBusy(name);
    try {
      await fn();
      await loadHub();
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Failed", "error");
    } finally {
      setRowBusy(null);
    }
  };

  const rows = hub?.plugins ?? [];
  const providers = hub?.providers;

  return (
    <div className="flex flex-col gap-4">
      <PluginSlot name="plugins:top" />

      <div className={cn("flex w-full flex-col gap-8")}>

        {providers && (
          <Card>
            <CardHeader>
              <CardTitle>{t.pluginsPage.providersHeading}</CardTitle>
              <p className="text-xs tracking-[0.08em] text-text-tertiary">
                {t.pluginsPage.providersHint}
              </p>
            </CardHeader>

            <CardContent className="flex flex-col gap-6">

              <div className="grid gap-6 sm:grid-cols-2 max-w-full">
              <div className="grid gap-2 min-w-0">
                <Label htmlFor="mem-provider">{t.pluginsPage.memoryProviderLabel}</Label>

                <Select
                  id="mem-provider"
                  className="w-full"
                  value={memorySel}
                  onValueChange={setMemorySel}
                >
                  <SelectOption value={MEMORY_PROVIDER_BUILTIN}>
                    {`(${t.pluginsPage.providerDefaults})`}
                  </SelectOption>

                  {providers.memory_options.map((o) => (
                    <SelectOption key={o.name} value={o.name}>
                      {o.name}
                    </SelectOption>
                  ))}
                </Select>
              </div>

              <div className="grid gap-2 min-w-0">
                <Label htmlFor="ctx-engine">{t.pluginsPage.contextEngineLabel}</Label>

                <Select
                  id="ctx-engine"
                  className="w-full"
                  value={contextSel}
                  onValueChange={setContextSel}
                >
                  <SelectOption value="compressor">compressor</SelectOption>

                  {providers.context_options
                    .filter((o) => o.name !== "compressor")
                    .map((o) => (
                      <SelectOption key={o.name} value={o.name}>
                        {o.name}
                      </SelectOption>
                    ))}
                </Select>
              </div>
              </div>

              <Button
                className="w-fit uppercase"
                size="sm"
                disabled={providerBusy}
                onClick={() => void onSaveProviders()}
                prefix={providerBusy ? <Spinner /> : undefined}
              >
                {t.common.save}
              </Button>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle>{t.pluginsPage.installHeading}</CardTitle>
            <p className="text-xs tracking-[0.08em] text-text-tertiary">
              {t.pluginsPage.installHint}
            </p>
          </CardHeader>


          <CardContent className="flex flex-col gap-4">

            <div className="flex flex-col gap-2">

              <Label htmlFor="install-url">{t.pluginsPage.identifierLabel}</Label>

              <Input
                className="font-mono-ui lowercase"
                id="install-url"
                placeholder="owner/repo, owner/repo/subdir, or https://..."
                spellCheck={false}
                value={installId}
                onChange={(e) => setInstallId(e.target.value)}
              />
            </div>


            <div className="flex flex-wrap items-center gap-8">

              <div className="flex items-center gap-3">

                <Switch checked={installForce} onCheckedChange={setInstallForce} />

                <span className="text-xs tracking-[0.06em] text-text-secondary">
                  {t.pluginsPage.forceReinstall}
                </span>
              </div>

              <div className="flex items-center gap-3">

                <Switch checked={installEnable} onCheckedChange={setInstallEnable} />

                <span className="text-xs tracking-[0.06em] text-text-secondary">
                  {t.pluginsPage.enableAfterInstall}
                </span>
              </div>
            </div>

            <Button
              className="w-fit uppercase"
              size="sm"
              disabled={installBusy}
              onClick={() => void onInstall()}
              prefix={installBusy ? <Spinner /> : undefined}
            >
              {t.pluginsPage.installBtn}
            </Button>

            <p className="text-xs tracking-[0.06em] text-text-tertiary">
              {t.pluginsPage.rescanHint}
            </p>

            <p className="text-xs tracking-[0.06em] text-text-tertiary">
              {t.pluginsPage.removeHint}
            </p>
          </CardContent>
        </Card>

        <div className="flex flex-col gap-3">

          <h3 className="font-mondwest text-display text-xs tracking-[0.12em] text-text-secondary">
            {t.pluginsPage.pluginListHeading}
          </h3>

          {loading ? (

            <div className="flex items-center gap-2 py-8 text-xs text-text-tertiary">

              <Spinner />
              <span>{t.common.loading}</span>
            </div>
          ) : rows.length === 0 ? (

            <p className="text-xs text-text-tertiary">{t.common.noResults}</p>
          ) : (

            <ul className="flex flex-col gap-3">

              {rows.map((row: HubAgentPluginRow) => (

                <li key={row.name}>


                  <PluginRowCard
                    {...{ row, rowBusy, setRuntimeLoading, showToast, t }}
                  />

                </li>
              ))}
            </ul>
          )}
        </div>

        {(hub?.orphan_dashboard_plugins?.length ?? 0) > 0 ? (


          <div className="flex flex-col gap-3 opacity-95">

            <h3 className="font-mondwest text-display text-xs tracking-[0.12em] text-text-secondary">
              {t.pluginsPage.orphanHeading}
            </h3>

            <ul className="flex flex-col gap-2 rounded border border-current/15 p-4">

              {hub!.orphan_dashboard_plugins.map((m) => (

                <li className="text-xs text-text-secondary" key={m.name}>


                  {m.label ?? m.name} — {m.description || m.tab?.path}


                  {!m.tab?.hidden ? (


                    <Link className="ml-3 inline-flex items-center gap-1 underline" to={m.tab.path}>


                      <ExternalLink className="h-3 w-3 opacity-65" />

                      {t.pluginsPage.openTab}
                    </Link>
                  ) : null}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>

      <Toast toast={toast} />
      <PluginSlot name="plugins:bottom" />
    </div>
  );
}

interface PluginRowCardProps {

  row: HubAgentPluginRow;
  rowBusy: string | null;
  setRuntimeLoading: (
    name: string,
    fn: () => Promise<unknown>,
  ) => Promise<void>;

  showToast: (msg: string, variant: "success" | "error") => void;
  t: Translations;
}

function PluginRowCard(props: PluginRowCardProps) {
  const {
    row,
    rowBusy,
    setRuntimeLoading,
    showToast,
    t,
  } = props;

  const dm = row.dashboard_manifest;

  const tabPath = dm?.tab && !dm.tab.hidden ? dm.tab.override ?? dm.tab.path : null;

  const busy = rowBusy === row.name;
  const [confirmRemove, setConfirmRemove] = useState(false);

  const badgeTone =
    row.runtime_status === "enabled"
      ? "success"
      : row.runtime_status === "disabled"
        ? "destructive"
        : "outline";

  return (

    <Card className={cn(busy ? "opacity-70" : undefined)}>


      <CardContent className="flex flex-col gap-4 px-6 py-4">


        <div className="flex flex-wrap items-start justify-between gap-4">

          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-3">

            <span className="truncate font-semibold">{row.name}</span>

            <Badge tone="outline">
              {t.pluginsPage.sourceBadge}: {row.source}
            </Badge>

            <Badge tone="outline">v{row.version || "—"}</Badge>

            <Badge tone={badgeTone}>{row.runtime_status}</Badge>

            {row.auth_required ? (
              <Badge tone="destructive">{t.pluginsPage.authRequired}</Badge>
            ) : null}
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            {row.runtime_status === "enabled" ? (
              <Button
                disabled={busy}
                ghost
                size="sm"
                onClick={() => {
                  void setRuntimeLoading(row.name, async () => {
                    await api.disableAgentPlugin(row.name);
                    showToast(t.pluginsPage.disableRuntime, "success");
                  });
                }}
              >
                {t.pluginsPage.disableRuntime}
              </Button>
            ) : (
              <Button
                disabled={busy}
                ghost
                size="sm"
                onClick={() => {
                  void setRuntimeLoading(row.name, async () => {
                    await api.enableAgentPlugin(row.name);
                    showToast(t.pluginsPage.enableRuntime, "success");
                  });
                }}
              >
                {t.pluginsPage.enableRuntime}
              </Button>
            )}

            {tabPath ? (

              <Link
                className={cn(
                  "inline-flex items-center rounded-none px-3 py-1.5",
                  "border border-current/25 hover:bg-current/10",
                  "font-mondwest text-display text-xs tracking-[0.1em]",
                )}
                to={tabPath}
              >
                {t.pluginsPage.openTab}
              </Link>
            ) : null}

            {row.can_update_git ? (

              <Button
                disabled={busy}
                ghost
                size="sm"
                onClick={() => {
                  void setRuntimeLoading(row.name, async () => {
                    await api.updateAgentPlugin(row.name);
                    showToast(t.pluginsPage.updateGit, "success");
                  });
                }}
              >
                {busy ? <Spinner /> : null}
                {t.pluginsPage.updateGit}
              </Button>
            ) : null}

            {row.has_dashboard_manifest ? (
              <Button
                disabled={busy}
                ghost
                size="sm"
                title={row.user_hidden ? t.pluginsPage.showInSidebar : t.pluginsPage.hideFromSidebar}
                onClick={() => {
                  void setRuntimeLoading(row.name, async () => {
                    await api.setPluginVisibility(row.name, !row.user_hidden);
                  });
                }}
              >
                {row.user_hidden ? (
                  <EyeOff className="h-3.5 w-3.5" />
                ) : (
                  <Eye className="h-3.5 w-3.5" />
                )}
                {row.user_hidden ? t.pluginsPage.showInSidebar : t.pluginsPage.hideFromSidebar}
              </Button>
            ) : null}

            {row.can_remove ? (


              <Button
                destructive
                disabled={busy}
                ghost
                size="sm"
                onClick={() => setConfirmRemove(true)}
              >

                {busy ? <Spinner /> : <Trash2 className="h-3.5 w-3.5" />}
              </Button>
            ) : null}
          </div>
        </div>

        {row.description ? (
          <p className="min-w-0 w-full text-xs tracking-[0.06em] text-text-secondary break-words">
            {row.description}
          </p>
        ) : null}

        {dm?.slots?.length ? (

          <p className="text-xs tracking-[0.05em] text-text-tertiary">
            {t.pluginsPage.dashboardSlots}: {dm.slots.join(", ")}
          </p>
        ) : null}

        {row.auth_required ? (
          <CommandBlock
            label={t.pluginsPage.authRequiredHint}
            code={row.auth_command}
          />
        ) : null}

        {!row.has_dashboard_manifest && !dm ? (


          <p className="text-xs italic text-text-disabled">
            {t.pluginsPage.noDashboardTab}
          </p>
        ) : null}
      </CardContent>

      <ConfirmDialog
        open={confirmRemove}
        onCancel={() => setConfirmRemove(false)}
        onConfirm={() => {
          setConfirmRemove(false);
          void setRuntimeLoading(row.name, async () => {
            await api.removeAgentPlugin(row.name);
            showToast(`${row.name} removed`, "success");
          });
        }}
        title={t.pluginsPage.removeConfirm}
        description={`This will remove the "${row.name}" plugin from your agent.`}
        destructive
        confirmLabel={t.common.delete}
      />
    </Card>
  );
}
