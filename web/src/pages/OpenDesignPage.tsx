import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import {
  ExternalLink,
  FolderOpen,
  Globe,
  Play,
  RefreshCw,
  Save,
  Square,
  Upload,
} from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Toast } from "@/components/Toast";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, type OpenDesignArtifact, type OpenDesignConfig, type OpenDesignStatusResponse } from "@/lib/api";
import { useToast } from "@/hooks/useToast";
import { useI18n } from "@/i18n";
import { PluginSlot } from "@/plugins";
import { usePageHeader } from "@/contexts/usePageHeader";

const DEFAULT_CONFIG: OpenDesignConfig = {
  repo_path: "",
  start_command: "pnpm tools-dev run web",
  ui_url: "http://127.0.0.1:3000",
  health_url: "http://127.0.0.1:3000",
  workspace_dir: "",
  artifacts_dir: "",
  env: {},
};

function formatDateTime(timestamp?: number | null): string {
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleString();
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function stateTone(state: string): "success" | "warning" | "destructive" | "secondary" {
  if (state === "running") return "success";
  if (state === "starting") return "warning";
  if (state === "error") return "destructive";
  return "secondary";
}

export default function OpenDesignPage() {
  const { t } = useI18n();
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();
  const [status, setStatus] = useState<OpenDesignStatusResponse | null>(null);
  const [config, setConfig] = useState<OpenDesignConfig>(DEFAULT_CONFIG);
  const [artifacts, setArtifacts] = useState<OpenDesignArtifact[]>([]);
  const [brief, setBrief] = useState("");
  const [projectName, setProjectName] = useState("");
  const [skill, setSkill] = useState("");
  const [designSystem, setDesignSystem] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [actionBusy, setActionBusy] = useState<string | null>(null);
  const [pushing, setPushing] = useState(false);

  const loadAll = useCallback(async () => {
    try {
      const [nextStatus, nextConfig, nextArtifacts] = await Promise.all([
        api.getOpenDesignStatus(),
        api.getOpenDesignConfig(),
        api.getOpenDesignArtifacts(50),
      ]);
      setStatus(nextStatus);
      setConfig(nextConfig);
      setArtifacts(nextArtifacts.items);
    } catch (error) {
      showToast(`${t.openDesign.loadFailed}: ${error}`, "error");
    } finally {
      setLoading(false);
    }
  }, [showToast, t.openDesign.loadFailed]);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  useLayoutEffect(() => {
    setEnd(
      <div className="flex items-center gap-2">
        <Button
          ghost
          size="sm"
          prefix={<RefreshCw />}
          onClick={loadAll}
          aria-label={t.common.refresh}
        >
          {t.common.refresh}
        </Button>
      </div>,
    );
    return () => setEnd(null);
  }, [loadAll, setEnd, t.common.refresh]);

  const envPairs = useMemo(
    () => Object.entries(config.env ?? {}).sort(([a], [b]) => a.localeCompare(b)),
    [config.env],
  );

  const updateField = <K extends keyof OpenDesignConfig>(key: K, value: OpenDesignConfig[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const saveConfig = async () => {
    setSaving(true);
    try {
      const saved = await api.saveOpenDesignConfig(config);
      setConfig(saved);
      showToast(t.openDesign.configSaved, "success");
      await loadAll();
    } catch (error) {
      showToast(`${t.openDesign.configSaveFailed}: ${error}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const runAction = async (
    kind: "start" | "stop" | "restart" | "open",
    action: () => Promise<unknown>,
    successMessage: string,
  ) => {
    setActionBusy(kind);
    try {
      await action();
      showToast(successMessage, "success");
      await loadAll();
    } catch (error) {
      showToast(`${t.status.error}: ${error}`, "error");
    } finally {
      setActionBusy(null);
    }
  };

  const pushBrief = async () => {
    if (!brief.trim()) {
      showToast(t.openDesign.briefRequired, "error");
      return;
    }
    setPushing(true);
    try {
      const result = await api.pushOpenDesignBrief({
        brief: brief.trim(),
        project_name: projectName.trim() || undefined,
        skill: skill.trim() || undefined,
        design_system: designSystem.trim() || undefined,
      });
      showToast(`${t.openDesign.briefPushed}: ${result.project_dir}`, "success");
      setBrief("");
      setProjectName("");
      setSkill("");
      setDesignSystem("");
      await loadAll();
    } catch (error) {
      showToast(`${t.openDesign.briefPushFailed}: ${error}`, "error");
    } finally {
      setPushing(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <PluginSlot name="open-design:top" />
      <Toast toast={toast} />

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Globe className="h-4 w-4" />
            {t.openDesign.title}
          </CardTitle>
          <CardDescription>{t.openDesign.description}</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className="grid gap-1">
            <span className="text-xs text-muted-foreground">{t.openDesign.statusLabel}</span>
            <div className="flex items-center gap-2">
              <Badge tone={stateTone(status?.state ?? "stopped")}>{status?.state ?? "stopped"}</Badge>
              {status?.health_ok ? <Badge tone="success">{t.openDesign.healthy}</Badge> : null}
            </div>
          </div>
          <div className="grid gap-1">
            <span className="text-xs text-muted-foreground">PID</span>
            <span className="text-sm">{status?.pid ?? "—"}</span>
          </div>
          <div className="grid gap-1">
            <span className="text-xs text-muted-foreground">{t.openDesign.lastStart}</span>
            <span className="text-sm">{formatDateTime(status?.last_start_at)}</span>
          </div>
          <div className="grid gap-1">
            <span className="text-xs text-muted-foreground">{t.openDesign.logPath}</span>
            <span className="truncate text-sm">{status?.log_path || "—"}</span>
          </div>

          <div className="flex flex-wrap gap-2 md:col-span-2 xl:col-span-4">
            <Button
              onClick={() => runAction("start", api.startOpenDesign, t.openDesign.started)}
              disabled={actionBusy !== null}
              prefix={<Play />}
            >
              {t.openDesign.start}
            </Button>
            <Button
              ghost
              onClick={() => runAction("stop", api.stopOpenDesign, t.openDesign.stopped)}
              disabled={actionBusy !== null}
              prefix={<Square />}
            >
              {t.openDesign.stop}
            </Button>
            <Button
              ghost
              onClick={() => runAction("restart", api.restartOpenDesign, t.openDesign.restarted)}
              disabled={actionBusy !== null}
              prefix={<RefreshCw />}
            >
              {t.openDesign.restart}
            </Button>
            <Button
              ghost
              onClick={() => runAction("open", api.openOpenDesignUi, t.openDesign.opened)}
              disabled={actionBusy !== null}
              prefix={<ExternalLink />}
            >
              {t.openDesign.openUi}
            </Button>
          </div>

          {status?.last_error ? (
            <div className="md:col-span-2 xl:col-span-4 rounded border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
              {status.last_error}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t.openDesign.connection}</CardTitle>
          <CardDescription>{t.openDesign.connectionDescription}</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 lg:grid-cols-2">
          <div className="grid gap-2">
            <Label htmlFor="od-repo">{t.openDesign.repoPath}</Label>
            <Input id="od-repo" value={config.repo_path} onChange={(e) => updateField("repo_path", e.target.value)} />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="od-command">{t.openDesign.startCommand}</Label>
            <Input id="od-command" value={config.start_command} onChange={(e) => updateField("start_command", e.target.value)} />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="od-ui-url">{t.openDesign.uiUrl}</Label>
            <Input id="od-ui-url" value={config.ui_url} onChange={(e) => updateField("ui_url", e.target.value)} />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="od-health-url">{t.openDesign.healthUrl}</Label>
            <Input id="od-health-url" value={config.health_url} onChange={(e) => updateField("health_url", e.target.value)} />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="od-workspace">{t.openDesign.workspaceDir}</Label>
            <Input id="od-workspace" value={config.workspace_dir} onChange={(e) => updateField("workspace_dir", e.target.value)} />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="od-artifacts">{t.openDesign.artifactsDir}</Label>
            <Input id="od-artifacts" value={config.artifacts_dir} onChange={(e) => updateField("artifacts_dir", e.target.value)} />
          </div>
          <div className="lg:col-span-2 flex justify-end">
            <Button onClick={saveConfig} disabled={saving} prefix={<Save />}>
              {saving ? t.common.saving : t.common.save}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t.openDesign.briefTitle}</CardTitle>
          <CardDescription>{t.openDesign.briefDescription}</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div className="grid gap-2 md:grid-cols-2">
            <div className="grid gap-2">
              <Label htmlFor="od-project-name">{t.openDesign.projectName}</Label>
              <Input id="od-project-name" value={projectName} onChange={(e) => setProjectName(e.target.value)} />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="od-skill">{t.openDesign.skillLabel}</Label>
              <Input id="od-skill" value={skill} onChange={(e) => setSkill(e.target.value)} />
            </div>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="od-brief">{t.openDesign.briefLabel}</Label>
            <textarea
              id="od-brief"
              className="flex min-h-[140px] w-full border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              placeholder={t.openDesign.briefPlaceholder}
              value={brief}
              onChange={(e) => setBrief(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="od-design-system">{t.openDesign.designSystem}</Label>
            <textarea
              id="od-design-system"
              className="flex min-h-[100px] w-full border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              placeholder={t.openDesign.designSystemPlaceholder}
              value={designSystem}
              onChange={(e) => setDesignSystem(e.target.value)}
            />
          </div>
          <div className="flex justify-end">
            <Button onClick={pushBrief} disabled={pushing} prefix={<Upload />}>
              {pushing ? t.openDesign.pushing : t.openDesign.pushBrief}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t.openDesign.artifactsTitle}</CardTitle>
          <CardDescription>{t.openDesign.artifactsDescription}</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3">
          {artifacts.length === 0 ? (
            <div className="text-sm text-muted-foreground">{t.openDesign.noArtifacts}</div>
          ) : (
            artifacts.map((artifact) => (
              <div
                key={artifact.path}
                className="flex flex-col gap-3 rounded border border-border p-3 md:flex-row md:items-center md:justify-between"
              >
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium">{artifact.name}</div>
                  <div className="truncate text-xs text-muted-foreground">{artifact.relative_path}</div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    {formatBytes(artifact.size)} · {new Date(artifact.mtime * 1000).toLocaleString()}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <a href={api.getOpenDesignArtifactUrl(artifact.path)} target="_blank" rel="noreferrer">
                    <Button ghost size="sm" prefix={<FolderOpen />}>
                      {t.openDesign.download}
                    </Button>
                  </a>
                </div>
              </div>
            ))
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t.openDesign.environmentTitle}</CardTitle>
          <CardDescription>{t.openDesign.environmentDescription}</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-2">
          {envPairs.length === 0 ? (
            <div className="text-sm text-muted-foreground">{t.openDesign.noEnv}</div>
          ) : (
            envPairs.map(([key, value]) => (
              <div key={key} className="grid gap-1 rounded border border-border p-3">
                <div className="font-mono-ui text-xs">{key}</div>
                <div className="break-all text-xs text-muted-foreground">{value}</div>
              </div>
            ))
          )}
        </CardContent>
      </Card>

      <PluginSlot name="open-design:bottom" />
    </div>
  );
}
