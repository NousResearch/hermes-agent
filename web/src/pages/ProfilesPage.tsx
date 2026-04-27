import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Download,
  Pencil,
  Plus,
  RotateCw,
  Settings2,
  Trash2,
  Upload,
  Users,
} from "lucide-react";
import { H2 } from "@nous-research/ui";
import { api } from "@/lib/api";
import type { ProfileInfo } from "@/lib/api";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { useToast } from "@/hooks/useToast";
import { useConfirmDelete } from "@/hooks/useConfirmDelete";
import { Toast } from "@/components/Toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectOption } from "@/components/ui/select";
import { Segmented } from "@/components/ui/segmented";
import { useI18n } from "@/i18n";

// Mirrors hermes_cli/profiles.py::_PROFILE_ID_RE so we can reject obviously
// invalid names (uppercase, spaces, …) before round-tripping a doomed
// PATCH/POST request and burning a toast cycle.
const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

// Friendly display names for env keys returned in shared_tokens — keep in
// sync with the `_EXCLUSIVE_TOKEN_ENV_KEYS` tuple in hermes_cli/web_server.py.
// These are user-facing brand names; deliberately not translated.
const TOKEN_PLATFORM_LABEL: Record<string, string> = {
  WEIXIN_TOKEN: "WeChat",
  TELEGRAM_BOT_TOKEN: "Telegram",
  DISCORD_BOT_TOKEN: "Discord",
};

export default function ProfilesPage() {
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [active, setActive] = useState<string>("default");
  const [loading, setLoading] = useState(true);
  const { toast, showToast } = useToast();
  const { t } = useI18n();

  // Create form. ``cloneSource === ""`` means a blank profile is created; any
  // other value is the explicit source profile to clone from. ``copyMode``
  // is irrelevant (and hidden from the UI) when the source is blank.
  const [newName, setNewName] = useState("");
  const [cloneSource, setCloneSource] = useState<string>("");
  const [copyMode, setCopyMode] = useState<"config" | "all">("config");
  const [creating, setCreating] = useState(false);

  // Rename state
  const [renamingFrom, setRenamingFrom] = useState<string | null>(null);
  const [renameTo, setRenameTo] = useState("");

  // Import state
  const [importPath, setImportPath] = useState("");
  const [importName, setImportName] = useState("");
  const [importing, setImporting] = useState(false);

  // Per-profile edit panel: which profile is expanded + cached form state.
  const [editingName, setEditingName] = useState<string | null>(null);
  const [soulText, setSoulText] = useState("");
  const [soulSaving, setSoulSaving] = useState(false);
  const [modelDraft, setModelDraft] = useState<{ model: string; provider: string }>({
    model: "",
    provider: "",
  });
  const [modelSaving, setModelSaving] = useState(false);

  const openEditor = useCallback(
    async (name: string) => {
      // Toggle off if clicking the already-open row.
      if (editingName === name) {
        setEditingName(null);
        return;
      }
      setEditingName(name);
      setSoulText("");
      setModelDraft({ model: "", provider: "" });
      try {
        const [soul, model] = await Promise.all([
          api.getProfileSoul(name),
          api.getProfileModel(name),
        ]);
        setSoulText(soul.content);
        setModelDraft({
          model: model.model ?? "",
          provider: model.provider ?? "",
        });
      } catch (e) {
        showToast(`${t.status.error}: ${e}`, "error");
      }
    },
    [editingName, showToast, t.status.error],
  );

  const handleSaveSoul = async (name: string) => {
    setSoulSaving(true);
    try {
      await api.updateProfileSoul(name, soulText);
      showToast(`${t.profiles.soulSaved}: ${name}`, "success");
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    } finally {
      setSoulSaving(false);
    }
  };

  const handleSaveModel = async (name: string) => {
    setModelSaving(true);
    try {
      const res = await api.updateProfileModel(name, {
        model: modelDraft.model.trim() || null,
        provider: modelDraft.provider.trim() || null,
      });
      showToast(`${t.profiles.modelSaved}: ${name}`, "success");
      // Reflect normalised values back so the row's badge / display refresh.
      setModelDraft({ model: res.model ?? "", provider: res.provider ?? "" });
      load();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    } finally {
      setModelSaving(false);
    }
  };

  const load = useCallback(() => {
    api
      .getProfiles()
      .then((res) => {
        setProfiles(res.profiles);
        setActive(res.active);
      })
      .catch((e) => showToast(`${t.status.error}: ${e}`, "error"))
      .finally(() => setLoading(false));
  }, [showToast, t.status.error]);

  useEffect(() => {
    load();
  }, [load]);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) {
      showToast(t.profiles.nameRequired, "error");
      return;
    }
    if (!PROFILE_NAME_RE.test(name)) {
      showToast(`${t.profiles.invalidName}: ${t.profiles.nameRule}`, "error");
      return;
    }
    setCreating(true);
    try {
      const hasSource = cloneSource !== "";
      await api.createProfile({
        name,
        clone_from: hasSource ? cloneSource : undefined,
        clone_all: hasSource && copyMode === "all",
        // Setting clone_config when clone_from is given is redundant — the
        // backend already copies config files whenever a source is set — but
        // we forward it for symmetry with the CLI flags.
        clone_config: hasSource && copyMode === "config",
      });
      showToast(`${t.profiles.created}: ${name}`, "success");
      setNewName("");
      setCloneSource("");
      setCopyMode("config");
      load();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    } finally {
      setCreating(false);
    }
  };

  // Profiles whose gateway-restart request is currently in flight. Drives the
  // spinner + disabled state on the row's restart button so the user gets
  // immediate visual feedback even when the toast is missed (the toast
  // auto-dismisses after 3s, the spinner stays until the post-restart reload
  // returns).
  const [restartingNames, setRestartingNames] = useState<Set<string>>(new Set());

  const handleRestartGateway = async (name: string) => {
    setRestartingNames((s) => new Set(s).add(name));
    try {
      await api.restartProfileGateway(name);
      showToast(`${t.profiles.gatewayRestarting}: ${name}`, "success");
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
      setRestartingNames((s) => {
        const next = new Set(s);
        next.delete(name);
        return next;
      });
      return;
    }

    // Poll the listing until the gateway shows running or we time out.
    // Spawn returns immediately but the gateway process itself can take
    // 5–15 s to write gateway.pid (clean shutdown + service registration
    // + init), so a single 3 s reload routinely caught the in-between
    // state and looked like the restart silently failed.
    const POLL_INTERVAL = 2000;
    const MAX_POLLS = 8; // ~16s ceiling
    let polls = 0;
    const clearSpinner = () =>
      setRestartingNames((s) => {
        const next = new Set(s);
        next.delete(name);
        return next;
      });

    const poll = async () => {
      polls += 1;
      try {
        const res = await api.getProfiles();
        setProfiles(res.profiles);
        setActive(res.active);
        const target = res.profiles.find((p) => p.name === name);
        if (target?.gateway_running) {
          clearSpinner();
          return;
        }
      } catch {
        /* swallow — keep polling, spinner stays visible */
      }
      if (polls >= MAX_POLLS) {
        clearSpinner();
        // Spawn returned 200 but the gateway never reported running. Most
        // commonly this is a startup conflict — Weixin/Telegram token in
        // use by another profile, missing optional dependency, missing API
        // keys, or a port collision. Point the user at the log instead of
        // silently clearing the spinner so it doesn't look like nothing
        // happened.
        showToast(
          `${t.profiles.gatewayDidNotStart}: ${name} — ${t.profiles.checkLog}`,
          "error",
        );
        return;
      }
      setTimeout(poll, POLL_INTERVAL);
    };

    setTimeout(poll, POLL_INTERVAL);
  };

  const handleRenameSubmit = async () => {
    if (!renamingFrom) return;
    const target = renameTo.trim();
    if (!target || target === renamingFrom) {
      setRenamingFrom(null);
      setRenameTo("");
      return;
    }
    if (!PROFILE_NAME_RE.test(target)) {
      // Keep the inline editor open so the user can fix the name in place.
      showToast(`${t.profiles.invalidName}: ${t.profiles.nameRule}`, "error");
      return;
    }
    try {
      await api.renameProfile(renamingFrom, target);
      showToast(`${t.profiles.renamed}: ${renamingFrom} → ${target}`, "success");
      setRenamingFrom(null);
      setRenameTo("");
      load();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    }
  };

  const handleExport = async (name: string) => {
    try {
      const res = await api.exportProfile(name);
      showToast(`${t.profiles.exported}: ${res.path}`, "success");
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    }
  };

  const handleImport = async () => {
    const path = importPath.trim();
    if (!path) {
      showToast(t.profiles.archivePathRequired, "error");
      return;
    }
    setImporting(true);
    try {
      const res = await api.importProfile(path, importName.trim() || undefined);
      showToast(`${t.profiles.imported}: ${res.name}`, "success");
      setImportPath("");
      setImportName("");
      load();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    } finally {
      setImporting(false);
    }
  };

  const profileDelete = useConfirmDelete<string>({
    onDelete: useCallback(
      async (name: string) => {
        try {
          await api.deleteProfile(name);
          showToast(`${t.profiles.deleted}: ${name}`, "success");
          load();
        } catch (e) {
          showToast(`${t.status.error}: ${e}`, "error");
          throw e;
        }
      },
      [load, showToast, t.profiles.deleted, t.status.error],
    ),
  });

  const pendingName = profileDelete.pendingId;
  const namedProfiles = profiles.filter((p) => !p.is_default);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    // The app shell sets ``uppercase`` on every page by default; override it
    // here because profile names, model slugs, and paths are case-sensitive
    // — rendering ``gf`` as ``GF`` falsely suggests uppercase is allowed.
    // Children that explicitly opt into ``uppercase`` (Badges, the Segmented
    // control, our small section headers) still apply it to themselves.
    <div className="flex flex-col gap-6 normal-case">
      <Toast toast={toast} />

      <DeleteConfirmDialog
        open={profileDelete.isOpen}
        onCancel={profileDelete.cancel}
        onConfirm={profileDelete.confirm}
        title={t.profiles.confirmDeleteTitle}
        description={
          pendingName
            ? t.profiles.confirmDeleteMessage.replace("{name}", pendingName)
            : t.profiles.confirmDeleteMessage
        }
        loading={profileDelete.isDeleting}
      />


      {/* Create new profile */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Plus className="h-4 w-4" />
            {t.profiles.newProfile}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            <div className="grid gap-2">
              <Label htmlFor="profile-name">{t.profiles.name}</Label>
              <Input
                id="profile-name"
                placeholder={t.profiles.namePlaceholder}
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                aria-invalid={
                  newName.trim() !== "" &&
                  !PROFILE_NAME_RE.test(newName.trim())
                }
              />
              <p className="text-xs text-muted-foreground">
                {t.profiles.nameRule}
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="grid gap-2">
                <Label htmlFor="profile-clone-source">
                  {t.profiles.cloneSource}
                </Label>
                <Select
                  id="profile-clone-source"
                  value={cloneSource}
                  onValueChange={(v) => setCloneSource(v)}
                >
                  <SelectOption value="">
                    {t.profiles.cloneSourceBlank}
                  </SelectOption>
                  {/* Wrap the .map() output in a Fragment so the Select's
                      flattenChildren walks props.children to find these
                      options — it doesn't recurse into bare arrays. */}
                  <>
                    {profiles.map((p) => (
                      <SelectOption key={p.name} value={p.name}>
                        {p.name}
                      </SelectOption>
                    ))}
                  </>
                </Select>
              </div>

              <div className="flex items-end">
                <Button
                  onClick={handleCreate}
                  disabled={creating}
                  className="w-full"
                >
                  <Plus className="h-3 w-3" />
                  {creating ? t.common.creating : t.common.create}
                </Button>
              </div>
            </div>

            {cloneSource !== "" && (
              <div className="grid gap-2">
                <Label>{t.profiles.copyMode}</Label>
                <Segmented<"config" | "all">
                  size="md"
                  value={copyMode}
                  onChange={setCopyMode}
                  options={[
                    { value: "config", label: t.profiles.cloneModeConfig },
                    { value: "all", label: t.profiles.cloneModeAll },
                  ]}
                />
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Import profile UI is hidden for now — needs UX work to handle
          server-side path entry and (eventually) browser file upload before
          it's safe to surface. Backend endpoint /api/profiles/import is
          still wired up, just no entry from this page. */}
      {/* eslint-disable-next-line no-constant-binary-expression */}
      {false && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Upload className="h-4 w-4" />
              {t.profiles.importTitle}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4">
              <div className="grid gap-2">
                <Label htmlFor="profile-import-path">
                  {t.profiles.archivePath}
                </Label>
                <Input
                  id="profile-import-path"
                  placeholder={t.profiles.archivePathPlaceholder}
                  value={importPath}
                  onChange={(e) => setImportPath(e.target.value)}
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="grid gap-2 sm:col-span-2">
                  <Label htmlFor="profile-import-name">
                    {t.profiles.importNameOptional}
                  </Label>
                  <Input
                    id="profile-import-name"
                    placeholder={t.profiles.importNamePlaceholder}
                    value={importName}
                    onChange={(e) => setImportName(e.target.value)}
                  />
                </div>
                <div className="flex items-end">
                  <Button
                    onClick={handleImport}
                    disabled={importing}
                    className="w-full"
                  >
                    <Upload className="h-3 w-3" />
                    {importing ? t.common.loading : t.profiles.importAction}
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Profiles list */}
      <div className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <Users className="h-4 w-4" />
          {t.profiles.allProfiles} ({profiles.length})
        </H2>

        {profiles.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              {t.profiles.noProfiles}
            </CardContent>
          </Card>
        )}

        {profiles.map((p) => {
          const isActive = p.name === active;
          const isRenaming = renamingFrom === p.name;
          const isEditing = editingName === p.name;
          return (
            <Card key={p.name}>
              <CardContent className="flex items-center gap-4 py-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    {isRenaming ? (
                      <Input
                        autoFocus
                        value={renameTo}
                        onChange={(e) => setRenameTo(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleRenameSubmit();
                          if (e.key === "Escape") setRenamingFrom(null);
                        }}
                        aria-invalid={
                          renameTo.trim() !== "" &&
                          renameTo.trim() !== p.name &&
                          !PROFILE_NAME_RE.test(renameTo.trim())
                        }
                        className="max-w-xs"
                      />
                    ) : (
                      <span className="font-medium text-sm truncate">
                        {p.name}
                      </span>
                    )}
                    {p.is_default && (
                      <Badge variant="secondary">{t.profiles.defaultBadge}</Badge>
                    )}
                    {isActive && (
                      <Badge variant="success">{t.profiles.activeBadge}</Badge>
                    )}
                    <Badge
                      variant={p.gateway_running ? "success" : "outline"}
                      className={
                        p.gateway_running
                          ? ""
                          : "text-foreground/80 border-foreground/30"
                      }
                    >
                      <span
                        aria-hidden
                        className={
                          "mr-1 inline-block h-2 w-2 rounded-full " +
                          (p.gateway_running
                            ? "bg-emerald-400"
                            : "bg-foreground/40")
                        }
                      />
                      {p.gateway_running
                        ? t.profiles.gatewayRunning
                        : t.profiles.gatewayStopped}
                    </Badge>
                    {p.has_env && (
                      <Badge variant="outline">{t.profiles.hasEnv}</Badge>
                    )}
                    {p.shared_tokens.map((s) => {
                      const label =
                        TOKEN_PLATFORM_LABEL[s.key] ?? s.key;
                      return (
                        <Badge
                          key={s.key}
                          variant="destructive"
                          title={`${s.key} — ${t.profiles.sharedWith} ${s.with.join(", ")}`}
                        >
                          <AlertTriangle
                            aria-hidden
                            className="mr-1 h-3 w-3"
                          />
                          {`${t.profiles.tokenConflict} (${label})`}
                        </Badge>
                      );
                    })}
                  </div>
                  {isRenaming &&
                    (() => {
                      const trimmed = renameTo.trim();
                      const invalid =
                        trimmed !== "" &&
                        trimmed !== p.name &&
                        !PROFILE_NAME_RE.test(trimmed);
                      return (
                        <p
                          className={
                            "text-xs mb-1 " +
                            (invalid
                              ? "text-destructive"
                              : "text-muted-foreground")
                          }
                        >
                          {invalid
                            ? `${t.profiles.invalidName}: ${t.profiles.nameRule}`
                            : t.profiles.nameRule}
                        </p>
                      );
                    })()}
                  <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
                    {p.model && (
                      <span>
                        {t.profiles.model}: {p.model}
                        {p.provider ? ` (${p.provider})` : ""}
                      </span>
                    )}
                    <span>
                      {t.profiles.skills}: {p.skill_count}
                    </span>
                    <span className="font-mono truncate max-w-[28rem]">
                      {p.path}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  {isRenaming ? (
                    <>
                      <Button
                        size="sm"
                        variant="default"
                        onClick={handleRenameSubmit}
                      >
                        {t.common.save}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setRenamingFrom(null)}
                      >
                        {t.common.cancel}
                      </Button>
                    </>
                  ) : (
                    <>
                      <Button
                        variant="ghost"
                        size="icon"
                        title={t.profiles.restartGateway}
                        aria-label={t.profiles.restartGateway}
                        disabled={restartingNames.has(p.name)}
                        onClick={() => handleRestartGateway(p.name)}
                      >
                        <RotateCw
                          className={
                            "h-4 w-4 " +
                            (restartingNames.has(p.name) ? "animate-spin" : "")
                          }
                        />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        title={t.profiles.editConfig}
                        aria-label={t.profiles.editConfig}
                        onClick={() => openEditor(p.name)}
                      >
                        {isEditing ? (
                          <ChevronDown className="h-4 w-4" />
                        ) : (
                          <Settings2 className="h-4 w-4" />
                        )}
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        title={t.profiles.exportAction}
                        aria-label={t.profiles.exportAction}
                        onClick={() => handleExport(p.name)}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                      {!p.is_default && (
                        <Button
                          variant="ghost"
                          size="icon"
                          title={t.profiles.rename}
                          aria-label={t.profiles.rename}
                          onClick={() => {
                            setRenamingFrom(p.name);
                            setRenameTo(p.name);
                          }}
                        >
                          <Pencil className="h-4 w-4" />
                        </Button>
                      )}
                      {!p.is_default && (
                        <Button
                          variant="ghost"
                          size="icon"
                          title={t.common.delete}
                          aria-label={t.common.delete}
                          onClick={() => profileDelete.requestDelete(p.name)}
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      )}
                    </>
                  )}
                </div>
              </CardContent>

              {isEditing && (
                <div className="border-t border-border px-4 pb-4 pt-3 flex flex-col gap-4">
                  <div className="grid gap-2">
                    <Label className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
                      <ChevronRight className="h-3 w-3" />
                      {t.profiles.modelSection}
                    </Label>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div className="grid gap-1 sm:col-span-2">
                        <Label htmlFor={`profile-${p.name}-model`}>
                          {t.profiles.modelSlug}
                        </Label>
                        <Input
                          id={`profile-${p.name}-model`}
                          placeholder={t.profiles.modelSlugPlaceholder}
                          value={modelDraft.model}
                          onChange={(e) =>
                            setModelDraft({ ...modelDraft, model: e.target.value })
                          }
                        />
                      </div>
                      <div className="grid gap-1">
                        <Label htmlFor={`profile-${p.name}-provider`}>
                          {t.profiles.modelProvider}
                        </Label>
                        <Input
                          id={`profile-${p.name}-provider`}
                          placeholder={t.profiles.modelProviderPlaceholder}
                          value={modelDraft.provider}
                          onChange={(e) =>
                            setModelDraft({
                              ...modelDraft,
                              provider: e.target.value,
                            })
                          }
                        />
                      </div>
                    </div>
                    <div>
                      <Button
                        size="sm"
                        onClick={() => handleSaveModel(p.name)}
                        disabled={modelSaving}
                      >
                        {modelSaving ? t.common.saving : t.profiles.saveModel}
                      </Button>
                    </div>
                  </div>

                  <div className="grid gap-2">
                    <Label className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
                      <ChevronRight className="h-3 w-3" />
                      {t.profiles.soulSection}
                    </Label>
                    <textarea
                      className="flex min-h-[180px] w-full border border-input bg-transparent px-3 py-2 text-sm font-mono shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      placeholder={t.profiles.soulPlaceholder}
                      value={soulText}
                      onChange={(e) => setSoulText(e.target.value)}
                    />
                    <div>
                      <Button
                        size="sm"
                        onClick={() => handleSaveSoul(p.name)}
                        disabled={soulSaving}
                      >
                        {soulSaving ? t.common.saving : t.profiles.saveSoul}
                      </Button>
                    </div>
                  </div>
                </div>
              )}
            </Card>
          );
        })}

        {namedProfiles.length === 0 && profiles.length > 0 && (
          <p className="text-xs text-muted-foreground">
            {t.profiles.onlyDefaultHint}
          </p>
        )}
      </div>
    </div>
  );
}
