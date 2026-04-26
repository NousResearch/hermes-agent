import { useCallback, useEffect, useState } from "react";
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Download,
  Pencil,
  Plus,
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
import { useI18n } from "@/i18n";

export default function ProfilesPage() {
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [active, setActive] = useState<string>("default");
  const [loading, setLoading] = useState(true);
  const { toast, showToast } = useToast();
  const { t } = useI18n();

  // Create form
  const [newName, setNewName] = useState("");
  const [cloneFrom, setCloneFrom] = useState<string>("");
  const [cloneMode, setCloneMode] = useState<"none" | "config" | "all">("none");
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
    setCreating(true);
    try {
      await api.createProfile({
        name,
        clone_from: cloneFrom || undefined,
        clone_all: cloneMode === "all",
        clone_config: cloneMode === "config",
      });
      showToast(`${t.profiles.created}: ${name}`, "success");
      setNewName("");
      setCloneFrom("");
      setCloneMode("none");
      load();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    } finally {
      setCreating(false);
    }
  };

  const handleActivate = async (name: string) => {
    try {
      await api.activateProfile(name);
      showToast(`${t.profiles.activated}: ${name}`, "success");
      load();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    }
  };

  const handleRenameSubmit = async () => {
    if (!renamingFrom) return;
    const target = renameTo.trim();
    if (!target || target === renamingFrom) {
      setRenamingFrom(null);
      setRenameTo("");
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
    <div className="flex flex-col gap-6">
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
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="grid gap-2">
                <Label htmlFor="profile-clone-from">
                  {t.profiles.cloneFrom}
                </Label>
                <Select
                  id="profile-clone-from"
                  value={cloneFrom}
                  onValueChange={(v) => setCloneFrom(v)}
                >
                  <SelectOption value="">
                    {t.profiles.cloneFromNone}
                  </SelectOption>
                  {profiles.map((p) => (
                    <SelectOption key={p.name} value={p.name}>
                      {p.name}
                    </SelectOption>
                  ))}
                </Select>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="profile-clone-mode">
                  {t.profiles.cloneMode}
                </Label>
                <Select
                  id="profile-clone-mode"
                  value={cloneMode}
                  onValueChange={(v) =>
                    setCloneMode(v as "none" | "config" | "all")
                  }
                >
                  <SelectOption value="none">
                    {t.profiles.cloneModeNone}
                  </SelectOption>
                  <SelectOption value="config">
                    {t.profiles.cloneModeConfig}
                  </SelectOption>
                  <SelectOption value="all">
                    {t.profiles.cloneModeAll}
                  </SelectOption>
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
          </div>
        </CardContent>
      </Card>

      {/* Import profile */}
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
                    {p.gateway_running && (
                      <Badge variant="outline">{t.profiles.gatewayRunning}</Badge>
                    )}
                    {p.has_env && (
                      <Badge variant="outline">{t.profiles.hasEnv}</Badge>
                    )}
                  </div>
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
                      {!isActive && (
                        <Button
                          variant="ghost"
                          size="icon"
                          title={t.profiles.activate}
                          aria-label={t.profiles.activate}
                          onClick={() => handleActivate(p.name)}
                        >
                          <CheckCircle2 className="h-4 w-4" />
                        </Button>
                      )}
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
