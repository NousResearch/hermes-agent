import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import {
  ChevronDown,
  Pencil,
  Terminal,
  Trash2,
  Users,
  X,
} from "lucide-react";
import spinners from "unicode-animations";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type { ProfileInfo } from "@/lib/api";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Checkbox } from "@nous-research/ui/ui/components/checkbox";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn, themedBody } from "@/lib/utils";

// Mirrors hermes_cli/profiles.py::_PROFILE_ID_RE so we can reject obviously
// invalid names (uppercase, spaces, …) before round-tripping a doomed POST.
const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

/** Braille unicode spinner (`unicode-animations`); static first frame when reduced motion is preferred. */
function ProfilesLoadingSpinner() {
  const { frames, interval } = spinners.braille;
  const [frameIndex, setFrameIndex] = useState(0);

  useEffect(() => {
    if (
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      return;
    }
    const id = window.setInterval(
      () => setFrameIndex((i) => (i + 1) % frames.length),
      interval,
    );
    return () => window.clearInterval(id);
  }, [frames.length, interval]);

  return (
    <span
      aria-hidden
      className="inline-block select-none font-mono text-xl leading-none text-muted-foreground"
    >
      {frames[frameIndex]}
    </span>
  );
}

const PROFILE_ROLE_HINTS: Record<string, string> = {
  "factory-orchestrator": "Coordina goals, lanes y gates del Factory.",
  "product-analyst": "Convierte objetivos en PRD, alcance y criterios.",
  "solution-architect": "Diseña arquitectura, ADRs y decisiones técnicas.",
  "implementation-planner": "Baja la solución a historias, incrementos y plan ejecutable.",
  "claude-builder": "Implementador principal para cambios multi-archivo y refactors.",
  "codex-builder": "Implementador/reviewer rápido para fixes acotados y QA.",
  "openhands-lab": "Sandbox aislado para builds pesados y validación independiente.",
  "quality-reviewer": "Revisión de calidad, mantenibilidad y deuda técnica.",
  "security-reviewer": "Revisión de seguridad, secretos y riesgos operativos.",
  "qa-verifier": "Verifica pruebas, regresiones y evidencia ejecutable.",
  "devops-release": "Deploy, runtime, infraestructura y release gates.",
  "factory-reporter": "Reportes, snapshots y síntesis de avance.",
};

function profileDisplayName(name: string): string {
  if (name === "default") return "Default Hermes";
  return name
    .split(/[-_]/g)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function profileInitials(name: string): string {
  const parts = name.split(/[-_]/g).filter(Boolean);
  return (parts[0]?.[0] ?? "H") + (parts[1]?.[0] ?? "");
}

function profileSummary(profile: ProfileInfo): string {
  if (PROFILE_ROLE_HINTS[profile.name]) return PROFILE_ROLE_HINTS[profile.name];
  if (profile.name === "default") return "Perfil base del dashboard y sesión principal.";
  return profile.model
    ? `Perfil Hermes con ${profile.model}${profile.provider ? ` vía ${profile.provider}` : ""}.`
    : "Perfil Hermes aislado con configuración y memoria propias.";
}

function profileAvatarUrl(name: string): string {
  return `/agent-avatars/${encodeURIComponent(name)}.webp`;
}

export default function ProfilesPage() {
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const { toast, showToast } = useToast();
  const { t } = useI18n();
  const { setEnd } = usePageHeader();

  // Create modal
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [cloneFromDefault, setCloneFromDefault] = useState(true);
  const [creating, setCreating] = useState(false);
  const closeCreateModal = useCallback(() => setCreateModalOpen(false), []);
  const createModalRef = useModalBehavior({
    open: createModalOpen,
    onClose: closeCreateModal,
  });

  // Inline rename state
  const [renamingFrom, setRenamingFrom] = useState<string | null>(null);
  const [renameTo, setRenameTo] = useState("");

  // Inline SOUL editor state
  const [editingSoulFor, setEditingSoulFor] = useState<string | null>(null);
  const [soulText, setSoulText] = useState("");
  const [soulSaving, setSoulSaving] = useState(false);
  // Tracks the latest SOUL request so out-of-order responses don't overwrite
  // newer state when the user switches profiles or closes the editor.
  const activeSoulRequest = useRef<string | null>(null);

  const load = useCallback(() => {
    api
      .getProfiles()
      .then((res) => setProfiles(res.profiles))
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
      await api.createProfile({ name, clone_from_default: cloneFromDefault });
      showToast(`${t.profiles.created}: ${name}`, "success");
      setNewName("");
      setCreateModalOpen(false);
      load();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    } finally {
      setCreating(false);
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
    if (!PROFILE_NAME_RE.test(target)) {
      showToast(`${t.profiles.invalidName}: ${t.profiles.nameRule}`, "error");
      return;
    }
    try {
      await api.renameProfile(renamingFrom, target);
      showToast(
        `${t.profiles.renamed}: ${renamingFrom} → ${target}`,
        "success",
      );
      setRenamingFrom(null);
      setRenameTo("");
      load();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    }
  };

  const openSoulEditor = useCallback(
    async (name: string) => {
      if (editingSoulFor === name) {
        activeSoulRequest.current = null;
        setEditingSoulFor(null);
        return;
      }
      setEditingSoulFor(name);
      setSoulText("");
      activeSoulRequest.current = name;
      try {
        const soul = await api.getProfileSoul(name);
        if (activeSoulRequest.current === name) {
          setSoulText(soul.content);
        }
      } catch (e) {
        if (activeSoulRequest.current === name) {
          showToast(`${t.status.error}: ${e}`, "error");
        }
      }
    },
    [editingSoulFor, showToast, t.status.error],
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

  const handleCopyTerminalCommand = async (name: string) => {
    let cmd: string;
    try {
      const res = await api.getProfileSetupCommand(name);
      cmd = res.command;
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
      return;
    }
    try {
      await navigator.clipboard.writeText(cmd);
      showToast(`${t.profiles.commandCopied}: ${cmd}`, "success");
    } catch {
      showToast(`${t.profiles.copyFailed}: ${cmd}`, "error");
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

  // Put "Create" button in page header
  useLayoutEffect(() => {
    setEnd(
      <Button
        className="uppercase"
        size="sm"
        onClick={() => setCreateModalOpen(true)}
      >
        {t.common.create}
      </Button>,
    );
    return () => {
      setEnd(null);
    };
  }, [setEnd, t.common.create, loading]);

  if (loading) {
    return (
      <div
        aria-busy="true"
        aria-live="polite"
        className="flex items-center justify-center py-24"
      >
        <span className="sr-only">{t.common.loading}</span>

        <ProfilesLoadingSpinner />
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

      {/* Create profile modal */}
      {createModalOpen && (
        <div
          ref={createModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
          onClick={(e) =>
            e.target === e.currentTarget && setCreateModalOpen(false)
          }
          role="dialog"
          aria-modal="true"
          aria-labelledby="create-profile-title"
        >
          <div className={cn(themedBody, "relative w-full max-w-md border border-border bg-card shadow-2xl flex flex-col")}>
            <Button
              ghost
              size="icon"
              onClick={() => setCreateModalOpen(false)}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label="Close"
            >
              <X />
            </Button>

            <header className="p-5 pb-3 border-b border-border">
              <h2
                id="create-profile-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                {t.profiles.newProfile}
              </h2>
            </header>

            <div className="p-5 grid gap-4">
              <div className="grid gap-2">
                <Label htmlFor="profile-name">{t.profiles.name}</Label>
                <Input
                  id="profile-name"
                  autoFocus
                  placeholder={t.profiles.namePlaceholder}
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleCreate();
                  }}
                  aria-invalid={
                    newName.trim() !== "" &&
                    !PROFILE_NAME_RE.test(newName.trim())
                  }
                />
                <p className="text-xs text-muted-foreground">
                  {t.profiles.nameRule}
                </p>
              </div>

              <div className="flex items-center gap-2.5">
                <Checkbox
                  checked={cloneFromDefault}
                  id="clone-from-default"
                  onCheckedChange={(checked) =>
                    setCloneFromDefault(checked === true)
                  }
                />

                <Label
                  className="font-mondwest normal-case tracking-normal text-sm cursor-pointer"
                  htmlFor="clone-from-default"
                >
                  {t.profiles.cloneFromDefault}
                </Label>
              </div>

              <div className="flex justify-end">
                <Button
                  className="uppercase"
                  size="sm"
                  onClick={handleCreate}
                  disabled={creating}
                >
                  {creating ? t.common.creating : t.common.create}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* List */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between gap-3">
          <H2
            variant="sm"
            className="flex items-center gap-2 text-muted-foreground"
          >
            <Users className="h-4 w-4" />
            {t.profiles.allProfiles} ({profiles.length})
          </H2>
          <p className="hidden text-xs text-muted-foreground md:block">
            Fichas visuales canónicas sobre los perfiles reales de Hermes.
          </p>
        </div>

        {profiles.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              {t.profiles.noProfiles}
            </CardContent>
          </Card>
        )}

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {profiles.map((p) => {
            const isRenaming = renamingFrom === p.name;
            const isEditingSoul = editingSoulFor === p.name;
            return (
              <Card
                key={p.name}
                className="group overflow-hidden border-border/80 bg-card/95 transition hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-xl hover:shadow-primary/5"
              >
                <CardContent className="flex min-h-[286px] flex-col p-0">
                  <div className="relative h-28 overflow-hidden border-b border-border bg-gradient-to-br from-primary/20 via-muted/30 to-background">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(255,255,255,0.22),transparent_32%),radial-gradient(circle_at_80%_0%,rgba(255,255,255,0.12),transparent_28%)]" />
                    <div className="absolute bottom-3 left-4 flex items-end gap-3">
                      <div className="relative flex h-20 w-20 items-center justify-center overflow-hidden rounded-2xl border border-border bg-background text-lg font-semibold uppercase shadow-lg">
                        <span className="absolute inset-0 flex items-center justify-center bg-primary/10 text-primary">
                          {profileInitials(p.name)}
                        </span>
                        <img
                          src={profileAvatarUrl(p.name)}
                          alt=""
                          className="relative h-full w-full object-cover"
                          onError={(event) => {
                            event.currentTarget.style.display = "none";
                          }}
                        />
                      </div>
                      <div className="mb-1 flex flex-wrap items-center gap-1.5">
                        {p.is_default && (
                          <Badge tone="secondary">{t.profiles.defaultBadge}</Badge>
                        )}
                        {p.has_env && <Badge tone="outline">{t.profiles.hasEnv}</Badge>}
                        {PROFILE_ROLE_HINTS[p.name] && <Badge tone="outline">Factory</Badge>}
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-1 flex-col gap-4 p-4">
                    <div className="min-w-0">
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
                        />
                      ) : (
                        <>
                          <h3 className="truncate text-lg font-semibold tracking-tight">
                            {profileDisplayName(p.name)}
                          </h3>
                          <p className="mt-0.5 truncate font-mono text-xs text-muted-foreground">
                            {p.name}
                          </p>
                        </>
                      )}

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
                                "mt-2 text-xs " +
                                (invalid ? "text-destructive" : "text-muted-foreground")
                              }
                            >
                              {invalid
                                ? `${t.profiles.invalidName}: ${t.profiles.nameRule}`
                                : t.profiles.nameRule}
                            </p>
                          );
                        })()}

                      <p className="mt-3 line-clamp-2 min-h-[2.5rem] text-sm text-muted-foreground">
                        {profileSummary(p)}
                      </p>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="rounded-lg border border-border bg-background/60 p-2">
                        <p className="uppercase tracking-[0.18em] text-muted-foreground">
                          {t.profiles.skills}
                        </p>
                        <p className="mt-1 text-base font-semibold">{p.skill_count}</p>
                      </div>
                      <div className="rounded-lg border border-border bg-background/60 p-2">
                        <p className="uppercase tracking-[0.18em] text-muted-foreground">
                          Provider
                        </p>
                        <p className="mt-1 truncate text-sm font-medium">
                          {p.provider || "—"}
                        </p>
                      </div>
                    </div>

                    {p.model && (
                      <div className="rounded-lg border border-border bg-muted/30 p-2 text-xs text-muted-foreground">
                        <span className="font-medium text-foreground">{t.profiles.model}:</span>{" "}
                        <span className="font-mono">{p.model}</span>
                      </div>
                    )}

                    <p className="truncate font-mono text-[11px] text-muted-foreground">
                      {p.path}
                    </p>

                    <div className="mt-auto flex items-center justify-between gap-2 border-t border-border pt-3">
                      {isRenaming ? (
                        <div className="flex items-center gap-2">
                          <Button size="sm" onClick={handleRenameSubmit}>
                            {t.common.save}
                          </Button>
                          <Button size="sm" ghost onClick={() => setRenamingFrom(null)}>
                            {t.common.cancel}
                          </Button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1">
                          <Button
                            ghost
                            size="icon"
                            title={t.profiles.editSoul}
                            aria-label={t.profiles.editSoul}
                            onClick={() => openSoulEditor(p.name)}
                          >
                            {isEditingSoul ? (
                              <ChevronDown className="h-4 w-4" />
                            ) : (
                              <span aria-hidden className="text-xs font-bold">S</span>
                            )}
                          </Button>
                          <Button
                            ghost
                            size="icon"
                            title={t.profiles.openInTerminal}
                            aria-label={t.profiles.openInTerminal}
                            onClick={() => handleCopyTerminalCommand(p.name)}
                          >
                            <Terminal className="h-4 w-4" />
                          </Button>
                          {!p.is_default && (
                            <Button
                              ghost
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
                        </div>
                      )}

                      {!isRenaming && !p.is_default && (
                        <Button
                          ghost
                          size="icon"
                          title={t.common.delete}
                          aria-label={t.common.delete}
                          onClick={() => profileDelete.requestDelete(p.name)}
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      )}
                    </div>
                  </div>
                </CardContent>

                {isEditingSoul && (
                  <div className="border-t border-border px-4 pb-4 pt-3 flex flex-col gap-2">
                    <Label
                      htmlFor={`soul-editor-${p.name}`}
                      className="flex items-center gap-2 font-mondwest text-display text-xs tracking-wider text-muted-foreground"
                    >
                      {t.profiles.soulSection}
                    </Label>
                    <textarea
                      id={`soul-editor-${p.name}`}
                      className="flex min-h-[180px] w-full border border-input bg-transparent px-3 py-2 text-sm font-mono shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      placeholder={t.profiles.soulPlaceholder}
                      value={soulText}
                      onChange={(e) => setSoulText(e.target.value)}
                    />
                    <div>
                      <Button
                        size="sm"
                        className="uppercase"
                        onClick={() => handleSaveSoul(p.name)}
                        disabled={soulSaving}
                      >
                        {soulSaving ? t.common.saving : t.common.save}
                      </Button>
                    </div>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
}
