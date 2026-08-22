import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useNavigate } from "react-router";
import { useProfileScope } from "@/contexts/useProfileScope";
import {
  AlignLeft,
  Check,
  ChevronDown,
  ChevronUp,
  Cpu,
  MoreVertical,
  Pencil,
  Package,
  Plus,
  Sparkles,
  Terminal,
  Trash2,
  Users,
  X,
} from "lucide-react";
import spinners from "unicode-animations";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type {
  ActiveProfileInfo,
  ProfileFallbackInfo,
  ProfileInfo,
} from "@/lib/api";
import { copyTextToClipboard } from "@/lib/clipboard";
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
import {
  Select,
  SelectOption,
} from "@nous-research/ui/ui/components/select";
import { Checkbox } from "@nous-research/ui/ui/components/checkbox";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn, themedBody } from "@/lib/utils";
import { EFFORT_OPTIONS, filterEffortOptions } from "@/lib/reasoning-effort";

// Mirrors hermes_cli/profiles.py::_PROFILE_ID_RE so we can reject obviously
// invalid names (uppercase, spaces, …) before round-tripping a doomed POST.
const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

type FallbackDraft = Omit<ProfileFallbackInfo, "source_index"> & {
  source_index: number | null;
};

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

/**
 * Per-card "⋯" actions menu. Holds every action for the profile (set active,
 * model, description, SOUL, copy command, rename, delete) so the card row stays
 * a single button. Mirrors the hand-rolled dropdown pattern used by ModelsPage's
 * "Use as" menu (button + absolute panel + outside-click close).
 */
function ProfileActionsMenu({
  isActive,
  isDefault,
  isEditingDesc,
  isEditingModel,
  isEditingSoul,
  labels,
  settingActive,
  onCopyCommand,
  onDelete,
  onEditDescription,
  onEditModel,
  onEditSoul,
  onManageSkills,
  onRename,
  onSetActive,
}: ProfileActionsMenuProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node | null;
      // Close only when the click lands outside *this* menu. Matching any
      // `[data-profile-actions]` would treat another card's menu as "inside"
      // and leave several menus open at once.
      if (target && !containerRef.current?.contains(target)) setOpen(false);
    };
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [open]);

  // Run the action, then collapse the menu. Toggle editors (model/description/
  // SOUL) expand the inline section below the card once the menu closes.
  const run = (fn: () => void) => () => {
    fn();
    setOpen(false);
  };

  const itemClass =
    "flex w-full items-center gap-2.5 px-3 py-2 text-xs uppercase tracking-wider hover:bg-muted/50 disabled:opacity-40";

  return (
    <div className="relative" data-profile-actions ref={containerRef}>
      <Button
        ghost
        size="icon"
        title={labels.actions}
        aria-label={labels.actions}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <MoreVertical className="h-4 w-4" />
      </Button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full z-50 mt-1 min-w-[200px] border border-border bg-card shadow-lg"
        >
          {!isActive && (
            <button
              type="button"
              role="menuitem"
              className={itemClass}
              disabled={settingActive}
              onClick={run(onSetActive)}
            >
              <Check className="h-4 w-4" />
              {labels.setActive}
            </button>
          )}

          <button
            type="button"
            role="menuitem"
            className={itemClass}
            onClick={run(onEditModel)}
          >
            {isEditingModel ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <Cpu className="h-4 w-4" />
            )}
            {labels.editModel}
          </button>

          <button
            type="button"
            role="menuitem"
            className={itemClass}
            onClick={run(onEditDescription)}
          >
            {isEditingDesc ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <AlignLeft className="h-4 w-4" />
            )}
            {labels.editDescription}
          </button>

          <button
            type="button"
            role="menuitem"
            className={itemClass}
            onClick={run(onEditSoul)}
          >
            {isEditingSoul ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <span aria-hidden className="w-4 text-center text-xs font-bold">
                S
              </span>
            )}
            {labels.editSoul}
          </button>

          <button
            type="button"
            role="menuitem"
            className={itemClass}
            onClick={run(onManageSkills)}
          >
            <Package className="h-4 w-4" />
            {labels.manageSkills}
          </button>

          <button
            type="button"
            role="menuitem"
            className={itemClass}
            onClick={run(onCopyCommand)}
          >
            <Terminal className="h-4 w-4" />
            {labels.openInTerminal}
          </button>

          {!isDefault && (
            <button
              type="button"
              role="menuitem"
              className={cn(itemClass, "border-t border-border/50")}
              onClick={run(onRename)}
            >
              <Pencil className="h-4 w-4" />
              {labels.rename}
            </button>
          )}

          {!isDefault && (
            <button
              type="button"
              role="menuitem"
              className={cn(itemClass, "text-destructive hover:bg-destructive/10")}
              onClick={run(onDelete)}
            >
              <Trash2 className="h-4 w-4" />
              {labels.delete}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default function ProfilesPage() {
  const navigate = useNavigate();
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [activeInfo, setActiveInfo] = useState<ActiveProfileInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const { toast, showToast } = useToast();
  const { t } = useI18n();
  const { setEnd } = usePageHeader();
  const { setProfile } = useProfileScope();

  // Locale strings with English fallbacks. The enriched keys are optional in
  // the i18n type so untranslated locales don't break the build — they render
  // the English literal until translated.
  const L = useMemo(() => {
    const p = t.profiles;
    return {
      activeProfile: p.activeProfile ?? "Active profile",
      activeBadge: p.activeBadge ?? "active",
      setActive: p.setActive ?? "Set as active",
      activeSet: p.activeSet ?? "Active profile set",
      gatewayRunning: p.gatewayRunning ?? "Gateway running",
      gatewayStopped: p.gatewayStopped ?? "Gateway stopped",
      gatewayRunningWarning:
        p.gatewayRunningWarning ??
        "This profile's gateway is running — it will be stopped.",
      aliasBadge: p.aliasBadge ?? "alias",
      description: p.description ?? "Description",
      descriptionPlaceholder:
        p.descriptionPlaceholder ??
        "What is this profile good at? Used to route kanban tasks by role.",
      noDescription: p.noDescription ?? "No description",
      editDescription: p.editDescription ?? "Edit description",
      descriptionSaved: p.descriptionSaved ?? "Description saved",
      reviewBadge: p.reviewBadge ?? "review",
      autoGenerate: p.autoGenerate ?? "Auto-generate",
      generating: p.generating ?? "Generating…",
      describeFailed: p.describeFailed ?? "Could not generate description",
      distribution: p.distribution ?? "Distribution",
      advancedOptions: p.advancedOptions ?? "Advanced options",
      cloneAll:
        p.cloneAll ?? "Clone everything (memories, sessions, skills, state)",
      noSkillsOption: p.noSkillsOption ?? "Don't seed bundled skills",
      descriptionOptional: p.descriptionOptional ?? "Description (optional)",
      modelOptional: p.modelOptional ?? "Model (optional)",
      modelInherit: p.modelInherit ?? "Inherit from clone / default",
      modelLoading: p.modelLoading ?? "Loading models…",
      modelNone:
        p.modelNone ?? "No authenticated providers — set a key first",
      editModel: p.editModel ?? "Change model",
      modelSaved: p.modelSaved ?? "Model updated",
      modelSelect: p.modelSelect ?? "Select a model",
      reasoningEffort: p.reasoningEffort ?? "Reasoning effort",
      reasoningInherit: p.reasoningInherit ?? "Inherit provider default",
      reasoningSaved: p.reasoningSaved ?? "Reasoning effort updated",
      reasoningUnset: p.reasoningUnset ?? "provider default",
      fallbacks: p.fallbacks ?? "Fallback chain",
      fallbackAdd: p.fallbackAdd ?? "Add fallback",
      fallbackRemove: p.fallbackRemove ?? "Remove fallback",
      fallbackMoveUp: p.fallbackMoveUp ?? "Move fallback up",
      fallbackMoveDown: p.fallbackMoveDown ?? "Move fallback down",
      fallbackSelect: p.fallbackSelect ?? "Select a fallback model",
      fallbackNone: p.fallbackNone ?? "No fallback models configured",
      fallbackLoadFailed:
        p.fallbackLoadFailed ??
        "Fallback chain could not be loaded; model and reasoning can still be saved.",
      fallbackSaved: p.fallbackSaved ?? "Fallback chain updated",
      fallbackCustomEndpoint:
        p.fallbackCustomEndpoint ?? "Custom endpoint retained",
      actions: p.actions ?? "Actions",
      manageSkills: p.manageSkills ?? "Manage skills & tools",
      activeSetHint:
        p.activeSetHint ??
        "Dashboard switched to manage {name}. New CLI/gateway runs will use this profile too.",
    };
  }, [t.profiles]);

  // Create modal
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [cloneFrom, setCloneFrom] = useState<string | null>("default");
  const [cloneAll, setCloneAll] = useState(false);
  const [noSkills, setNoSkills] = useState(false);
  const [newDescription, setNewDescription] = useState("");
  const [creating, setCreating] = useState(false);
  // Model picker (lazy-loaded the first time a picker is opened). modelChoice
  // is a "slug\u0000model" key, or "" to inherit from clone/default.
  const [modelChoices, setModelChoices] = useState<
    | {
        provider: string;
        model: string;
        label: string;
        reasoningLevels?: string[] | null;
      }[]
    | null
  >(null);
  // The model catalog is profile-scoped: provider/model availability can differ
  // between profiles because each profile owns its own credentials/config.
  const [modelChoicesScope, setModelChoicesScope] = useState<string | null>(null);
  const modelChoicesLoading = useRef<string | null>(null);
  const modelChoicesRequestGeneration = useRef(0);
  const [modelChoice, setModelChoice] = useState("");
  const closeCreateModal = useCallback(() => {
    modelChoicesRequestGeneration.current += 1;
    modelChoicesLoading.current = null;
    setCreateModalOpen(false);
  }, []);
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

  // Inline description editor state
  const [editingDescFor, setEditingDescFor] = useState<string | null>(null);
  const [descText, setDescText] = useState("");
  const [descSaving, setDescSaving] = useState(false);
  const [describing, setDescribing] = useState(false);
  // Tracks the latest description request (save / auto-describe) so a late
  // response can't overwrite state for a different, newly-opened editor.
  const activeDescRequest = useRef<string | null>(null);
  // Counts in-flight save / auto-describe requests so the saving indicator
  // is only cleared when the last concurrent request settles.
  const descSavingCount = useRef(0);
  const describingCount = useRef(0);

  // Inline model editor state
  const [editingModelFor, setEditingModelFor] = useState<string | null>(null);
  const [modelEditChoice, setModelEditChoice] = useState("");
  const [modelEditInitialChoice, setModelEditInitialChoice] = useState("");
  const [reasoningEditChoice, setReasoningEditChoice] = useState("");
  const [fallbackEdits, setFallbackEdits] = useState<FallbackDraft[]>([]);
  const [fallbacksLoading, setFallbacksLoading] = useState(false);
  const [fallbacksLoaded, setFallbacksLoaded] = useState(false);
  const [fallbacksLoadFailed, setFallbacksLoadFailed] = useState(false);
  const [modelSaving, setModelSaving] = useState(false);
  // Tracks the latest fallback-chain request so out-of-order responses don't
  // overwrite state when the user switches profiles or closes the editor.
  const activeFallbackRequest = useRef<string | null>(null);
  const fallbackRequestGeneration = useRef(0);
  // Invalidates an in-flight model save when the editor changes or closes so
  // a late response cannot affect a newer editor.
  const activeModelRequest = useRef(0);
  const invalidateModelSave = useCallback(() => {
    activeModelRequest.current += 1;
    setModelSaving(false);
  }, []);

  // Per-profile "set active" in-flight name
  const [settingActive, setSettingActive] = useState<string | null>(null);

  const modelKey = (provider: string | null, model: string | null) =>
    model ? `${provider ?? ""}\u0000${model}` : "";

  const loadModelChoices = useCallback(
    (profile?: string) => {
      const scope = profile ?? "";
      if (modelChoices !== null && modelChoicesScope === scope) return;
      if (modelChoicesLoading.current === scope) return;

      // Clear the previous profile's catalog immediately. Keeping it visible
      // while the new request is in flight could present profile A's models
      // while editing profile B.
      setModelChoices(null);
      setModelChoicesScope(scope);
      const requestId = ++modelChoicesRequestGeneration.current;
      modelChoicesLoading.current = scope;
      api
        .getModelOptions(profile)
        .then((res) => {
          if (
            modelChoicesLoading.current !== scope
            || modelChoicesRequestGeneration.current !== requestId
          ) return;
          const flat: {
            provider: string;
            model: string;
            label: string;
            reasoningLevels?: string[] | null;
          }[] = [];
          for (const prov of res.providers ?? []) {
            const caps = prov.capabilities ?? {};
            for (const m of prov.models ?? []) {
              flat.push({
                provider: prov.slug,
                model: m,
                label: `${prov.name} · ${m}`,
                reasoningLevels: caps[m]?.reasoning_levels,
              });
            }
          }
          setModelChoices(flat);
        })
        .catch(() => {
          if (
            modelChoicesLoading.current === scope
            && modelChoicesRequestGeneration.current === requestId
          ) setModelChoices([]);
        })
        .finally(() => {
          if (
            modelChoicesLoading.current === scope
            && modelChoicesRequestGeneration.current === requestId
          ) {
            modelChoicesLoading.current = null;
          }
        });
    },
    [modelChoices, modelChoicesScope],
  );

  const load = useCallback(() => {
    Promise.all([api.getProfiles(), api.getActiveProfile().catch(() => null)])
      .then(([res, active]) => {
        setProfiles(res.profiles);
        setActiveInfo(active);
      })
      .catch((e) => showToast(`${t.status.error}: ${e}`, "error"))
      .finally(() => setLoading(false));
  }, [showToast, t.status.error]);

  useEffect(() => {
    load();
  }, [load]);

  // Lazily load the model picker the first time the create modal opens.
  useEffect(() => {
    if (createModalOpen) loadModelChoices();
  }, [createModalOpen, loadModelChoices]);

  const isActive = useCallback(
    (p: ProfileInfo) =>
      activeInfo != null &&
      (activeInfo.active === p.name ||
        (activeInfo.active === "default" && p.is_default)),
    [activeInfo],
  );

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
      const cloning = cloneFrom !== null;
      const picked = modelChoice
        ? modelChoices?.find(
            (c) => `${c.provider}\u0000${c.model}` === modelChoice,
          )
        : undefined;
      const res = await api.createProfile({
        name,
        clone_from: cloneFrom,
        clone_all: cloning && cloneAll,
        no_skills: cloning ? false : noSkills,
        description: newDescription.trim() || undefined,
        provider: picked?.provider,
        model: picked?.model,
      });
      showToast(`${t.profiles.created}: ${name}`, "success");
      if (picked && res.model_set === false) {
        showToast(
          `Profile created, but the model could not be saved — set it from the profile editor.`,
          "error",
        );
      }
      setNewName("");
      setNewDescription("");
      setNoSkills(false);
      setCloneAll(false);
      setCloneFrom("default");
      setModelChoice("");
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
      showToast(`${t.profiles.renamed}: ${renamingFrom} → ${target}`, "success");
      setRenamingFrom(null);
      setRenameTo("");
      load();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    }
  };

  const handleSetActive = async (name: string) => {
    setSettingActive(name);
    try {
      // The backend normalizes/validates the name; trust the canonical
      // value it returns rather than the raw input.
      const { active } = await api.setActiveProfile(name);
      setProfile(active);
      showToast(
        `${L.activeSet}: ${active} — ${L.activeSetHint.replace("{name}", active)}`,
        "success",
      );
      setActiveInfo((prev) =>
        prev ? { ...prev, active } : { active, current: active },
      );
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    } finally {
      setSettingActive(null);
    }
  };

  // Closes whichever editor dialog is open (model / description / SOUL).
  const closeEditor = useCallback(() => {
    activeSoulRequest.current = null;
    activeDescRequest.current = null;
    activeFallbackRequest.current = null;
    modelChoicesRequestGeneration.current += 1;
    modelChoicesLoading.current = null;
    invalidateModelSave();
    setEditingModelFor(null);
    setEditingDescFor(null);
    setEditingSoulFor(null);
  }, [invalidateModelSave]);

  const openSoulEditor = useCallback(
    async (name: string) => {
      // Re-selecting the action for the already-open editor collapses it,
      // matching the chevron-down affordance in the actions menu.
      if (editingSoulFor === name) {
        closeEditor();
        return;
      }
      invalidateModelSave();
      setEditingDescFor(null);
      setEditingModelFor(null);
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
    [closeEditor, editingSoulFor, invalidateModelSave, showToast, t.status.error],
  );

  const handleSaveSoul = async (name: string) => {
    setSoulSaving(true);
    try {
      await api.updateProfileSoul(name, soulText);
      showToast(`${t.profiles.soulSaved}: ${name}`, "success");
      activeSoulRequest.current = null;
      setEditingSoulFor(null);
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    } finally {
      setSoulSaving(false);
    }
  };

  const openDescEditor = useCallback(
    (p: ProfileInfo) => {
      if (editingDescFor === p.name) {
        closeEditor();
        return;
      }
      invalidateModelSave();
      activeDescRequest.current = p.name;
      setEditingSoulFor(null);
      setEditingModelFor(null);
      setEditingDescFor(p.name);
      setDescText(p.description ?? "");
    },
    [closeEditor, editingDescFor, invalidateModelSave],
  );

  const handleSaveDesc = async (name: string) => {
    descSavingCount.current += 1;
    setDescSaving(true);
    activeDescRequest.current = name;
    try {
      const res = await api.updateProfileDescription(name, descText);
      // Profile-list state always reflects the persisted result, but only
      // touch the open editor if it's still showing this profile.
      setProfiles((prev) =>
        prev.map((p) =>
          p.name === name
            ? {
                ...p,
                description: res.description,
                description_auto: res.description_auto,
              }
            : p,
        ),
      );
      if (activeDescRequest.current === name) {
        showToast(`${L.descriptionSaved}: ${name}`, "success");
        setEditingDescFor(null);
      }
    } catch (e) {
      if (activeDescRequest.current === name) {
        showToast(`${t.status.error}: ${e}`, "error");
      }
    } finally {
      descSavingCount.current -= 1;
      if (descSavingCount.current === 0) setDescSaving(false);
    }
  };

  const handleAutoDescribe = async (name: string) => {
    describingCount.current += 1;
    setDescribing(true);
    activeDescRequest.current = name;
    try {
      const res = await api.describeProfileAuto(name);
      const current = activeDescRequest.current === name;
      if (res.ok && res.description != null) {
        if (current) setDescText(res.description);
        setProfiles((prev) =>
          prev.map((p) =>
            p.name === name
              ? {
                  ...p,
                  description: res.description ?? "",
                  description_auto: res.description_auto,
                }
              : p,
          ),
        );
        if (current) showToast(`${L.descriptionSaved}: ${name}`, "success");
      } else if (current) {
        showToast(`${L.describeFailed}: ${res.reason}`, "error");
      }
    } catch (e) {
      if (activeDescRequest.current === name) {
        showToast(`${t.status.error}: ${e}`, "error");
      }
    } finally {
      describingCount.current -= 1;
      if (describingCount.current === 0) setDescribing(false);
    }
  };

  const openModelEditor = useCallback(
    (p: ProfileInfo) => {
      if (editingModelFor === p.name) {
        closeEditor();
        return;
      }
      invalidateModelSave();
      setEditingSoulFor(null);
      setEditingDescFor(null);
      setEditingModelFor(p.name);
      const initialChoice = modelKey(p.provider, p.model);
      setModelEditChoice(initialChoice);
      setModelEditInitialChoice(initialChoice);
      setReasoningEditChoice(p.reasoning_effort ?? "");
      setFallbackEdits([]);
      setFallbacksLoaded(false);
      setFallbacksLoadFailed(false);
      setFallbacksLoading(true);
      const fallbackRequestToken = `${p.name}:${++fallbackRequestGeneration.current}`;
      activeFallbackRequest.current = fallbackRequestToken;
      loadModelChoices(p.name);
      void api
        .getProfileFallbacks(p.name)
        .then((response) => {
          if (activeFallbackRequest.current !== fallbackRequestToken) return;
          setFallbackEdits(
            response.fallbacks.map((entry) => ({
              source_index: entry.source_index,
              source_provider: entry.source_provider ?? null,
              source_model: entry.source_model ?? null,
              source_base_url: entry.source_base_url ?? null,
              source_api_mode: entry.source_api_mode ?? null,
              provider: entry.provider,
              model: entry.model,
              reasoning_effort: entry.reasoning_effort ?? "",
              base_url: entry.base_url,
              api_mode: entry.api_mode,
            })),
          );
          setFallbacksLoadFailed(false);
          setFallbacksLoaded(true);
        })
        .catch((error) => {
          if (activeFallbackRequest.current === fallbackRequestToken) {
            setFallbacksLoadFailed(true);
            showToast(`${t.status.error}: ${error}`, "error");
          }
        })
        .finally(() => {
          if (activeFallbackRequest.current === fallbackRequestToken) {
            setFallbacksLoading(false);
          }
        });
    },
    [
      closeEditor,
      editingModelFor,
      invalidateModelSave,
      loadModelChoices,
      showToast,
      t.status.error,
    ],
  );

  // Keep a persisted custom/unknown model visible in the picker even when the
  // dashboard catalog has no matching option. It remains a display-only
  // option: the save path below still only writes catalog selections.
  const unlistedInitialModel =
    modelChoices !== null &&
    modelEditInitialChoice !== "" &&
    !modelChoices.some(
      (c) => `${c.provider}\u0000${c.model}` === modelEditInitialChoice,
    );

  const fallbackModelOptions = (entry: FallbackDraft) => {
    const current = modelKey(entry.provider, entry.model);
    const listed = modelChoices ?? [];
    if (
      current &&
      !listed.some((choice) => `${choice.provider}\u0000${choice.model}` === current)
    ) {
      return [
        {
          provider: entry.provider,
          model: entry.model,
          label: `${entry.provider} · ${entry.model}`,
        },
        ...listed,
      ];
    }
    return listed;
  };

  // Reasoning dial options for a selected model: provider-known levels when
  // the catalog carries them, full list when unknown, empty when the model
  // has no dial (caller hides the select).
  const reasoningOptionsFor = (
    provider: string | null,
    model: string | null,
    savedEffort?: string,
  ) => {
    if (!provider || !model) return EFFORT_OPTIONS;
    const choice = (modelChoices ?? []).find(
      (c) => `${c.provider}\u0000${c.model}` === `${provider}\u0000${model}`,
    );
    return filterEffortOptions(choice?.reasoningLevels, savedEffort);
  };

  const addFallback = () => {
    setFallbackEdits((previous) => [
      ...previous,
      {
        source_index: null,
        source_provider: null,
        source_model: null,
        source_base_url: null,
        source_api_mode: null,
        provider: "",
        model: "",
        reasoning_effort: "",
        base_url: null,
        api_mode: null,
      },
    ]);
  };

  const updateFallback = (index: number, patch: Partial<FallbackDraft>) => {
    setFallbackEdits((previous) =>
      previous.map((entry, entryIndex) =>
        entryIndex === index ? { ...entry, ...patch } : entry,
      ),
    );
  };

  const removeFallback = (index: number) => {
    setFallbackEdits((previous) => previous.filter((_, i) => i !== index));
  };

  const moveFallback = (index: number, direction: -1 | 1) => {
    setFallbackEdits((previous) => {
      const nextIndex = index + direction;
      if (nextIndex < 0 || nextIndex >= previous.length) return previous;
      const next = [...previous];
      [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
      return next;
    });
  };

  const handleSaveModel = async (name: string) => {
    const picked = modelEditChoice
      ? modelChoices?.find(
          (c) => `${c.provider}\u0000${c.model}` === modelEditChoice,
        )
      : undefined;
    const unchangedUnlistedModel =
      modelEditChoice !== "" && modelEditChoice === modelEditInitialChoice;
    if (modelEditChoice && modelChoices?.length && !picked && !unchangedUnlistedModel) {
      return;
    }
    if (
      fallbacksLoaded &&
      fallbackEdits.some((entry) => !entry.provider.trim() || !entry.model.trim())
    ) {
      return;
    }
    const changedModel =
      picked !== undefined && modelEditChoice !== modelEditInitialChoice ? picked : null;
    const requestId = ++activeModelRequest.current;
    setModelSaving(true);
    try {
      const settings = await api.setProfileSettings(
        name,
        changedModel?.provider ?? null,
        changedModel?.model ?? null,
        reasoningEditChoice,
      );
      if (activeModelRequest.current !== requestId) return;
      setProfiles((prev) =>
        prev.map((p) =>
          p.name === name
            ? {
                ...p,
                ...(changedModel
                  ? { model: changedModel.model, provider: changedModel.provider }
                  : {}),
                reasoning_effort: settings.reasoning_effort,
              }
            : p,
        ),
      );
      if (changedModel) {
        showToast(`${L.modelSaved}: ${changedModel.model}`, "success");
      }
      showToast(
        `${L.reasoningSaved}: ${settings.reasoning_effort || L.reasoningUnset}`,
        "success",
      );
      if (fallbacksLoaded) {
        const fallbackResponse = await api.updateProfileFallbacks(
          name,
          fallbackEdits.map(
            ({
              source_index,
              source_provider,
              source_model,
              source_base_url,
              source_api_mode,
              provider,
              model,
              reasoning_effort,
              base_url,
              api_mode,
            }) => ({
              source_index,
              source_provider,
              source_model,
              source_base_url,
              source_api_mode,
              provider,
              model,
              reasoning_effort,
              base_url,
              api_mode,
            }),
          ),
        );
        if (activeModelRequest.current !== requestId) return;
        setFallbackEdits(fallbackResponse.fallbacks);
        showToast(L.fallbackSaved, "success");
      }
      setEditingModelFor(null);
    } catch (e) {
      if (activeModelRequest.current === requestId) {
        showToast(`${t.status.error}: ${e}`, "error");
      }
    } finally {
      if (activeModelRequest.current === requestId) {
        setModelSaving(false);
      }
    }
  };

  // Exactly one editor is open at a time; derive which profile + kind so a
  // single dialog can render the right body.
  const editorName = editingModelFor ?? editingDescFor ?? editingSoulFor;
  const editorKind: "model" | "desc" | "soul" | null = editingModelFor
    ? "model"
    : editingDescFor
      ? "desc"
      : editingSoulFor
        ? "soul"
        : null;
  const editorModalRef = useModalBehavior({
    open: editorName != null,
    onClose: closeEditor,
  });

  const mainModelParts = modelEditChoice
    ? modelEditChoice.split("\u0000")
    : [];
  const mainReasoningOptions = reasoningOptionsFor(
    mainModelParts[0] || null,
    mainModelParts[1] || null,
    reasoningEditChoice || undefined,
  );

  const handleCopyTerminalCommand = async (name: string) => {
    let cmd: string;
    try {
      const res = await api.getProfileSetupCommand(name);
      cmd = res.command;
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
      return;
    }
    if (await copyTextToClipboard(cmd)) {
      showToast(`${t.profiles.commandCopied}: ${cmd}`, "success");
    } else {
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
  const pendingProfile = pendingName
    ? profiles.find((p) => p.name === pendingName)
    : undefined;
  const deleteMessage = (() => {
    if (!pendingName) return t.profiles.confirmDeleteMessage;
    const base = t.profiles.confirmDeleteMessage.replace("{name}", pendingName);
    return pendingProfile?.gateway_running
      ? `${base}\n\n${L.gatewayRunningWarning}`
      : base;
  })();

  // Put "Build" (full builder) + "Create" (quick modal) buttons in header
  useLayoutEffect(() => {
    setEnd(
      <div className="flex items-center gap-2">
        <Button
          className="uppercase"
          size="sm"
          outlined
          onClick={() => navigate("/profiles/new")}
        >
          Build
        </Button>
        <Button
          className="uppercase"
          size="sm"
          onClick={() => setCreateModalOpen(true)}
        >
          {t.common.create}
        </Button>
      </div>,
    );
    return () => {
      setEnd(null);
    };
  }, [setEnd, t.common.create, loading, navigate]);

  const cloning = cloneFrom !== null;

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
        description={deleteMessage}
        loading={profileDelete.isDeleting}
      />

      {/* Create profile modal */}
      {createModalOpen && (
        <div
          ref={createModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
          onClick={(e) =>
            e.target === e.currentTarget && setCreateModalOpen(false)
          }
          role="dialog"
          aria-modal="true"
          aria-labelledby="create-profile-title"
        >
          <div
            className={cn(
              themedBody,
              "relative w-full max-w-md border border-border bg-card shadow-2xl flex flex-col max-h-[90vh]",
            )}
          >
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

            <div className="min-h-0 overflow-y-auto p-5 grid gap-4">
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

              <div className="grid gap-2">
                <Label htmlFor="clone-from">{t.profiles.cloneFrom}</Label>
                <Select
                  id="clone-from"
                  value={cloneFrom ?? ""}
                  onValueChange={(v) => {
                    const next = v || null;
                    setCloneFrom(next);
                    if (next === null) setCloneAll(false);
                  }}
                >
                  <SelectOption value="">{t.profiles.cloneFromNone}</SelectOption>
                  {profiles.map((profile) => (
                    <SelectOption key={profile.name} value={profile.name}>
                      {profile.name}
                    </SelectOption>
                  ))}
                </Select>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="profile-description">
                  {L.descriptionOptional}
                </Label>

                <textarea
                  id="profile-description"
                  className="flex min-h-[64px] w-full border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  placeholder={L.descriptionPlaceholder}
                  value={newDescription}
                  onChange={(e) => setNewDescription(e.target.value)}
                />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="profile-model">{L.modelOptional}</Label>

                <Select
                  id="profile-model"
                  value={modelChoice}
                  disabled={modelChoices === null}
                  onValueChange={setModelChoice}
                >
                  <SelectOption value="">
                    {modelChoices === null ? L.modelLoading : L.modelInherit}
                  </SelectOption>

                  {(modelChoices ?? []).map((c) => (
                    <SelectOption
                      key={`${c.provider}\u0000${c.model}`}
                      value={`${c.provider}\u0000${c.model}`}
                    >
                      {c.label}
                    </SelectOption>
                  ))}
                </Select>

                {modelChoices !== null && modelChoices.length === 0 && (
                  <p className="text-xs text-muted-foreground">{L.modelNone}</p>
                )}
              </div>

              <fieldset className="grid gap-3 border-t border-border pt-4">
                <legend className="font-mondwest text-display text-xs tracking-wider text-muted-foreground">
                  {L.advancedOptions}
                </legend>

                <div className="flex items-center gap-2.5">
                  <Checkbox
                    checked={cloneAll}
                    disabled={!cloning}
                    id="clone-all"
                    onCheckedChange={(checked) => setCloneAll(checked === true)}
                  />

                  <Label
                    className={cn(
                      "font-mondwest normal-case tracking-normal text-sm cursor-pointer",
                      !cloning && "opacity-50",
                    )}
                    htmlFor="clone-all"
                  >
                    {L.cloneAll}
                  </Label>
                </div>

                <div className="flex items-center gap-2.5">
                  <Checkbox
                    checked={noSkills}
                    id="no-skills"
                    disabled={cloning}
                    onCheckedChange={(checked) => setNoSkills(checked === true)}
                  />

                  <Label
                    className={cn(
                      "font-mondwest normal-case tracking-normal text-sm cursor-pointer",
                      cloning && "opacity-50",
                    )}
                    htmlFor="no-skills"
                  >
                    {L.noSkillsOption}
                  </Label>
                </div>
              </fieldset>

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

      {/* Active profile banner */}
      {activeInfo && (
        <Card>
          <CardContent className="flex flex-wrap items-center gap-x-4 gap-y-1 py-3 text-xs">
            <span className="flex items-center gap-2 text-muted-foreground">
              <Check className="h-3.5 w-3.5 text-success" />

              <span>
                {L.activeProfile}:{" "}
                <span className="font-medium text-foreground">
                  {activeInfo.active}
                </span>
              </span>
            </span>

            {activeInfo.current !== activeInfo.active && (
              <span className="font-mono text-muted-foreground/80">
                ({activeInfo.current})
              </span>
            )}
          </CardContent>
        </Card>
      )}

      {/* List */}
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

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {profiles.map((p) => {
            const isRenaming = renamingFrom === p.name;
            const isEditingSoul = editingSoulFor === p.name;
            const isEditingDesc = editingDescFor === p.name;
            const isEditingModel = editingModelFor === p.name;
            const active = isActive(p);
            return (
              <Card key={p.name} className="h-full">
                <CardContent className="flex h-full flex-col gap-2 py-4">
                  {isRenaming ? (
                    <div className="flex flex-col gap-2">
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

                      {(() => {
                        const trimmed = renameTo.trim();
                        const invalid =
                          trimmed !== "" &&
                          trimmed !== p.name &&
                          !PROFILE_NAME_RE.test(trimmed);
                        return (
                          <p
                            className={cn(
                              "text-xs",
                              invalid
                                ? "text-destructive"
                                : "text-muted-foreground",
                            )}
                          >
                            {invalid
                              ? `${t.profiles.invalidName}: ${t.profiles.nameRule}`
                              : t.profiles.nameRule}
                          </p>
                        );
                      })()}

                      <div className="flex gap-1.5">
                        <Button size="sm" onClick={handleRenameSubmit}>
                          {t.common.save}
                        </Button>

                        <Button
                          size="sm"
                          ghost
                          onClick={() => setRenamingFrom(null)}
                        >
                          {t.common.cancel}
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="flex items-start gap-2">
                        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
                          <span className="font-medium text-sm truncate">
                            {p.display_name?.trim()
                              ? `${p.display_name.trim()} (${p.name})`
                              : p.name}
                          </span>

                          {active && (
                            <Badge tone="success">{L.activeBadge}</Badge>
                          )}

                          {p.is_default && (
                            <Badge tone="secondary">
                              {t.profiles.defaultBadge}
                            </Badge>
                          )}

                          {p.has_alias && (
                            <Badge tone="outline">{L.aliasBadge}</Badge>
                          )}

                          {p.has_env && (
                            <Badge tone="outline">{t.profiles.hasEnv}</Badge>
                          )}

                          {p.distribution_name && (
                            <Badge tone="outline" className="gap-1">
                              <Package className="h-3 w-3" />
                              {p.distribution_name}
                              {p.distribution_version
                                ? `@${p.distribution_version}`
                                : ""}
                            </Badge>
                          )}
                        </div>

                        <ProfileActionsMenu
                          isActive={active}
                          isDefault={p.is_default}
                          isEditingDesc={isEditingDesc}
                          isEditingModel={isEditingModel}
                          isEditingSoul={isEditingSoul}
                          settingActive={settingActive === p.name}
                          labels={{
                            actions: L.actions,
                            setActive: L.setActive,
                            editModel: L.editModel,
                            editDescription: L.editDescription,
                            editSoul: t.profiles.editSoul,
                            manageSkills: L.manageSkills,
                            openInTerminal: t.profiles.openInTerminal,
                            rename: t.profiles.rename,
                            delete: t.common.delete,
                          }}
                          onCopyCommand={() =>
                            handleCopyTerminalCommand(p.name)
                          }
                          onDelete={() => profileDelete.requestDelete(p.name)}
                          onEditDescription={() => openDescEditor(p)}
                          onEditModel={() => openModelEditor(p)}
                          onEditSoul={() => openSoulEditor(p.name)}
                          onManageSkills={() =>
                            navigate(
                              `/skills?profile=${encodeURIComponent(p.name)}`,
                            )
                          }
                          onRename={() => {
                            setRenamingFrom(p.name);
                            setRenameTo(p.name);
                          }}
                          onSetActive={() => handleSetActive(p.name)}
                        />
                      </div>

                      <div className="flex items-center gap-1.5 text-xs">
                        <span
                          className={cn(
                            "h-1.5 w-1.5 rounded-full",
                            p.gateway_running
                              ? "bg-success"
                              : "bg-muted-foreground/40",
                          )}
                        />

                        <span
                          className={cn(
                            p.gateway_running
                              ? "text-success"
                              : "text-muted-foreground",
                          )}
                        >
                          {p.gateway_running
                            ? L.gatewayRunning
                            : L.gatewayStopped}
                        </span>
                      </div>

                      <div className="flex items-start gap-2 text-xs">
                        <span
                          className={cn(
                            "line-clamp-2",
                            p.description
                              ? "text-muted-foreground"
                              : "text-muted-foreground/60 italic",
                          )}
                        >
                          {p.description || L.noDescription}
                        </span>

                        {p.description && p.description_auto && (
                          <Badge tone="warning" className="shrink-0">
                            {L.reviewBadge}
                          </Badge>
                        )}
                      </div>

                      <div className="mt-auto flex flex-col gap-0.5 pt-1 text-xs text-muted-foreground">
                        {p.model && (
                          <span className="truncate">
                            {t.profiles.model}: {p.model}
                            {p.provider ? ` (${p.provider})` : ""}
                          </span>
                        )}

                        <span>
                          {t.profiles.skills}: {p.skill_count}
                        </span>

                        <span className="truncate">
                          {L.reasoningEffort}: {p.reasoning_effort || L.reasoningUnset}
                        </span>

                        <span className="font-mono truncate">{p.path}</span>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Editor dialog — model / description / SOUL for the selected profile */}
      {editorName && (
        <div
          ref={editorModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
          onClick={(e) => e.target === e.currentTarget && closeEditor()}
          role="dialog"
          aria-modal="true"
          aria-labelledby="profile-editor-title"
        >
          <div
            className={cn(
              themedBody,
              "relative w-full max-w-2xl border border-border bg-card shadow-2xl flex flex-col max-h-[90vh]",
            )}
          >
            <Button
              ghost
              size="icon"
              onClick={closeEditor}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label="Close"
            >
              <X />
            </Button>

            <header className="p-5 pb-3 border-b border-border">
              <h2
                id="profile-editor-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                {editorKind === "model"
                  ? L.editModel
                  : editorKind === "desc"
                    ? L.description
                    : t.profiles.soulSection}
                <span className="text-muted-foreground"> · {editorName}</span>
              </h2>
            </header>

            <div
              className={cn(
                "p-5 grid gap-4",
                (editorKind === "model" || editorKind === "soul") &&
                  "min-h-0 overflow-y-auto",
              )}
            >
              {editorKind === "model" && (
                <>
                  {modelChoices !== null && modelChoices.length === 0 ? (
                    <p className="text-xs text-muted-foreground">{L.modelNone}</p>
                  ) : (
                    <Select
                      value={modelEditChoice}
                      disabled={modelChoices === null}
                      placeholder={
                        modelChoices === null ? L.modelLoading : L.modelSelect
                      }
                      onValueChange={setModelEditChoice}
                    >
                      {unlistedInitialModel && (
                        <SelectOption value={modelEditInitialChoice}>
                          {modelEditInitialChoice.replace("\u0000", " · ")}
                        </SelectOption>
                      )}
                      {(modelChoices ?? []).map((c) => (
                        <SelectOption
                          key={`${c.provider}\u0000${c.model}`}
                          value={`${c.provider}\u0000${c.model}`}
                        >
                          {c.label}
                        </SelectOption>
                      ))}
                    </Select>
                  )}

                  {mainReasoningOptions.length > 0 && (
                    <div className="grid gap-1.5">
                      <Label htmlFor="profile-reasoning-effort">
                        {L.reasoningEffort}
                      </Label>
                      <Select
                        id="profile-reasoning-effort"
                        value={reasoningEditChoice}
                        disabled={modelSaving}
                        onValueChange={setReasoningEditChoice}
                      >
                        <SelectOption value="">{L.reasoningInherit}</SelectOption>
                        {mainReasoningOptions.map((option) => (
                          <SelectOption key={option.value} value={option.value}>
                            {option.label}
                          </SelectOption>
                        ))}
                      </Select>
                    </div>
                  )}

                  <fieldset
                    className="grid gap-3 border-t border-border pt-4"
                    data-profile-fallbacks
                  >
                    <div className="flex items-center justify-between gap-2">
                      <legend className="font-mondwest text-display text-xs tracking-wider text-muted-foreground">
                        {L.fallbacks}
                      </legend>
                      <Button
                        ghost
                        size="sm"
                        className="gap-1.5"
                        onClick={addFallback}
                        disabled={
                          modelSaving ||
                          fallbacksLoading ||
                          !fallbacksLoaded ||
                          modelChoices === null ||
                          modelChoices.length === 0
                        }
                      >
                        <Plus className="h-3.5 w-3.5" />
                        {L.fallbackAdd}
                      </Button>
                    </div>

                    {fallbacksLoading && (
                      <p className="text-xs text-muted-foreground">{L.modelLoading}</p>
                    )}

                    {!fallbacksLoading && fallbacksLoadFailed && (
                      <p className="text-xs text-destructive">
                        {L.fallbackLoadFailed}
                      </p>
                    )}

                    {!fallbacksLoading && fallbacksLoaded && fallbackEdits.length === 0 && (
                      <p className="text-xs text-muted-foreground">{L.fallbackNone}</p>
                    )}

                    {fallbackEdits.map((entry, index) => {
                      const current = modelKey(entry.provider, entry.model);
                      const fallbackReasoningOptions = reasoningOptionsFor(
                        entry.provider || null,
                        entry.model || null,
                        entry.reasoning_effort || undefined,
                      );
                      return (
                        <div
                          key={index}
                          className="grid gap-2 border border-border p-3"
                          data-fallback-entry
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                              #{index + 1}
                            </span>
                            <div className="flex items-center gap-1">
                              <Button
                                ghost
                                size="icon"
                                title={L.fallbackMoveUp}
                                aria-label={L.fallbackMoveUp}
                                disabled={index === 0 || modelSaving || !fallbacksLoaded}
                                onClick={() => moveFallback(index, -1)}
                              >
                                <ChevronUp className="h-3.5 w-3.5" />
                              </Button>
                              <Button
                                ghost
                                size="icon"
                                title={L.fallbackMoveDown}
                                aria-label={L.fallbackMoveDown}
                                disabled={
                                  index === fallbackEdits.length - 1 ||
                                  modelSaving ||
                                  !fallbacksLoaded
                                }
                                onClick={() => moveFallback(index, 1)}
                              >
                                <ChevronDown className="h-3.5 w-3.5" />
                              </Button>
                              <Button
                                ghost
                                size="icon"
                                title={L.fallbackRemove}
                                aria-label={L.fallbackRemove}
                                disabled={modelSaving || !fallbacksLoaded}
                                onClick={() => removeFallback(index)}
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </Button>
                            </div>
                          </div>

                          <Select
                            value={current}
                            disabled={
                              modelChoices === null ||
                              modelSaving ||
                              !fallbacksLoaded
                            }
                            placeholder={L.fallbackSelect}
                            onValueChange={(value) => {
                              const separator = value.indexOf("\u0000");
                              if (separator < 1) return;
                              updateFallback(index, {
                                provider: value.slice(0, separator),
                                model: value.slice(separator + 1),
                              });
                            }}
                          >
                            {fallbackModelOptions(entry).map((choice) => (
                              <SelectOption
                                key={`${choice.provider}\u0000${choice.model}`}
                                value={`${choice.provider}\u0000${choice.model}`}
                              >
                                {choice.label}
                              </SelectOption>
                            ))}
                          </Select>

                          {fallbackReasoningOptions.length > 0 && (
                            <Select
                              value={entry.reasoning_effort}
                              disabled={modelSaving || !fallbacksLoaded}
                              onValueChange={(value) =>
                                updateFallback(index, { reasoning_effort: value })
                              }
                            >
                              <SelectOption value="">{L.reasoningInherit}</SelectOption>
                              {fallbackReasoningOptions.map((option) => (
                                <SelectOption key={option.value} value={option.value}>
                                  {option.label}
                                </SelectOption>
                              ))}
                            </Select>
                          )}

                          {entry.base_url && (
                            <p className="truncate text-[11px] text-muted-foreground">
                              {L.fallbackCustomEndpoint}: {entry.base_url}
                            </p>
                          )}
                        </div>
                      );
                    })}
                  </fieldset>

                  <div className="flex justify-end">
                    <Button
                      size="sm"
                      className="uppercase"
                      onClick={() => handleSaveModel(editorName)}
                      disabled={
                        modelSaving ||
                        (fallbacksLoaded &&
                          fallbackEdits.some(
                            (entry) => !entry.provider.trim() || !entry.model.trim(),
                          )) ||
                        (modelEditChoice !== "" &&
                          (modelChoices === null ||
                            (modelChoices.length > 0 &&
                              !modelChoices.some(
                                (c) =>
                                  `${c.provider}\u0000${c.model}` === modelEditChoice,
                              ) &&
                              modelEditChoice !== modelEditInitialChoice)))
                      }
                    >
                      {modelSaving ? t.common.saving : t.common.save}
                    </Button>
                  </div>
                </>
              )}

              {editorKind === "desc" && (
                <>
                  <div className="flex items-center justify-between gap-2">
                    <Label
                      htmlFor="profile-desc-editor"
                      className="font-mondwest text-display text-xs tracking-wider text-muted-foreground"
                    >
                      {L.description}
                    </Label>

                    <Button
                      size="sm"
                      ghost
                      className="gap-1.5"
                      disabled={describing}
                      onClick={() => handleAutoDescribe(editorName)}
                    >
                      <Sparkles className="h-3.5 w-3.5" />
                      {describing ? L.generating : L.autoGenerate}
                    </Button>
                  </div>

                  <textarea
                    id="profile-desc-editor"
                    className="flex min-h-[96px] w-full border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    placeholder={L.descriptionPlaceholder}
                    value={descText}
                    onChange={(e) => setDescText(e.target.value)}
                  />

                  <div className="flex justify-end">
                    <Button
                      size="sm"
                      className="uppercase"
                      onClick={() => handleSaveDesc(editorName)}
                      disabled={descSaving}
                    >
                      {descSaving ? t.common.saving : t.common.save}
                    </Button>
                  </div>
                </>
              )}

              {editorKind === "soul" && (
                <>
                  <Label
                    htmlFor="profile-soul-editor"
                    className="font-mondwest text-display text-xs tracking-wider text-muted-foreground"
                  >
                    {t.profiles.soulSection}
                  </Label>

                  <textarea
                    id="profile-soul-editor"
                    className="flex min-h-[280px] w-full border border-input bg-transparent px-3 py-2 text-sm font-mono shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    placeholder={t.profiles.soulPlaceholder}
                    value={soulText}
                    onChange={(e) => setSoulText(e.target.value)}
                  />

                  <div className="flex justify-end">
                    <Button
                      size="sm"
                      className="uppercase"
                      onClick={() => handleSaveSoul(editorName)}
                      disabled={soulSaving}
                    >
                      {soulSaving ? t.common.saving : t.common.save}
                    </Button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface ProfileActionsMenuProps {
  isActive: boolean;
  isDefault: boolean;
  isEditingDesc: boolean;
  isEditingModel: boolean;
  isEditingSoul: boolean;
  labels: {
    actions: string;
    delete: string;
    editDescription: string;
    editModel: string;
    editSoul: string;
    manageSkills: string;
    openInTerminal: string;
    rename: string;
    setActive: string;
  };
  settingActive: boolean;
  onCopyCommand: () => void;
  onDelete: () => void;
  onEditDescription: () => void;
  onEditModel: () => void;
  onEditSoul: () => void;
  onManageSkills: () => void;
  onRename: () => void;
  onSetActive: () => void;
}
