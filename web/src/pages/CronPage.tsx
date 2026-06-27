import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { Clock, Pause, Pencil, Play, Trash2, X, Zap } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type {
  CronJob,
  CronDeliveryTarget,
  ModelOptionsResponse,
  ProfileInfo,
  SkillInfo,
  ToolsetInfo,
} from "@/lib/api";
import {
  buildCronJobPayload,
  cronJobHasExecutionContent,
  cronJobFormFromJob,
  type CronJobFormState,
} from "@/lib/cron-job";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import {
  DEFAULT_SCHEDULE_STATE,
  ScheduleBuilder,
} from "@/components/ScheduleBuilder";
import {
  buildScheduleString,
  describeSchedule,
  englishOrdinal,
  parseScheduleString,
  type ScheduleBuilderState,
  type ScheduleDescribeStrings,
} from "@/lib/schedule";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { PluginSlot } from "@/plugins";
import { Segmented } from "@nous-research/ui/ui/components/segmented";
import { AutomationBlueprints } from "@/components/AutomationBlueprints";
import { cn, themedBody } from "@/lib/utils";

function formatTime(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString();
}

function asText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function truncateText(value: string, maxLength: number): string {
  return value.length > maxLength
    ? value.slice(0, maxLength) + "..."
    : value;
}

function getJobPrompt(job: CronJob): string {
  return asText(job.prompt);
}

function NameCheckboxPicker({
  id,
  available,
  selected,
  onChange,
  emptyLabel,
}: {
  id: string;
  available: Array<{ name: string; description?: string | null }>;
  selected: string[];
  onChange: (names: string[]) => void;
  emptyLabel: string;
}) {
  const names = available.map((item) => item.name);
  const orphaned = selected.filter((s) => !names.includes(s));
  const all = [...orphaned.map((name) => ({ name, description: "" })), ...available];

  if (all.length === 0) {
    return <p className="text-xs text-muted-foreground">{emptyLabel}</p>;
  }

  const toggle = (name: string, checked: boolean) => {
    if (checked) onChange([...selected, name]);
    else onChange(selected.filter((s) => s !== name));
  };

  return (
    <div
      id={id}
      className="max-h-36 overflow-y-auto border border-border bg-background/40 p-1"
    >
      {all.map((item) => (
        <label
          key={item.name}
          className="flex cursor-pointer items-center gap-2 px-2 py-1 text-xs hover:bg-muted/40"
          title={item.description || undefined}
        >
          <input
            type="checkbox"
            className="accent-foreground"
            checked={selected.includes(item.name)}
            onChange={(e) => toggle(item.name, e.target.checked)}
          />
          <span className="font-mono-ui truncate">{item.name}</span>
        </label>
      ))}
    </div>
  );
}

interface CronJobEditorState extends CronJobFormState {
  scheduleState: ScheduleBuilderState;
}

interface CronJobFormResources {
  availableSkills: SkillInfo[];
  availableToolsets: ToolsetInfo[];
  modelOptions: ModelOptionsResponse | null;
  deliveryTargets: CronDeliveryTarget[];
}

function emptyCronJobForm(): CronJobEditorState {
  return {
    name: "",
    prompt: "",
    schedule: "",
    deliver: "local",
    skills: [],
    provider: "",
    model: "",
    base_url: "",
    script: "",
    no_agent: false,
    context_from: "",
    enabled_toolsets: [],
    workdir: "",
    scheduleState: { ...DEFAULT_SCHEDULE_STATE },
  };
}

function editorFormFromJob(job: CronJob): CronJobEditorState {
  const form = cronJobFormFromJob(job);
  return { ...form, scheduleState: parseScheduleString(form.schedule) };
}

function buildCronJobPayloadFromEditor(form: CronJobEditorState) {
  const { scheduleState, ...payloadForm } = form;
  return buildCronJobPayload({
    ...payloadForm,
    schedule: buildScheduleString(scheduleState),
  });
}

function selectOptions(
  current: string,
  options: Array<{ value: string; label: string }>,
) {
  const known = new Set(options.map((option) => option.value));
  return [
    ...options.map((option) => (
      <SelectOption key={option.value} value={option.value}>
        {option.label}
      </SelectOption>
    )),
    ...(current && !known.has(current)
      ? [
          <SelectOption key={current} value={current}>
            {current}
          </SelectOption>,
        ]
      : []),
  ];
}

function CronAdvancedFields({
  idPrefix,
  form,
  onChange,
  modelOptions,
  availableToolsets,
}: {
  idPrefix: string;
  form: CronJobEditorState;
  onChange: (form: CronJobEditorState) => void;
  modelOptions: ModelOptionsResponse | null;
  availableToolsets: ToolsetInfo[];
}) {
  const update = <K extends keyof CronJobEditorState,>(
    key: K,
    next: CronJobEditorState[K],
  ) => {
    onChange({ ...form, [key]: next });
  };

  const providers = (modelOptions?.providers ?? []).filter(
    (p) => p.authenticated !== false,
  );
  const selectedProvider = providers.find((p) => p.slug === form.provider);
  const models = selectedProvider?.models ?? [];

  return (
    <details className="border border-border bg-background/30 p-3" open>
      <summary className="cursor-pointer text-xs font-medium uppercase tracking-wide text-muted-foreground">
        Advanced fields
      </summary>
      <div className="mt-3 grid gap-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="grid gap-1">
            <Label htmlFor={`${idPrefix}-provider`}>Provider</Label>
            <Select
              id={`${idPrefix}-provider`}
              value={form.provider}
              onValueChange={(v) => {
                onChange({ ...form, provider: v, model: "" });
              }}
            >
              <SelectOption value="">Default</SelectOption>
              {selectOptions(
                form.provider,
                providers.map((p) => ({ value: p.slug, label: p.name })),
              )}
            </Select>
          </div>
          <div className="grid gap-1">
            <Label htmlFor={`${idPrefix}-model`}>Model</Label>
            <Select
              id={`${idPrefix}-model`}
              value={form.model}
              onValueChange={(v) => update("model", v)}
            >
              <SelectOption value="">Default</SelectOption>
              {selectOptions(
                form.model,
                models.map((model) => ({ value: model, label: model })),
              )}
            </Select>
          </div>
        </div>

        <div className="grid gap-1">
          <Label htmlFor={`${idPrefix}-base-url`}>Base URL override</Label>
          <Input
            id={`${idPrefix}-base-url`}
            placeholder="https://api.example.com/v1"
            value={form.base_url}
            onChange={(e) => update("base_url", e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 items-end">
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              className="accent-foreground"
              checked={form.no_agent}
              onChange={(e) => update("no_agent", e.target.checked)}
            />
            no_agent: run the script only and deliver stdout verbatim
          </label>
          <div className="grid gap-1">
            <Label htmlFor={`${idPrefix}-script`}>Script</Label>
            <Input
              id={`${idPrefix}-script`}
              value={form.script}
              onChange={(e) => update("script", e.target.value)}
              placeholder="relative/path/in/scripts"
            />
          </div>
        </div>

        <div className="grid gap-1">
          <Label htmlFor={`${idPrefix}-workdir`}>Workdir</Label>
          <Input
            id={`${idPrefix}-workdir`}
            value={form.workdir}
            onChange={(e) => update("workdir", e.target.value)}
            placeholder="/absolute/project/path"
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="grid gap-1">
            <Label htmlFor={`${idPrefix}-context-from`}>context_from job IDs</Label>
            <textarea
              id={`${idPrefix}-context-from`}
              className="flex min-h-[64px] w-full border border-border bg-background/40 px-3 py-2 text-xs font-courier shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 focus-visible:border-foreground/25"
              placeholder="one job id per line"
              value={form.context_from}
              onChange={(e) => update("context_from", e.target.value)}
            />
          </div>
          <div className="grid gap-1">
            <Label htmlFor={`${idPrefix}-toolsets`}>enabled_toolsets</Label>
            <NameCheckboxPicker
              id={`${idPrefix}-toolsets`}
              available={availableToolsets}
              selected={form.enabled_toolsets}
              onChange={(v) => update("enabled_toolsets", v)}
              emptyLabel="No toolsets available."
            />
          </div>
        </div>
      </div>
    </details>
  );
}

interface CronJobFormFieldsProps {
  idPrefix: string;
  autoFocus?: boolean;
  form: CronJobEditorState;
  resources: CronJobFormResources;
  onChange: (form: CronJobEditorState) => void;
}

function CronJobFormFields({
  idPrefix,
  autoFocus,
  form,
  resources,
  onChange,
}: CronJobFormFieldsProps) {
  const { t } = useI18n();
  const { availableSkills, availableToolsets, deliveryTargets, modelOptions } = resources;
  const update = <K extends keyof CronJobEditorState,>(
    key: K,
    next: CronJobEditorState[K],
  ) => {
    onChange({ ...form, [key]: next });
  };
  const onlyLocalAvailable =
    deliveryTargets.filter((target) => target.id !== "local").length === 0;

  const deliveryOptions = selectOptions(
    form.deliver,
    deliveryTargets.map((target) => {
      const base = target.id === "local" ? t.cron.delivery.local : target.name;
      if (target.id !== "local" && !target.home_target_set) {
        const hint = t.cron.delivery.needsHomeChannel ?? "set a home channel first";
        return { value: target.id, label: `${base} — ${hint}` };
      }
      return { value: target.id, label: base };
    }),
  );

  return (
    <>
      <div className="grid gap-2">
        <Label htmlFor={`${idPrefix}-name`}>{t.cron.nameOptional}</Label>
        <Input
          id={`${idPrefix}-name`}
          autoFocus={autoFocus}
          placeholder={t.cron.namePlaceholder}
          value={form.name}
          onChange={(e) => update("name", e.target.value)}
        />
      </div>

      <div className="grid gap-2">
        <Label htmlFor={`${idPrefix}-prompt`}>{t.cron.prompt}</Label>
        <textarea
          id={`${idPrefix}-prompt`}
          className="flex min-h-[80px] w-full border border-border bg-background/40 px-3 py-2 text-sm font-courier shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 focus-visible:border-foreground/25"
          placeholder={t.cron.promptPlaceholder}
          value={form.prompt}
          onChange={(e) => update("prompt", e.target.value)}
        />
      </div>

      <ScheduleBuilder
        value={form.scheduleState}
        onChange={(state) => update("scheduleState", state)}
      />

      <div className="grid gap-2">
        <Label htmlFor={`${idPrefix}-deliver`}>{t.cron.deliverTo}</Label>
        <Select
          id={`${idPrefix}-deliver`}
          value={form.deliver}
          onValueChange={(v) => update("deliver", v)}
        >
          {deliveryOptions}
        </Select>
        {onlyLocalAvailable && (
          <p className="text-xs text-muted-foreground">
            {t.cron.delivery.noneConfigured ??
              "No messaging platforms configured. Set one up under Channels to deliver reports."}
          </p>
        )}
      </div>

      <div className="grid gap-2">
        <Label htmlFor={`${idPrefix}-skills`}>Skills (optional)</Label>
        <NameCheckboxPicker
          id={`${idPrefix}-skills`}
          available={availableSkills}
          selected={form.skills}
          onChange={(skills) => update("skills", skills)}
          emptyLabel="No skills installed for this profile."
        />
        <p className="text-xs text-muted-foreground">
          Selected skills are loaded before the prompt runs — the cron
          sets when, the skill sets how.
        </p>
      </div>

      <CronAdvancedFields
        idPrefix={`${idPrefix}-advanced`}
        form={form}
        onChange={onChange}
        modelOptions={modelOptions}
        availableToolsets={availableToolsets}
      />
    </>
  );
}

function getJobName(job: CronJob): string {
  return asText(job.name).trim();
}

function getJobTitle(job: CronJob): string {
  const name = getJobName(job);
  if (name) return name;

  const prompt = getJobPrompt(job);
  if (prompt) return truncateText(prompt, 60);

  const script = asText(job.script);
  if (script) return truncateText(script, 60);

  return job.id || "Cron job";
}

function getJobScheduleDisplay(
  job: CronJob,
  strings: ScheduleDescribeStrings,
): string {
  // Prefer a structured render so cron expressions like
  // ``30 14 * * 1,3,5`` surface as "Weekly on Mon, Wed, Fri at 14:30"
  // in the list instead of the raw five-field gibberish. Falls back
  // through the existing chain (``schedule_display`` from the backend,
  // then the structured ``display`` field, then the raw ``expr``) so
  // legacy job rows still render *something* meaningful.
  return describeSchedule(
    job.schedule,
    asText(job.schedule_display) || asText(job.schedule?.display),
    strings,
  );
}

function getJobState(job: CronJob): string {
  return asText(job.state) || (job.enabled === false ? "disabled" : "scheduled");
}

function getRepeatDisplay(job: CronJob): string {
  const repeat = job.repeat;
  if (!repeat || repeat.times == null) return "forever";
  const completed = repeat.completed ?? 0;
  return completed > 0 ? `${completed}/${repeat.times}` : `${repeat.times} times`;
}

function getJobMode(job: CronJob): string {
  if (job.no_agent) return "no_agent";
  if (job.script) return "script+agent";
  return "agent";
}

function getModelDisplay(job: CronJob): string {
  const provider = asText(job.provider);
  const model = asText(job.model);
  if (provider && model) return `${provider}/${model}`;
  return model || provider;
}

function getJobProfile(job: CronJob): string {
  return asText(job.profile) || asText(job.profile_name) || "default";
}

function getJobKey(job: CronJob): string {
  return `${getJobProfile(job)}:${job.id}`;
}

function splitJobKey(key: string): { profile: string; id: string } {
  const idx = key.indexOf(":");
  if (idx === -1) return { profile: "default", id: key };
  return { profile: key.slice(0, idx) || "default", id: key.slice(idx + 1) };
}

function profileLabel(profile: string): string {
  return profile === "default" ? "default" : profile;
}

const STATUS_TONE: Record<string, "success" | "warning" | "destructive"> = {
  enabled: "success",
  scheduled: "success",
  paused: "warning",
  error: "destructive",
  completed: "destructive",
};

export default function CronPage() {
  const [jobs, setJobs] = useState<CronJob[]>([]);
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [selectedProfile, setSelectedProfile] = useState("all");
  const [view, setView] = useState<"jobs" | "blueprints">("jobs");
  const [loading, setLoading] = useState(true);
  const { toast, showToast } = useToast();
  const { t, locale } = useI18n();
  const { setEnd } = usePageHeader();

  // Translation surface for the human-readable schedule describer.
  // English ordinals are a special case ("1st", "2nd", "23rd"); every
  // other locale falls back to the plain numeric form, which avoids
  // shipping incorrect grammar (e.g. naive "1th"/"2th" suffixes that
  // don't exist in most languages).
  //
  // Built inline (not memoized) — the cron page renders a small job
  // list, this is single-digit microseconds, and a useMemo here would
  // just add boilerplate.
  const scheduleDescribeStrings: ScheduleDescribeStrings = {
    ...t.cron.scheduleDescribe,
    weekdaysShort: t.cron.scheduleModes.weekdaysShort,
    ordinal: locale === "en" ? englishOrdinal : (n: number) => String(n),
  };

  // New job modal state
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [createProfile, setCreateProfile] = useState("default");
  const [createForm, setCreateForm] = useState<CronJobEditorState>(
    emptyCronJobForm,
  );
  const closeCreateModal = useCallback(() => setCreateModalOpen(false), []);
  const createModalRef = useModalBehavior({
    open: createModalOpen,
    onClose: closeCreateModal,
  });
  const [deliveryTargets, setDeliveryTargets] = useState<CronDeliveryTarget[]>([
    { id: "local", name: "Local", home_target_set: true, home_env_var: null },
  ]);
  const [creating, setCreating] = useState(false);

  // Edit job modal state
  const [editJob, setEditJob] = useState<CronJob | null>(null);
  const [editForm, setEditForm] = useState<CronJobEditorState>(
    emptyCronJobForm,
  );
  const [saving, setSaving] = useState(false);
  const closeEditModal = useCallback(() => setEditJob(null), []);
  const editModalRef = useModalBehavior({
    open: editJob !== null,
    onClose: closeEditModal,
  });

  // Skills installed in the profile a job will run under, for the
  // attach-skill selector (parity with `hermes cron edit --add-skill`).
  // Keyed on the create-modal profile; the edit modal reuses the list —
  // a job's current skills are always shown even if not in it.
  const [availableSkills, setAvailableSkills] = useState<SkillInfo[]>([]);
  const [availableToolsets, setAvailableToolsets] = useState<ToolsetInfo[]>([]);
  const [modelOptions, setModelOptions] = useState<ModelOptionsResponse | null>(null);

  const resourceProfile = editJob ? getJobProfile(editJob) : createProfile;

  const openEditModal = useCallback((job: CronJob) => {
    setEditJob(job);
    setEditForm(editorFormFromJob(job));
  }, []);

  const loadJobs = useCallback(() => {
    api
      .getCronJobs(selectedProfile)
      .then(setJobs)
      .catch(() => showToast(t.common.loading, "error"))
      .finally(() => setLoading(false));
  }, [selectedProfile, showToast, t.common.loading]);

  useEffect(() => {
    api
      .getProfiles()
      .then((res) => setProfiles(res.profiles))
      .catch(() => setProfiles([]));
  }, []);

  useEffect(() => {
    api
      .getCronDeliveryTargets()
      .then((res) => setDeliveryTargets(res.targets))
      .catch(() =>
        // Fall back to local-only so the modal still works if the endpoint fails.
        setDeliveryTargets([
          { id: "local", name: "Local", home_target_set: true, home_env_var: null },
        ]),
      );
  }, []);

  useEffect(() => {
    loadJobs();
  }, [loadJobs]);

  // Load resources from the profile the create/edit form actually targets.
  // Pass "default" explicitly so the global dashboard profile switch cannot
  // redirect a default-profile cron form to some other profile.
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      api.getSkills(resourceProfile).catch(() => []),
      api.getToolsets(resourceProfile).catch(() => []),
      api.getModelOptions(resourceProfile).catch(() => null),
    ]).then(([skills, toolsets, options]) => {
      if (cancelled) return;
      setAvailableSkills([...skills].sort((a, b) => a.name.localeCompare(b.name)));
      setAvailableToolsets([...toolsets].sort((a, b) => a.name.localeCompare(b.name)));
      setModelOptions(options);
    });
    return () => {
      cancelled = true;
    };
  }, [resourceProfile]);

  const handleCreate = async () => {
    const payload = buildCronJobPayloadFromEditor(createForm);
    if (
      !payload.schedule ||
      (!payload.no_agent && !cronJobHasExecutionContent(payload))
    ) {
      showToast(`${t.cron.prompt} & ${t.cron.schedule} required`, "error");
      return;
    }
    if (payload.no_agent && !payload.script) {
      showToast("no_agent jobs require a script", "error");
      return;
    }
    setCreating(true);
    try {
      await api.createCronJob(payload, createProfile);
      showToast(t.common.create + " ✓", "success");
      setCreateForm(emptyCronJobForm());
      setCreateModalOpen(false);
      loadJobs();
    } catch (e) {
      showToast(`${t.config.failedToSave}: ${e}`, "error");
    } finally {
      setCreating(false);
    }
  };

  const handleEdit = async () => {
    if (!editJob) return;
    const payload = buildCronJobPayloadFromEditor(editForm);
    if (
      !payload.schedule ||
      (!payload.no_agent && !cronJobHasExecutionContent(payload))
    ) {
      showToast(`${t.cron.prompt} & ${t.cron.schedule} required`, "error");
      return;
    }
    if (payload.no_agent && !payload.script) {
      showToast("no_agent jobs require a script", "error");
      return;
    }
    setSaving(true);
    try {
      await api.updateCronJob(
        editJob.id,
        payload,
        getJobProfile(editJob),
      );
      showToast("Saved changes ✓", "success");
      setEditJob(null);
      loadJobs();
    } catch (e) {
      showToast(`${t.config.failedToSave}: ${e}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const handlePauseResume = async (job: CronJob) => {
    try {
      const isPaused = getJobState(job) === "paused";
      const profile = getJobProfile(job);
      if (isPaused) {
        await api.resumeCronJob(job.id, profile);
        showToast(
          `${t.cron.resume}: "${truncateText(getJobTitle(job), 30)}"`,
          "success",
        );
      } else {
        await api.pauseCronJob(job.id, profile);
        showToast(
          `${t.cron.pause}: "${truncateText(getJobTitle(job), 30)}"`,
          "success",
        );
      }
      loadJobs();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    }
  };

  const handleTrigger = async (job: CronJob) => {
    try {
      await api.triggerCronJob(job.id, getJobProfile(job));
      showToast(
        `${t.cron.triggerNow}: "${truncateText(getJobTitle(job), 30)}"`,
        "success",
      );
      loadJobs();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    }
  };

  const jobDelete = useConfirmDelete({
    onDelete: useCallback(
      async (key: string) => {
        const { profile, id } = splitJobKey(key);
        const job = jobs.find((j) => getJobKey(j) === key);
        try {
          await api.deleteCronJob(id, profile);
          showToast(
            `${t.common.delete}: "${job ? truncateText(getJobTitle(job), 30) : id}"`,
            "success",
          );
          loadJobs();
        } catch (e) {
          showToast(`${t.status.error}: ${e}`, "error");
          throw e;
        }
      },
      [jobs, loadJobs, showToast, t.common.delete, t.status.error],
    ),
  });

  // Put "Create" button in page header
  useLayoutEffect(() => {
    setEnd(
      <Button
        className="uppercase"
        size="sm"
        onClick={() => {
          setCreateProfile(selectedProfile === "all" ? "default" : selectedProfile);
          setCreateModalOpen(true);
        }}
      >
        {t.common.create}
      </Button>,
    );
    return () => {
      setEnd(null);
    };
  }, [setEnd, t.common.create, loading, selectedProfile]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  const pendingJob = jobDelete.pendingId
    ? jobs.find((j) => getJobKey(j) === jobDelete.pendingId)
    : null;

  return (
    <div className="flex flex-col gap-6">
      <PluginSlot name="cron:top" />
      <Toast toast={toast} />

      <Segmented
        value={view}
        onChange={(v) => setView(v as "jobs" | "blueprints")}
        options={[
          { value: "jobs", label: "Jobs" },
          { value: "blueprints", label: "Blueprints" },
        ]}
      />

      {view === "blueprints" && (
        <AutomationBlueprints
          profile={selectedProfile === "all" ? "default" : selectedProfile}
          onCreated={loadJobs}
        />
      )}


      <DeleteConfirmDialog
        open={jobDelete.isOpen}
        onCancel={jobDelete.cancel}
        onConfirm={jobDelete.confirm}
        title={t.cron.confirmDeleteTitle}
        description={
          pendingJob
            ? `"${truncateText(getJobTitle(pendingJob), 40)}" — ${
                t.cron.confirmDeleteMessage
              }`
            : t.cron.confirmDeleteMessage
        }
        loading={jobDelete.isDeleting}
      />

      {/* Create job modal */}
      {createModalOpen && (
        <div
          ref={createModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
          onClick={(e) => e.target === e.currentTarget && setCreateModalOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="create-cron-title"
        >
          <div className={cn(themedBody, "relative w-full max-w-3xl max-h-[90vh] border border-border bg-card shadow-2xl flex flex-col")}>
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
                id="create-cron-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                {t.cron.newJob}
              </h2>
            </header>

            <div className="min-h-0 overflow-y-auto p-5 grid gap-4">
              <div className="grid gap-2">
                <Label htmlFor="cron-profile">Profile</Label>
                <Select
                  id="cron-profile"
                  value={createProfile}
                  onValueChange={(v) => setCreateProfile(v)}
                >
                  {profiles.map((profile) => (
                    <SelectOption key={profile.name} value={profile.name}>
                      {profileLabel(profile.name)}
                    </SelectOption>
                  ))}
                </Select>
              </div>

              <CronJobFormFields
                idPrefix="cron"
                autoFocus
                form={createForm}
                onChange={setCreateForm}
                resources={{
                  availableSkills,
                  availableToolsets,
                  modelOptions,
                  deliveryTargets,
                }}
              />

              <div className="flex justify-end">
                <Button
                  className="uppercase"
                  size="sm"
                  onClick={handleCreate}
                  disabled={creating}
                  prefix={creating ? <Spinner /> : undefined}
                >
                  {creating ? t.common.creating : t.common.create}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Edit job modal */}
      {editJob && (
        <div
          ref={editModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
          onClick={(e) => e.target === e.currentTarget && setEditJob(null)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="edit-cron-title"
        >
          <div className={cn(themedBody, "relative w-full max-w-3xl max-h-[90vh] border border-border bg-card shadow-2xl flex flex-col")}>
            <Button
              ghost
              size="icon"
              onClick={() => setEditJob(null)}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label="Close"
            >
              <X />
            </Button>

            <header className="p-5 pb-3 border-b border-border">
              <h2
                id="edit-cron-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                Edit job
              </h2>
            </header>

            <div className="min-h-0 overflow-y-auto p-5 grid gap-4">
              <CronJobFormFields
                idPrefix="edit-cron"
                autoFocus
                form={editForm}
                onChange={setEditForm}
                resources={{
                  availableSkills,
                  availableToolsets,
                  modelOptions,
                  deliveryTargets,
                }}
              />

              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground font-mono-ui truncate pr-4">
                  {editJob.id}
                </span>
                <Button
                  className="uppercase"
                  size="sm"
                  onClick={handleEdit}
                  disabled={saving}
                  prefix={saving ? <Spinner /> : undefined}
                >
                  {saving ? t.common.loading : "Save changes"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {view === "jobs" && (
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <H2
            variant="sm"
            className="flex items-center gap-2 text-muted-foreground"
          >
            <Clock className="h-4 w-4" />
            {t.cron.scheduledJobs} ({jobs.length})
          </H2>

          <div className="grid gap-1 min-w-[220px]">
            <Label htmlFor="cron-profile-filter">Profile</Label>
            <Select
              id="cron-profile-filter"
              value={selectedProfile}
              onValueChange={(v) => setSelectedProfile(v)}
            >
              <SelectOption value="all">All profiles</SelectOption>
              {profiles.map((profile) => (
                <SelectOption key={profile.name} value={profile.name}>
                  {profileLabel(profile.name)}
                </SelectOption>
              ))}
            </Select>
          </div>
        </div>

        {jobs.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              {t.cron.noJobs}
            </CardContent>
          </Card>
        )}

        {jobs.map((job) => {
          const state = getJobState(job);
          const promptText = getJobPrompt(job);
          const title = getJobTitle(job);
          const hasName = Boolean(getJobName(job));
          const deliver = asText(job.deliver);
          const profile = getJobProfile(job);
          const jobKey = getJobKey(job);
          const mode = getJobMode(job);
          const modelDisplay = getModelDisplay(job);
          const toolsets = Array.isArray(job.enabled_toolsets)
            ? job.enabled_toolsets.filter(Boolean)
            : [];

          return (
            <Card key={jobKey}>
              <CardContent className="flex items-start gap-4 py-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-sm truncate">
                      {title}
                    </span>
                    <Badge tone={STATUS_TONE[state] ?? "secondary"}>
                      {state}
                    </Badge>
                    <Badge tone="outline">{profileLabel(profile)}</Badge>
                    {deliver && deliver !== "local" && (
                      <Badge tone="outline">{deliver}</Badge>
                    )}
                    {Array.isArray(job.skills) && job.skills.length > 0 && (
                      <Badge tone="outline" title={job.skills.join(", ")}>
                        {job.skills.length === 1
                          ? job.skills[0]
                          : `${job.skills.length} skills`}
                      </Badge>
                    )}
                    {mode !== "agent" && (
                      <Badge tone="outline">{mode}</Badge>
                    )}
                    {modelDisplay && (
                      <Badge tone="outline" title={modelDisplay}>
                        model
                      </Badge>
                    )}
                    {toolsets.length > 0 && (
                      <Badge tone="outline" title={toolsets.join(", ")}>
                        {toolsets.length} toolsets
                      </Badge>
                    )}
                  </div>
                  {hasName && promptText && (
                    <p className="text-xs text-muted-foreground truncate mb-1">
                      {truncateText(promptText, 100)}
                    </p>
                  )}
                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    <span className="font-mono-ui">
                      {getJobScheduleDisplay(job, scheduleDescribeStrings)}
                    </span>
                    <span>repeat: {getRepeatDisplay(job)}</span>
                    <span>
                      {t.cron.last}: {formatTime(job.last_run_at)}
                    </span>
                    <span>
                      {t.cron.next}: {formatTime(job.next_run_at)}
                    </span>
                  </div>
                  {job.last_delivery_error && (
                    <p className="text-xs text-destructive mt-1">
                      delivery: {job.last_delivery_error}
                    </p>
                  )}
                  {job.last_error && (
                    <p className="text-xs text-destructive mt-1">
                      {job.last_error}
                    </p>
                  )}
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  <Button
                    ghost
                    size="icon"
                    title={state === "paused" ? t.cron.resume : t.cron.pause}
                    aria-label={
                      state === "paused" ? t.cron.resume : t.cron.pause
                    }
                    onClick={() => handlePauseResume(job)}
                    className={
                      state === "paused" ? "text-success" : "text-warning"
                    }
                  >
                    {state === "paused" ? <Play /> : <Pause />}
                  </Button>

                  <Button
                    ghost
                    size="icon"
                    title={t.cron.triggerNow}
                    aria-label={t.cron.triggerNow}
                    onClick={() => handleTrigger(job)}
                  >
                    <Zap />
                  </Button>

                  <Button
                    ghost
                    size="icon"
                    title="Edit job"
                    aria-label="Edit job"
                    onClick={() => openEditModal(job)}
                  >
                    <Pencil />
                  </Button>

                  <Button
                    ghost
                    destructive
                    size="icon"
                    title={t.common.delete}
                    aria-label={t.common.delete}
                    onClick={() => jobDelete.requestDelete(jobKey)}
                  >
                    <Trash2 />
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
      )}

      <PluginSlot name="cron:bottom" />
    </div>
  );
}
