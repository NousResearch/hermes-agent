import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { Clock, Pause, Pencil, Play, Plus, Save, Trash2, X, Zap } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Checkbox } from "@nous-research/ui/ui/components/checkbox";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@/components/NouiTypography";
import { api } from "@/lib/api";
import type { CronJob, CronJobCreatePayload, CronJobUpdate, ProfileInfo } from "@/lib/api";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { useToast } from "@/hooks/useToast";
import { useConfirmDelete } from "@/hooks/useConfirmDelete";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { Toast } from "@/components/Toast";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { PluginSlot } from "@/plugins";
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

function getJobScheduleDisplay(job: CronJob): string {
  return (
    asText(job.schedule_display) ||
    asText(job.schedule?.display) ||
    asText(job.schedule?.expr) ||
    "—"
  );
}

function getJobState(job: CronJob): string {
  return asText(job.state) || (job.enabled === false ? "disabled" : "scheduled");
}

function getJobProfile(job: CronJob): string {
  return asText(job.profile_name) || asText(job.profile) || "default";
}

function getJobRunProfile(job: CronJob): string {
  return asText(job.run_profile);
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

interface CronJobConfigForm {
  name: string;
  prompt: string;
  schedule: string;
  deliver: string;
  repeat: string;
  skills: string;
  script: string;
  noAgent: boolean;
  workdir: string;
  runProfile: string;
  model: string;
  provider: string;
  baseUrl: string;
  contextFrom: string;
  enabledToolsets: string;
}

function emptyCronForm(): CronJobConfigForm {
  return {
    name: "",
    prompt: "",
    schedule: "",
    deliver: "local",
    repeat: "",
    skills: "",
    script: "",
    noAgent: false,
    workdir: "",
    runProfile: "",
    model: "",
    provider: "",
    baseUrl: "",
    contextFrom: "",
    enabledToolsets: "",
  };
}

function joinList(value: unknown): string {
  if (Array.isArray(value)) return value.map(String).join("\n");
  return asText(value);
}

function splitList(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function nullableText(value: string): string | null {
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function repeatTimes(job: CronJob): string {
  const times = job.repeat?.times;
  return typeof times === "number" && times > 0 ? String(times) : "";
}

function jobToEditForm(job: CronJob): CronJobConfigForm {
  return {
    name: asText(job.name),
    prompt: asText(job.prompt),
    schedule: getJobScheduleDisplay(job),
    deliver: asText(job.deliver) || "local",
    repeat: repeatTimes(job),
    skills: joinList(job.skills ?? job.skill ?? ""),
    script: asText(job.script),
    noAgent: job.no_agent === true,
    workdir: asText(job.workdir),
    runProfile: getJobRunProfile(job),
    model: asText(job.model),
    provider: asText(job.provider),
    baseUrl: asText(job.base_url),
    contextFrom: joinList(job.context_from),
    enabledToolsets: joinList(job.enabled_toolsets),
  };
}

function parseCronForm(form: CronJobConfigForm) {
  const schedule = form.schedule.trim();
  if (!schedule) throw new Error("Schedule is required");

  const repeatRaw = form.repeat.trim();
  let repeat: number | null = null;
  if (repeatRaw) {
    const parsed = Number(repeatRaw);
    if (!Number.isInteger(parsed) || parsed < 1) {
      throw new Error("Repeat count must be a positive integer");
    }
    repeat = parsed;
  }

  const script = nullableText(form.script);
  if (form.noAgent && !script) {
    throw new Error("No-agent mode requires a script");
  }

  const skills = splitList(form.skills);
  const contextFrom = splitList(form.contextFrom);
  const enabledToolsets = splitList(form.enabledToolsets);

  return {
    name: form.name.trim(),
    prompt: form.prompt.trim(),
    schedule,
    deliver: form.deliver.trim() || "local",
    repeat,
    skills,
    script,
    no_agent: form.noAgent,
    workdir: nullableText(form.workdir),
    profile: nullableText(form.runProfile),
    model: nullableText(form.model),
    provider: nullableText(form.provider),
    base_url: nullableText(form.baseUrl),
    context_from: contextFrom.length ? contextFrom : null,
    enabled_toolsets: enabledToolsets.length ? enabledToolsets : null,
  };
}

function buildCronCreate(form: CronJobConfigForm): CronJobCreatePayload {
  return parseCronForm(form);
}

function buildCronUpdate(job: CronJob, form: CronJobConfigForm): CronJobUpdate {
  const parsed = parseCronForm(form);
  return {
    ...parsed,
    repeat: {
      times: parsed.repeat,
      completed: job.repeat?.completed ?? 0,
    },
  };
}

function CronJobConfigFields({
  form,
  setField,
  profiles,
  idPrefix,
  autoFocusName = false,
}: {
  form: CronJobConfigForm;
  setField<K extends keyof CronJobConfigForm>(key: K, value: CronJobConfigForm[K]): void;
  profiles: ProfileInfo[];
  idPrefix: string;
  autoFocusName?: boolean;
}) {
  const { t } = useI18n();
  const profileOptionsId = `${idPrefix}-profile-options`;
  const textAreaClass =
    "flex min-h-[72px] w-full border border-border bg-background/40 px-3 py-2 text-sm font-courier shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 focus-visible:border-foreground/25";
  return (
    <>
      <section className="grid gap-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Core
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-name`}>{t.cron.nameOptional}</Label>
            <Input
              id={`${idPrefix}-name`}
              autoFocus={autoFocusName}
              placeholder={t.cron.namePlaceholder}
              value={form.name}
              onChange={(e) => setField("name", e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-schedule`}>{t.cron.schedule}</Label>
            <Input
              id={`${idPrefix}-schedule`}
              placeholder={t.cron.schedulePlaceholder}
              value={form.schedule}
              onChange={(e) => setField("schedule", e.target.value)}
            />
          </div>
        </div>

        <div className="grid gap-2">
          <Label htmlFor={`${idPrefix}-prompt`}>{t.cron.prompt}</Label>
          <textarea
            id={`${idPrefix}-prompt`}
            className={textAreaClass}
            placeholder={t.cron.promptPlaceholder}
            value={form.prompt}
            onChange={(e) => setField("prompt", e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-deliver`}>{t.cron.deliverTo}</Label>
            <Input
              id={`${idPrefix}-deliver`}
              value={form.deliver}
              onChange={(e) => setField("deliver", e.target.value)}
            />
            <p className="text-[11px] text-muted-foreground">
              local, telegram, discord, signal, email, or platform:chat_id
            </p>
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-repeat`}>Repeat count</Label>
            <Input
              id={`${idPrefix}-repeat`}
              inputMode="numeric"
              placeholder="blank = unlimited"
              value={form.repeat}
              onChange={(e) => setField("repeat", e.target.value)}
            />
          </div>
        </div>
      </section>

      <section className="grid gap-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Execution
        </h3>
        <div className="grid gap-2">
          <Label htmlFor={`${idPrefix}-skills`}>Skills</Label>
          <textarea
            id={`${idPrefix}-skills`}
            className={textAreaClass}
            placeholder="one skill per line"
            value={form.skills}
            onChange={(e) => setField("skills", e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-script`}>Script</Label>
            <Input
              id={`${idPrefix}-script`}
              placeholder="relative to ~/.hermes/scripts/"
              value={form.script}
              onChange={(e) => setField("script", e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-workdir`}>Workdir</Label>
            <Input
              id={`${idPrefix}-workdir`}
              placeholder="/absolute/project/path"
              value={form.workdir}
              onChange={(e) => setField("workdir", e.target.value)}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="flex items-center gap-2.5">
            <Checkbox
              checked={form.noAgent}
              id={`${idPrefix}-no-agent`}
              onCheckedChange={(checked) =>
                setField("noAgent", checked === true)
              }
            />
            <Label
              className="font-sans normal-case tracking-normal text-sm cursor-pointer"
              htmlFor={`${idPrefix}-no-agent`}
            >
              No-agent mode
            </Label>
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-run-profile`}>Run profile</Label>
            <Input
              id={`${idPrefix}-run-profile`}
              list={profileOptionsId}
              placeholder="blank = scheduler default"
              value={form.runProfile}
              onChange={(e) => setField("runProfile", e.target.value)}
            />
            <datalist id={profileOptionsId}>
              {profiles.map((profile) => (
                <option key={profile.name} value={profile.name} />
              ))}
            </datalist>
          </div>
        </div>
      </section>

      <section className="grid gap-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Advanced model and context
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-provider`}>Provider</Label>
            <Input
              id={`${idPrefix}-provider`}
              value={form.provider}
              onChange={(e) => setField("provider", e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-model`}>Model</Label>
            <Input
              id={`${idPrefix}-model`}
              value={form.model}
              onChange={(e) => setField("model", e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-base-url`}>Base URL</Label>
            <Input
              id={`${idPrefix}-base-url`}
              value={form.baseUrl}
              onChange={(e) => setField("baseUrl", e.target.value)}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-context-from`}>Context from</Label>
            <textarea
              id={`${idPrefix}-context-from`}
              className={textAreaClass}
              placeholder="one job ID per line"
              value={form.contextFrom}
              onChange={(e) => setField("contextFrom", e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${idPrefix}-toolsets`}>Enabled toolsets</Label>
            <textarea
              id={`${idPrefix}-toolsets`}
              className={textAreaClass}
              placeholder="one toolset per line"
              value={form.enabledToolsets}
              onChange={(e) => setField("enabledToolsets", e.target.value)}
            />
          </div>
        </div>
      </section>
    </>
  );
}

function CronJobEditModal({
  job,
  onClose,
  onSaved,
  profiles,
  showToast,
}: {
  job: CronJob;
  onClose(): void;
  onSaved(): void;
  profiles: ProfileInfo[];
  showToast(message: string, tone: "success" | "error"): void;
}) {
  const { t } = useI18n();
  const [form, setForm] = useState<CronJobConfigForm>(() => jobToEditForm(job));
  const [saving, setSaving] = useState(false);
  const modalRef = useModalBehavior({ open: true, onClose });
  const storageProfile = getJobProfile(job);

  const setField = <K extends keyof CronJobConfigForm>(key: K, value: CronJobConfigForm[K]) => {
    setForm((current) => ({ ...current, [key]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.updateCronJob(job.id, buildCronUpdate(job, form), storageProfile);
      showToast(`${t.common.save} ✓`, "success");
      onSaved();
      onClose();
    } catch (e) {
      showToast(`${t.config.failedToSave}: ${e}`, "error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      ref={modalRef}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
      role="dialog"
      aria-modal="true"
      aria-labelledby="edit-cron-title"
    >
      <div className="relative w-full max-w-4xl max-h-[88vh] border border-border bg-card shadow-2xl flex flex-col">
        <Button
          ghost
          size="icon"
          onClick={onClose}
          className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
          aria-label={t.common.close}
        >
          <X />
        </Button>

        <header className="p-5 pb-3 border-b border-border">
          <h2
            id="edit-cron-title"
            className="font-display text-base tracking-wider uppercase"
          >
            Edit Cron Job
          </h2>
          <p className="mt-1 text-xs text-muted-foreground">
            ID <span className="font-mono">{job.id}</span> · stored in{" "}
            <span className="font-mono">{storageProfile}</span>
          </p>
        </header>

        <div className="p-5 overflow-y-auto grid gap-5">
          <CronJobConfigFields
            form={form}
            setField={setField}
            profiles={profiles}
            idPrefix="edit-cron"
            autoFocusName
          />
        </div>

        <footer className="p-4 border-t border-border flex justify-end gap-2">
          <Button outlined size="sm" onClick={onClose} disabled={saving}>
            {t.common.cancel}
          </Button>
          <Button
            size="sm"
            onClick={handleSave}
            disabled={saving}
            prefix={saving ? <Spinner /> : <Save className="h-3 w-3" />}
          >
            {saving ? t.common.saving : t.common.save}
          </Button>
        </footer>
      </div>
    </div>
  );
}

export default function CronPage() {
  const [jobs, setJobs] = useState<CronJob[]>([]);
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [selectedProfile, setSelectedProfile] = useState("all");
  const [loading, setLoading] = useState(true);
  const { toast, showToast } = useToast();
  const { t } = useI18n();
  const { setEnd } = usePageHeader();

  // New job modal state
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [createForm, setCreateForm] = useState<CronJobConfigForm>(() => emptyCronForm());
  const [createStorageProfile, setCreateStorageProfile] = useState("default");
  const closeCreateModal = useCallback(() => setCreateModalOpen(false), []);
  const createModalRef = useModalBehavior({
    open: createModalOpen,
    onClose: closeCreateModal,
  });
  const [creating, setCreating] = useState(false);
  const [editingJob, setEditingJob] = useState<CronJob | null>(null);

  const setCreateField = <K extends keyof CronJobConfigForm>(
    key: K,
    value: CronJobConfigForm[K],
  ) => {
    setCreateForm((current) => ({ ...current, [key]: value }));
  };

  const openCreateModal = useCallback(() => {
    setCreateStorageProfile(selectedProfile === "all" ? "default" : selectedProfile);
    setCreateForm(emptyCronForm());
    setCreateModalOpen(true);
  }, [selectedProfile]);

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
    loadJobs();
  }, [loadJobs]);

  const handleCreate = async () => {
    setCreating(true);
    try {
      await api.createCronJob(buildCronCreate(createForm), createStorageProfile);
      showToast(t.common.create + " ✓", "success");
      setCreateForm(emptyCronForm());
      setCreateModalOpen(false);
      loadJobs();
    } catch (e) {
      showToast(`${t.config.failedToSave}: ${e}`, "error");
    } finally {
      setCreating(false);
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
        onClick={openCreateModal}
      >
        {t.common.create}
      </Button>,
    );
    return () => {
      setEnd(null);
    };
  }, [setEnd, t.common.create, loading, openCreateModal]);

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

      {editingJob && (
        <CronJobEditModal
          key={getJobKey(editingJob)}
          job={editingJob}
          onClose={() => setEditingJob(null)}
          onSaved={loadJobs}
          profiles={profiles}
          showToast={showToast}
        />
      )}

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
          <div
            className={cn(
              themedBody,
              "relative w-full max-w-4xl max-h-[88vh] border border-border bg-card shadow-2xl flex flex-col",
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
                id="create-cron-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                {t.cron.newJob}
              </h2>
            </header>

            <div className="p-5 overflow-y-auto grid gap-5">
              <div className="grid gap-2">
                <Label htmlFor="cron-profile">Profile</Label>
                <Select
                  id="cron-profile"
                  value={createStorageProfile}
                  onValueChange={(v) => setCreateStorageProfile(v)}
                >
                  {profiles.map((profile) => (
                    <SelectOption key={profile.name} value={profile.name}>
                      {profileLabel(profile.name)}
                    </SelectOption>
                  ))}
                </Select>
              </div>

              <CronJobConfigFields
                form={createForm}
                setField={setCreateField}
                profiles={profiles}
                idPrefix="create-cron"
                autoFocusName
              />
            </div>

            <footer className="p-4 border-t border-border flex justify-end gap-2">
              <Button
                outlined
                size="sm"
                onClick={() => setCreateModalOpen(false)}
                disabled={creating}
              >
                {t.common.cancel}
              </Button>
              <Button
                size="sm"
                onClick={handleCreate}
                disabled={creating}
                prefix={creating ? <Spinner /> : <Plus />}
              >
                {creating ? t.common.creating : t.common.create}
              </Button>
            </footer>
          </div>
        </div>
      )}

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
          const runProfile = getJobRunProfile(job);
          const jobKey = getJobKey(job);

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
                    {runProfile && runProfile !== profile && (
                      <Badge tone="outline">run: {profileLabel(runProfile)}</Badge>
                    )}
                    {deliver && deliver !== "local" && (
                      <Badge tone="outline">{deliver}</Badge>
                    )}
                  </div>
                  {hasName && promptText && (
                    <p className="text-xs text-muted-foreground truncate mb-1">
                      {truncateText(promptText, 100)}
                    </p>
                  )}
                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    <span className="font-mono">{getJobScheduleDisplay(job)}</span>
                    <span>
                      {t.cron.last}: {formatTime(job.last_run_at)}
                    </span>
                    <span>
                      {t.cron.next}: {formatTime(job.next_run_at)}
                    </span>
                  </div>
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
                    title="Edit"
                    aria-label="Edit"
                    onClick={() => setEditingJob(job)}
                  >
                    <Pencil />
                  </Button>

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

      <PluginSlot name="cron:bottom" />
    </div>
  );
}
