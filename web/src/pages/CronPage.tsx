import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import { Clock, Pause, Play, Plus, Trash2, X, Zap } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@/components/NouiTypography";
import { api } from "@/lib/api";
import type { CronJob, ProfileInfo } from "@/lib/api";
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

// Sentinel value for the dropdown's "All profiles" option. Kept as a value
// here (not a name conflict with any installed profile) so the existing
// profile-name validation rules — which forbid uppercase — protect it.
const ALL_PROFILES = "ALL" as const;

// Annotated row carrying the owning profile and that profile's live
// gateway state. The "All" view merges N per-profile responses into a
// flat list of these; the per-profile view annotates with the filter's
// own profile + gateway_running.
type CronJobRow = CronJob & {
  profile: string;
  gateway_running: boolean;
};

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

// Render the same recurrence marker `hermes cron list` shows, so the
// "4320m" (one-shot) vs "every 4320m" (recurring) foot-gun is visible at
// a glance. Returns "" for jobs without a recognisable schedule kind.
function getJobRepeatLabel(job: CronJob): string {
  const kind = job.schedule?.kind;
  if (kind === "once") return "Repeat: 1/1";
  if (kind === "interval" || kind === "cron") return "Repeat: ∞";
  return "";
}

const STATUS_TONE: Record<string, "success" | "warning" | "destructive"> = {
  enabled: "success",
  scheduled: "success",
  paused: "warning",
  error: "destructive",
  completed: "destructive",
};

export default function CronPage() {
  const [jobs, setJobs] = useState<CronJobRow[]>([]);
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<string>(ALL_PROFILES);
  const [loading, setLoading] = useState(true);
  const [jobsLoading, setJobsLoading] = useState(false);
  const { toast, showToast } = useToast();
  const { t } = useI18n();
  const { setEnd } = usePageHeader();

  // Active-profile fallback chain: is_active (modern gateways) → is_default
  // (older gateways without is_active) → first profile (last-resort).
  // Matches the SkillsPage pattern so the active marker stays consistent
  // across older daemons.
  const activeProfileName = useMemo(() => {
    const active = profiles.find((p) => p.is_active);
    if (active) return active.name;
    const dflt = profiles.find((p) => p.is_default);
    return dflt?.name ?? profiles[0]?.name ?? "";
  }, [profiles]);

  const showProfileColumn =
    selectedProfile === ALL_PROFILES && profiles.length > 1;

  // New job modal state. ``createProfile`` defaults to the active profile;
  // when the user is filtered to a specific profile the form targets that
  // one and hides the selector.
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [schedule, setSchedule] = useState("");
  const [name, setName] = useState("");
  const [deliver, setDeliver] = useState("local");
  const [createProfile, setCreateProfile] = useState("");
  const [creating, setCreating] = useState(false);
  const closeCreateModal = useCallback(() => setCreateModalOpen(false), []);
  const createModalRef = useModalBehavior({
    open: createModalOpen,
    onClose: closeCreateModal,
  });

  const openCreateModal = useCallback(() => {
    setCreateProfile(
      selectedProfile === ALL_PROFILES ? activeProfileName : selectedProfile,
    );
    setCreateModalOpen(true);
  }, [selectedProfile, activeProfileName]);

  // Fetch jobs for the current selection. ``profileList`` is passed in
  // explicitly because callers (initial load + dropdown change) want to
  // pass either freshly-fetched profiles or the already-set state.
  const loadJobs = useCallback(
    async (profileList: ProfileInfo[], selection: string) => {
      setJobsLoading(true);
      try {
        if (selection === ALL_PROFILES) {
          const settled = await Promise.allSettled(
            profileList.map((p) =>
              api
                .getProfileCronJobs(p.name)
                .then((rows): CronJobRow[] =>
                  rows.map((j) => ({
                    ...j,
                    profile: p.name,
                    gateway_running: Boolean(p.gateway_running),
                  })),
                ),
            ),
          );
          const merged: CronJobRow[] = [];
          settled.forEach((res) => {
            if (res.status === "fulfilled") {
              merged.push(...res.value);
            }
          });
          setJobs(merged);
        } else {
          const target = profileList.find((p) => p.name === selection);
          const rows = await api.getProfileCronJobs(selection);
          setJobs(
            rows.map((j) => ({
              ...j,
              profile: selection,
              gateway_running: Boolean(target?.gateway_running),
            })),
          );
        }
      } catch (e) {
        showToast(`${t.status.error}: ${e}`, "error");
      } finally {
        setJobsLoading(false);
      }
    },
    [showToast, t.status.error],
  );

  // Initial load: profiles + jobs for "All" view. The "All" default differs
  // from SkillsPage (which defaults to the active profile) because cron is
  // operations work that benefits from cross-profile visibility; skills is
  // per-profile config work.
  useEffect(() => {
    api
      .getProfiles()
      .then(async ({ profiles: profileList }) => {
        setProfiles(profileList);
        await loadJobs(profileList, ALL_PROFILES);
      })
      .catch(() => showToast(t.common.loading, "error"))
      .finally(() => setLoading(false));
    // loadJobs depends on showToast/t.status.error — but we deliberately
    // run this only on mount. Subsequent loads happen via the selection
    // effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refetch when the dropdown changes (skip the initial mount, which the
  // load effect above already handled).
  useEffect(() => {
    if (loading) return;
    loadJobs(profiles, selectedProfile);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProfile]);

  const refreshJobs = useCallback(() => {
    loadJobs(profiles, selectedProfile);
  }, [loadJobs, profiles, selectedProfile]);

  const handleCreate = async () => {
    if (!prompt.trim() || !schedule.trim()) {
      showToast(`${t.cron.prompt} & ${t.cron.schedule} required`, "error");
      return;
    }
    const target = createProfile || activeProfileName;
    if (!target) {
      showToast(`${t.status.error}: no profile selected`, "error");
      return;
    }
    setCreating(true);
    try {
      await api.createProfileCronJob(target, {
        prompt: prompt.trim(),
        schedule: schedule.trim(),
        name: name.trim() || undefined,
        deliver,
      });
      showToast(t.common.create + " ✓", "success");
      setPrompt("");
      setSchedule("");
      setName("");
      setDeliver("local");
      setCreateModalOpen(false);
      refreshJobs();
    } catch (e) {
      showToast(`${t.config.failedToSave}: ${e}`, "error");
    } finally {
      setCreating(false);
    }
  };

  const handlePauseResume = async (job: CronJobRow) => {
    try {
      const isPaused = getJobState(job) === "paused";
      if (isPaused) {
        await api.resumeProfileCronJob(job.profile, job.id);
        showToast(
          `${t.cron.resume}: "${truncateText(getJobTitle(job), 30)}"`,
          "success",
        );
      } else {
        await api.pauseProfileCronJob(job.profile, job.id);
        showToast(
          `${t.cron.pause}: "${truncateText(getJobTitle(job), 30)}"`,
          "success",
        );
      }
      refreshJobs();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    }
  };

  const handleTrigger = async (job: CronJobRow) => {
    try {
      await api.triggerProfileCronJob(job.profile, job.id);
      showToast(
        `${t.cron.triggerNow}: "${truncateText(getJobTitle(job), 30)}"`,
        "success",
      );
      refreshJobs();
    } catch (e) {
      showToast(`${t.status.error}: ${e}`, "error");
    }
  };

  const jobDelete = useConfirmDelete({
    onDelete: useCallback(
      async (id: string) => {
        const job = jobs.find((j) => j.id === id);
        if (!job) return;
        try {
          await api.deleteProfileCronJob(job.profile, id);
          showToast(
            `${t.common.delete}: "${truncateText(getJobTitle(job), 30)}"`,
            "success",
          );
          refreshJobs();
        } catch (e) {
          showToast(`${t.status.error}: ${e}`, "error");
          throw e;
        }
      },
      [jobs, refreshJobs, showToast, t.common.delete, t.status.error],
    ),
  });

  // Put "Create" button in page header
  useLayoutEffect(() => {
    setEnd(
      <Button size="sm" onClick={openCreateModal}>
        <Plus className="h-3 w-3" />
        {t.common.create}
      </Button>,
    );
    return () => {
      setEnd(null);
    };
  }, [setEnd, t.common.create, openCreateModal]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  const pendingJob = jobDelete.pendingId
    ? jobs.find((j) => j.id === jobDelete.pendingId)
    : null;

  const showProfileSelectorInCreate =
    profiles.length > 1 && selectedProfile === ALL_PROFILES;

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
          <div className="relative w-full max-w-lg border border-border bg-card shadow-2xl flex flex-col">
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
                className="font-display text-base tracking-wider uppercase"
              >
                {t.cron.newJob}
              </h2>
            </header>

            <div className="p-5 grid gap-4">
              {showProfileSelectorInCreate && (
                <div className="grid gap-2">
                  <Label htmlFor="cron-profile">Profile</Label>
                  <Select
                    id="cron-profile"
                    value={createProfile}
                    onValueChange={(v) => setCreateProfile(v)}
                  >
                    {profiles.map((p) => (
                      <SelectOption key={p.name} value={p.name}>
                        {p.name === activeProfileName
                          ? `${p.name} (active)`
                          : p.name}
                      </SelectOption>
                    ))}
                  </Select>
                </div>
              )}

              <div className="grid gap-2">
                <Label htmlFor="cron-name">{t.cron.nameOptional}</Label>
                <Input
                  id="cron-name"
                  autoFocus
                  placeholder={t.cron.namePlaceholder}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="cron-prompt">{t.cron.prompt}</Label>
                <textarea
                  id="cron-prompt"
                  className="flex min-h-[80px] w-full border border-border bg-background/40 px-3 py-2 text-sm font-courier shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 focus-visible:border-foreground/25"
                  placeholder={t.cron.promptPlaceholder}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="cron-schedule">{t.cron.schedule}</Label>
                  <Input
                    id="cron-schedule"
                    placeholder={t.cron.schedulePlaceholder}
                    value={schedule}
                    onChange={(e) => setSchedule(e.target.value)}
                  />
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="cron-deliver">{t.cron.deliverTo}</Label>
                  <Select
                    id="cron-deliver"
                    value={deliver}
                    onValueChange={(v) => setDeliver(v)}
                  >
                    <SelectOption value="local">
                      {t.cron.delivery.local}
                    </SelectOption>
                    <SelectOption value="telegram">
                      {t.cron.delivery.telegram}
                    </SelectOption>
                    <SelectOption value="discord">
                      {t.cron.delivery.discord}
                    </SelectOption>
                    <SelectOption value="slack">
                      {t.cron.delivery.slack}
                    </SelectOption>
                    <SelectOption value="email">
                      {t.cron.delivery.email}
                    </SelectOption>
                  </Select>
                </div>
              </div>

              <div className="flex justify-end">
                <Button
                  size="sm"
                  onClick={handleCreate}
                  disabled={creating}
                  prefix={creating ? <Spinner /> : <Plus />}
                >
                  {creating ? t.common.creating : t.common.create}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {profiles.length > 1 && (
        <div className="flex items-center gap-3 border border-border bg-muted/20 px-3 py-2">
          <span className="font-mondwest text-[0.65rem] tracking-[0.12em] uppercase text-muted-foreground">
            Profile
          </span>
          <Select
            value={selectedProfile}
            onValueChange={(v) => setSelectedProfile(v)}
            aria-label="Profile filter"
            className="w-48 text-xs [&>button]:h-7 [&>button]:py-0"
          >
            <SelectOption value={ALL_PROFILES}>All</SelectOption>
            {profiles.map((p) => (
              <SelectOption key={p.name} value={p.name}>
                {p.name === activeProfileName
                  ? `${p.name} (active)`
                  : p.name}
              </SelectOption>
            ))}
          </Select>
          {jobsLoading ? (
            <Spinner className="text-xs text-muted-foreground" />
          ) : null}
        </div>
      )}

      <div className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <Clock className="h-4 w-4" />
          {t.cron.scheduledJobs} ({jobs.length})
        </H2>

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
          const repeatLabel = getJobRepeatLabel(job);
          const triggerDisabled = !job.gateway_running;
          const triggerTooltip = triggerDisabled
            ? `Gateway not running for profile ${job.profile}`
            : t.cron.triggerNow;

          return (
            <Card key={`${job.profile}:${job.id}`}>
              <CardContent className="flex items-center gap-4 py-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className="font-medium text-sm truncate">
                      {title}
                    </span>
                    <Badge tone={STATUS_TONE[state] ?? "secondary"}>
                      {state}
                    </Badge>
                    {showProfileColumn && (
                      <Badge tone="outline">
                        {job.profile === activeProfileName
                          ? `${job.profile} (active)`
                          : job.profile}
                      </Badge>
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
                  <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
                    <span className="font-mono">{getJobScheduleDisplay(job)}</span>
                    {repeatLabel && (
                      <span className="font-mono">{repeatLabel}</span>
                    )}
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
                    title={triggerTooltip}
                    aria-label={triggerTooltip}
                    onClick={() => handleTrigger(job)}
                    disabled={triggerDisabled}
                  >
                    <Zap />
                  </Button>

                  <Button
                    ghost
                    destructive
                    size="icon"
                    title={t.common.delete}
                    aria-label={t.common.delete}
                    onClick={() => jobDelete.requestDelete(job.id)}
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
