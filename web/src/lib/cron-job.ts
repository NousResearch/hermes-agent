import type { CronJob, CronJobMutation } from "./api";

export interface CronJobFormState {
  name: string;
  prompt: string;
  schedule: string;
  deliver: string;
  skills: string[];
  provider: string;
  model: string;
  base_url: string;
  script: string;
  no_agent: boolean;
  context_from: string;
  continuity: boolean;
  enabled_toolsets: string[];
  workdir: string;
}

/** Split a comma/newline list (or array) into trimmed, non-empty items. */
export function splitCronList(value: unknown): string[] {
  const items = Array.isArray(value)
    ? value
    : typeof value === "string"
      ? value.split(/[\n,]/)
      : [];
  return items.map((item) => String(item).trim()).filter(Boolean);
}

/** Trim to a non-empty string, or null. Optionally strip trailing slashes
 * (base URLs). Mirrors the backend's `_cron_optional_text`. */
function optionalText(value: string, stripTrailingSlash = false): string | null {
  const text = stripTrailingSlash ? value.trim().replace(/\/+$/, "") : value.trim();
  return text || null;
}

/** Read a stored string field as a plain string ("" when absent). */
function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/** Build the create/update payload. Optional fields collapse to null so an
 * update explicitly clears them rather than leaving stale values. */
export function buildCronJobPayload(form: CronJobFormState): CronJobMutation {
  // The `continuity` toggle is stored as the reserved "self" entry in
  // context_from (the job's own previous output). Users never type "self" —
  // the checkbox is the surface; strip any hand-typed variant first.
  const contextFrom = splitCronList(form.context_from).filter(
    (item) => item.toLowerCase() !== "self",
  );
  if (form.continuity) contextFrom.push("self");
  const enabledToolsets = form.enabled_toolsets.filter(Boolean);
  return {
    name: form.name.trim(),
    prompt: form.prompt.trim(),
    schedule: form.schedule.trim(),
    deliver: form.deliver.trim() || "local",
    skills: form.skills.filter(Boolean),
    provider: optionalText(form.provider),
    model: optionalText(form.model),
    base_url: optionalText(form.base_url, true),
    script: optionalText(form.script),
    no_agent: Boolean(form.no_agent),
    context_from: contextFrom.length > 0 ? contextFrom : null,
    enabled_toolsets: enabledToolsets.length > 0 ? enabledToolsets : null,
    workdir: optionalText(form.workdir),
  };
}

export function cronJobHasExecutionContent(
  job: Pick<CronJobMutation, "prompt" | "skills" | "script">,
): boolean {
  const skills = Array.isArray(job.skills) ? job.skills.filter(Boolean) : [];
  return Boolean(asString(job.prompt).trim() || asString(job.script).trim() || skills.length);
}

export function cronJobFormFromJob(job: CronJob): CronJobFormState {
  const storedRefs = splitCronList(job.context_from);
  // Raw store records carry the reserved "self" entry inside context_from;
  // tool/RPC-formatted records strip it and set an explicit continuity flag.
  const continuity =
    Boolean((job as { continuity?: boolean }).continuity) ||
    storedRefs.some((item) => item.toLowerCase() === "self");
  const externalRefs = storedRefs.filter((item) => item.toLowerCase() !== "self");
  return {
    name: asString(job.name),
    prompt: asString(job.prompt),
    schedule:
      asString(job.schedule?.expr) ||
      asString(job.schedule?.run_at) ||
      asString(job.schedule_display),
    deliver: asString(job.deliver) || "local",
    skills: Array.isArray(job.skills) ? job.skills.filter(Boolean) : [],
    provider: asString(job.provider),
    model: asString(job.model),
    base_url: asString(job.base_url),
    script: asString(job.script),
    no_agent: Boolean(job.no_agent),
    context_from: externalRefs.join("\n"),
    continuity,
    enabled_toolsets: splitCronList(job.enabled_toolsets),
    workdir: asString(job.workdir),
  };
}

/** How a job's `last_status` should render. The scheduler writes a small,
 *  closed set of literals; every literal maps to an explicit tone here so a
 *  new status can never fall through to a neutral "unknown"-looking badge.
 *  In particular `delivery_failed` (agent run succeeded, output never reached
 *  the target) is amber, not green and not the same red as a run error, and
 *  its detail lives in `last_delivery_error` (last_error is null for it). */
export type CronLastResultTone = "success" | "warning" | "destructive";

export interface CronLastResult {
  status: string;
  tone: CronLastResultTone;
  /** Human detail to show next to the badge; null when nothing to add. */
  detail: string | null;
}

const CRON_LAST_RESULT_TONE: Record<string, CronLastResultTone> = {
  ok: "success",
  delivery_failed: "warning",
  blocked_config: "warning",
  error: "destructive",
};

export function cronLastResult(
  job: Pick<CronJob, "last_status" | "last_error" | "last_delivery_error">,
): CronLastResult | null {
  const status = asString(job.last_status).trim();
  if (!status) return null;
  const tone = CRON_LAST_RESULT_TONE[status] ?? "destructive";
  if (status === "ok") return { status, tone, detail: null };
  const detail =
    status === "delivery_failed"
      ? asString(job.last_delivery_error).trim() || asString(job.last_error).trim()
      : asString(job.last_error).trim() || asString(job.last_delivery_error).trim();
  return { status, tone, detail: detail || null };
}
