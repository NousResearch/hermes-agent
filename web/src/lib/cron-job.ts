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
  enabled_toolsets: string[];
  workdir: string;
}

export function splitCronList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  if (typeof value !== "string") return [];
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function optionalText(value: string): string | null {
  const text = value.trim();
  return text || null;
}

function optionalBaseUrl(value: string): string | null {
  const text = optionalText(value);
  return text ? text.replace(/\/+$/, "") : null;
}

function listToText(value: unknown, separator: string): string {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean).join(separator);
  }
  return typeof value === "string" ? value : "";
}

export function buildCronJobPayload(form: CronJobFormState): CronJobMutation {
  const contextFrom = splitCronList(form.context_from);
  const enabledToolsets = form.enabled_toolsets.filter(Boolean);
  return {
    name: form.name.trim(),
    prompt: form.prompt.trim(),
    schedule: form.schedule.trim(),
    deliver: form.deliver.trim() || "local",
    skills: form.skills.filter(Boolean),
    provider: optionalText(form.provider),
    model: optionalText(form.model),
    base_url: optionalBaseUrl(form.base_url),
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
  const prompt = typeof job.prompt === "string" ? job.prompt.trim() : "";
  const script = typeof job.script === "string" ? job.script.trim() : "";
  const skills = Array.isArray(job.skills)
    ? job.skills.map((skill) => String(skill).trim()).filter(Boolean)
    : [];
  return Boolean(prompt || script || skills.length > 0);
}

export function cronJobFormFromJob(job: CronJob): CronJobFormState {
  return {
    name: typeof job.name === "string" ? job.name : "",
    prompt: typeof job.prompt === "string" ? job.prompt : "",
    schedule:
      (typeof job.schedule?.expr === "string" && job.schedule.expr) ||
      (typeof job.schedule?.run_at === "string" && job.schedule.run_at) ||
      (typeof job.schedule_display === "string" ? job.schedule_display : ""),
    deliver: typeof job.deliver === "string" && job.deliver ? job.deliver : "local",
    skills: Array.isArray(job.skills) ? job.skills.filter(Boolean) : [],
    provider: typeof job.provider === "string" ? job.provider : "",
    model: typeof job.model === "string" ? job.model : "",
    base_url: typeof job.base_url === "string" ? job.base_url : "",
    script: typeof job.script === "string" ? job.script : "",
    no_agent: Boolean(job.no_agent),
    context_from: listToText(job.context_from, "\n"),
    enabled_toolsets: Array.isArray(job.enabled_toolsets)
      ? job.enabled_toolsets.filter(Boolean)
      : splitCronList(job.enabled_toolsets),
    workdir: typeof job.workdir === "string" ? job.workdir : "",
  };
}
