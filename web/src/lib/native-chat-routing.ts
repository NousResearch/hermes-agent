export type NativeReasoningLevel = "auto" | "low" | "medium" | "high" | "max";

export interface NativeModelChoice {
  label: string;
  model: string;
  provider: string;
}

export interface NativeRoutingSelection {
  model?: NativeModelChoice;
  reasoning: NativeReasoningLevel;
}

export interface ModelOptionsProvider {
  slug?: string;
  name?: string;
  models?: string[];
}

export interface ModelOptionsCatalog {
  providers?: ModelOptionsProvider[];
}

export const NATIVE_REASONING_OPTIONS: ReadonlyArray<{ value: NativeReasoningLevel; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
  { value: "max", label: "Max" },
];

export function nativeChatSessionCreateParams(
  profile: string | undefined,
  selection: NativeRoutingSelection,
): Record<string, unknown> {
  const params: Record<string, unknown> = {
    close_on_disconnect: true,
    source: "dashboard",
    ...(profile ? { profile } : {}),
  };
  if (selection.model) {
    params.model = selection.model.model;
    params.provider = selection.model.provider;
  }
  if (selection.reasoning !== "auto") params.reasoning_effort = selection.reasoning;
  return params;
}

function modelLabel(model: string): string {
  const id = model.split("/").at(-1) ?? model;
  if (id === "gpt-5.6-luna") return "GPT Luna";
  if (id === "gpt-5.6-sol") return "GPT Sol";
  if (id === "minimax-m3:free" || id === "minimax-m3-free") return "MiniMax M3 Free";
  return id;
}

/** Select the requested compact native-chat choices from the live model catalog. */
export function nativeChatModelChoices(catalog: ModelOptionsCatalog): NativeModelChoice[] {
  const choices: NativeModelChoice[] = [];
  for (const provider of catalog.providers ?? []) {
    const slug = String(provider.slug ?? "").trim();
    if (!slug) continue;
    for (const model of provider.models ?? []) {
      const id = String(model).trim();
      const lower = id.toLowerCase();
      const wanted = lower === "gpt-5.6-luna" || lower.endsWith("/gpt-5.6-luna") ||
        lower === "gpt-5.6-sol" || lower.endsWith("/gpt-5.6-sol") ||
        lower === "minimax-m3:free" || lower.endsWith("/minimax-m3:free");
      if (!wanted || choices.some((choice) => choice.model === id && choice.provider === slug)) continue;
      choices.push({ label: modelLabel(id), model: id, provider: slug });
    }
  }
  return choices;
}

export function selectionFromSearchParams(
  model: string | null,
  provider: string | null,
  reasoning: string | null,
  choices: NativeModelChoice[],
): NativeRoutingSelection {
  const selectedModel = model && provider
    ? choices.find((choice) => choice.model === model && choice.provider === provider) ?? {
        model,
        provider,
        label: modelLabel(model),
      }
    : undefined;
  const validReasoning = NATIVE_REASONING_OPTIONS.some((option) => option.value === reasoning)
    ? reasoning as NativeReasoningLevel
    : "auto";
  return { model: selectedModel, reasoning: validReasoning };
}
