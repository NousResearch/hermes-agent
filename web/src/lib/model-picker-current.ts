/**
 * Resolve what the dashboard model picker should treat as "current".
 *
 * `/api/model/options` always returns the *main* chat model. Auxiliary and
 * MoA pickers reuse that catalog, so they must pass the slot assignment or
 * the header/highlight will show GLM (or whatever the chat model is) while
 * the task list shows the real Vision/MoA pin.
 */

export interface PickerCurrent {
  model: string;
  provider: string;
}

export function isAutoPickerCurrent(current: PickerCurrent): boolean {
  const provider = current.provider.trim().toLowerCase();
  return !provider || provider === "auto";
}

/** Map an auxiliary-task row onto picker current. Auto/empty → auto. */
export function assignmentToPickerCurrent(
  assignment: { model?: string; provider?: string } | null | undefined,
): PickerCurrent {
  const provider = String(assignment?.provider ?? "").trim();
  const model = String(assignment?.model ?? "").trim();
  if (!provider || provider.toLowerCase() === "auto") {
    return { model: "", provider: "auto" };
  }
  return { model, provider };
}

/**
 * `override` wins when the picker is for a non-main slot. Missing override
 * keeps the catalog's main model (chat-session / Set Main Model).
 */
export function resolvePickerCurrent(
  options: { model?: string; provider?: string } | null | undefined,
  override?: { model?: string; provider?: string } | null,
): PickerCurrent {
  if (override) {
    return {
      model: String(override.model ?? ""),
      provider: String(override.provider ?? ""),
    };
  }
  return {
    model: String(options?.model ?? ""),
    provider: String(options?.provider ?? ""),
  };
}

export function formatPickerCurrentLabel(current: PickerCurrent): string {
  if (isAutoPickerCurrent(current)) {
    return "auto (use main model)";
  }
  const model = current.model.trim() || "(unknown)";
  const provider = current.provider.trim();
  return provider ? `${model} · ${provider}` : model;
}

export function resolveInitialProviderSlug(
  providers: readonly { slug: string; is_current?: boolean }[],
  currentProvider: string,
): string {
  const slug = currentProvider.trim();
  if (slug && slug.toLowerCase() !== "auto") {
    const match = providers.find((p) => p.slug === slug);
    if (match) return match.slug;
  }
  return (providers.find((p) => p.is_current) ?? providers[0])?.slug ?? "";
}
