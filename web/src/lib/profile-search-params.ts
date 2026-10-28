function effectiveProfile(selection: string, currentProfile: string): string {
  return selection || currentProfile || "default";
}

export function profileSelectionSearchParams(
  previous: URLSearchParams,
  selection: string,
  previousSelection: string,
  currentProfile: string,
): URLSearchParams {
  const next = new URLSearchParams(previous);

  if (selection) next.set("profile", selection);
  else next.delete("profile");

  if (
    effectiveProfile(selection, currentProfile) !==
    effectiveProfile(previousSelection, currentProfile)
  ) {
    next.delete("resume");
  }

  return next;
}
