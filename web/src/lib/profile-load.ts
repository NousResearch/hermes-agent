export interface ProfileStatusLoad<T> {
  loadedProfile: string;
  status: T | null;
  error: string | null;
}

export function resolveProfileStatusLoad<T>(
  profile: string,
  result: PromiseSettledResult<T>,
): ProfileStatusLoad<T> {
  if (result.status === "fulfilled") {
    return { loadedProfile: profile, status: result.value, error: null };
  }
  const error =
    result.reason instanceof Error ? result.reason.message : String(result.reason);
  return { loadedProfile: profile, status: null, error };
}
