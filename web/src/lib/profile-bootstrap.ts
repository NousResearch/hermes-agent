declare global {
  interface Window {
    __HERMES_INITIAL_PROFILE__?: string;
  }
}

export function dashboardInitialProfile(): string {
  if (typeof window === "undefined") return "";
  return window.__HERMES_INITIAL_PROFILE__ ?? "";
}

export function initialProfileScope(
  searchParams: URLSearchParams,
  bootstrapProfile = dashboardInitialProfile(),
): string {
  const urlProfile = searchParams.get("profile");
  return urlProfile === null ? bootstrapProfile : urlProfile;
}

export function shouldAdoptActiveProfile(
  urlProfile: string | null,
  bootstrapProfile: string,
  currentProfile: string,
  activeProfile: string,
): boolean {
  return (
    urlProfile === null &&
    !bootstrapProfile &&
    activeProfile !== currentProfile
  );
}
