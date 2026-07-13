export function shouldProbeDashboardAuth(): boolean {
  return (
    typeof window !== "undefined" &&
    window.__HERMES_AUTH_REQUIRED__ === true
  );
}
