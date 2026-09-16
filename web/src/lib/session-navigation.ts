export function buildChatResumePath(sessionId: string, profile?: string): string {
  const profileQuery = profile ? `&profile=${encodeURIComponent(profile)}` : "";
  return `/chat?resume=${encodeURIComponent(sessionId)}${profileQuery}`;
}
