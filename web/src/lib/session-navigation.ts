export function buildChatResumePath(sessionId: string): string {
  return `/chat?resume=${encodeURIComponent(sessionId)}`;
}
