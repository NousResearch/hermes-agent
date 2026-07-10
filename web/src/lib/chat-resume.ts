export const RESUME_NONCE_PARAM = "resumeNonce";

export function buildResumeInChatUrl(sessionId: string): string {
  const params = new URLSearchParams({
    resume: sessionId,
    [RESUME_NONCE_PARAM]: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
  });
  return `/chat?${params.toString()}`;
}

export function buildChatAttachScope(
  sessionId: string | null,
  profile: string | null | undefined,
): string {
  return `${profile ?? "default"}::${sessionId ?? "fresh"}`;
}
