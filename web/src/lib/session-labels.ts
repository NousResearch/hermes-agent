import type { SessionInfo } from "@/lib/api";

export function telegramThreadLabel(session: SessionInfo): string | null {
  if (session.source !== "telegram") return null;

  let topic: string | null = null;
  try {
    const origin = session.origin_json ? JSON.parse(session.origin_json) : null;
    const rawTopic = origin?.chat_topic ?? origin?.topic_name ?? null;
    if (typeof rawTopic === "string" && rawTopic.trim()) topic = rawTopic.trim();
  } catch {
    // Best-effort display only; malformed origin JSON should not break the list.
  }

  const parts = ["telegram"];
  const who = session.display_name?.trim();
  if (who) parts.push(who);
  if (topic) {
    parts.push(topic);
  } else if (session.thread_id !== null && session.thread_id !== undefined && `${session.thread_id}`) {
    parts.push(`thread ${session.thread_id}`);
  }
  return parts.join(" · ");
}
