export type SearchableTranscriptMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
};

export function filterTranscriptMessages<T extends SearchableTranscriptMessage>(
  messages: readonly T[],
  query: string,
): T[] {
  const normalized = query.trim().toLocaleLowerCase();
  if (!normalized) return [...messages];
  return messages.filter((message) => message.text.toLocaleLowerCase().includes(normalized));
}
