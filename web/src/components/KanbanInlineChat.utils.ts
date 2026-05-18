export type StreamEvent = {
  type?: string;
  card_id?: string;
  content?: string;
  status?: string;
  ok?: boolean;
};

export function parseSseChunk(buffer: string): { events: StreamEvent[]; rest: string } {
  const parts = buffer.split(/\n\n/);
  const rest = parts.pop() ?? "";
  const events = parts.flatMap((part) => {
    const data = part
      .split(/\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (!data) return [];
    try {
      return [JSON.parse(data) as StreamEvent];
    } catch {
      return [{ type: "result", content: data }];
    }
  });
  return { events, rest };
}
