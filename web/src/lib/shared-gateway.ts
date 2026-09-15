import type { StatusResponse } from "@/lib/api";

/** Profiles a gateway restart would blip when the managed profile is carried by the shared
 *  multiplexer (default first). `null` for a standalone gateway or an older backend without
 *  `gateway_shared_with`; those keep the plain restart copy. */
export function sharedGatewayProfiles(
  status: Pick<StatusResponse, "gateway_shared_with"> | null | undefined,
): string[] | null {
  const shared = status?.gateway_shared_with;
  if (!Array.isArray(shared)) return null;
  const names = [...new Set(shared.map((n) => String(n).trim()).filter(Boolean))];
  if (names.length < 2) return null;
  return names.sort((a, b) =>
    a === "default" ? -1 : b === "default" ? 1 : a.localeCompare(b),
  );
}

export function sharedGatewayRestartDescription(profiles: string[]): string {
  return `All bots on this device reconnect: ${profiles.join(", ")}`;
}

export function sharedGatewayRestartedMessage(count: number): string {
  return `Shared gateway restarted (${count} ${count === 1 ? "bot" : "bots"})`;
}

/** The REST layer throws `"<status>: <body>"`; a 409 on gateway start/stop for a served profile
 *  carries the multiplexer explanation in `detail`. Return it as a plain sentence, else null. */
export function servedProfileRefusal(error: unknown): string | null {
  const text = error instanceof Error ? error.message : String(error ?? "");
  if (!text.startsWith("409")) return null;
  const detail = text.match(/"detail"\s*:\s*"([^"]+)"/)?.[1];
  return detail ?? text.replace(/^409:\s*/, "");
}
