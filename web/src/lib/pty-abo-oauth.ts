/** OAuth Abo identity for dashboard /chat — never the API-key ``openai`` route. */

const OAUTH_ABOS = new Set(["xai-oauth", "anthropic", "openai-codex"]);

export function ptyAboOauthParams(search: URLSearchParams): Record<string, string> {
  const provider = (search.get("provider") || "").trim();
  const model = (search.get("model") || "").trim();
  const mode = (search.get("chatgpt_mode") || "").trim();
  const out: Record<string, string> = {};
  if (OAUTH_ABOS.has(provider)) out.provider = provider;
  if (model) out.model = model;
  if (mode === "chat" || mode === "codex") out.chatgpt_mode = mode;
  return out;
}

export function ptyAboOauthChannelKey(search: URLSearchParams): string {
  const p = ptyAboOauthParams(search);
  return `${p.provider ?? ""}\0${p.model ?? ""}\0${p.chatgpt_mode ?? ""}`;
}
