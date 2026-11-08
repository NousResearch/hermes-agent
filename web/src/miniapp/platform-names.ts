// Explicit overrides for keys that plain word-splitting can't get right --
// acronyms (CLI, API) or specific capitalization a title-case pass wouldn't
// know about. Anything not listed falls through to formatPlatformName's
// generic snake_case -> Title Case split, which is a reasonable default for
// the rest (telegram_inline -> "Telegram Inline", feishu -> "Feishu", etc.).
const PLATFORM_NAME_OVERRIDES: Record<string, string> = {
  cli: "CLI",
  api_server: "API Server",
};

// Was `textTransform: "capitalize"` (CSS) directly on the raw key -- that
// only uppercases the first letter of the WHOLE string (CSS doesn't treat
// "_" as a word boundary), so "cli" -> "Cli" and "api_server" -> "Api_server"
// instead of "CLI" / "API Server". This does real word-splitting plus an
// override table for cases plain title-casing can't get right.
export function formatPlatformName(name: string): string {
  const override = PLATFORM_NAME_OVERRIDES[name];
  if (override) return override;
  return name
    .split("_")
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : word))
    .join(" ");
}
