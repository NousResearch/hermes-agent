import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(__dirname, "..");
const page = readFileSync(resolve(webRoot, "src/pages/VoiceDispatchPage.tsx"), "utf8");
const api = readFileSync(resolve(webRoot, "src/lib/api.ts"), "utf8");

assert.doesNotMatch(page, /call\.name\s*===\s*["']approve_delegate_action["']/, "Grok-callable approve_delegate_action must not be handled by the browser");
assert.doesNotMatch(page, /call\.name\s*===\s*["']approve_hermes_action["']/, "Grok-callable approve_hermes_action must not be handled by the browser");
assert.doesNotMatch(page, /"always"\s*,\s*"deny"/, "Voice approval UI must not offer always approval in v1");
assert.match(page, /explicit\s+Hermes\s+UI\s+click/i, "Voice instructions should tell Grok approval requires an explicit UI click");
assert.match(page, /approveFromUi\s*=\s*useCallback\(async \(choice: "once" \| "session" \| "deny"\)/, "UI approval handler must only accept once/session/deny");
assert.match(api, /export type VoiceApprovalChoice = "once" \| "session" \| "deny";/, "API should expose a narrowed voice approval choice type");
assert.doesNotMatch(api, /choice:\s*"once" \| "session" \| "always" \| "deny"/, "API approval requests must not accept always");

console.log("voiceDispatchApprovalSafety tests passed");
