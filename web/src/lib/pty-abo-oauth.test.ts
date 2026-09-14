import { describe, expect, it } from "vitest";

import { ptyAboOauthChannelKey, ptyAboOauthParams } from "@/lib/pty-abo-oauth";

describe("ptyAboOauthParams", () => {
  it("binds ChatGPT Codex OAuth and ignores the API-key openai slug", () => {
    const abo = ptyAboOauthParams(
      new URLSearchParams("provider=openai-codex&model=gpt-5.6-terra&chatgpt_mode=codex"),
    );
    expect(abo).toEqual({
      provider: "openai-codex",
      model: "gpt-5.6-terra",
      chatgpt_mode: "codex",
    });
    expect(ptyAboOauthParams(new URLSearchParams("provider=openai&model=gpt-4o")).provider).toBeUndefined();
  });

  it("keeps Grok, Claude and ChatGPT Chat as distinct PTY identities", () => {
    const grok = ptyAboOauthChannelKey(new URLSearchParams("provider=xai-oauth&model=grok-4.6"));
    const claude = ptyAboOauthChannelKey(new URLSearchParams("provider=anthropic&model=claude-sonnet-4"));
    const chat = ptyAboOauthChannelKey(
      new URLSearchParams("provider=openai-codex&model=gpt-5.4&chatgpt_mode=chat"),
    );
    const codex = ptyAboOauthChannelKey(
      new URLSearchParams("provider=openai-codex&model=gpt-5.6-terra&chatgpt_mode=codex"),
    );
    expect(new Set([grok, claude, chat, codex]).size).toBe(4);
  });
});
