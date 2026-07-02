---
sidebar_position: 20
title: "KakaoTalk"
description: "Set up Hermes Agent as a KakaoTalk channel chatbot via Kakao i Open Builder"
---

# KakaoTalk Setup

Run Hermes Agent as a [KakaoTalk](https://www.kakaocorp.com/page/service/service/KakaoTalk) channel chatbot via [Kakao i Open Builder](https://i.kakao.com/)'s skill-server webhook model. The adapter lives as a bundled platform plugin under `plugins/platforms/kakao/` — no core edits, just enable it like any other platform.

KakaoTalk is South Korea's dominant messenger, used by the overwhelming majority of the country. If your users live there, this is how they reach you.

No business registration is required: a personal Kakao account, a free channel, and a free Open Builder bot are enough.

> Run `hermes gateway setup` and pick **Kakao** for a guided walk-through.

## How the bot responds

Kakao's skill-server model is **strictly request-response**: Kakao POSTs each user utterance to your webhook, and the only ways to answer are the HTTP response itself (within Kakao's **5-second SLA**) or a **one-shot callback URL**. There is no free-text push in this model, so the adapter answers in three tiers:

| Answer ready within | Delivery |
|---|---|
| `KAKAO_SYNC_TIMEOUT` (default 4s) | Immediate synchronous reply |
| `KAKAO_CALLBACK_TIMEOUT` (default 50s, measured after the sync budget) | One-shot callback POST (requires the callback option on the block) |
| Later | A "still working" bubble with an **답변 확인** (get answer) quick-reply button — also delivered via the callback; the finished answer is held and delivered on the user's next message or button tap |

With the callback option off, the last two tiers collapse into a plain "try again later" notice plus the held answer. Held answers stay eligible for 30 minutes, then drop as stale; if several pile up (e.g. a second bubble after an approval ack), they are joined (capped at 6000 chars) and delivered together, subject to the 3-bubble response limit. Inbound is text-only (`userRequest.utterance`); outbound is `simpleText` bubbles.

---

## Step 1: Create a KakaoTalk channel and a chatbot

1. Create a **KakaoTalk channel** at the [Kakao channel admin center](https://center-pf.kakao.com/) (this console has an official English UI — language toggle in the top bar): sign in with your personal Kakao account → **create a new channel** (**새 채널 만들기**) → fill in the profile: channel name, **search ID** (**검색용 아이디** — permanent once set; this is what users type into KakaoTalk search to find your bot in Step 5), and a category → create. Free and instant. The console's **Business Review** ("Apply for Review") is a separate opt-in requiring business verification (a business registration certificate for corporations; individual owners can verify with a KakaoTalk electronic certificate) that unlocks Business Channel features (Alimtalk/Brand Message, the Event API — see the Cron section). Running a bot does not need it, and the "business info not verified" notice on plain personal channels does not block anything.
2. Go to the [chatbot admin center (Kakao i Open Builder)](https://i.kakao.com/) (Korean-only UI — hence the Korean labels alongside the English throughout this guide) → the create-bot button (**봇 만들기** or **+ 채널 챗봇 만들기**, depending on console version) → **KakaoTalk chatbot** (**카카오톡 챗봇**).
3. You do **not** need to author intent blocks — Hermes is the brain. Everything routes through the fallback block (Step 4).

:::note
Kakao's *published* skill-callback guide still describes an "AI 챗봇 전환" (AI chatbot conversion) review as a prerequisite for the callback option used in Step 4 — apply via Settings → AI chatbot management (**[설정] → AI 챗봇 관리**), ~1–2 business days to approve — but that documentation is stale. Confirmed hands-on while writing this guide: newly created bots in 2026 do not show that gate at all, and the callback option is available on the block directly, no application step. If your bot is older, or you still see an **AI 챗봇 관리** tab, it's harmless to check/apply there, but a fresh bot should not need it.
:::

---

## Step 2: Expose the webhook port

Kakao delivers skill requests over public HTTPS. The default port is `8647` — override with `KAKAO_PORT` if needed.

```bash
# Cloudflare quick tunnel (dev — random hostname on every run;
# use a named tunnel in production for a fixed one)
cloudflared tunnel --url http://localhost:8647

# ngrok (good for dev)
ngrok http 8647
```

Copy the `https://...` URL. **Leave the tunnel running** while testing. For production, set up a fixed Cloudflare named tunnel so the skill URL doesn't change on restart — Kakao's skill URL must be re-saved (and the bot re-deployed) every time your public hostname changes.

Mind the latency budget: Kakao's 5-second SLA is measured on **Kakao's side**, so tunnel round-trip time counts against it. The adapter's default 4s sync budget leaves headroom for ~0.5s of tunnel latency; on a slow link, lower `KAKAO_SYNC_TIMEOUT` further.

---

## Step 3: Configure Hermes

Add to `~/.hermes/.env`:

```env
# Required. Kakao does not sign skill requests, so this shared secret —
# sent as a custom header you configure on the skill — is the only
# authentication. Use a long random value.
KAKAO_SKILL_SECRET=some-long-random-string

# Allowlist — botUserKey values (or KAKAO_ALLOW_ALL_USERS=true for dev).
# Don't know your key yet? Leave this unset; after completing Steps 4-5,
# message the bot once and copy the full key from the gateway log's
# "rejecting unauthorized user" warning — then set it here and restart.
KAKAO_ALLOWED_USERS=5af4d8cf...
```

Setting `KAKAO_SKILL_SECRET` is enough to enable the platform — the bundled-plugin scan picks up `plugins/platforms/kakao/` automatically, provided `aiohttp` is importable (it ships with the `messaging` extra; on a minimal install run `pip install aiohttp`). If you prefer explicit config, the equivalent `~/.hermes/config.yaml` block also works:

```yaml
gateway:
  platforms:
    kakao:
      enabled: true
```

---

## Step 4: Wire up the bot in Open Builder (skill, fallback block, channel, deploy)

In the Open Builder console:

1. Open the **Skills** menu (**[스킬]**) → create a skill with URL `https://<your-tunnel>/kakao/webhook` (note the `/kakao/webhook` path — the adapter listens there).
2. In the skill's **Headers** (**헤더**) section, add a custom header: name `X-Hermes-Kakao-Secret` (or your `KAKAO_SECRET_HEADER` override), value = your `KAKAO_SKILL_SECRET`. Requests without it get a 401.
3. Open **Scenario → fallback block** (**[시나리오] → 폴백 블록**): in the block's parameter settings pick your skill, and set the bot response to **skill data** (**스킬데이터**) so the skill's response is what the user sees.
4. On the same block, **enable the callback option** (**콜백 설정**, in the block's bot-response settings). Without it Kakao never issues a `callbackUrl`, and any answer slower than the sync budget is reduced to a "try again later" notice — the eventual answer is still held and delivered on the user's next message. On a fresh bot this option is just there (see the note in Step 1); if it's greyed out on an older bot, check Settings → AI chatbot management (**[설정] → AI 챗봇 관리**) first.
5. Open **Settings → KakaoTalk channel connection** (**[설정] → 카카오톡 채널 연동**) and connect your operational channel.
6. **Deploy**: left sidebar **Deploy** (**[배포]**) → keep **full deploy** (**전체 배포**) → click the deploy button (the release-note field is optional). Nothing reaches the real channel until you do this, and you must **re-deploy after *every* console change**, including the channel connection above — the web bot tester runs your draft, but the real channel only runs the last deployed version.

---

## Step 5: Run the gateway

```bash
hermes gateway
```

Verify the webhook end-to-end before blaming Kakao:

```bash
curl -i https://<your-tunnel>/kakao/webhook/health
# → {"status": "ok", "platform": "kakao"}
```

Now add the channel as a friend in the KakaoTalk app (search for the channel's search ID from Step 1) and message it. Expect this first-run sequence: your first message is **rejected** (fail-closed allowlist — copy your `botUserKey` from the gateway log, set `KAKAO_ALLOWED_USERS`, restart); the next message may be consumed by Hermes' home-channel onboarding prompt — answer or ignore it; real answers start from the message after that. (Running `/sethome` in a Kakao chat is pointless anyway: see Limitations.)

---

## Slow LLM responses

Kakao's callback token is single-use and — despite the docs saying "valid for 5 minutes" — **expires after roughly 1 minute in practice** (POSTs after that fail with `Invalid Callback token`). The adapter therefore budgets `KAKAO_CALLBACK_TIMEOUT` (default 50s) for the callback leg.

When the LLM is still running past that budget, the adapter spends the one-shot callback on a notice with a quick-reply button:

> 답변을 아직 만들고 있어요. 잠시 후 아래 버튼을 누르거나 아무 메시지나 보내 주시면 준비된 답변을 보여드릴게요.
> *("Still working on your answer. Tap the button below in a moment — or send any message — and I'll show you the finished answer.")*
>
> [ 답변 확인 ]

The user taps **답변 확인** (or sends any message) when convenient — that new utterance retrieves the held answer immediately. Same idea as the [LINE adapter](./line.md)'s slow-response postback button, adapted to Kakao's turn model.

While the callback is pending, the user sees an interim bubble (`KAKAO_SYNC_TIMEOUT` elapsed): "🤔 답변을 만들고 있어요. 잠시만 기다려 주세요." (*"Working on the answer — one moment."*) — customizable via `callback_waiting_text` in the platform's `extra` config.

---

## Cron / notification delivery

**Not supported — by the platform, not by the adapter.** Kakao's skill-server model can only answer a user's own utterance inside that request's window. Kakao does sell separate outbound products, but none can carry agent-generated free text, so the adapter does not use them: the Event API (bot-initiated messages — needs a business-verified channel and app, pre-registered event blocks, and the user must be a channel friend) and the Alimtalk / Brand Message template products (알림톡/브랜드메시지 — business verification, pre-reviewed templates, and per-message fees through broker agencies; Brand Message additionally requires channel-friend status, while Alimtalk can reach non-friends). `deliver: kakao` cron jobs and out-of-process `send_message` calls fail with a descriptive error; in-gateway sends that arrive after a turn's delivery slot is spent are held and served on the user's next message instead of being dropped. Point cron delivery at a push-capable platform ([Telegram](./telegram.md), [LINE](./line.md), [ntfy](./ntfy.md), …) and keep Kakao for conversational use.

---

## Environment variable reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `KAKAO_SKILL_SECRET` | yes | — | Shared secret verified (constant-time) against the skill's custom header |
| `KAKAO_SECRET_HEADER` | no | `X-Hermes-Kakao-Secret` | Header name carrying the secret; must match the skill config |
| `KAKAO_HOST` | no | `0.0.0.0` | Webhook bind host |
| `KAKAO_PORT` | no | `8647` | Webhook bind port |
| `KAKAO_PUBLIC_URL` | no | — | Public HTTPS base URL — informational (startup log) only |
| `KAKAO_ALLOWED_USERS` | one of | — | Comma-separated `botUserKey` values allowed to talk to the bot |
| `KAKAO_ALLOW_ALL_USERS` | dev only | `false` | Skip the allowlist entirely |
| `KAKAO_SYNC_TIMEOUT` | no | `4.0` | Seconds to wait for the answer before going async (hard platform cap: 5s incl. network) |
| `KAKAO_CALLBACK_TIMEOUT` | no | `50` | Seconds to wait after `useCallback` before spending the callback on the still-working notice |
| `KAKAO_BOT_ID` | no | — | Open Builder bot ID — informational only |

---

## Troubleshooting

**Kakao's default "I can't do that" reply (제가 할 수 있는 일이 아니에요).** The fallback block isn't wired to your skill. Re-do Step 4.3 (pick the skill *and* set the response to 스킬데이터), then re-deploy.

**Works in the web bot tester, silent on the real channel.** The tester runs your draft; the channel runs the deployed version. Deploy (again), and confirm the operational channel is connected.

**Skill-server timeout — "사용자의 스킬 서버에서 타임아웃이 발생했습니다. (1001)" in the skill error log** (Open Builder console → Skills → error history, **스킬 → 오류 내역**). Your sync response exceeded Kakao's 5s SLA — usually tunnel latency stacking on the sync budget. Lower `KAKAO_SYNC_TIMEOUT` (e.g. `3.5`).

**"네트워크 접근에 실패했습니다. 공인IP 또는 공중망 도메인을 사용해 주시기 바랍니다. (1005)" in the skill error log.** Kakao couldn't reach the skill URL at all — the tunnel isn't running, the quick-tunnel hostname rotated since you registered the skill (Step 2), or the URL points at a private/local address. Confirm `curl -i https://<your-tunnel>/kakao/webhook/health` works from *outside* your machine, and re-save the skill URL if the tunnel hostname changed.

Note: **오류 내역 only records errors from the operational channel** — failed bot-tester (봇테스트) runs never show up here, so use the tester for quick iteration but confirm real fixes against this log or the deployed channel.

**Approved a command, got "Command approved. The agent is resuming...", then nothing.** The approval ack spends the turn's single answer slot, so the actual result is held — send any message (even "?") to retrieve it. The gateway logs `no open skill request ... holding the message` when this happens; that line is informational, not an error.

**Delayed-answer notice — "답변 생성이 예상보다 오래 걸리고 있어요" ("the answer is taking longer than expected") on every long answer.** Kakao issued no `callbackUrl` — the block's callback option is off, you enabled it and didn't re-deploy, or (on an older bot) the callback is still gated behind the AI-chatbot review — see the note in Step 1.

**`callback POST failed (400) ... Invalid Callback token` in the gateway log.** In the **bot tester** this is normal — the tester doesn't support callbacks (`cbtest:` tokens are never accepted); test on the deployed channel. On the real channel it means the one-shot token was already spent or has expired (~1 minute); the adapter holds the answer for the user's next message (within the 30-minute hold window), so nothing is lost.

**401 `invalid secret`.** The skill's custom header name/value doesn't match `KAKAO_SECRET_HEADER` / `KAKAO_SKILL_SECRET`.

**Bot replies, but only with a system notice.** No LLM provider is configured — set `ANTHROPIC_API_KEY` (or another provider key) in `~/.hermes/.env` and restart the gateway.

**Finding your `botUserKey` for the allowlist.** With the allowlist unset the adapter rejects everyone (fail-closed) and logs a warning with the full key: `Kakao: rejecting unauthorized user <botUserKey> -- add this botUserKey to KAKAO_ALLOWED_USERS to allow them.` Message the bot once, copy the key from the log, set `KAKAO_ALLOWED_USERS`, restart. Bot-tester and real-channel keys differ; allowlist both if you use both.

---

## Limitations

* **Request-response only.** One answer per user utterance, delivered in that request's window (or held for the next one). No proactive messages, no cron delivery, no home-channel semantics — the skill-server model has no push, and Kakao's separate outbound products (Event API, Alimtalk) can't carry agent-generated text (see Cron section).
* **Answers can require one extra tap.** Anything slower than the ~1-minute callback window arrives via the 답변 확인 button / next message — including results that follow a command-approval ack, since the ack itself spends the turn's answer. That's the platform ceiling, not a config issue.
* **Text in, text out.** v1 handles text utterances and replies with `simpleText` bubbles (1000 chars each, max 3 per response, chunked at ~900). No media send/receive, no typing indicator, no message editing / streaming.
* **No Markdown rendering.** Formatting is stripped before sending; URLs are preserved.
* **Bot-scoped user IDs.** `botUserKey` identifies a user *per bot* — it can't be correlated with the same person on another channel or with Kakao Login accounts.
