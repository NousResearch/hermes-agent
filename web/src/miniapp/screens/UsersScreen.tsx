import { useEffect, useState } from "react";
import { del, get, post } from "../api";
import { useMiniApp } from "../context";
import { needsRestartBanner } from "../restart-banner";
import type { MiniAppStatusExtra, TelegramAllowlistEntry, TelegramAllowlistResponse } from "../types";

function displayFor(u: TelegramAllowlistEntry): string {
  const handle = u.username ? `@${u.username}` : null;
  if (handle && u.name) return `${handle} · ${u.name}`;
  return handle || u.name || "name unavailable";
}

function initialsFor(u: TelegramAllowlistEntry): string {
  return (u.name || u.username || u.user_id).slice(0, 2).toUpperCase();
}

function subLineFor(u: TelegramAllowlistEntry): string {
  if (u.added_at == null) return u.user_id;
  const d = new Date(u.added_at * 1000);
  const added = d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return `${u.user_id} · added ${added}`;
}

export function UsersScreen({ statusExtra }: { statusExtra: MiniAppStatusExtra }) {
  const { showToast, askConfirm, askRestartGateway, refreshStatus } = useMiniApp();
  const [allowlist, setAllowlist] = useState<TelegramAllowlistEntry[] | null>(null);
  const [newUser, setNewUser] = useState("");

  const load = () =>
    get<TelegramAllowlistResponse>("/api/telegram/allowlist")
      .then((r) => setAllowlist(r.allowlist))
      .catch(() => setAllowlist([]));

  useEffect(() => {
    load();
  }, []);

  if (!allowlist) return null;

  // Extracted to restart-banner.ts (unit-tested there, one case per null
  // operand + the true/false comparison cases) rather than inlined here --
  // see that file for the fail-closed null-check rationale.
  const showRestartBanner = needsRestartBanner(statusExtra);

  const addUser = async () => {
    const v = newUser.trim();
    if (!/^\d{5,15}$/.test(v)) {
      showToast("Enter a numeric Telegram user ID");
      return;
    }
    if (allowlist.some((u) => u.user_id === v)) {
      showToast("Already listed");
      return;
    }
    try {
      await post("/api/telegram/allowlist", { user_id: v });
      setNewUser("");
      showToast("User added");
      load();
      refreshStatus();
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Couldn't add user");
    }
  };

  const removeUser = (u: TelegramAllowlistEntry) =>
    askConfirm({
      title: "Remove this user?",
      body: `${u.username ? "@" + u.username : "User " + u.user_id} loses dashboard access and can no longer chat with the bot.`,
      label: "Remove user",
      destructive: true,
      run: async () => {
        try {
          await del(`/api/telegram/allowlist/${u.user_id}`);
          showToast("User removed");
          load();
          refreshStatus();
        } catch {
          showToast("Couldn't remove user");
        }
      },
    });

  return (
    <div style={{ padding: "16px 14px 24px", display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--t3)", fontFamily: "var(--mono)", padding: "0 4px" }}>
        Telegram users
      </div>
      <div style={{ background: "var(--card)", border: "1px solid var(--line)", borderRadius: 14, padding: "11px 14px", fontSize: 12, color: "var(--t2)", lineHeight: 1.55 }}>
        This list can also be changed by <b style={{ color: "var(--mid)" }}>asking Hermes in chat</b> to add or remove Telegram users and groups.
      </div>

      {showRestartBanner && (
        <div
          style={{
            background: "color-mix(in srgb, var(--warning) 10%, var(--bg))",
            border: "1px solid color-mix(in srgb, var(--warning) 50%, transparent)",
            borderRadius: 14,
            padding: "11px 14px",
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          <div style={{ fontSize: 12, color: "var(--mid)", lineHeight: 1.5 }}>
            <b style={{ color: "var(--warning)" }}>Restart needed:</b> Telegram user/chat list changed since
            the gateway last started — changes won't take effect until it's restarted.
          </div>
          <button
            onClick={askRestartGateway}
            style={{
              alignSelf: "flex-start",
              fontFamily: "var(--mono)",
              fontSize: 11,
              letterSpacing: "0.05em",
              padding: "6px 12px",
              borderRadius: 8,
              border: "1px solid var(--warning)",
              background: "transparent",
              cursor: "pointer",
              color: "var(--warning)",
            }}
          >
            Restart gateway
          </button>
        </div>
      )}

      <div
        style={{
          background: "color-mix(in srgb, var(--warning) 8%, var(--bg))",
          border: "1px solid color-mix(in srgb, var(--warning) 45%, transparent)",
          borderRadius: 14,
          padding: "11px 14px",
          fontSize: 12,
          color: "var(--t2)",
          lineHeight: 1.55,
        }}
      >
        <b style={{ color: "var(--warning)" }}>Warning:</b> anyone on this list can also chat with the bot directly.
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={newUser}
          onChange={(e) => setNewUser(e.target.value)}
          placeholder="Telegram user ID"
          inputMode="numeric"
          style={{
            flex: 1,
            minWidth: 0,
            boxSizing: "border-box",
            background: "var(--card)",
            border: "1px solid var(--line)",
            borderRadius: 11,
            padding: "10px 13px",
            fontFamily: "var(--mono)",
            fontSize: 13,
            color: "var(--mid)",
            outline: "none",
          }}
        />
        <button
          onClick={addUser}
          style={{
            padding: "10px 17px",
            borderRadius: 11,
            border: "none",
            background: "var(--accent)",
            color: "var(--on-accent)",
            fontSize: 13,
            fontWeight: 650,
            cursor: "pointer",
          }}
        >
          Add
        </button>
      </div>

      {allowlist.map((u) => (
        <div
          key={u.user_id}
          style={{ background: "var(--card)", border: "1px solid var(--line)", borderRadius: 14, padding: "11px 14px", display: "flex", alignItems: "center", gap: 11 }}
        >
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 99,
              background: "var(--card2)",
              border: "1px solid var(--line)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontFamily: "var(--mono)",
              fontSize: 11,
              color: "var(--t2)",
              flexShrink: 0,
            }}
          >
            {initialsFor(u)}
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 13.5, fontWeight: 600, color: "var(--mid)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {displayFor(u)}
            </div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--t3)", marginTop: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {subLineFor(u)}
            </div>
          </div>
          <button
            onClick={() => removeUser(u)}
            aria-label="Remove"
            style={{
              width: 32,
              height: 32,
              borderRadius: 9,
              border: "1px solid var(--line)",
              background: "transparent",
              cursor: "pointer",
              color: "var(--t3)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
              <path d="M1.5 1.5l9 9M10.5 1.5l-9 9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  );
}
