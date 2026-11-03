import { useCallback, useEffect, useRef, useState } from "react";
import { Copy, ExternalLink, Link2 } from "lucide-react";
import * as QRCode from "qrcode";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import type { MessagingPlatform, RelayLinkCode, RelayLinkedAccount } from "@/lib/api";
import { errorMessage } from "@/lib/api-error";

const POLL_MS = 3000;

function accountLabel(account: RelayLinkedAccount): string {
  const name = account.displayName || (account.handle ? `@${account.handle.replace(/^@/, "")}` : "");
  return name || account.platformUserId;
}

/**
 * Team Gateway setup for a platform the relay's shared bot carries (#141546).
 *
 * On such an instance a bot of the user's own is switched off at every gateway start, so the
 * working setup is linking their chat account to the shared bot: mint a one-time code here, send
 * `/link <code>` to the bot, and the relay routes that account's messages to this agent.
 */
export function RelayLinkPanel({
  platform,
  showToast,
}: {
  platform: MessagingPlatform;
  showToast: (message: string, type: "success" | "error") => void;
}) {
  const [accounts, setAccounts] = useState<RelayLinkedAccount[] | null>(null);
  const [botUrl, setBotUrl] = useState<string | null>(null);
  const [link, setLink] = useState<RelayLinkCode | null>(null);
  const [qrDataUrl, setQrDataUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // Accounts linked when the code was minted: a redeemed code shows up as one more.
  const linkedBefore = useRef(0);

  const loadAccounts = useCallback(async () => {
    try {
      const res = await api.getRelayLinks();
      const row = res.platforms.find((p) => p.platform === platform.id);
      setAccounts(row?.accounts ?? []);
      setBotUrl(row?.bot_url ?? null);
      setError("");
      return row?.accounts ?? [];
    } catch (loadError) {
      setAccounts(null);
      setError(errorMessage(loadError));
      return null;
    }
  }, [platform.id]);

  useEffect(() => {
    void loadAccounts();
  }, [loadAccounts]);

  useEffect(() => {
    if (!link) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      try {
        const state = await api.getRelayLinkState(link.code);
        if (cancelled) return;
        if (state.state === "pending") {
          timer = setTimeout(poll, POLL_MS);
          return;
        }
        setLink(null);
        setQrDataUrl("");
        if (state.state === "refused") {
          setError(`${platform.name} refused the link${state.reason ? ` (${state.reason})` : ""}. Get a new code and try again.`);
          return;
        }
        const after = await loadAccounts();
        if (!cancelled && after && after.length > linkedBefore.current) {
          showToast(`${platform.name} account linked. Message the bot to talk to your agent.`, "success");
        } else if (!cancelled) {
          setError("That code expired before it was used. Get a new code and try again.");
        }
      } catch {
        if (!cancelled) timer = setTimeout(poll, POLL_MS);
      }
    };
    timer = setTimeout(poll, POLL_MS);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [link, loadAccounts, platform.name, showToast]);

  const start = async () => {
    setBusy(true);
    setError("");
    try {
      const res = await api.requestRelayLink(platform.id);
      linkedBefore.current = accounts?.length ?? 0;
      setLink(res);
      const target = res.bot_url ?? botUrl;
      setQrDataUrl(
        target ? await QRCode.toDataURL(target, { errorCorrectionLevel: "M", margin: 1, width: 192 }) : "",
      );
    } catch (startError) {
      setError(errorMessage(startError));
    } finally {
      setBusy(false);
    }
  };

  const unlink = async (account: RelayLinkedAccount) => {
    try {
      await api.unlinkRelayAccount(platform.id, account.platformUserId);
      await loadAccounts();
    } catch (unlinkError) {
      showToast(errorMessage(unlinkError), "error");
    }
  };

  const copyCommand = async () => {
    if (!link) return;
    try {
      await navigator.clipboard.writeText(link.command);
      showToast("Copied", "success");
    } catch {
      showToast(`Copy failed. Type ${link.command} instead.`, "error");
    }
  };

  const openUrl = link?.bot_url ?? botUrl;

  return (
    <div className="flex flex-col gap-3 border-t border-border pt-3" data-testid={`relay-link-${platform.id}`}>
      <div className="flex items-center gap-2 flex-wrap">
        <Badge tone="secondary">Team Gateway</Badge>
        <span className="text-xs text-muted-foreground">
          {platform.name} reaches this agent through the shared Hermes bot. Link your {platform.name}{" "}
          account to it; no bot token needed.
        </span>
      </div>

      {accounts && accounts.length > 0 && (
        <ul className="flex flex-col gap-1">
          {accounts.map((account) => (
            <li key={account.platformUserId} className="flex items-center justify-between gap-2 text-sm">
              <span className="truncate">{accountLabel(account)}</span>
              <Button ghost size="sm" onClick={() => void unlink(account)}>
                Unlink
              </Button>
            </li>
          ))}
        </ul>
      )}

      {link ? (
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
          {qrDataUrl && (
            <img
              src={qrDataUrl}
              alt={`${platform.name} shared bot QR code`}
              className="h-40 w-40 shrink-0 rounded border border-border bg-white p-1"
            />
          )}
          <div className="flex flex-col gap-2 text-sm">
            <span>
              1. {qrDataUrl ? "Scan the code with your phone, or open" : "Open"} the bot in {platform.name}.
            </span>
            <span>
              2. Send it this message from the account you want to link:
            </span>
            <code className="w-fit rounded bg-muted px-2 py-1 font-mono">{link.command}</code>
            <div className="flex flex-wrap gap-2">
              {openUrl && (
                <Button size="sm" onClick={() => window.open(openUrl, "_blank", "noopener,noreferrer")} prefix={<ExternalLink className="h-4 w-4" />}>
                  Open {platform.name}
                </Button>
              )}
              <Button ghost size="sm" onClick={() => void copyCommand()} prefix={<Copy className="h-4 w-4" />}>
                Copy message
              </Button>
              <Button ghost size="sm" onClick={() => { setLink(null); setQrDataUrl(""); }}>
                Cancel
              </Button>
            </div>
            <span className="flex items-center gap-2 text-xs text-muted-foreground">
              <Spinner /> Waiting for the bot to see your message. The code works once.
            </span>
          </div>
        </div>
      ) : (
        <Button
          size="sm"
          className="w-fit uppercase"
          onClick={() => void start()}
          disabled={busy}
          prefix={busy ? <Spinner /> : <Link2 className="h-4 w-4" />}
        >
          {accounts && accounts.length > 0 ? `Link another ${platform.name} account` : `Link ${platform.name}`}
        </Button>
      )}

      {error && <span className="text-xs text-destructive">{error}</span>}
    </div>
  );
}
