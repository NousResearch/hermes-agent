import { useCallback, useEffect, useRef, useState } from "react";
import { Copy, ExternalLink, Link2 } from "lucide-react";
import * as QRCode from "qrcode";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import type { MessagingPlatform, RelayLinkCode, RelayLinkedAccount } from "@/lib/api";
import { errorMessage } from "@/lib/api-error";
import { en } from "@/i18n/en";
import { useI18n } from "@/i18n";

const POLL_MS = 3000;
// Consecutive poll failures before the panel stops waiting and shows the error.
const MAX_POLL_ERRORS = 5;

function hasExpired(link: RelayLinkCode): boolean {
  const expiresAt = link.expires_at ? Date.parse(link.expires_at) : Number.NaN;
  return Number.isFinite(expiresAt) && Date.now() >= expiresAt;
}

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
 * A Portal-managed agent takes links only from the Portal (a code minted here would bind nothing),
 * so there the panel lists the linked accounts and points to the Portal.
 */
export function RelayLinkPanel({
  platform,
  showToast,
}: {
  platform: MessagingPlatform;
  showToast: (message: string, type: "success" | "error") => void;
}) {
  const { t } = useI18n();
  const relayUi = t.channelsRelayUi ?? en.channelsRelayUi!;
  const withPlatform = useCallback(
    (text: string) => text.replaceAll("{platform}", platform.name),
    [platform.name],
  );
  const [accounts, setAccounts] = useState<RelayLinkedAccount[] | null>(null);
  const [botUrl, setBotUrl] = useState<string | null>(null);
  const [link, setLink] = useState<RelayLinkCode | null>(null);
  const [qrDataUrl, setQrDataUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [portalUrl, setPortalUrl] = useState<string | null>(null);
  // Accounts linked when the code was minted: only an id not in this set proves a new link.
  const linkedBefore = useRef<Set<string>>(new Set());

  const loadAccounts = useCallback(async () => {
    try {
      const res = await api.getRelayLinks();
      const row = res.platforms.find((p) => p.platform === platform.id);
      setPortalUrl(res.managed_by === "portal" ? res.portal_url ?? null : null);
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
    let failures = 0;
    const poll = async () => {
      try {
        const state = await api.getRelayLinkState(link.code);
        failures = 0;
        if (cancelled) return;
        if (state.state === "pending") {
          timer = setTimeout(poll, POLL_MS);
          return;
        }
        if (state.state === "refused") {
          setLink(null);
          setQrDataUrl("");
          setError(
            withPlatform(relayUi.linkRefused).replace(
              "{reason}",
              state.reason ? ` (${state.reason})` : "",
            ),
          );
          return;
        }
        // Reload before closing the code: closing re-runs this effect and cancels the rest of it.
        const after = await loadAccounts();
        if (cancelled) return;
        setLink(null);
        setQrDataUrl("");
        if (!after) return;
        // `inactive` alone means used, expired or unknown: only a new account id proves the link.
        if (after.some((account) => !linkedBefore.current.has(account.platformUserId))) {
          showToast(withPlatform(relayUi.accountLinked), "success");
        } else if (hasExpired(link)) {
          setError(relayUi.codeExpired);
        } else {
          setNotice(withPlatform(relayUi.inactiveCode));
        }
      } catch (pollError) {
        if (cancelled) return;
        failures += 1;
        if (failures >= MAX_POLL_ERRORS || hasExpired(link)) {
          setLink(null);
          setQrDataUrl("");
          setError(errorMessage(pollError));
          return;
        }
        timer = setTimeout(poll, POLL_MS);
      }
    };
    timer = setTimeout(poll, POLL_MS);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [link, loadAccounts, relayUi, showToast, withPlatform]);

  const start = async () => {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      // Snapshot what is linked right now: a failed earlier load must not make every account "new".
      const before = await loadAccounts();
      if (!before) return;
      const res = await api.requestRelayLink(platform.id);
      linkedBefore.current = new Set(before.map((account) => account.platformUserId));
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
      showToast(relayUi.copied, "success");
    } catch {
      showToast(relayUi.copyFailed.replace("{command}", link.command), "error");
    }
  };

  const openUrl = link?.bot_url ?? botUrl;

  return (
    <div className="flex flex-col gap-3 border-t border-border pt-3" data-testid={`relay-link-${platform.id}`}>
      <div className="flex items-center gap-2 flex-wrap">
        <Badge tone="secondary">{relayUi.teamGateway}</Badge>
        <span className="text-xs text-muted-foreground">
          {portalUrl
            ? withPlatform(relayUi.managedHint)
            : withPlatform(relayUi.linkHint)}
        </span>
      </div>

      {accounts && accounts.length > 0 && (
        <ul className="flex flex-col gap-1">
          {accounts.map((account) => (
            <li key={account.platformUserId} className="flex items-center justify-between gap-2 text-sm">
              <span className="truncate">{accountLabel(account)}</span>
              {!portalUrl && (
                <Button ghost size="sm" onClick={() => void unlink(account)}>
                  {relayUi.unlink}
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}

      {portalUrl ? (
        <Button
          size="sm"
          className="w-fit uppercase"
          onClick={() => window.open(portalUrl, "_blank", "noopener,noreferrer")}
          prefix={<ExternalLink className="h-4 w-4" />}
        >
          {relayUi.linkInPortal}
        </Button>
      ) : link ? (
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
          {qrDataUrl && (
            <img
              src={qrDataUrl}
              alt={withPlatform(relayUi.qrCodeAlt)}
              className="h-40 w-40 shrink-0 rounded border border-border bg-white p-1"
            />
          )}
          <div className="flex flex-col gap-2 text-sm">
            <span>
              {qrDataUrl ? withPlatform(relayUi.scanOrOpen) : withPlatform(relayUi.openBot)}
            </span>
            <span>
              {relayUi.sendMessage}
            </span>
            <code className="w-fit rounded bg-muted px-2 py-1 font-mono">{link.command}</code>
            <div className="flex flex-wrap gap-2">
              {openUrl && (
                <Button size="sm" onClick={() => window.open(openUrl, "_blank", "noopener,noreferrer")} prefix={<ExternalLink className="h-4 w-4" />}>
                  {withPlatform(relayUi.openPlatform)}
                </Button>
              )}
              <Button ghost size="sm" onClick={() => void copyCommand()} prefix={<Copy className="h-4 w-4" />}>
                {relayUi.copyMessage}
              </Button>
              <Button ghost size="sm" onClick={() => { setLink(null); setQrDataUrl(""); }}>
                {relayUi.cancel}
              </Button>
            </div>
            <span className="flex items-center gap-2 text-xs text-muted-foreground">
              <Spinner /> {relayUi.waiting}
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
          {accounts && accounts.length > 0
            ? withPlatform(relayUi.linkAnother)
            : withPlatform(relayUi.link)}
        </Button>
      )}

      {notice && <span className="text-xs text-muted-foreground">{notice}</span>}
      {error && <span className="text-xs text-destructive">{error}</span>}
    </div>
  );
}
