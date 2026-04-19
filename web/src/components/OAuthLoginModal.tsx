import { useEffect, useRef, useState } from "react";
import { ExternalLink, Copy, X, Check, Loader2 } from "lucide-react";
import { Typography } from "@nous-research/ui";
import { api, type OAuthProvider, type OAuthStartResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useI18n } from "@/i18n";

interface Props {
  provider: OAuthProvider;
  onClose: () => void;
  onSuccess: (msg: string) => void;
  onError: (msg: string) => void;
}

type Phase =
  | "idle"
  | "starting"
  | "awaiting_user"
  | "submitting"
  | "polling"
  | "approved"
  | "error";

export function OAuthLoginModal({
  provider,
  onClose,
  onSuccess,
  onError,
}: Props) {
  const [phase, setPhase] = useState<Phase>("starting");
  const [start, setStart] = useState<OAuthStartResponse | null>(null);
  const [pkceCode, setPkceCode] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [secondsLeft, setSecondsLeft] = useState<number | null>(null);
  const [codeCopied, setCodeCopied] = useState(false);
  const isMounted = useRef(true);
  const pollTimer = useRef<number | null>(null);
  const { t } = useI18n();

  // Initiate flow on mount
  useEffect(() => {
    isMounted.current = true;
    api
      .startOAuthLogin(provider.id)
      .then((resp) => {
        if (!isMounted.current) return;
        setStart(resp);
        setSecondsLeft(resp.expires_in);
        setPhase(resp.flow === "device_code" ? "polling" : "awaiting_user");
        if (resp.flow === "pkce") {
          window.open(resp.auth_url, "_blank", "noopener,noreferrer");
        } else {
          window.open(resp.verification_url, "_blank", "noopener,noreferrer");
        }
      })
      .catch((e) => {
        if (!isMounted.current) return;
        setPhase("error");
        setErrorMsg(`Failed to start login: ${e}`);
      });
    return () => {
      isMounted.current = false;
      if (pollTimer.current !== null) window.clearInterval(pollTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Tick the countdown
  useEffect(() => {
    if (secondsLeft === null) return;
    if (phase === "approved" || phase === "error") return;
    const tick = window.setInterval(() => {
      if (!isMounted.current) return;
      setSecondsLeft((s) => {
        if (s !== null && s <= 1) {
          setPhase("error");
          setErrorMsg(t.oauth.sessionExpired);
          return 0;
        }
        return s !== null && s > 0 ? s - 1 : 0;
      });
    }, 1000);
    return () => window.clearInterval(tick);
  }, [secondsLeft, phase, t]);

  // Device-code: poll backend every 2s
  useEffect(() => {
    if (!start || start.flow !== "device_code" || phase !== "polling") return;
    const sid = start.session_id;
    pollTimer.current = window.setInterval(async () => {
      try {
        const resp = await api.pollOAuthSession(provider.id, sid);
        if (!isMounted.current) return;
        if (resp.status === "approved") {
          setPhase("approved");
          if (pollTimer.current !== null)
            window.clearInterval(pollTimer.current);
          onSuccess(`${provider.name} connected`);
          window.setTimeout(() => isMounted.current && onClose(), 1500);
        } else if (resp.status !== "pending") {
          setPhase("error");
          setErrorMsg(resp.error_message || `Login ${resp.status}`);
          if (pollTimer.current !== null)
            window.clearInterval(pollTimer.current);
        }
      } catch (e) {
        if (!isMounted.current) return;
        setPhase("error");
        setErrorMsg(`Polling failed: ${e}`);
        if (pollTimer.current !== null) window.clearInterval(pollTimer.current);
      }
    }, 2000);
    return () => {
      if (pollTimer.current !== null) window.clearInterval(pollTimer.current);
    };
  }, [start, phase, provider.id, provider.name, onSuccess, onClose]);

  const handleSubmitPkceCode = async () => {
    if (!start || start.flow !== "pkce") return;
    if (!pkceCode.trim()) return;
    setPhase("submitting");
    setErrorMsg(null);
    try {
      const resp = await api.submitOAuthCode(
        provider.id,
        start.session_id,
        pkceCode.trim(),
      );
      if (!isMounted.current) return;
      if (resp.ok && resp.status === "approved") {
        setPhase("approved");
        onSuccess(`${provider.name} connected`);
        window.setTimeout(() => isMounted.current && onClose(), 1500);
      } else {
        setPhase("error");
        setErrorMsg(resp.message || "Token exchange failed");
      }
    } catch (e) {
      if (!isMounted.current) return;
      setPhase("error");
      setErrorMsg(`Submit failed: ${e}`);
    }
  };

  const handleClose = async () => {
    if (start && phase !== "approved" && phase !== "error") {
      try {
        await api.cancelOAuthSession(start.session_id);
      } catch {
        // ignore
      }
    }
    onClose();
  };

  const handleCopyUserCode = async (code: string) => {
    try {
      await navigator.clipboard.writeText(code);
      setCodeCopied(true);
      window.setTimeout(() => isMounted.current && setCodeCopied(false), 1500);
    } catch {
      onError("Clipboard write failed");
    }
  };

  const handleBackdrop = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) handleClose();
  };

  const fmtTime = (s: number | null) => {
    if (s === null) return "";
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}:${String(r).padStart(2, "0")}`;
  };

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
      onClick={handleBackdrop}
      role="dialog"
      aria-modal="true"
      aria-labelledby="oauth-modal-title"
    >
      <div className="relative w-full max-w-md border border-border bg-card shadow-2xl">
        <button
          type="button"
          onClick={handleClose}
          className="absolute right-3 top-3 text-muted-foreground hover:text-foreground transition-colors"
          aria-label={t.common.close}
        >
          <X className="h-5 w-5" />
        </button>
        <div className="p-6 flex flex-col gap-4">
          <div>
            <Typography
              as="h2"
              mondwest
              id="oauth-modal-title"
              className="text-base tracking-wider uppercase"
            >
              {t.oauth.connect} {provider.name}
            </Typography>
            {secondsLeft !== null &&
              phase !== "approved" &&
              phase !== "error" && (
                <p className="text-xs text-muted-foreground mt-1">
                  {t.oauth.sessionExpires.replace(
                    "{time}",
                    fmtTime(secondsLeft),
                  )}
                </p>
              )}
          </div>

          {/* ── starting ───────────────────────────────────── */}
          {phase === "starting" && (
            <div className="flex items-center gap-3 py-6 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              {t.oauth.initiatingLogin}
            </div>
          )}

          {/* ── PKCE: paste code ───────────────────────────── */}
          {start?.flow === "pkce" && phase === "awaiting_user" && (
            <>
              <ol className="text-sm space-y-2 list-decimal list-inside text-muted-foreground">
                <li>{t.oauth.pkceStep1}</li>
                <li>{t.oauth.pkceStep2}</li>
                <li>{t.oauth.pkceStep3}</li>
              </ol>
              <div className="flex flex-col gap-2">
                <Input
                  value={pkceCode}
                  onChange={(e) => setPkceCode(e.target.value)}
                  placeholder={t.oauth.pasteCode}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmitPkceCode()}
                  autoFocus
                />
                <div className="flex items-center gap-2 justify-between">
                  <a
                    href={
                      (start as Extract<OAuthStartResponse, { flow: "pkce" }>)
                        .auth_url
                    }
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
                  >
                    <ExternalLink className="h-3 w-3" />
                    {t.oauth.reOpenAuth}
                  </a>
                  <Button
                    onClick={handleSubmitPkceCode}
                    disabled={!pkceCode.trim()}
                    size="sm"
                  >
                    {t.oauth.submitCode}
                  </Button>
                </div>
              </div>
            </>
          )}

          {/* ── PKCE: submitting exchange ──────────────────── */}
          {phase === "submitting" && (
            <div className="flex items-center gap-3 py-6 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              {t.oauth.exchangingCode}
            </div>
          )}

          {/* ── Device code: show code + URL, polling ──────── */}
          {start?.flow === "device_code" && phase === "polling" && (
            <>
              <p className="text-sm text-muted-foreground">
                {t.oauth.enterCodePrompt}
              </p>
              <div className="flex items-center justify-between gap-2 border border-border bg-secondary/30 p-4">
                <code className="font-mono-ui text-2xl tracking-widest text-foreground">
                  {
                    (
                      start as Extract<
                        OAuthStartResponse,
                        { flow: "device_code" }
                      >
                    ).user_code
                  }
                </code>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    handleCopyUserCode(
                      (
                        start as Extract<
                          OAuthStartResponse,
                          { flow: "device_code" }
                        >
                      ).user_code,
                    )
                  }
                  className="text-xs"
                >
                  {codeCopied ? (
                    <Check className="h-3 w-3" />
                  ) : (
                    <Copy className="h-3 w-3" />
                  )}
                </Button>
              </div>
              <a
                href={
                  (
                    start as Extract<
                      OAuthStartResponse,
                      { flow: "device_code" }
                    >
                  ).verification_url
                }
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
              >
                <ExternalLink className="h-3 w-3" />
                {t.oauth.reOpenVerification}
              </a>
              <div className="flex items-center gap-2 text-xs text-muted-foreground border-t border-border pt-3">
                <Loader2 className="h-3 w-3 animate-spin" />
                {t.oauth.waitingAuth}
              </div>
            </>
          )}

          {/* ── approved ───────────────────────────────────── */}
          {phase === "approved" && (
            <div className="flex items-center gap-3 py-6 text-sm text-success">
              <Check className="h-5 w-5" />
              {t.oauth.connectedClosing}
            </div>
          )}

          {/* ── error ──────────────────────────────────────── */}
          {phase === "error" && (
            <>
              <div className="border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                {errorMsg || t.oauth.loginFailed}
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={handleClose}>
                  {t.common.close}
                </Button>
                <Button
                  size="sm"
                  onClick={() => {
                    if (start?.session_id) {
                      api.cancelOAuthSession(start.session_id).catch(() => {});
                    }
                    setErrorMsg(null);
                    setStart(null);
                    setPkceCode("");
                    setPhase("starting");
                    api
                      .startOAuthLogin(provider.id)
                      .then((resp) => {
                        if (!isMounted.current) return;
                        setStart(resp);
                        setSecondsLeft(resp.expires_in);
                        setPhase(
                          resp.flow === "device_code"
                            ? "polling"
                            : "awaiting_user",
                        );
                        if (resp.flow === "pkce") {
                          window.open(
                            resp.auth_url,
                            "_blank",
                            "noopener,noreferrer",
                          );
                        } else {
                          window.open(
                            resp.verification_url,
                            "_blank",
                            "noopener,noreferrer",
                          );
                        }
                      })
                      .catch((e) => {
                        if (!isMounted.current) return;
                        setPhase("error");
                        setErrorMsg(`${t.common.retry} failed: ${e}`);
                      });
                  }}
                >
                  {t.common.retry}
                </Button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
