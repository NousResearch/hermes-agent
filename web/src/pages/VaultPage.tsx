import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Database,
  KeyRound,
  Lock,
  LockOpen,
  Plus,
  RefreshCw,
  ShieldCheck,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import { api, type VaultItem, type VaultSource } from "@/lib/api";
import { useProfileScope } from "@/contexts/useProfileScope";
import { usePageHeader } from "@/contexts/usePageHeader";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { errorMessage } from "@/lib/api-error";
import { cn, themedBody } from "@/lib/utils";

const KINDS = ["login", "payment", "address"] as const;
const ID_TYPES = ["email", "phone", "username"] as const;
const PAYMENT_FIELDS = [
  "card_number",
  "cardholder_name",
  "exp_month",
  "exp_year",
  "cvc",
  "billing_postal_code",
];
const ADDRESS_FIELDS = [
  "address_line1",
  "address_line2",
  "city",
  "state",
  "postal_code",
  "country",
];
const REQUIRED: Record<string, string[]> = {
  payment: ["card_number", "exp_month", "exp_year", "cvc"],
  address: ["address_line1", "city", "postal_code", "country"],
};

const TH_CLASS = "py-2 pr-4 text-left text-xs font-medium";
const TD_CLASS = "py-2.5 pr-4 align-middle";

function SourceStatus({
  source,
  state,
}: {
  source: VaultSource;
  state: "installed" | "enabled" | "unlocked";
}) {
  if (state === "installed") {
    return (
      <Badge tone={source.installed ? "success" : "outline"}>
        {source.installed ? "installed" : "not installed"}
      </Badge>
    );
  }
  if (state === "enabled") {
    if (source.name === "local") {
      return <span className="text-xs text-muted-foreground">always</span>;
    }
    return source.enabled ? (
      <Badge tone="success">enabled</Badge>
    ) : (
      <Badge tone="outline">disabled</Badge>
    );
  }
  if (!source.needs_unlock) {
    return <span className="text-xs text-muted-foreground">n/a</span>;
  }
  return source.unlocked ? (
    <Badge tone="success">unlocked</Badge>
  ) : (
    <Badge tone="warning">locked</Badge>
  );
}

export default function VaultPage() {
  const { profile } = useProfileScope();
  const { setEnd } = usePageHeader();
  const [items, setItems] = useState<VaultItem[]>([]);
  const [sources, setSources] = useState<VaultSource[]>([]);
  const [locked, setLocked] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [kind, setKind] = useState<string>("login");
  const [label, setLabel] = useState("");
  const [origin, setOrigin] = useState("");
  const [loginIdType, setLoginIdType] = useState<string>("email");
  const [loginIdentifier, setLoginIdentifier] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [loginOtp, setLoginOtp] = useState("");
  const [fields, setFields] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);

  const [unlockName, setUnlockName] = useState<string | null>(null);
  const [unlockPw, setUnlockPw] = useState("");
  const [unlocking, setUnlocking] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [it, src] = await Promise.all([api.getVaultItems(), api.getVaultSources()]);
      setItems(it.items || []);
      setLocked(it.locked || []);
      setSources(src.sources || []);
    } catch (e) {
      setError(errorMessage(e) || "Failed to load vault");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial + profile-scoped fetch
    load();
  }, [load, profile]);

  useEffect(() => {
    setEnd(
      <Button outlined size="sm" onClick={load} disabled={loading} prefix={<RefreshCw />}>
        Refresh
      </Button>,
    );
    return () => {
      setEnd(null);
    };
  }, [load, loading, setEnd]);

  const addItem = async () => {
    setSubmitting(true);
    setError(null);
    try {
      let secret: Record<string, string> = {};
      if (kind === "login") {
        secret = {
          identifier_type: loginIdType,
          identifier: loginIdentifier,
          password: loginPassword,
        };
        if (loginOtp.trim()) secret.otp_secret = loginOtp.trim();
      } else {
        const fs = kind === "payment" ? PAYMENT_FIELDS : ADDRESS_FIELDS;
        for (const f of fs) if (fields[f]?.trim()) secret[f] = fields[f].trim();
      }
      await api.addVaultItem(kind, label, origin, secret);
      setLabel("");
      setOrigin("");
      setLoginIdentifier("");
      setLoginPassword("");
      setLoginOtp("");
      setFields({});
      await load();
    } catch (e) {
      setError(errorMessage(e) || "Failed to add item");
    } finally {
      setSubmitting(false);
    }
  };

  const removeItem = async (id: string) => {
    if (!window.confirm("Remove this vault item?")) return;
    try {
      await api.removeVaultItem(id);
      await load();
    } catch (e) {
      setError(errorMessage(e) || "Failed to remove item");
    }
  };

  const toggleSource = async (name: string, enabled: boolean) => {
    try {
      await api.setVaultSourceEnabled(name, enabled);
      await load();
    } catch (e) {
      setError(errorMessage(e) || "Failed to toggle source");
    }
  };

  const doUnlock = async () => {
    if (!unlockName) return;
    setUnlocking(true);
    setError(null);
    try {
      await api.unlockVaultSource(unlockName, unlockPw);
      setUnlockName(null);
      setUnlockPw("");
      await load();
    } catch (e) {
      setError(errorMessage(e) || "Unlock failed");
    } finally {
      setUnlocking(false);
    }
  };

  const fieldList = useMemo(
    () =>
      kind === "payment" ? PAYMENT_FIELDS : kind === "address" ? ADDRESS_FIELDS : [],
    [kind],
  );

  return (
    <div className={cn("flex flex-col gap-4", themedBody)}>
      <p className="max-w-3xl text-sm text-muted-foreground">
        Encrypted local vault for browser autofill plus external password-manager
        backends. Secrets are stored encrypted and are never shown to the model.
      </p>

      {error && (
        <div className="flex items-start gap-3 border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <TriangleAlert className="mt-0.5 size-4 shrink-0" />
          <span className="min-w-0 flex-1 break-words">{error}</span>
          <Button ghost size="sm" onClick={() => setError(null)} className="shrink-0">
            Dismiss
          </Button>
        </div>
      )}

      {locked.length > 0 && (
        <div className="flex items-center gap-3 border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning">
          <Lock className="size-4 shrink-0" />
          <span>Locked managers: {locked.join(", ")}</span>
        </div>
      )}

      {/* Sources */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Database className="size-4 text-muted-foreground" />
            <CardTitle>Sources</CardTitle>
          </div>
          <CardDescription>
            Hermes vault is always on. External managers must be installed, enabled,
            and unlocked before their items appear below.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4 px-6 py-4">
          <div className="overflow-x-auto">
            <table className="w-full font-mondwest normal-case text-sm">
              <thead>
                <tr className="border-b border-border text-muted-foreground">
                  <th className={TH_CLASS}>Source</th>
                  <th className={TH_CLASS}>Installed</th>
                  <th className={TH_CLASS}>Enabled</th>
                  <th className={TH_CLASS}>Unlocked</th>
                  <th className={TH_CLASS}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {sources.length === 0 && (
                  <tr>
                    <td
                      colSpan={5}
                      className="py-6 text-center text-sm text-muted-foreground"
                    >
                      {loading ? "Loading sources…" : "No sources reported."}
                    </td>
                  </tr>
                )}
                {sources.map((s) => (
                  <tr
                    key={s.name}
                    className="border-b border-border/50 transition-colors hover:bg-secondary/20"
                  >
                    <td className={cn(TD_CLASS, "font-medium")}>{s.display_name}</td>
                    <td className={TD_CLASS}>
                      <SourceStatus source={s} state="installed" />
                    </td>
                    <td className={TD_CLASS}>
                      {s.name === "local" ? (
                        <span className="text-xs text-muted-foreground">always</span>
                      ) : (
                        <Switch
                          checked={s.enabled}
                          disabled={!s.installed}
                          onCheckedChange={(next) => toggleSource(s.name, next)}
                        />
                      )}
                    </td>
                    <td className={TD_CLASS}>
                      <SourceStatus source={s} state="unlocked" />
                    </td>
                    <td className={TD_CLASS}>
                      <div className="flex items-center gap-1">
                        {s.name !== "local" && s.enabled && s.needs_unlock && !s.unlocked && (
                          <Button
                            outlined
                            size="sm"
                            prefix={<LockOpen />}
                            onClick={() => setUnlockName(s.name)}
                          >
                            Unlock
                          </Button>
                        )}
                        {s.name !== "local" && s.enabled && s.unlocked && (
                          <Button
                            ghost
                            size="sm"
                            prefix={<Lock />}
                            onClick={async () => {
                              try {
                                await api.lockVaultSources(s.name);
                                await load();
                              } catch (e) {
                                setError(errorMessage(e) || "Lock failed");
                              }
                            }}
                          >
                            Lock
                          </Button>
                        )}
                        {s.name === "local" && (
                          <span className="text-xs text-muted-foreground">
                            no master password
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {unlockName && (
            <div className="flex flex-col gap-3 border border-border bg-background/40 p-3 sm:flex-row sm:items-end">
              <div className="grid flex-1 gap-2">
                <Label htmlFor="vault-unlock-pw">
                  Master password for {unlockName}
                </Label>
                <Input
                  id="vault-unlock-pw"
                  type="password"
                  autoComplete="current-password"
                  value={unlockPw}
                  onChange={(e) => setUnlockPw(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") doUnlock();
                  }}
                />
              </div>
              <div className="flex items-center gap-2">
                <Button size="sm" onClick={doUnlock} disabled={unlocking}>
                  {unlocking ? "Unlocking…" : "Unlock"}
                </Button>
                <Button
                  ghost
                  size="sm"
                  onClick={() => {
                    setUnlockName(null);
                    setUnlockPw("");
                  }}
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Saved items */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <KeyRound className="size-4 text-muted-foreground" />
            <CardTitle>Saved items</CardTitle>
          </div>
          <CardDescription>
            {items.length > 0
              ? `${items.length} item${items.length === 1 ? "" : "s"} available for browser autofill.`
              : "Credentials and cards available for browser autofill."}
          </CardDescription>
        </CardHeader>
        <CardContent className="px-6 py-4">
          {loading ? (
            <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
              <Spinner />
              <span>Loading vault items…</span>
            </div>
          ) : items.length === 0 ? (
            <div className="flex flex-col items-center gap-2 border border-dashed border-border/70 px-4 py-10 text-center">
              <ShieldCheck className="size-5 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                No vault items yet. Add your first login below.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full font-mondwest normal-case text-sm">
                <thead>
                  <tr className="border-b border-border text-muted-foreground">
                    <th className={TH_CLASS}>Label</th>
                    <th className={TH_CLASS}>Kind</th>
                    <th className={TH_CLASS}>Identifier</th>
                    <th className={TH_CLASS}>Origin</th>
                    <th className={TH_CLASS}>Source</th>
                    <th className={TH_CLASS}>2FA</th>
                    <th className={cn(TH_CLASS, "text-right pr-0")}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((it) => (
                    <tr
                      key={it.id}
                      className="border-b border-border/50 transition-colors hover:bg-secondary/20"
                    >
                      <td className={cn(TD_CLASS, "font-medium")}>
                        {it.label}
                        <span className="ml-2 font-mono text-[0.6875rem] text-muted-foreground">
                          {it.id}
                        </span>
                      </td>
                      <td className={TD_CLASS}>
                        <Badge tone="secondary">{it.kind}</Badge>
                      </td>
                      <td className={cn(TD_CLASS, "font-mono text-xs")}>
                        {it.identifier ? (
                          <>
                            <span className="text-muted-foreground">
                              {it.identifier_type}:
                            </span>{" "}
                            {it.identifier}
                          </>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </td>
                      <td className={cn(TD_CLASS, "text-xs")}>
                        {it.origin || <span className="text-muted-foreground">—</span>}
                      </td>
                      <td className={cn(TD_CLASS, "text-xs text-muted-foreground")}>
                        {it.backend || "local"}
                      </td>
                      <td className={TD_CLASS}>
                        {it.has_otp ? (
                          <Badge tone="secondary">auto</Badge>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </td>
                      <td className={cn(TD_CLASS, "pr-0 text-right")}>
                        <Button
                          ghost
                          destructive
                          size="sm"
                          prefix={<Trash2 />}
                          onClick={() => removeItem(it.id)}
                        >
                          Remove
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Add form */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Plus className="size-4 text-muted-foreground" />
            <CardTitle>Add item</CardTitle>
          </div>
          <CardDescription>
            Stored encrypted on this machine. Values are write-only — the dashboard
            never displays them back.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4 px-6 py-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="grid gap-2">
              <Label htmlFor="vault-kind">Kind</Label>
              <Select id="vault-kind" value={kind} onValueChange={setKind}>
                {KINDS.map((k) => (
                  <SelectOption key={k} value={k}>
                    {k}
                  </SelectOption>
                ))}
              </Select>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="vault-label">Label</Label>
              <Input
                id="vault-label"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="e.g. GitHub work account"
              />
            </div>
            <div className="grid gap-2 sm:col-span-2">
              <Label htmlFor="vault-origin">Site origin</Label>
              <Input
                id="vault-origin"
                value={origin}
                onChange={(e) => setOrigin(e.target.value)}
                placeholder="https://github.com"
              />
            </div>
          </div>

          {kind === "login" && (
            <div className="grid gap-4 border-t border-border/60 pt-4 sm:grid-cols-2">
              <div className="grid gap-2">
                <Label htmlFor="vault-id-type">Identifier type</Label>
                <Select
                  id="vault-id-type"
                  value={loginIdType}
                  onValueChange={setLoginIdType}
                >
                  {ID_TYPES.map((t) => (
                    <SelectOption key={t} value={t}>
                      {t}
                    </SelectOption>
                  ))}
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="vault-identifier">Identifier</Label>
                <Input
                  id="vault-identifier"
                  value={loginIdentifier}
                  onChange={(e) => setLoginIdentifier(e.target.value)}
                  placeholder="you@example.com"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="vault-password">Password</Label>
                <Input
                  id="vault-password"
                  type="password"
                  autoComplete="new-password"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="vault-otp">Authenticator key (optional)</Label>
                <Input
                  id="vault-otp"
                  type="password"
                  autoComplete="off"
                  value={loginOtp}
                  onChange={(e) => setLoginOtp(e.target.value)}
                  placeholder="setup key or otpauth:// link"
                />
              </div>
            </div>
          )}

          {fieldList.length > 0 && (
            <div className="grid gap-4 border-t border-border/60 pt-4 sm:grid-cols-2">
              {fieldList.map((f) => (
                <div key={f} className="grid gap-2">
                  <Label htmlFor={`vault-field-${f}`}>
                    {f.replace(/_/g, " ")}
                    {REQUIRED[kind]?.includes(f) ? " *" : ""}
                  </Label>
                  <Input
                    id={`vault-field-${f}`}
                    type={
                      f.includes("cvc") || f.includes("number") ? "password" : "text"
                    }
                    autoComplete="off"
                    value={fields[f] || ""}
                    onChange={(e) =>
                      setFields((prev) => ({ ...prev, [f]: e.target.value }))
                    }
                  />
                </div>
              ))}
            </div>
          )}

          <div className="flex items-center justify-between gap-3 border-t border-border/60 pt-4">
            <p className="text-xs text-muted-foreground">
              Secrets never leave this machine unless a tool explicitly asks for
              autofill.
            </p>
            <Button
              onClick={addItem}
              disabled={submitting || !label.trim()}
              prefix={<Plus />}
            >
              {submitting ? "Saving…" : "Add to vault"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
