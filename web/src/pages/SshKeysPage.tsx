import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import {
  Copy,
  KeyRound,
  Plus,
  Server,
  Trash2,
  Upload,
} from "lucide-react";
import { api } from "@/lib/api";
import type { SshHostInfo, SshKeyInfo } from "@/lib/api";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { useI18n } from "@/i18n";
import { en } from "@/i18n/en";
import { usePageHeader } from "@/contexts/usePageHeader";

export default function SshKeysPage() {
  const { t } = useI18n();
  const s = t.ssh ?? en.ssh!;
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();

  const [keys, setKeys] = useState<SshKeyInfo[]>([]);
  const [hosts, setHosts] = useState<SshHostInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);

  const [genName, setGenName] = useState("id_ed25519");
  const [genComment, setGenComment] = useState("hermes-agent");
  const [importName, setImportName] = useState("id_ed25519");
  const [importKey, setImportKey] = useState("");

  const [hostAlias, setHostAlias] = useState("");
  const [hostName, setHostName] = useState("");
  const [hostUser, setHostUser] = useState("");
  const [hostPort, setHostPort] = useState("22");
  const [hostIdentity, setHostIdentity] = useState("id_ed25519");

  const loadAll = useCallback(() => {
    setLoading(true);
    Promise.all([api.getSshKeys(), api.getSshHosts()])
      .then(([keyRes, hostRes]) => {
        setKeys(keyRes.keys);
        setHosts(hostRes.hosts);
      })
      .catch(() => showToast(s.loadFailed, "error"))
      .finally(() => setLoading(false));
  }, [showToast, s.loadFailed]);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  useLayoutEffect(() => {
    setEnd(
      <Button size="sm" outlined onClick={loadAll} disabled={loading}>
        {t.common.refresh}
      </Button>,
    );
    return () => setEnd(null);
  }, [loadAll, loading, setEnd, t.common.refresh]);

  const keyDelete = useConfirmDelete({
    onDelete: useCallback(
      async (name: string) => {
        setBusy(name);
        try {
          await api.deleteSshKey(name);
          showToast(s.keyDeleted.replace("{name}", name), "success");
          loadAll();
        } catch (e) {
          showToast(`${t.common.failedToRemove} ${name}: ${e}`, "error");
          throw e;
        } finally {
          setBusy(null);
        }
      },
      [loadAll, showToast, t.common.failedToRemove, s.keyDeleted],
    ),
  });

  const hostDelete = useConfirmDelete({
    onDelete: useCallback(
      async (alias: string) => {
        setBusy(alias);
        try {
          await api.deleteSshHost(alias);
          showToast(s.hostDeleted.replace("{alias}", alias), "success");
          loadAll();
        } catch (e) {
          showToast(`${t.common.failedToRemove} ${alias}: ${e}`, "error");
          throw e;
        } finally {
          setBusy(null);
        }
      },
      [loadAll, showToast, t.common.failedToRemove, s.hostDeleted],
    ),
  });

  const handleGenerate = async () => {
    setBusy("generate");
    try {
      await api.generateSshKey(genName, genComment);
      showToast(s.keyGenerated, "success");
      loadAll();
    } catch (e) {
      showToast(`${s.generateFailed}: ${e}`, "error");
    } finally {
      setBusy(null);
    }
  };

  const handleImport = async () => {
    if (!importKey.trim()) return;
    setBusy("import");
    try {
      await api.importSshKey(importName, importKey);
      setImportKey("");
      showToast(s.keyImported, "success");
      loadAll();
    } catch (e) {
      showToast(`${s.importFailed}: ${e}`, "error");
    } finally {
      setBusy(null);
    }
  };

  const handleSaveHost = async () => {
    setBusy("host");
    try {
      await api.upsertSshHost({
        alias: hostAlias,
        host_name: hostName,
        user: hostUser,
        port: Number(hostPort) || 22,
        identity_file: hostIdentity,
      });
      showToast(s.hostSaved, "success");
      loadAll();
    } catch (e) {
      showToast(`${s.hostSaveFailed}: ${e}`, "error");
    } finally {
      setBusy(null);
    }
  };

  const handleTestHost = async (alias: string) => {
    setBusy(`test:${alias}`);
    try {
      const res = await api.testSshHost(alias);
      showToast(res.message, res.ok ? "success" : "error");
    } catch (e) {
      showToast(`${s.testFailed}: ${e}`, "error");
    } finally {
      setBusy(null);
    }
  };

  const copyPublicKey = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      showToast(s.publicKeyCopied, "success");
    } catch {
      showToast(s.copyFailed, "error");
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      <p className="text-sm text-muted-foreground">{s.description}</p>

      <Card>
        <CardHeader className="border-b border-border bg-card">
          <div className="flex items-center gap-2">
            <KeyRound className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-base">{s.keysTitle}</CardTitle>
          </div>
          <CardDescription>{s.keysHint}</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 pt-4">
          {keys.length === 0 ? (
            <p className="text-sm text-text-tertiary">{s.noKeys}</p>
          ) : (
            keys.map((key) => (
              <div
                key={key.name}
                className="grid gap-2 border border-border p-4"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="font-mono-ui text-sm font-semibold">
                      {key.name}
                    </span>
                    {key.key_type && (
                      <Badge tone="secondary">{key.key_type}</Badge>
                    )}
                  </div>
                  <div className="flex gap-2">
                    {key.public_key && (
                      <Button
                        size="sm"
                        outlined
                        prefix={<Copy />}
                        onClick={() => copyPublicKey(key.public_key!)}
                      >
                        {s.copyPublic}
                      </Button>
                    )}
                    <Button
                      size="sm"
                      outlined
                      destructive
                      prefix={<Trash2 />}
                      disabled={busy === key.name || keyDelete.isOpen}
                      onClick={() => keyDelete.requestDelete(key.name)}
                    >
                      {t.common.delete}
                    </Button>
                  </div>
                </div>
                {key.fingerprint && (
                  <p className="font-mono-ui text-xs text-text-tertiary break-all">
                    {key.fingerprint}
                  </p>
                )}
                {key.public_key && (
                  <p className="font-mono-ui text-xs break-all text-muted-foreground">
                    {key.public_key}
                  </p>
                )}
              </div>
            ))
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader className="border-b border-border bg-card">
            <CardTitle className="text-base">{s.generateTitle}</CardTitle>
            <CardDescription>{s.generateHint}</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 pt-4">
            <div className="grid gap-1">
              <Label>{s.keyName}</Label>
              <Input
                value={genName}
                onChange={(e) => setGenName(e.target.value)}
                placeholder="id_ed25519"
              />
            </div>
            <div className="grid gap-1">
              <Label>{s.comment}</Label>
              <Input
                value={genComment}
                onChange={(e) => setGenComment(e.target.value)}
              />
            </div>
            <Button
              prefix={<Plus />}
              onClick={handleGenerate}
              disabled={busy === "generate"}
            >
              {busy === "generate" ? "..." : s.generateAction}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="border-b border-border bg-card">
            <CardTitle className="text-base">{s.importTitle}</CardTitle>
            <CardDescription>{s.importHint}</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 pt-4">
            <div className="grid gap-1">
              <Label>{s.keyName}</Label>
              <Input
                value={importName}
                onChange={(e) => setImportName(e.target.value)}
              />
            </div>
            <div className="grid gap-1">
              <Label>{s.privateKey}</Label>
              <textarea
                className="min-h-[120px] w-full rounded-none border border-border bg-background px-3 py-2 font-mono-ui text-xs"
                value={importKey}
                onChange={(e) => setImportKey(e.target.value)}
                placeholder="-----BEGIN OPENSSH PRIVATE KEY-----"
              />
            </div>
            <Button
              prefix={<Upload />}
              onClick={handleImport}
              disabled={busy === "import" || !importKey.trim()}
            >
              {busy === "import" ? "..." : s.importAction}
            </Button>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="border-b border-border bg-card">
          <div className="flex items-center gap-2">
            <Server className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-base">{s.hostsTitle}</CardTitle>
          </div>
          <CardDescription>{s.hostsHint}</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 pt-4">
          {hosts.map((host) => (
            <div
              key={host.alias}
              className="flex flex-wrap items-center justify-between gap-3 border border-border p-4"
            >
              <div>
                <p className="font-semibold">{host.alias}</p>
                <p className="font-mono-ui text-xs text-muted-foreground">
                  {host.user ? `${host.user}@` : ""}
                  {host.host_name}
                  {host.port !== 22 ? `:${host.port}` : ""} · {host.identity_file}
                </p>
              </div>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  outlined
                  onClick={() => handleTestHost(host.alias)}
                  disabled={busy === `test:${host.alias}`}
                >
                  {s.testConnection}
                </Button>
                <Button
                  size="sm"
                  outlined
                  destructive
                  prefix={<Trash2 />}
                  onClick={() => hostDelete.requestDelete(host.alias)}
                >
                  {t.common.delete}
                </Button>
              </div>
            </div>
          ))}

          <div className="grid gap-3 border border-dashed border-border p-4">
            <p className="text-xs font-semibold tracking-wide">{s.addHost}</p>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="grid gap-1">
                <Label>{s.hostAlias}</Label>
                <Input value={hostAlias} onChange={(e) => setHostAlias(e.target.value)} placeholder="prod" />
              </div>
              <div className="grid gap-1">
                <Label>{s.hostName}</Label>
                <Input value={hostName} onChange={(e) => setHostName(e.target.value)} placeholder="10.0.0.120" />
              </div>
              <div className="grid gap-1">
                <Label>{s.hostUser}</Label>
                <Input value={hostUser} onChange={(e) => setHostUser(e.target.value)} />
              </div>
              <div className="grid gap-1">
                <Label>{s.hostPort}</Label>
                <Input value={hostPort} onChange={(e) => setHostPort(e.target.value)} />
              </div>
              <div className="grid gap-1 sm:col-span-2">
                <Label>{s.identityFile}</Label>
                <Input value={hostIdentity} onChange={(e) => setHostIdentity(e.target.value)} />
              </div>
            </div>
            <Button onClick={handleSaveHost} disabled={busy === "host" || !hostAlias || !hostName}>
              {busy === "host" ? "..." : s.saveHost}
            </Button>
          </div>
        </CardContent>
      </Card>

      <DeleteConfirmDialog
        open={keyDelete.isOpen}
        onCancel={keyDelete.cancel}
        onConfirm={keyDelete.confirm}
        title={s.deleteKeyTitle}
        description={s.deleteKeyMessage}
        loading={keyDelete.isDeleting}
      />
      <DeleteConfirmDialog
        open={hostDelete.isOpen}
        onCancel={hostDelete.cancel}
        onConfirm={hostDelete.confirm}
        title={s.deleteHostTitle}
        description={s.deleteHostMessage}
        loading={hostDelete.isDeleting}
      />
    </div>
  );
}
