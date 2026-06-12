import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { Package, Power, Server, Trash2, X, Zap } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type {
  McpCatalogDiagnostic,
  McpCatalogEntry,
  McpServer,
  McpServerCreate,
  McpTestResult,
} from "@/lib/api";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useI18n } from "@/i18n";
import { en } from "@/i18n/en";
import { cn, themedBody } from "@/lib/utils";

type Transport = "http" | "stdio";

function truncateText(value: string, maxLength: number): string {
  return value.length > maxLength ? value.slice(0, maxLength) + "..." : value;
}

function interpolate(
  template: string,
  values: Record<string, string | number>,
): string {
  return Object.entries(values).reduce(
    (result, [key, value]) => result.replaceAll(`{${key}}`, String(value)),
    template,
  );
}

function parseArgs(raw: string): string[] {
  return raw
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function parseEnv(raw: string): Record<string, string> {
  const env: Record<string, string> = {};
  raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .forEach((line) => {
      const idx = line.indexOf("=");
      if (idx === -1) return;
      const key = line.slice(0, idx).trim();
      const value = line.slice(idx + 1).trim();
      if (key) env[key] = value;
    });
  return env;
}

const TRANSPORT_TONE: Record<string, "success" | "warning" | "secondary"> = {
  http: "success",
  stdio: "warning",
  unknown: "secondary",
};

export default function McpPage() {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [catalog, setCatalog] = useState<McpCatalogEntry[]>([]);
  const [diagnostics, setDiagnostics] = useState<McpCatalogDiagnostic[]>([]);
  const [loading, setLoading] = useState(true);
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();
  const { t } = useI18n();
  const copy = t.mcpPage ?? en.mcpPage!;

  // Add server modal state
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [name, setName] = useState("");
  const [transport, setTransport] = useState<Transport>("http");
  const [url, setUrl] = useState("");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [env, setEnv] = useState("");
  const [creating, setCreating] = useState(false);
  const closeCreateModal = useCallback(() => setCreateModalOpen(false), []);
  const createModalRef = useModalBehavior({
    open: createModalOpen,
    onClose: closeCreateModal,
  });

  // Test results keyed by server name
  const [testing, setTesting] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, McpTestResult>>(
    {},
  );

  // Enable/disable state
  const [togglingName, setTogglingName] = useState<string | null>(null);
  const [restartNote, setRestartNote] = useState<string | null>(null);

  // Catalog install modal state
  const [installEntry, setInstallEntry] = useState<McpCatalogEntry | null>(
    null,
  );
  const [installEnv, setInstallEnv] = useState<Record<string, string>>({});
  const [installingName, setInstallingName] = useState<string | null>(null);
  const closeInstallModal = useCallback(() => setInstallEntry(null), []);
  const installModalRef = useModalBehavior({
    open: installEntry !== null,
    onClose: closeInstallModal,
  });

  const loadServers = useCallback(() => {
    return api
      .getMcpServers()
      .then((res) => setServers(res.servers))
      .catch((e) =>
        showToast(interpolate(copy.error, { error: String(e) }), "error"),
      );
  }, [copy, showToast]);

  const loadCatalog = useCallback(() => {
    return api
      .getMcpCatalog()
      .then((res) => {
        setCatalog(res.entries);
        setDiagnostics(res.diagnostics);
      })
      .catch((e) =>
        showToast(interpolate(copy.error, { error: String(e) }), "error"),
      );
  }, [copy, showToast]);

  useEffect(() => {
    Promise.all([loadServers(), loadCatalog()]).finally(() =>
      setLoading(false),
    );
  }, [loadServers, loadCatalog]);

  const handleCreate = async () => {
    if (!name.trim()) {
      showToast(copy.nameRequired, "error");
      return;
    }
    if (transport === "http" && !url.trim()) {
      showToast(copy.urlRequired, "error");
      return;
    }
    if (transport === "stdio" && !command.trim()) {
      showToast(copy.commandRequired, "error");
      return;
    }
    setCreating(true);
    try {
      const body: McpServerCreate = { name: name.trim() };
      if (transport === "http") {
        body.url = url.trim();
      } else {
        body.command = command.trim();
        const argList = parseArgs(args);
        if (argList.length) body.args = argList;
      }
      const envMap = parseEnv(env);
      if (Object.keys(envMap).length) body.env = envMap;

      await api.addMcpServer(body);
      showToast(copy.added, "success");
      setName("");
      setUrl("");
      setCommand("");
      setArgs("");
      setEnv("");
      setTransport("http");
      setCreateModalOpen(false);
      loadServers();
    } catch (e) {
      showToast(interpolate(copy.addFailed, { error: String(e) }), "error");
    } finally {
      setCreating(false);
    }
  };

  const handleTest = async (server: McpServer) => {
    setTesting(server.name);
    try {
      const result = await api.testMcpServer(server.name);
      setTestResults((prev) => ({ ...prev, [server.name]: result }));
      if (result.ok) {
        showToast(
          interpolate(copy.toolsCount, {
            name: server.name,
            count: result.tools.length,
          }),
          "success",
        );
      } else {
        showToast(`${server.name}: ${result.error ?? copy.failed}`, "error");
      }
    } catch (e) {
      showToast(interpolate(copy.error, { error: String(e) }), "error");
    } finally {
      setTesting(null);
    }
  };

  const handleToggleEnabled = async (server: McpServer) => {
    const next = !server.enabled;
    setTogglingName(server.name);
    try {
      await api.setMcpServerEnabled(server.name, next);
      setServers((prev) =>
        prev.map((s) => (s.name === server.name ? { ...s, enabled: next } : s)),
      );
      setRestartNote(copy.restartNote);
    } catch (e) {
      showToast(interpolate(copy.error, { error: String(e) }), "error");
    } finally {
      setTogglingName(null);
    }
  };

  const serverDelete = useConfirmDelete({
    onDelete: useCallback(
      async (serverName: string) => {
        try {
          await api.removeMcpServer(serverName);
          showToast(
            interpolate(copy.deleted, {
              name: truncateText(serverName, 30),
            }),
            "success",
          );
          setTestResults((prev) => {
            const next = { ...prev };
            delete next[serverName];
            return next;
          });
          loadServers();
        } catch (e) {
          showToast(interpolate(copy.error, { error: String(e) }), "error");
          throw e;
        }
      },
      [copy, loadServers, showToast],
    ),
  });

  // ── Catalog install ──────────────────────────────────────────────────
  const runInstall = useCallback(
    async (entry: McpCatalogEntry, envMap: Record<string, string>) => {
      setInstallingName(entry.name);
      try {
        const res = await api.installMcpCatalogEntry(entry.name, envMap, true);
        if (res.background) {
          showToast(copy.installingBackground, "success");
        } else {
          showToast(
            interpolate(copy.installedNamed, {
              name: truncateText(entry.name, 30),
            }),
            "success",
          );
        }
        setInstallEntry(null);
        setInstallEnv({});
        await Promise.all([loadServers(), loadCatalog()]);
      } catch (e) {
        showToast(
          interpolate(copy.installFailed, { error: String(e) }),
          "error",
        );
      } finally {
        setInstallingName(null);
      }
    },
    [copy, loadServers, loadCatalog, showToast],
  );

  const handleInstallClick = (entry: McpCatalogEntry) => {
    if (entry.required_env.length > 0) {
      const initial: Record<string, string> = {};
      entry.required_env.forEach((item) => {
        initial[item.name] = "";
      });
      setInstallEnv(initial);
      setInstallEntry(entry);
    } else {
      void runInstall(entry, {});
    }
  };

  const handleInstallSubmit = () => {
    if (!installEntry) return;
    const missing = installEntry.required_env.filter(
      (item) => item.required && !(installEnv[item.name] ?? "").trim(),
    );
    if (missing.length > 0) {
      showToast(
        interpolate(copy.fieldRequired, { field: missing[0].prompt }),
        "error",
      );
      return;
    }
    const envMap: Record<string, string> = {};
    Object.entries(installEnv).forEach(([k, v]) => {
      if (v.trim()) envMap[k] = v.trim();
    });
    void runInstall(installEntry, envMap);
  };

  // Put "Add Server" button in page header
  useLayoutEffect(() => {
    setEnd(
      <Button
        className="uppercase"
        size="sm"
        onClick={() => setCreateModalOpen(true)}
      >
        {copy.addServer}
      </Button>,
    );
    return () => {
      setEnd(null);
    };
  }, [copy.addServer, setEnd, loading]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  const diagnosticsByName: Record<string, McpCatalogDiagnostic[]> = {};
  diagnostics.forEach((d) => {
    (diagnosticsByName[d.name] ??= []).push(d);
  });

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      <DeleteConfirmDialog
        open={serverDelete.isOpen}
        onCancel={serverDelete.cancel}
        onConfirm={serverDelete.confirm}
        title={copy.removeServer}
        description={
          serverDelete.pendingId
            ? interpolate(copy.removeNamedDescription, {
                name: truncateText(serverDelete.pendingId, 40),
              })
            : copy.removeDescription
        }
        loading={serverDelete.isDeleting}
      />

      {/* Add server modal */}
      {createModalOpen && (
        <div
          ref={createModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
          onClick={(e) =>
            e.target === e.currentTarget && setCreateModalOpen(false)
          }
          role="dialog"
          aria-modal="true"
          aria-labelledby="create-mcp-title"
        >
          <div
            className={cn(
              themedBody,
              "relative w-full max-w-lg border border-border bg-card shadow-2xl flex flex-col",
            )}
          >
            <Button
              ghost
              size="icon"
              onClick={() => setCreateModalOpen(false)}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label={copy.close}
            >
              <X />
            </Button>

            <header className="p-5 pb-3 border-b border-border">
              <h2
                id="create-mcp-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                {copy.addServer}
              </h2>
            </header>

            <div className="p-5 grid gap-4">
              <div className="grid gap-2">
                <Label htmlFor="mcp-name">{copy.name}</Label>
                <Input
                  id="mcp-name"
                  autoFocus
                  placeholder="my-server"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="mcp-transport">{copy.transport}</Label>
                <Select
                  id="mcp-transport"
                  value={transport}
                  onValueChange={(v) => setTransport(v as Transport)}
                >
                  <SelectOption value="http">HTTP/SSE</SelectOption>
                  <SelectOption value="stdio">stdio</SelectOption>
                </Select>
              </div>

              {transport === "http" ? (
                <div className="grid gap-2">
                  <Label htmlFor="mcp-url">{copy.url}</Label>
                  <Input
                    id="mcp-url"
                    placeholder="https://example.com/mcp"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                  />
                </div>
              ) : (
                <>
                  <div className="grid gap-2">
                    <Label htmlFor="mcp-command">{copy.command}</Label>
                    <Input
                      id="mcp-command"
                      placeholder="npx"
                      value={command}
                      onChange={(e) => setCommand(e.target.value)}
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="mcp-args">{copy.args}</Label>
                    <Input
                      id="mcp-args"
                      placeholder="-y @modelcontextprotocol/server-foo"
                      value={args}
                      onChange={(e) => setArgs(e.target.value)}
                    />
                  </div>
                </>
              )}

              <div className="grid gap-2">
                <Label htmlFor="mcp-env">{copy.environment}</Label>
                <textarea
                  id="mcp-env"
                  className="flex min-h-[80px] w-full border border-border bg-background/40 px-3 py-2 text-sm font-courier shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 focus-visible:border-foreground/25"
                  placeholder={"API_KEY=secret\nDEBUG=1"}
                  value={env}
                  onChange={(e) => setEnv(e.target.value)}
                />
              </div>

              <div className="flex justify-end">
                <Button
                  className="uppercase"
                  size="sm"
                  onClick={handleCreate}
                  disabled={creating}
                  prefix={creating ? <Spinner /> : undefined}
                >
                  {creating ? copy.adding : copy.add}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Catalog install modal (required env vars) */}
      {installEntry && (
        <div
          ref={installModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
          onClick={(e) => e.target === e.currentTarget && setInstallEntry(null)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="install-mcp-title"
        >
          <div
            className={cn(
              themedBody,
              "relative w-full max-w-lg border border-border bg-card shadow-2xl flex flex-col",
            )}
          >
            <Button
              ghost
              size="icon"
              onClick={() => setInstallEntry(null)}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label={copy.close}
            >
              <X />
            </Button>

            <header className="p-5 pb-3 border-b border-border">
              <h2
                id="install-mcp-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                {interpolate(copy.installNamed, { name: installEntry.name })}
              </h2>
            </header>

            <div className="p-5 grid gap-4">
              <p className="text-xs text-muted-foreground">
                {copy.installRequirements}
              </p>
              {installEntry.required_env.map((item) => (
                <div className="grid gap-2" key={item.name}>
                  <Label htmlFor={`install-env-${item.name}`}>
                    {item.prompt}
                    {item.required ? " *" : ""}
                  </Label>
                  <Input
                    id={`install-env-${item.name}`}
                    type="password"
                    placeholder={item.name}
                    value={installEnv[item.name] ?? ""}
                    onChange={(e) =>
                      setInstallEnv((prev) => ({
                        ...prev,
                        [item.name]: e.target.value,
                      }))
                    }
                  />
                </div>
              ))}

              <div className="flex justify-end">
                <Button
                  className="uppercase"
                  size="sm"
                  onClick={handleInstallSubmit}
                  disabled={installingName === installEntry.name}
                  prefix={
                    installingName === installEntry.name ? (
                      <Spinner />
                    ) : undefined
                  }
                >
                  {installingName === installEntry.name
                    ? copy.installing
                    : copy.install}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Your MCP servers ── */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <H2
            variant="sm"
            className="flex items-center gap-2 text-muted-foreground"
          >
            <Server className="h-4 w-4" />
            {interpolate(copy.yourServers, { count: servers.length })}
          </H2>
        </div>

        {restartNote && <p className="text-xs text-warning">{restartNote}</p>}

        {servers.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              {copy.noServers}
            </CardContent>
          </Card>
        )}

        {servers.map((server) => {
          const envCount = Object.keys(server.env ?? {}).length;
          const result = testResults[server.name];

          return (
            <Card key={server.name}>
              <CardContent
                className={cn(
                  "flex items-start gap-4 py-4",
                  !server.enabled && "opacity-60",
                )}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-sm truncate">
                      {server.name}
                    </span>
                    <Badge
                      tone={TRANSPORT_TONE[server.transport] ?? "secondary"}
                    >
                      {server.transport}
                    </Badge>
                    {!server.enabled && (
                      <Badge tone="outline">{copy.disabled}</Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    {server.transport === "http" ? (
                      <span className="font-mono truncate">
                        {server.url ?? "—"}
                      </span>
                    ) : (
                      <span className="font-mono truncate">
                        {[server.command, ...(server.args ?? [])]
                          .filter(Boolean)
                          .join(" ") || "—"}
                      </span>
                    )}
                    {envCount > 0 && (
                      <span>
                        {interpolate(copy.envCount, { count: envCount })}
                      </span>
                    )}
                  </div>
                  {result && (
                    <div className="mt-2 text-xs">
                      {result.ok ? (
                        <p className="text-success">
                          {result.tools.length === 0
                            ? copy.connectedNoTools
                            : interpolate(copy.tools, {
                                tools: result.tools
                                  .map((tool) => tool.name)
                                  .join(", "),
                              })}
                        </p>
                      ) : (
                        <p className="text-destructive">
                          {result.error ?? copy.connectionFailed}
                        </p>
                      )}
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  <Button
                    ghost
                    size="sm"
                    title={server.enabled ? copy.disable : copy.enable}
                    aria-label={server.enabled ? copy.disable : copy.enable}
                    onClick={() => handleToggleEnabled(server)}
                    disabled={togglingName === server.name}
                    prefix={
                      togglingName === server.name ? <Spinner /> : <Power />
                    }
                    className={server.enabled ? "text-success" : undefined}
                  >
                    {server.enabled ? copy.disable : copy.enable}
                  </Button>

                  <Button
                    ghost
                    size="icon"
                    title={copy.testConnection}
                    aria-label={copy.testConnection}
                    onClick={() => handleTest(server)}
                    disabled={testing === server.name}
                  >
                    {testing === server.name ? <Spinner /> : <Zap />}
                  </Button>

                  <Button
                    ghost
                    destructive
                    size="icon"
                    title={copy.delete}
                    aria-label={copy.delete}
                    onClick={() => serverDelete.requestDelete(server.name)}
                  >
                    <Trash2 />
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* ── Catalog ── */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <H2
            variant="sm"
            className="flex items-center gap-2 text-muted-foreground"
          >
            <Package className="h-4 w-4" />
            {interpolate(copy.catalog, { count: catalog.length })}
          </H2>
        </div>

        <p className="text-xs text-muted-foreground">
          {copy.catalogDescription}
        </p>

        {catalog.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              {copy.noCatalog}
            </CardContent>
          </Card>
        )}

        {catalog.map((entry) => {
          const entryDiags = diagnosticsByName[entry.name] ?? [];
          const isInstalling = installingName === entry.name;

          return (
            <Card key={entry.name}>
              <CardContent className="flex items-start gap-4 py-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className="font-medium text-sm truncate">
                      {entry.name}
                    </span>
                    <Badge
                      tone={TRANSPORT_TONE[entry.transport] ?? "secondary"}
                    >
                      {entry.transport}
                    </Badge>
                    <Badge tone="outline">
                      {entry.source === "official"
                        ? copy.official
                        : entry.source}
                    </Badge>
                    {entry.installed && (
                      <Badge tone="success">{copy.installed}</Badge>
                    )}
                    {entry.installed && !entry.enabled && (
                      <Badge tone="outline">{copy.disabled}</Badge>
                    )}
                  </div>
                  {entry.description && (
                    <p className="text-xs text-muted-foreground">
                      {entry.description}
                    </p>
                  )}
                  {entryDiags.map((d, i) => (
                    <p
                      key={`${entry.name}-diag-${i}`}
                      className="text-xs text-warning mt-1"
                    >
                      {d.message}
                    </p>
                  ))}
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  {entry.installed ? (
                    <Badge tone="success">{copy.installed}</Badge>
                  ) : (
                    <Button
                      className="uppercase"
                      size="sm"
                      onClick={() => handleInstallClick(entry)}
                      disabled={isInstalling}
                      prefix={isInstalling ? <Spinner /> : undefined}
                    >
                      {isInstalling ? copy.installing : copy.install}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
