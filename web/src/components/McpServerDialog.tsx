import { useCallback, useState } from "react";
import { X } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import type { McpHttpAuth, McpServer } from "@/lib/mcp-server-types";
import {
  emptyMcpServerDraft,
  mcpServerEditDraft,
  type McpServerDraft,
  type McpTransport,
} from "@/lib/mcp-server-create";
import { cn, themedBody } from "@/lib/utils";

interface McpServerDialogProps {
  server?: McpServer;
  saving: boolean;
  onClose: () => void;
  onSave: (draft: McpServerDraft) => Promise<void>;
}

export function McpServerDialog({ server, saving, onClose, onSave }: McpServerDialogProps) {
  const [draft, setDraft] = useState(() => server ? mcpServerEditDraft(server) : emptyMcpServerDraft());
  const { name, transport, url, httpAuth, bearerToken, command, args, env } = draft;
  const update = (changes: Partial<McpServerDraft>) => setDraft((previous) => ({ ...previous, ...changes }));
  const close = useCallback(() => {
    if (!saving) onClose();
  }, [onClose, saving]);
  const dialogRef = useModalBehavior({ open: true, onClose: close });

  return (
    <div
      ref={dialogRef}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
      onClick={(e) => e.target === e.currentTarget && close()}
      role="dialog"
      aria-modal="true"
      aria-labelledby="mcp-server-dialog-title"
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
          onClick={close}
          className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
          aria-label="Close"
        >
          <X />
        </Button>

        <header className="p-5 pb-3 border-b border-border">
          <h2
            id="mcp-server-dialog-title"
            className="font-mondwest text-display text-base tracking-wider"
          >
            {server ? "Edit MCP server" : "Add MCP server"}
          </h2>
        </header>

        <div className="p-5 grid gap-4">
          <div className="grid gap-2">
            <Label htmlFor="mcp-name">Name</Label>
            <Input
              id="mcp-name"
              autoFocus={!server}
              placeholder="my-server"
              value={name}
              readOnly={!!server}
              onChange={(e) => update({ name: e.target.value })}
            />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="mcp-transport">Transport</Label>
            <Select
              id="mcp-transport"
              value={transport}
              disabled={!!server}
              onValueChange={(value) => {
                const nextTransport = value as McpTransport;
                update({ transport: nextTransport, bearerToken: "" });
              }}
            >
              <SelectOption value="http">HTTP/SSE</SelectOption>
              <SelectOption value="stdio">stdio</SelectOption>
            </Select>
          </div>

          {transport === "http" ? (
            <>
              <div className="grid gap-2">
                <Label htmlFor="mcp-url">URL</Label>
                <Input
                  id="mcp-url"
                  placeholder="https://example.com/mcp"
                  value={url}
                  autoFocus={!!server}
                  onChange={(e) => update({ url: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="mcp-auth">Authentication</Label>
                <Select
                  id="mcp-auth"
                  value={httpAuth}
                  disabled={!!server?.auth && server.auth !== "header" && server.auth !== "oauth"}
                  onValueChange={(value) => {
                    const nextAuth = value as McpHttpAuth;
                    update({ httpAuth: nextAuth, bearerToken: nextAuth === "header" ? bearerToken : "" });
                  }}
                >
                  <SelectOption value="none">None</SelectOption>
                  <SelectOption value="header">Bearer token</SelectOption>
                  <SelectOption value="oauth">OAuth</SelectOption>
                </Select>
              </div>
              {httpAuth === "header" && (
                <div className="grid gap-2">
                  <Label htmlFor="mcp-bearer-token">Bearer token</Label>
                  <Input
                    id="mcp-bearer-token"
                    type="password"
                    autoComplete="new-password"
                    placeholder={server?.auth === "header" ? "Leave blank to keep the current token" : "Token or Bearer token"}
                    value={bearerToken}
                    onChange={(e) => update({ bearerToken: e.target.value })}
                  />
                  <p className="text-xs text-muted-foreground">
                    Stored in this profile&apos;s .env; config.yaml keeps
                    only an environment-variable reference.
                  </p>
                </div>
              )}
              {httpAuth === "oauth" && (
                <p className="text-xs text-muted-foreground">
                  {server ? "After saving, use Authenticate if prompted." : "Add the server, then use Authenticate."} Hermes opens the
                  OAuth browser on the machine running the Dashboard
                  backend.
                </p>
              )}
            </>
          ) : (
            <>
              <div className="grid gap-2">
                <Label htmlFor="mcp-command">Command</Label>
                <Input
                  id="mcp-command"
                  placeholder="npx"
                  value={command}
                  autoFocus={!!server}
                  onChange={(e) => update({ command: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="mcp-args">{server ? "Arguments (JSON list)" : "Args"}</Label>
                <Input
                  id="mcp-args"
                  placeholder={server ? '["-y", "package"]' : "-y @modelcontextprotocol/server-foo"}
                  value={args}
                  onChange={(e) => update({ args: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="mcp-env">
                  Environment (KEY=VALUE per line)
                </Label>
                <textarea
                  id="mcp-env"
                  className="flex min-h-[80px] w-full border border-border bg-background/40 px-3 py-2 text-sm font-courier shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 focus-visible:border-foreground/25"
                  placeholder={"API_KEY=secret\nDEBUG=1"}
                  value={env}
                  onChange={(e) => update({ env: e.target.value })}
                />
                {server && (
                  <p className="text-xs text-muted-foreground">
                    Leave a listed value blank to keep it. Delete its line to remove it.
                  </p>
                )}
              </div>
            </>
          )}

          <div className="flex justify-end">
            <Button
              className="uppercase"
              size="sm"
              onClick={() => void onSave(draft)}
              disabled={saving}
              prefix={saving ? <Spinner /> : undefined}
            >
              {saving ? (server ? "Saving..." : "Adding...") : (server ? "Save" : "Add")}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
