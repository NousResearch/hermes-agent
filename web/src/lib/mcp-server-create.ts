import type { McpHttpAuth, McpServer, McpServerCreate, McpServerEdit } from "@/lib/mcp-server-types";

export type McpTransport = "http" | "stdio";

export interface McpServerDraft {
  name: string;
  transport: McpTransport;
  url: string;
  httpAuth: McpHttpAuth;
  bearerToken: string;
  command: string;
  args: string;
  env: string;
}

export function emptyMcpServerDraft(): McpServerDraft {
  return {
    name: "",
    transport: "http",
    url: "",
    httpAuth: "none",
    bearerToken: "",
    command: "",
    args: "",
    env: "",
  };
}

function parseArgs(raw: string): string[] {
  return raw
    .split(/[\s,]+/)
    .map((value) => value.trim())
    .filter(Boolean);
}

function parseEnv(raw: string): Record<string, string> {
  const env: Record<string, string> = Object.create(null);
  for (const rawLine of raw.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    const separator = line.indexOf("=");
    if (separator === -1) continue;
    const key = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trim();
    if (key) env[key] = value;
  }
  return env;
}

export function mcpServerEditDraft(server: McpServer): McpServerDraft {
  return {
    ...emptyMcpServerDraft(),
    name: server.name,
    transport: server.transport === "stdio" ? "stdio" : "http",
    url: server.url ?? "",
    httpAuth: server.auth === "header" || server.auth === "oauth" ? server.auth : "none",
    command: server.command ?? "",
    args: JSON.stringify(server.args),
    env: Object.keys(server.env).map((key) => `${key}=`).join("\n"),
  };
}

export function buildMcpServerEdit(draft: McpServerDraft, server: McpServer): McpServerEdit {
  const initial = mcpServerEditDraft(server);
  const edit: McpServerEdit = {};
  if (draft.transport === "http") {
    const url = draft.url.trim();
    if (!url) throw new Error("URL required");
    if (url !== initial.url) edit.url = url;
    if (draft.httpAuth !== initial.httpAuth) edit.auth = draft.httpAuth;
    if (draft.httpAuth === "header" && draft.bearerToken.trim()) {
      edit.auth = "header";
      edit.bearer_token = draft.bearerToken;
    }
    if (edit.auth === "header" && !edit.bearer_token) {
      throw new Error("Bearer token required");
    }
    return edit;
  }

  const command = draft.command.trim();
  if (!command) throw new Error("Command required");
  if (command !== initial.command) edit.command = command;
  if (draft.args !== initial.args) edit.args = parseEditArgs(draft.args);
  const env = parseEditEnv(draft.env, server.env);
  if (Object.keys(env).length) edit.env = env;
  return edit;
}

function parseEditArgs(raw: string): string[] {
  const message = 'Enter arguments as a JSON list of strings, for example ["-y", "package"].';
  let args: unknown;
  try {
    args = raw.trim() ? JSON.parse(raw) : [];
  } catch {
    throw new Error(message);
  }
  if (!Array.isArray(args) || !args.every((arg) => typeof arg === "string")) {
    throw new Error(message);
  }
  return args;
}

function parseEditEnv(raw: string, stored: Record<string, string>): Record<string, string | null> {
  if (raw.split("\n").some((line) => line.trim() && line.indexOf("=") < 1)) {
    throw new Error("Enter each environment variable as KEY=VALUE.");
  }
  const values = parseEnv(raw);
  const changes: Record<string, string | null> = Object.create(null);
  for (const [key, value] of Object.entries(values)) {
    if (value || !Object.hasOwn(stored, key)) changes[key] = value;
  }
  for (const key of Object.keys(stored)) {
    if (!Object.hasOwn(values, key)) changes[key] = null;
  }
  return changes;
}

export function buildMcpServerCreate(draft: McpServerDraft): McpServerCreate {
  const name = draft.name.trim();
  if (!name) throw new Error("Name required");

  if (draft.transport === "http") {
    const url = draft.url.trim();
    if (!url) throw new Error("URL required");
    if (draft.httpAuth === "header" && !draft.bearerToken.trim()) {
      throw new Error("Bearer token required");
    }

    const server: McpServerCreate = { name, url };
    if (draft.httpAuth !== "none") server.auth = draft.httpAuth;
    if (draft.httpAuth === "header") {
      server.bearer_token = draft.bearerToken;
    }
    return server;
  }

  const command = draft.command.trim();
  if (!command) throw new Error("Command required");

  const server: McpServerCreate = { name, command };
  const args = parseArgs(draft.args);
  if (args.length) server.args = args;
  const env = parseEnv(draft.env);
  if (Object.keys(env).length) server.env = env;
  return server;
}
