export interface McpServer {
  name: string;
  transport: "http" | "stdio" | "unknown";
  url: string | null;
  command: string | null;
  args: string[];
  env: Record<string, string>;
  auth: string | null;
  enabled: boolean;
  tools: string[] | null;
  source: "config" | "plugin";
}

export type McpHttpAuth = "none" | "header" | "oauth";

export interface McpServerCreate {
  name: string;
  url?: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  auth?: McpHttpAuth;
  bearer_token?: string;
}

export interface McpServerEdit {
  url?: string;
  command?: string;
  args?: string[];
  env?: Record<string, string | null>;
  auth?: McpHttpAuth;
  bearer_token?: string;
}
