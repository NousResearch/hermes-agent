import { describe, expect, it } from "vitest";

import type { McpServer } from "./mcp-server-types";
import { buildMcpServerCreate, buildMcpServerEdit, emptyMcpServerDraft, mcpServerEditDraft } from "./mcp-server-create";

describe("editing an MCP server", () => {
  const server: McpServer = {
    name: "local", transport: "stdio", url: null, command: "uvx", args: ["package", "two words", ""],
    env: { API_KEY: "secr...cret", DEBUG: "***" }, auth: null, enabled: false, tools: null, source: "config",
  };

  it("preserves argument boundaries and leaves saved values blank", () => {
    const draft = mcpServerEditDraft(server);
    expect(JSON.parse(draft.args)).toEqual(server.args);
    expect(draft.env).toBe("API_KEY=\nDEBUG=");
    expect(buildMcpServerEdit(draft, server)).toEqual({});
    expect(buildMcpServerEdit({
      ...draft, args: '["next", "two words"]', env: "API_KEY=\nPORT=8080",
    }, server)).toEqual({ args: ["next", "two words"], env: { DEBUG: null, PORT: "8080" } });
  });

  it("only includes a bearer token when a replacement is entered", () => {
    const remote: McpServer = {
      ...server, transport: "http", url: "https://example.com/mcp", command: null, args: [], env: {}, auth: "header",
    };
    const draft = mcpServerEditDraft(remote);
    expect(draft.bearerToken).toBe("");
    expect(buildMcpServerEdit({ ...draft, url: "https://example.com/next" }, remote)).toEqual({
      url: "https://example.com/next",
    });
    expect(buildMcpServerEdit({ ...draft, bearerToken: "new-token" }, remote)).toEqual({
      auth: "header", bearer_token: "new-token",
    });
  });
});

describe("buildMcpServerCreate", () => {
  it("builds an HTTP Bearer request without stdio fields", () => {
    const server = buildMcpServerCreate({
      ...emptyMcpServerDraft(),
      name: " Linear ",
      url: " https://mcp.linear.app/mcp ",
      httpAuth: "header",
      bearerToken: "Bearer secret-token",
      command: "ignored",
      args: "--ignored",
      env: "IGNORED=value",
    });

    expect(server).toEqual({
      name: "Linear",
      url: "https://mcp.linear.app/mcp",
      auth: "header",
      bearer_token: "Bearer secret-token",
    });
  });

  it("builds OAuth and unauthenticated HTTP requests without a token", () => {
    expect(
      buildMcpServerCreate({
        ...emptyMcpServerDraft(),
        name: "oauth",
        url: "https://example.com/mcp",
        httpAuth: "oauth",
      }),
    ).toEqual({
      name: "oauth",
      url: "https://example.com/mcp",
      auth: "oauth",
    });

    expect(
      buildMcpServerCreate({
        ...emptyMcpServerDraft(),
        name: "public",
        url: "https://example.com/mcp",
      }),
    ).toEqual({
      name: "public",
      url: "https://example.com/mcp",
    });
  });

  it("parses stdio arguments and environment assignments", () => {
    const server = buildMcpServerCreate({
      ...emptyMcpServerDraft(),
      name: "local",
      transport: "stdio",
      command: " uvx ",
      args: "mcp-server, --debug",
      env: "API_KEY=secret\nURL=https://example.com?a=b\nINVALID",
    });

    expect(server).toEqual({
      name: "local",
      command: "uvx",
      args: ["mcp-server", "--debug"],
      env: {
        API_KEY: "secret",
        URL: "https://example.com?a=b",
      },
    });
  });

  it("rejects missing transport fields and Bearer tokens", () => {
    expect(() => buildMcpServerCreate(emptyMcpServerDraft())).toThrow(
      "Name required",
    );
    expect(() =>
      buildMcpServerCreate({
        ...emptyMcpServerDraft(),
        name: "remote",
      }),
    ).toThrow("URL required");
    expect(() =>
      buildMcpServerCreate({
        ...emptyMcpServerDraft(),
        name: "remote",
        url: "https://example.com/mcp",
        httpAuth: "header",
      }),
    ).toThrow("Bearer token required");
    expect(() =>
      buildMcpServerCreate({
        ...emptyMcpServerDraft(),
        name: "local",
        transport: "stdio",
      }),
    ).toThrow("Command required");
  });
});
