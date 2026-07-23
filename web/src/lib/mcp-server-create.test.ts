import { describe, expect, it } from "vitest";

import {
  buildMcpServerCreate,
  emptyMcpServerDraft,
  McpServerCreateError,
} from "./mcp-server-create";

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

  it.each([
    {
      draft: emptyMcpServerDraft(),
      code: "name-required" as const,
    },
    {
      draft: { ...emptyMcpServerDraft(), name: "remote" },
      code: "url-required" as const,
    },
    {
      draft: {
        ...emptyMcpServerDraft(),
        name: "remote",
        url: "https://example.com/mcp",
        httpAuth: "header" as const,
      },
      code: "bearer-token-required" as const,
    },
    {
      draft: {
        ...emptyMcpServerDraft(),
        name: "local",
        transport: "stdio" as const,
      },
      code: "command-required" as const,
    },
  ])("rejects invalid drafts with stable code $code", ({ draft, code }) => {
    try {
      buildMcpServerCreate(draft);
      throw new Error("expected validation to fail");
    } catch (error) {
      expect(error).toBeInstanceOf(McpServerCreateError);
      expect((error as McpServerCreateError).code).toBe(code);
    }
  });
});
