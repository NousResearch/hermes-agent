import { describe, expect, it } from "vitest";

import { buildMcpServerCreate, emptyMcpServerDraft } from "./mcp-server-create";

describe("MCP target shared by Add Server and Profile Builder", () => {
  it.each(["local", "windows"] as const)(
    "retains %s on Streamable HTTP and SSE",
    network => {
      for (const httpTransport of ["http", "sse"] as const) {
        const body = buildMcpServerCreate({
          ...emptyMcpServerDraft(),
          name: "unity",
          url: "http://localhost:8080/mcp",
          network,
          httpTransport,
        });
        expect(body.network).toBe(network);
        expect(body.transport ?? "http").toBe(httpTransport);
        expect(body).not.toHaveProperty("command");
      }
    },
  );

  it("persists the new-entry auto choice and never leaks HTTP settings into stdio", () => {
    const draft = {
      ...emptyMcpServerDraft(),
      name: "unity",
      url: "http://localhost:8080/mcp",
    };
    expect(buildMcpServerCreate(draft)).toEqual({
      name: "unity",
      url: draft.url,
      network: "auto",
    });
    const body = buildMcpServerCreate({
      ...draft,
      network: "windows",
      httpTransport: "sse",
      transport: "stdio",
      command: "python",
    });
    expect(body).not.toHaveProperty("network");
    expect(body).not.toHaveProperty("transport");
  });
});
