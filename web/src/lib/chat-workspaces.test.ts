import { describe, expect, it } from "vitest";

import type { ChatWorkspacesResponse } from "@/lib/api";
import { abbreviateHomePath, workspaceOptions } from "@/lib/chat-workspaces";

const base: ChatWorkspacesResponse = {
  projects: [],
  repos: [],
  default_cwd: "/home/u/.hermes",
  home: "/home/u",
  scan_enabled: true,
};

describe("workspaceOptions", () => {
  it("lists project folders first, then repos by recency, deduped on path", () => {
    const data: ChatWorkspacesResponse = {
      ...base,
      projects: [
        {
          id: "p1",
          slug: "site",
          name: "Site",
          primary_path: "/home/u/code/site",
          archived: false,
          folders: [
            { path: "/home/u/code/site", label: null, is_primary: true },
            { path: "/home/u/code/site-infra", label: "infra", is_primary: false },
          ],
        },
        {
          id: "p2",
          slug: "old",
          name: "Old",
          primary_path: "/home/u/code/old",
          archived: true,
          folders: [{ path: "/home/u/code/old", label: null, is_primary: true }],
        },
      ],
      repos: [
        { root: "/home/u/code/site", label: "site", sessions: 3, last_active: 50 },
        { root: "/home/u/code/lib", label: "lib", sessions: 1, last_active: 10 },
        { root: "/home/u/code/app", label: "app", sessions: 2, last_active: 90 },
      ],
    };

    expect(workspaceOptions(data)).toEqual([
      { value: "/home/u/code/site", label: "Site · ~/code/site" },
      { value: "/home/u/code/site-infra", label: "Site · infra" },
      { value: "/home/u/code/app", label: "app  ~/code/app" },
      { value: "/home/u/code/lib", label: "lib  ~/code/lib" },
    ]);
  });
});

describe("abbreviateHomePath", () => {
  it("collapses the home prefix only on a path boundary", () => {
    expect(abbreviateHomePath("/home/u/code/x", "/home/u")).toBe("~/code/x");
    expect(abbreviateHomePath("/home/u", "/home/u/")).toBe("~");
    expect(abbreviateHomePath("/home/user2/code", "/home/u")).toBe("/home/user2/code");
  });
});
