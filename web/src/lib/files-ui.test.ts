import { describe, expect, it } from "vitest";

import type { ManagedFileEntry } from "./api";
import {
  extractManagedFileErrorPath,
  filterManagedEntries,
  joinPath,
  parentPathOf,
} from "./files-ui";

const entry = (name: string, path = `/tmp/${name}`): ManagedFileEntry => ({
  name,
  path,
  is_directory: false,
  size: 1,
  mtime: 0,
  mime_type: "text/plain",
});

describe("files-ui path helpers", () => {
  it("joins POSIX and Windows-looking paths without duplicating separators", () => {
    expect(joinPath("/tmp/hermes", "notes.md")).toBe("/tmp/hermes/notes.md");
    expect(joinPath("/tmp/hermes/", "/notes.md")).toBe("/tmp/hermes/notes.md");
    expect(joinPath("C:\\Users\\example", "Desktop")).toBe("C:\\Users\\example\\Desktop");
  });

  it("extracts the failed absolute path from raw managed-file API errors", () => {
    expect(
      extractManagedFileErrorPath(
        "Error: 500: {\"detail\":\"Could not stat path: [Errno 2] No such file or directory: '/tmp/hermes/missing-root'\"}",
      ),
    ).toBe("/tmp/hermes/missing-root");
  });

  it("returns the parent recovery path for a failed file-browser root", () => {
    expect(parentPathOf("/tmp/hermes/missing-root")).toBe("/tmp/hermes");
    expect(parentPathOf("/")).toBeNull();
  });
});

describe("filterManagedEntries", () => {
  it("filters by file name or path case-insensitively", () => {
    const entries = [
      entry("README.md", "/repo/README.md"),
      entry("server.log", "/var/log/server.log"),
      entry("photo.png", "/tmp/hermes/Pictures/photo.png"),
    ];

    expect(filterManagedEntries(entries, "read").map((item) => item.name)).toEqual([
      "README.md",
    ]);
    expect(filterManagedEntries(entries, "pictures").map((item) => item.name)).toEqual([
      "photo.png",
    ]);
    expect(filterManagedEntries(entries, "")).toBe(entries);
  });
});
