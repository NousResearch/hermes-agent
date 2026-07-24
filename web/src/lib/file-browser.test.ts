import { describe, expect, it } from "vitest";

import {
  buildBreadcrumbs,
  decodeTextDataUrl,
  previewKind,
} from "./file-browser";

describe("previewKind", () => {
  it("recognizes Markdown independently of MIME type", () => {
    expect(previewKind("README.md", "application/octet-stream", 42)).toBe("markdown");
    expect(previewKind("guide.MARKDOWN", "text/plain", 42)).toBe("markdown");
  });

  it("recognizes source files, images, binary files, and oversized files", () => {
    expect(previewKind("config.yaml", "application/octet-stream", 42)).toBe("text");
    expect(previewKind("notes.txt", "text/plain", 42)).toBe("text");
    expect(previewKind("diagram.png", "image/png", 42)).toBe("image");
    expect(previewKind("archive.zip", "application/zip", 42)).toBe("binary");
    expect(previewKind("large.md", "text/markdown", 2 * 1024 * 1024 + 1)).toBe("large");
  });
});

describe("decodeTextDataUrl", () => {
  it("decodes UTF-8 base64 payloads", () => {
    expect(decodeTextDataUrl("data:text/plain;base64,SGVybWVzIOKckw==")).toBe("Hermes ✓");
  });
});

describe("buildBreadcrumbs", () => {
  it("builds navigable segments below a managed POSIX root", () => {
    expect(buildBreadcrumbs("/srv/hermes/webdav", "/srv/hermes/webdav/docs/guides")).toEqual([
      { label: "webdav", path: "/srv/hermes/webdav" },
      { label: "docs", path: "/srv/hermes/webdav/docs" },
      { label: "guides", path: "/srv/hermes/webdav/docs/guides" },
    ]);
  });

  it("supports Windows paths", () => {
    expect(buildBreadcrumbs("C:\\Users\\hermes", "C:\\Users\\hermes\\docs")).toEqual([
      { label: "hermes", path: "C:\\Users\\hermes" },
      { label: "docs", path: "C:\\Users\\hermes\\docs" },
    ]);
  });
});
