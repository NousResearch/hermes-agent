import { describe, expect, it } from "vitest";

import { buildManagedFileDownloadUrl } from "./api";

describe("buildManagedFileDownloadUrl", () => {
  it("targets the streaming download endpoint with encoded path and token", () => {
    expect(
      buildManagedFileDownloadUrl("/tmp/Hermes files/report 1.mp3", {
        basePath: "/hermes/",
        token: "session token",
      }),
    ).toBe(
      "/hermes/api/files/download?path=%2Ftmp%2FHermes+files%2Freport+1.mp3&token=session+token",
    );
  });

  it("omits the token query parameter when no token is available", () => {
    expect(
      buildManagedFileDownloadUrl("relative/file.txt", {
        basePath: "",
        token: null,
      }),
    ).toBe("/api/files/download?path=relative%2Ffile.txt");
  });
});
