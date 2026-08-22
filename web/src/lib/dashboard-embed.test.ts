import { describe, expect, it, vi } from "vitest";

import {
  parseDashboardEmbedRequest,
  postDashboardEmbedEvent,
} from "./dashboard-embed";

describe("dashboard embedded chat contract", () => {
  it("accepts only configured parent origins and keeps the embed id opaque", () => {
    const request = parseDashboardEmbedRequest(
      new URLSearchParams(
        "embed=console-wolf&profile=wolf&parent_origin=https%3A%2F%2Fconsole.runi.services",
      ),
      ["https://console.runi.services"],
      { "console-wolf": "wolf" },
    );

    expect(request).toEqual({
      authBridge: false,
      embedId: "console-wolf",
      parentOrigin: "https://console.runi.services",
      profile: "wolf",
    });
    expect(
      parseDashboardEmbedRequest(
        new URLSearchParams(
          "embed=console-wolf&parent_origin=https%3A%2F%2Fevil.example",
        ),
        ["https://console.runi.services"],
        { "console-wolf": "wolf" },
      ),
    ).toBeNull();
  });

  it("rejects unknown embed ids and client profile drift", () => {
    const origins = ["https://console.runi.services"];
    const profiles = { "console-wolf": "wolf" };
    expect(
      parseDashboardEmbedRequest(
        new URLSearchParams(
          "embed=console-other&profile=wolf&parent_origin=https%3A%2F%2Fconsole.runi.services",
        ),
        origins,
        profiles,
      ),
    ).toBeNull();
    expect(
      parseDashboardEmbedRequest(
        new URLSearchParams(
          "embed=console-wolf&profile=torkil&parent_origin=https%3A%2F%2Fconsole.runi.services",
        ),
        origins,
        profiles,
      ),
    ).toBeNull();
    expect(
      parseDashboardEmbedRequest(
        new URLSearchParams(
          "embed=console-default&profile=default&parent_origin=https%3A%2F%2Fconsole.runi.services",
        ),
        origins,
        { "console-default": "" },
      )?.profile,
    ).toBe("");
  });

  it("posts bounded lifecycle events only to the validated parent", () => {
    const postMessage = vi.fn();
    postDashboardEmbedEvent(
      { postMessage },
      "https://console.runi.services",
      "ready",
      "console-wolf",
    );

    expect(postMessage).toHaveBeenCalledWith(
      {
        type: "hermes.dashboard.embed",
        event: "ready",
        embedId: "console-wolf",
      },
      "https://console.runi.services",
    );
  });

  it("marks the OAuth return bridge without starting a terminal", () => {
    expect(
      parseDashboardEmbedRequest(
        new URLSearchParams(
          "embed=console-hugin&auth_bridge=1&parent_origin=https%3A%2F%2Fconsole.runi.services",
        ),
        ["https://console.runi.services"],
        { "console-hugin": "" },
      )?.authBridge,
    ).toBe(true);
  });
});
