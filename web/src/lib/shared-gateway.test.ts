import { describe, expect, it } from "vitest";

import { ApiError } from "./api-error";
import {
  gatewayActionFailedMessage,
  servedProfileRefusal,
  sharedGatewayProfiles,
} from "./shared-gateway";

describe("sharedGatewayProfiles", () => {
  it("names every bot on the shared multiplexer, default first", () => {
    expect(
      sharedGatewayProfiles({ gateway_shared_with: ["beta", "default", "alpha"] }),
    ).toEqual(["default", "alpha", "beta"]);
  });

  it("is null for standalone gateways, older backends and a lone default", () => {
    expect(sharedGatewayProfiles({ gateway_shared_with: null })).toBeNull();
    expect(sharedGatewayProfiles({})).toBeNull();
    expect(sharedGatewayProfiles({ gateway_shared_with: ["default"] })).toBeNull();
  });
});

describe("servedProfileRefusal", () => {
  it("unwraps the 409 detail into a sentence and ignores other failures", () => {
    const err = new Error(
      '409: {"detail":"The default gateway already serves profile \'alpha\' as a multiplexer; stop it from the default profile instead of a separate gateway for this profile."}',
    );
    expect(servedProfileRefusal(err)).toMatch(/^The default gateway already serves profile 'alpha'/);
    expect(servedProfileRefusal(new Error("500: boom"))).toBeNull();
    const apiErr = new ApiError("The default gateway already serves profile 'beta'", {
      status: 409,
      body: "",
      url: "/api/gateway/start",
    });
    expect(servedProfileRefusal(apiErr)).toMatch(/^The default gateway already serves profile 'beta'/);
    expect(
      servedProfileRefusal(new ApiError("boom", { status: 500, body: "", url: "/x" })),
    ).toBeNull();
  });
});

describe("gatewayActionFailedMessage", () => {
  it("does not double punctuation when the detail already ends a sentence", () => {
    const text = gatewayActionFailedMessage("restart", "Is `hermes dashboard` still running?");
    expect(text).not.toMatch(/[?!.]\./);
    expect(text).toMatch(/^Could not restart the gateway: /);
    expect(text).toContain("still running? Open Logs");
  });

  it("skips the Logs pointer when the dashboard itself is unreachable", () => {
    const unreachable = new ApiError("down", { status: 0, body: "", url: "/api/gateway/start" });
    expect(gatewayActionFailedMessage("start", "down", unreachable)).not.toContain("Logs");

    const serverError = new ApiError("boom", { status: 500, body: "", url: "/api/gateway/start" });
    expect(gatewayActionFailedMessage("start", "boom.", serverError)).toContain("Open Logs for details.");
    expect(gatewayActionFailedMessage("start", "boom.", serverError)).not.toContain("boom..");
  });
});
