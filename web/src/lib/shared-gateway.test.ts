import { describe, expect, it } from "vitest";

import {
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
  });
});
