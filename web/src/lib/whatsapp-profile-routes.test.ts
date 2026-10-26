import { describe, expect, it, vi } from "vitest";

import {
  extractProfileRoutes,
  formatProfileRouteOption,
  formatWhatsAppLastSeen,
  mergeWhatsAppProfileRoutes,
  multiplexProfilesFlag,
  persistWhatsAppProfileRoutes,
  profileRoutesSavePartial,
  unassignedWhatsAppChats,
  validateWhatsAppRouteChatId,
  whatsappChatLabel,
  whatsappMultiplexerEngaged,
  whatsappProfileRoutesFromList,
  whatsappRouteManager,
  whatsappRouteCountsByProfile,
  whatsappRouteNameForChat,
} from "./whatsapp-profile-routes";

describe("whatsapp profile_routes merge", () => {
  it("round-trips whatsapp rows while preserving other platforms", async () => {
    const existing = [
      { name: "discord-main", platform: "discord", guild_id: "1", profile: "acme" },
      { name: "old-wa", platform: "whatsapp", chat_id: "111@g.us", profile: "a" },
    ];
    const edited = [
      { name: "wa-group", platform: "whatsapp" as const, chat_id: "120363@g.us", profile: "team" },
    ];

    const merged = mergeWhatsAppProfileRoutes(existing, edited);
    expect(merged).toHaveLength(2);
    expect(merged[0]).toMatchObject({ platform: "discord", guild_id: "1" });
    expect(merged[1]).toMatchObject(edited[0]);

    const getConfig = vi.fn().mockResolvedValue({ gateway: { profile_routes: existing } });
    const saveConfig = vi.fn().mockResolvedValue({ ok: true });
    const restartGateway = vi.fn();

    await persistWhatsAppProfileRoutes({ getConfig, saveConfig }, "default", edited);

    expect(getConfig).toHaveBeenCalledWith("default");
    expect(saveConfig).toHaveBeenCalledWith(
      { gateway: { profile_routes: merged } },
      "default",
    );
    expect(restartGateway).not.toHaveBeenCalled();
  });

  it("filters whatsapp rows for display", () => {
    const routes = [
      { platform: "telegram", chat_id: "-100", profile: "x" },
      { platform: "whatsapp", name: "g", chat_id: "9@g.us", profile: "p" },
    ];
    expect(whatsappProfileRoutesFromList(routes)).toHaveLength(1);
    expect(whatsappProfileRoutesFromList(routes)[0].chat_id).toBe("9@g.us");
  });
});

describe("whatsappRouteCountsByProfile", () => {
  it("counts chats per profile and labels unused profiles as zero", () => {
    const counts = whatsappRouteCountsByProfile([
      { profile: "market" },
      { profile: " market " },
      { profile: "ops" },
      { profile: "" },
    ]);
    expect(counts.get("market")).toBe(2);
    expect(counts.get("ops")).toBe(1);
    expect(formatProfileRouteOption("market", counts.get("market") ?? 0)).toBe(
      "market (2)",
    );
    expect(formatProfileRouteOption("default", counts.get("default") ?? 0)).toBe("default (0)");
  });
});

describe("whatsapp chat labels", () => {
  it("shows a real name, and the number when the name is missing", () => {
    expect(whatsappChatLabel("120363@g.us", "Design")).toBe("Design");
    expect(whatsappChatLabel("120363@g.us", "120363@g.us")).toBe("120363");
    expect(whatsappChatLabel("120363000000000001@g.us", "unknown")).toBe(
      "120363000000000001",
    );
    expect(whatsappChatLabel("1555@s.whatsapp.net", "")).toBe("1555");
    expect(whatsappChatLabel("15551234567@s.whatsapp.net", "unknown number")).toBe(
      "15551234567",
    );
    expect(whatsappChatLabel("15551234567", "15551234567")).toBe("15551234567");
  });

  it("offers only chats that are not already routed", () => {
    const seen = [
      { chat_id: "120363@g.us", name: "Design" },
      { chat_id: "999@g.us", name: "Reports" },
    ];
    expect(unassignedWhatsAppChats(seen, [{ chat_id: "120363@g.us" }])).toEqual([
      { chat_id: "999@g.us", name: "Reports" },
    ]);
  });

  it("builds a unique internal route name from the display label", () => {
    const taken = new Set(["design"]);
    expect(whatsappRouteNameForChat("Design", "120363abcd@g.us", taken)).toBe(
      "design-bcdgus",
    );
  });

  it("formats last seen as a local timestamp", () => {
    const formatted = formatWhatsAppLastSeen(Date.UTC(2026, 8, 18, 17, 5) / 1000);
    expect(formatted).toMatch(/^2026-09-18 \d{2}:\d{2}$/);
  });
});

describe("profile route location", () => {
  const whatsapp = [
    { name: "wa-group", platform: "whatsapp" as const, chat_id: "120363@g.us", profile: "team" },
  ];

  it("uses a top-level list even when it is empty and the nested list is not", () => {
    const config = {
      profile_routes: [],
      gateway: {
        profile_routes: [
          { name: "hidden", platform: "whatsapp", chat_id: "1@g.us", profile: "acme" },
        ],
      },
    };
    expect(extractProfileRoutes(config)).toEqual([]);
    expect(profileRoutesSavePartial(config, whatsapp)).toEqual({
      profile_routes: whatsapp,
      gateway: { profile_routes: whatsapp },
    });
  });

  it("keeps non-whatsapp rows in each list that already exists", () => {
    const config = {
      profile_routes: [{ name: "discord-main", platform: "discord", guild_id: "1", profile: "acme" }],
      gateway: {
        profile_routes: [{ name: "tg", platform: "telegram", chat_id: "-100", profile: "team" }],
      },
    };
    expect(extractProfileRoutes(config)).toEqual(config.profile_routes);
    const partial = profileRoutesSavePartial(config, whatsapp);
    expect(partial.profile_routes).toEqual([...config.profile_routes, ...whatsapp]);
    expect((partial.gateway as { profile_routes: unknown[] }).profile_routes).toEqual([
      ...config.gateway.profile_routes,
      ...whatsapp,
    ]);
  });

  it("writes the nested list when the file has neither", () => {
    expect(profileRoutesSavePartial({}, whatsapp)).toEqual({
      gateway: { profile_routes: whatsapp },
    });
  });

  it("reads the multiplex flag from the raw file, not a missing key", () => {
    expect(multiplexProfilesFlag({})).toBeNull();
    expect(multiplexProfilesFlag({ multiplex_profiles: false })).toBe(false);
    expect(multiplexProfilesFlag({ gateway: { multiplex_profiles: true } })).toBe(true);
    expect(multiplexProfilesFlag({ multiplex_profiles: false, gateway: { multiplex_profiles: true } })).toBe(
      false,
    );
  });
});

describe("whatsapp multiplexer link", () => {
  const routes = [{ profile: "ops" }, { profile: "market" }];

  it("treats the selected profile as linked when it is a route target", () => {
    expect(whatsappRouteManager({
      selectedProfile: "ops",
      hostProfile: "default",
      routes,
    })).toEqual({ manager: "default", linked: true });
    expect(whatsappRouteManager({
      selectedProfile: "",
      hostProfile: "default",
      routes,
    })).toEqual({ manager: "default", linked: false });
  });

  it("is engaged when the flag is on or the gateway already serves several profiles", () => {
    expect(whatsappMultiplexerEngaged(true, null, null)).toBe(true);
    expect(whatsappMultiplexerEngaged(null, ["default", "ops"], null)).toBe(true);
    expect(whatsappMultiplexerEngaged(null, ["default"], "a secondary gateway is still running")).toBe(false);
    expect(whatsappMultiplexerEngaged(false, null, null)).toBe(false);
  });
});

describe("validateWhatsAppRouteChatId", () => {
  it("accepts group JIDs ending in @g.us", () => {
    expect(validateWhatsAppRouteChatId("120363123456789012@g.us")).toBeNull();
  });

  it("accepts phone numbers and user JIDs", () => {
    expect(validateWhatsAppRouteChatId("15551234567")).toBeNull();
    expect(validateWhatsAppRouteChatId("15551234567@s.whatsapp.net")).toBeNull();
    expect(validateWhatsAppRouteChatId("abc@lid")).toBeNull();
  });

  it("rejects empty, whitespace, and invalid group ids", () => {
    expect(validateWhatsAppRouteChatId("")).toMatch(/required/i);
    expect(validateWhatsAppRouteChatId("120363 @g.us")).toMatch(/whitespace/i);
    expect(validateWhatsAppRouteChatId("120363@g.us.extra")).toMatch(/@g\.us/i);
    expect(validateWhatsAppRouteChatId("@foo.com")).toMatch(/group JID/i);
  });
});
