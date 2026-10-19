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
  whatsappProfileRoutesFromList,
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
      { profile: "cmp_market" },
      { profile: " cmp_market " },
      { profile: "jbq_orchestrator" },
      { profile: "" },
    ]);
    expect(counts.get("cmp_market")).toBe(2);
    expect(counts.get("jbq_orchestrator")).toBe(1);
    expect(formatProfileRouteOption("cmp_market", counts.get("cmp_market") ?? 0)).toBe(
      "cmp_market (2)",
    );
    expect(formatProfileRouteOption("default", counts.get("default") ?? 0)).toBe("default (0)");
  });
});

describe("whatsapp chat labels", () => {
  it("shows a group name and hides the JID", () => {
    expect(whatsappChatLabel("120363@g.us", "Job Quest")).toBe("Job Quest");
    expect(whatsappChatLabel("120363@g.us", "120363@g.us")).toBe("Unnamed group");
    expect(whatsappChatLabel("120363431353336060@g.us", "120363431353336060")).toBe(
      "Unnamed group",
    );
    expect(whatsappChatLabel("1555@s.whatsapp.net", "")).toBe("Unnamed contact");
  });

  it("offers only chats that are not already routed", () => {
    const seen = [
      { chat_id: "120363@g.us", name: "Job Quest" },
      { chat_id: "999@g.us", name: "Reportes" },
    ];
    expect(unassignedWhatsAppChats(seen, [{ chat_id: "120363@g.us" }])).toEqual([
      { chat_id: "999@g.us", name: "Reportes" },
    ]);
  });

  it("builds a unique internal route name from the display label", () => {
    const taken = new Set(["job-quest"]);
    expect(whatsappRouteNameForChat("Job Quest", "120363abcd@g.us", taken)).toBe(
      "job-quest-bcdgus",
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
