import { describe, expect, it, vi } from "vitest";

import {
  mergeWhatsAppProfileRoutes,
  persistWhatsAppProfileRoutes,
  validateWhatsAppRouteChatId,
  whatsappProfileRoutesFromList,
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
