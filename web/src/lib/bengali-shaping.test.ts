import { describe, expect, it, vi } from "vitest";

import {
  type CharacterJoinRange,
  DASHBOARD_CHAT_TERMINAL_FONT_FAMILY,
  getBengaliCharacterJoinRanges,
  registerBengaliCharacterJoiner,
} from "./bengali-shaping";

describe("DASHBOARD_CHAT_TERMINAL_FONT_FAMILY", () => {
  it("keeps Bengali-capable fonts ahead of generic monospace fallback", () => {
    expect(DASHBOARD_CHAT_TERMINAL_FONT_FAMILY).toContain(
      "'Noto Sans Bengali'",
    );
    expect(
      DASHBOARD_CHAT_TERMINAL_FONT_FAMILY.indexOf("'Noto Sans Bengali'"),
    ).toBeLessThan(DASHBOARD_CHAT_TERMINAL_FONT_FAMILY.indexOf("monospace"));
  });
});

describe("getBengaliCharacterJoinRanges", () => {
  it("joins Bengali conjunct samples as script runs", () => {
    const text =
      "প্রযুক্তি ক্লিপবোর্ড যুক্তাক্ষর শ্রদ্ধা কর্তৃপক্ষ";

    const joined = getBengaliCharacterJoinRanges(text).map(([start, end]) =>
      text.slice(start, end),
    );

    expect(joined).toEqual([
      "প্রযুক্তি",
      "ক্লিপবোর্ড",
      "যুক্তাক্ষর",
      "শ্রদ্ধা",
      "কর্তৃপক্ষ",
    ]);
  });

  it("does not join Latin text or isolated Bengali characters", () => {
    const text = "run ক test বাংলা";

    const joined = getBengaliCharacterJoinRanges(text).map(([start, end]) =>
      text.slice(start, end),
    );

    expect(joined).toEqual(["বাংলা"]);
  });
});

describe("registerBengaliCharacterJoiner", () => {
  it("registers and deregisters the Bengali joiner", () => {
    let registeredHandler: (text: string) => CharacterJoinRange[] = () => [];
    const term = {
      registerCharacterJoiner: vi.fn(
        (handler: (text: string) => CharacterJoinRange[]) => {
          registeredHandler = handler;
          return 17;
        },
      ),
      deregisterCharacterJoiner: vi.fn(),
    };

    const dispose = registerBengaliCharacterJoiner(term);

    expect(registeredHandler("প্রযুক্তি")).toEqual([
      [0, "প্রযুক্তি".length],
    ]);

    dispose();

    expect(term.deregisterCharacterJoiner).toHaveBeenCalledWith(17);
  });
});
