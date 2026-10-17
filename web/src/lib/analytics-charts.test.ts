import { describe, expect, it } from "vitest";

import { inputOutputRatio, OTHER_MODELS, stackDailyByModel } from "./analytics-charts";

const row = (day: string, model: string, input: number, output = 0) => ({
  day,
  model,
  input_tokens: input,
  output_tokens: output,
});

describe("stackDailyByModel (#20412)", () => {
  const rows = [
    row("2026-05-01", "a", 900, 100),
    row("2026-05-01", "b", 50),
    row("2026-05-01", "c", 10),
    row("2026-05-02", "b", 600),
    row("2026-05-02", "d", 5),
    row("2026-05-02", "", 30),
  ];

  it("keeps each day's total equal to the sum of its model rows, folded or not", () => {
    for (const maxSeries of [1, 2, 5]) {
      const { days } = stackDailyByModel(rows, maxSeries);
      for (const d of days) {
        const raw = rows.filter((r) => r.day === d.day).reduce((s, r) => s + r.input_tokens + r.output_tokens, 0);
        expect(d.total).toBe(raw);
        expect(d.segments.reduce((s, x) => s + x.tokens, 0)).toBe(raw);
      }
    }
  });

  it("ranks series once per period so a model keeps its slot on every day", () => {
    const { series, days } = stackDailyByModel(rows, 2);
    expect(series).toEqual(["a", "b", OTHER_MODELS]);
    for (const d of days) {
      const keys = d.segments.map((s) => s.key);
      expect(keys).toEqual(series.filter((k) => keys.includes(k)));
    }
  });

  it("never folds when every model fits", () => {
    const { series } = stackDailyByModel(rows, 10);
    expect(series).not.toContain(OTHER_MODELS);
  });
});

describe("inputOutputRatio", () => {
  it("is undefined (null) without output rather than Infinity", () => {
    expect(inputOutputRatio(500, 0)).toBeNull();
    expect(inputOutputRatio(500, 50)).toBe(10);
  });
});
