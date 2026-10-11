import { describe, expect, it } from "vitest";

import {
  MODEL_SERIES_COLORS,
  OTHER_MODELS,
  ioRatio,
  periodRatio,
  stackModelsByDay,
} from "./analytics-series";

const row = (day: string, model: string, input: number, output = 0) => ({
  day,
  model,
  input_tokens: input,
  output_tokens: output,
});

describe("ioRatio / periodRatio", () => {
  it("is undefined for a day with no output", () => {
    expect(ioRatio(1000, 0)).toBeNull();
    expect(ioRatio(1000, 50)).toBe(20);
  });

  it("weights the period average by tokens, not by day", () => {
    const daily = [
      { day: "d1", input_tokens: 900, output_tokens: 100 }, // 9x
      { day: "d2", input_tokens: 100, output_tokens: 100 }, // 1x
    ];
    expect(periodRatio(daily)).toBe(5);
    expect(periodRatio([])).toBeNull();
  });
});

describe("stackModelsByDay", () => {
  it("aligns to the given days and ranks models by period volume", () => {
    const stack = stackModelsByDay(
      [row("d1", "small", 10), row("d1", "big", 500, 50), row("d3", "small", 30)],
      ["d1", "d2", "d3"],
    );

    expect(stack.models).toEqual(["big", "small"]);
    expect(stack.days.map((d) => [d.day, d.total])).toEqual([
      ["d1", 560],
      ["d2", 0],
      ["d3", 30],
    ]);
    expect(stack.days[0].segments.map((s) => s.model)).toEqual(["big", "small"]);
    expect(stack.max).toBe(560);
  });

  it("folds models past the palette into Other instead of cycling hues", () => {
    const n = MODEL_SERIES_COLORS.length + 2;
    const rows = Array.from({ length: n }, (_, i) => row("d1", `m${i}`, (n - i) * 100));
    const stack = stackModelsByDay(rows, ["d1"]);

    expect(stack.models).toHaveLength(MODEL_SERIES_COLORS.length);
    expect(stack.models.at(-1)).toBe(OTHER_MODELS);
    const other = stack.days[0].segments.find((s) => s.model === OTHER_MODELS)!;
    expect(other.input_tokens).toBe(300 + 200 + 100);
    expect(stack.days[0].total).toBe(rows.reduce((s, r) => s + r.input_tokens, 0));
  });

  it("keeps every name when the models exactly fill the palette", () => {
    const rows = MODEL_SERIES_COLORS.map((_, i) => row("d1", `m${i}`, 100 - i));
    expect(stackModelsByDay(rows, ["d1"]).models).not.toContain(OTHER_MODELS);
  });
});
