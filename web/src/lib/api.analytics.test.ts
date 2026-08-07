import { describe, expect, it } from "vitest";

import { normalizeAnalyticsResponse, type AnalyticsResponse } from "./api";

const totals = {
  total_input: 0,
  total_output: 0,
  total_cache_read: 0,
  total_reasoning: 0,
  total_estimated_cost: 0,
  total_actual_cost: 0,
  total_sessions: 0,
  total_api_calls: 0,
};

describe("normalizeAnalyticsResponse", () => {
  it("fills in missing skill analytics for older backends", () => {
    const raw: AnalyticsResponse = {
      daily: [],
      by_model: [],
      totals,
    };

    const normalized = normalizeAnalyticsResponse(raw);

    expect(normalized.skills).toEqual({
      summary: {
        total_skill_loads: 0,
        total_skill_edits: 0,
        total_skill_actions: 0,
        distinct_skills_used: 0,
      },
      top_skills: [],
    });
  });

  it("fills in partially-missing skill analytics fields", () => {
    const raw: AnalyticsResponse = {
      daily: [],
      by_model: [],
      totals,
      skills: {},
    };

    const normalized = normalizeAnalyticsResponse(raw);

    expect(normalized.skills.summary.total_skill_loads).toBe(0);
    expect(normalized.skills.top_skills).toEqual([]);
  });

  it("preserves populated skill analytics", () => {
    const raw: AnalyticsResponse = {
      daily: [],
      by_model: [],
      totals,
      skills: {
        summary: {
          total_skill_loads: 2,
          total_skill_edits: 1,
          total_skill_actions: 3,
          distinct_skills_used: 2,
        },
        top_skills: [
          {
            skill: "systematic-debugging",
            view_count: 2,
            manage_count: 1,
            total_count: 3,
            percentage: 100,
            last_used_at: 1713900000,
          },
        ],
      },
    };

    const normalized = normalizeAnalyticsResponse(raw);

    expect(normalized.skills).toEqual(raw.skills);
  });
});
