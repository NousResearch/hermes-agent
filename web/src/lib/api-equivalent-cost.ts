export interface AnalyticsStat {
  label: string;
  value: string;
}

export function apiEquivalentCostStat(
  amount: number | null | undefined,
  label: string,
  unpricedTokens = 0,
): AnalyticsStat | null {
  if (amount === null || amount === undefined) return null;
  if (amount === 0 && unpricedTokens > 0) {
    return { label, value: "N/A" };
  }
  return {
    label,
    value: `$${amount.toFixed(2)}${unpricedTokens > 0 ? "+" : ""}`,
  };
}
