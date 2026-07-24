export interface TokenBreakdown {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  reasoning: number;
}

export interface TokenSegment {
  key: "cacheRead" | "cacheWrite" | "reasoning" | "input" | "output";
  label: string;
  value: number;
}

export interface TokenBarBreakdown {
  total: number;
  segments: TokenSegment[];
  metadata: TokenSegment[];
}

export function tokenBarBreakdown({
  input,
  output,
  cacheRead,
  cacheWrite,
  reasoning,
}: TokenBreakdown): TokenBarBreakdown {
  // CanonicalUsage.total_tokens is prompt + output. Reasoning tokens are
  // separately reported metadata and may already be represented in output.
  const total = input + output + cacheRead + cacheWrite;
  const segments = [
    { key: "cacheRead", label: "Cache Read", value: cacheRead },
    { key: "cacheWrite", label: "Cache Write", value: cacheWrite },
    { key: "input", label: "Input", value: input },
    { key: "output", label: "Output", value: output },
  ].filter((segment) => segment.value > 0) as TokenSegment[];
  const metadata = [
    { key: "reasoning", label: "Reasoning", value: reasoning },
  ].filter((segment) => segment.value > 0) as TokenSegment[];

  return { total, segments, metadata };
}
