import { describe, expect, it } from "vitest";
import { reportArtifactPath, rewriteReportLinks } from "./report-download";

const artifacts = [{ filename: "status-comparison.csv", sha256: "verified" }];

describe("reportArtifactPath", () => {
  it("returns a scoped URL only for an artifact in the selected report", () => {
    expect(reportArtifactPath("as24-de-partial-20260927-v4", "status-comparison.csv", artifacts)).toBe(
      "/api/reports/as24-de-partial-20260927-v4/status-comparison.csv",
    );
    expect(reportArtifactPath("as24-de-partial-20260927-v4", "other.csv", artifacts)).toBeNull();
    expect(reportArtifactPath("as24-de-partial-20260927-v4", "../status-comparison.csv", artifacts)).toBeNull();
  });
});

describe("rewriteReportLinks", () => {
  it("bridges a bound standalone detail and relative downloads under the configured base path", () => {
    const html = rewriteReportLinks(
      '<head></head><a href="/reports/report-1/f001-details.html">Details</a><a href="findings.csv">CSV</a>',
      "report-1", "/hermes",
    );
    expect(html).toContain('href="/hermes/api/reports/report-1/f001-details.html"');
    expect(html).toContain('href="/hermes/api/reports/report-1/findings.csv"');
    expect(html).toContain("hermes-report-detail");
    expect(html).toContain("Content-Security-Policy");
  });
});
