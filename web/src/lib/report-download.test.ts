import { describe, expect, it } from "vitest";
import { reportArtifactPath, rewriteReportLinks } from "./report-download";

const artifacts = [{ filename: "status-comparison.csv", sha256: "verified" }];

describe("reportArtifactPath", () => {
  it("returns a scoped URL only for an artifact in the selected report", () => {
    expect(reportArtifactPath("sample-report", "status-comparison.csv", artifacts)).toBe(
      "/api/reports/sample-report/status-comparison.csv",
    );
    expect(reportArtifactPath("sample-report", "other.csv", artifacts)).toBeNull();
    expect(reportArtifactPath("sample-report", "../status-comparison.csv", artifacts)).toBeNull();
  });
});

describe("rewriteReportLinks", () => {
  it("closes the restrictive CSP attribute before report markup with no head element", () => {
    const html = rewriteReportLinks("<!doctype html><html><body><h1>Executive summary</h1></body></html>", "report-1", "");
    expect(html).toMatch(/<meta http-equiv="Content-Security-Policy" content="sandbox allow-scripts; default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'">/);
    expect(html.indexOf("<h1>Executive summary</h1>")).toBeGreaterThan(html.indexOf("<meta "));
  });

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
