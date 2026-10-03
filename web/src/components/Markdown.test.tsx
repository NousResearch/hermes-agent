// GFM table rendering in the dashboard Markdown renderer: detection,
// structure, ragged-row normalisation, and paragraph non-interference.
import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";

import { Markdown } from "./Markdown";

const TABLE_MD = [
  "| Method | Speed | Cost |",
  "| :--- | ---: | :---: |",
  "| api | fast | $$ |",
  "| batch | slow | $ |",
].join("\n");

function render(md: string): string {
  return renderToStaticMarkup(<Markdown content={md} />);
}

describe("Markdown GFM tables", () => {
  it("renders a GFM table as a single real <table> with header and rows", () => {
    const html = render(TABLE_MD);

    const tableCount = html.split("<table").length - 1;
    expect(tableCount).toBe(1);

    // '<th[ >]' so we don't count the <thead> wrapper
    const thCount = html.split("<th").length - 1 - (html.includes("<thead") ? 1 : 0);
    const tdCount = html.split("<td").length - 1;
    const trCount = html.split("<tr").length - 1;
    expect(thCount).toBe(3);
    expect(tdCount).toBe(6);
    expect(trCount).toBe(3); // 1 header + 2 body rows

    expect(html).toContain("Method");
    expect(html).toContain("api");
  });

  it("does not leak the literal divider row into the output", () => {
    const html = render(TABLE_MD);
    expect(html).not.toContain("---:");
    expect(html).not.toContain(":---");
    expect(html).not.toMatch(/<t[hd][^>]*>\s*[-:]{3,}/);
  });

  it("applies per-column alignment from the divider cells", () => {
    const html = render(TABLE_MD);
    const thClasses = [...html.matchAll(/<th class="([^"]*)"/g)].map((m) => m[1]);
    expect(thClasses).toHaveLength(3);
    expect(thClasses[0]).toContain("text-left");
    expect(thClasses[1]).toContain("text-right");
    expect(thClasses[2]).toContain("text-center");
  });

  it("normalises ragged rows to the header width", () => {
    const md = [
      "| A | B | C |",
      "| --- | --- | --- |",
      "| only-one |",
      "| x | y | z | extra |",
    ].join("\n");
    const html = render(md);

    // Short row padded, long row clamped: every body <tr> carries exactly 3
    // cells; the clamped overflow cell is dropped, per the TUI numCols contract
    const trs = html.split("<tr").slice(1);
    expect(trs).toHaveLength(3);
    expect(trs[0]!.split("<th").length - 1).toBe(3);
    for (const tr of trs.slice(1)) {
      expect(tr.split("<td").length - 1).toBe(3);
    }
    expect(html).not.toContain("extra");
  });

  it("renders inline formatting inside cells through InlineContent", () => {
    const md = ["| Name | Note |", "| --- | --- |", "| **bold** | `code` |"].join("\n");
    const html = render(md);
    expect(html).toContain("<strong");
    expect(html).toContain("<code");
  });

  it("does not false-trigger a table on a plain paragraph containing '|'", () => {
    const html = render("a || b is not a table\nplain continuation");
    expect(html.split("<table").length - 1).toBe(0);
    expect(html).toContain("<p>");
    // The pipes survive as literal text inside the paragraph
    expect(html).toContain("a || b is not a table");
  });

  it("terminates an in-progress paragraph when a table starts", () => {
    const md = [
      "intro paragraph line",
      "| H1 | H2 |",
      "| --- | --- |",
      "| a | b |",
    ].join("\n");
    const html = render(md);

    expect(html.split("<table").length - 1).toBe(1);
    // The header cells must not appear inside the preceding <p>
    const pMatch = html.match(/<p>(.*?)<\/p>/s);
    expect(pMatch).not.toBeNull();
    expect(pMatch![1]).not.toContain("H1");
    expect(pMatch![1]).toContain("intro paragraph line");
  });

  it("stops the table at a blank line and resumes paragraph parsing", () => {
    const md = [
      "| H | J |",
      "| --- | --- |",
      "| v | w |",
      "",
      "after the table",
    ].join("\n");
    const html = render(md);

    expect(html.split("<table").length - 1).toBe(1);
    const pMatch = html.match(/<p>(.*?)<\/p>/s);
    expect(pMatch).not.toBeNull();
    expect(pMatch![1]).toContain("after the table");
  });

  it("leaves non-table markdown untouched", () => {
    const html = render("# Title\n\n- item\n\ntext with **bold**");
    expect(html).toContain("<h1");
    expect(html).toContain("<ul");
    expect(html).toContain("<strong");
    expect(html.split("<table").length - 1).toBe(0);
  });
});
