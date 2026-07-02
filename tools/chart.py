import argparse
import csv
import math
import os
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


STYLE = {
    "figure.figsize": (11, 5.5),
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
}

PALETTE = [
    "#2E86AB", "#E63946", "#06A77D", "#F4A261", "#6A4C93",
    "#FF6B9D", "#118AB2", "#EF476F", "#06D6A0", "#FFD166",
]


def read_csv(path):
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
    return headers, rows


def try_parse_date(value):
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def detect_chart_type(headers):
    if len(headers) == 2:
        return "bar"
    if len(headers) == 3:
        return "grouped_bar"
    return "facet"


def _is_numeric(value):
    if value is None or value == "":
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def detect_wide_format(headers, rows):
    """Wide = one column per series, one row per x.

    Signature: 3+ columns, every column after the first is fully numeric on
    every row, AND the first column has no duplicates. Long format has
    duplicate x values (one row per (x, series)), so uniqueness of col 0 is
    the distinguishing tell.
    """
    if len(headers) < 3 or not rows:
        return False
    for row in rows:
        for v in row[1:]:
            if not _is_numeric(v):
                return False
    first_col = [row[0] for row in rows]
    return len(first_col) == len(set(first_col))


def wide_format_error(headers, input_path):
    x_col = headers[0]
    series_cols = headers[1:]
    series_list = ", ".join(series_cols)
    union_lines = "\n".join(
        f"SELECT {x_col}, '{s}' AS series, {s} AS value FROM <source>"
        + (" UNION ALL" if i < len(series_cols) - 1 else "")
        for i, s in enumerate(series_cols)
    )
    return (
        f"ERROR: {input_path} appears to be in WIDE format "
        f"(x='{x_col}', series columns={series_list}).\n"
        f"chart.py requires LONG format for multi-series charts — one row per (x, series, value).\n"
        f"If this CSV is rendered as-is, col 2 ('{series_cols[0]}') becomes the 'group' label and "
        f"every value becomes a distinct legend entry, producing an exploded legend.\n"
        f"\n"
        f"Reshape the source query to long format, e.g.:\n"
        f"{union_lines}\n"
        f"ORDER BY {x_col}, series;\n"
        f"\n"
        f"Or use UNPIVOT:\n"
        f"SELECT * FROM <wide_source> UNPIVOT (value FOR series IN ({series_list}));\n"
        f"\n"
        f"See yallaplay-wiki/reference/analytics_rules.md and the analytics skill output rules.\n"
        f"To override this check (e.g. facet chart with intentional wide shape), pass --allow-wide."
    )


def drop_null_rows(headers, rows, numeric_col_indexes):
    """Return (clean_rows, dropped_rows) where any numeric column is null/empty."""
    clean, dropped = [], []
    for i, row in enumerate(rows, start=2):  # start=2 so index matches 1-based CSV line (header=1)
        if any(row[j] is None or row[j] == "" for j in numeric_col_indexes):
            dropped.append(i)
        else:
            clean.append(row)
    return clean, dropped


def find_null_cells(headers, rows, numeric_col_indexes):
    """Return list of (row_idx_1based_line, col_name) for the first few nulls."""
    hits = []
    for i, row in enumerate(rows, start=2):
        for j in numeric_col_indexes:
            if row[j] is None or row[j] == "":
                hits.append((i, headers[j]))
                if len(hits) >= 3:
                    return hits
    return hits


def validate_data(headers, rows, chart_type, value_format, input_path, allow_nulls):
    """Run hard guardrails. Prints ERROR + exits 2 on violation, else returns cleaned rows."""
    # Determine which columns carry numeric values for this chart type.
    if chart_type == "bar":
        numeric_cols = [1]
    elif chart_type in ("grouped_bar", "line", "heatmap"):
        numeric_cols = [2]
    elif chart_type == "facet":
        numeric_cols = [3] if len(headers) >= 4 else [2]
    else:
        numeric_cols = [len(headers) - 1]

    # 1. NULL / empty values in numeric cells.
    null_hits = find_null_cells(headers, rows, numeric_cols)
    if null_hits:
        if allow_nulls:
            rows, dropped = drop_null_rows(headers, rows, numeric_cols)
            print(
                f"warning: dropped {len(dropped)} row(s) with null/empty numeric cells "
                f"(lines {dropped[:5]}{'...' if len(dropped) > 5 else ''})",
                file=sys.stderr,
            )
        else:
            samples = ", ".join(f"line {ln} col '{col}'" for ln, col in null_hits)
            print(
                f"ERROR: {input_path} has null/empty numeric cells ({samples}).\n"
                f"Fix the source query: drop the rows with WHERE IS NOT NULL, "
                f"or impute via COALESCE(col, 0). Do not leave nulls implicit.\n"
                f"To skip null rows and proceed with a warning, pass --allow-nulls.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Need at least one row after null-drop.
    if not rows:
        print(f"ERROR: {input_path} has no data rows after validation.", file=sys.stderr)
        sys.exit(2)

    # 2. Minimum-row guards per chart type.
    min_needed = 2
    if chart_type == "bar" and len(rows) < min_needed:
        print(
            f"ERROR: bar chart needs at least {min_needed} rows, got {len(rows)}. "
            f"Nothing to compare.",
            file=sys.stderr,
        )
        sys.exit(2)
    if chart_type in ("grouped_bar", "line"):
        distinct_x = len({row[0] for row in rows})
        if distinct_x < 2:
            print(
                f"ERROR: {chart_type} needs at least 2 distinct values of '{headers[0]}', "
                f"got {distinct_x}. Nothing to compare across x.",
                file=sys.stderr,
            )
            sys.exit(2)

    # 3. Percent-format out of range. Catches forgot-to-multiply-by-100 and mis-counted rates.
    if value_format == "pct":
        bad_range = []
        all_vals = []
        for i, row in enumerate(rows, start=2):
            for j in numeric_cols:
                try:
                    v = float(row[j])
                except (ValueError, TypeError):
                    continue
                all_vals.append(v)
                if v < -5 or v > 105:
                    bad_range.append((i, headers[j], v))
        if bad_range:
            samples = "; ".join(f"line {ln} '{col}'={v}" for ln, col, v in bad_range[:3])
            print(
                f"ERROR: column inferred as percent ('{headers[numeric_cols[0]]}') has values outside [0, 100] "
                f"({samples}).\n"
                f"Likely cause: values are not scaled percentages, or a miscount.\n"
                f"Multiply by 100 in SQL (e.g. 100.0 * retained / cohort_size), or pass --format num "
                f"if this is not a percent.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Decimals-not-scaled heuristic: all values in [0, 1] and at least one strictly >0
        # is almost certainly unscaled (e.g. 0.28 meant as 28%). A real 100%-bounded percent
        # series will have at least one value >= 1 in practice.
        if all_vals and all(0 <= v <= 1 for v in all_vals) and any(v > 0 for v in all_vals):
            max_v = max(all_vals)
            print(
                f"ERROR: column inferred as percent ('{headers[numeric_cols[0]]}') has all values in [0, 1] "
                f"(max={max_v}).\n"
                f"Looks like decimals that weren't scaled — a percent should be in the 0–100 range.\n"
                f"Multiply by 100 in SQL (e.g. 100.0 * retained / cohort_size), or pass --format num "
                f"if this is really meant as a unitless fraction.",
                file=sys.stderr,
            )
            sys.exit(2)

    # 4. Too many categories on grouped_bar.
    if chart_type == "grouped_bar":
        distinct_groups = len({row[1] for row in rows})
        if distinct_groups > 12:
            print(
                f"ERROR: grouped_bar has {distinct_groups} distinct values of '{headers[1]}' "
                f"(max 12 for readability). Aggregate smaller groups into 'other' in SQL, "
                f"or use --type facet to split into sub-plots.",
                file=sys.stderr,
            )
            sys.exit(2)

    return rows


def title_from_path(path):
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-?", "", base)
    base = base.replace("_", " ").replace("-", " ")
    return " ".join(w.capitalize() for w in base.split())


def format_value(v, value_format):
    if value_format == "pct":
        return f"{v:.1f}%"
    if value_format == "int":
        return f"{int(v):,}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def axis_formatter(value_format):
    if value_format == "pct":
        return FuncFormatter(lambda v, _: f"{v:.0f}%")
    if value_format == "int":
        return FuncFormatter(lambda v, _: f"{int(v):,}")
    return FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}")


def sort_rows(rows, by_value, descending=True):
    if not by_value:
        return rows
    try:
        return sorted(rows, key=lambda r: float(r[-1]), reverse=descending)
    except (ValueError, IndexError):
        return rows


def apply_footnote(fig, footnote):
    if footnote:
        fig.text(0.5, -0.02, footnote, ha="center", va="top",
                 fontsize=9, color="#666666", style="italic")


def plot_bar(headers, rows, title, output, value_format, show_values, sort, footnote):
    rows = sort_rows(rows, sort)
    labels = [r[0] for r in rows]
    values = [float(r[1]) for r in rows]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=PALETTE[0], width=0.7)
    ax.set_xlabel(headers[0])
    ax.set_ylabel(headers[1])
    ax.set_title(title)
    ax.yaxis.set_major_formatter(axis_formatter(value_format))
    ax.grid(axis="x", visible=False)

    if show_values:
        max_v = max(values) if values else 0
        pad = max_v * 0.015
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                    format_value(v, value_format), ha="center", va="bottom", fontsize=9)
        ax.set_ylim(bottom=0, top=max_v * 1.12)
    else:
        ax.set_ylim(bottom=0)

    plt.xticks(rotation=35, ha="right")
    apply_footnote(fig, footnote)
    plt.savefig(output)
    plt.close()


def plot_grouped_bar(headers, rows, title, output, value_format, show_values, sort, footnote):
    x_col, group_col, value_col = headers[0], headers[1], headers[2]

    if sort:
        totals = {}
        for row in rows:
            totals[row[0]] = totals.get(row[0], 0) + float(row[2])
        ordered_x = [x for x, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)]
    else:
        ordered_x = []
        for row in rows:
            if row[0] not in ordered_x:
                ordered_x.append(row[0])

    groups = []
    data = {}
    for row in rows:
        x, g, v = row[0], row[1], float(row[2])
        if g not in groups:
            groups.append(g)
        data[(x, g)] = v

    fig, ax = plt.subplots()
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)
    xs = list(range(len(ordered_x)))

    for i, g in enumerate(groups):
        offsets = [x + (i - (n_groups - 1) / 2) * width for x in xs]
        values = [data.get((xl, g), 0) for xl in ordered_x]
        bars = ax.bar(offsets, values, width=width, label=g, color=PALETTE[i % len(PALETTE)])
        if show_values:
            max_v = max((v for v in data.values()), default=0)
            pad = max_v * 0.012
            for bar, v in zip(bars, values):
                if v != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                            format_value(v, value_format), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels(ordered_x, rotation=35, ha="right")
    ax.set_xlabel(x_col)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(axis_formatter(value_format))
    ax.grid(axis="x", visible=False)
    ax.legend(title=group_col, loc="best")

    if show_values:
        max_v = max((v for v in data.values()), default=0)
        ax.set_ylim(top=max_v * 1.15)

    apply_footnote(fig, footnote)
    plt.savefig(output)
    plt.close()


def plot_line(headers, rows, title, output, value_format, footnote):
    x_col, group_col, value_col = headers[0], headers[1], headers[2]
    use_dates = try_parse_date(rows[0][0]) is not None

    series = {}
    for row in rows:
        x = try_parse_date(row[0]) if use_dates else row[0]
        group = row[1]
        value = float(row[2])
        series.setdefault(group, ([], []))
        series[group][0].append(x)
        series[group][1].append(value)

    fig, ax = plt.subplots()
    for i, (group, (xs, ys)) in enumerate(series.items()):
        ax.plot(xs, ys, marker="o", label=group, color=PALETTE[i % len(PALETTE)])

    ax.set_xlabel(x_col)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(axis_formatter(value_format))
    ax.legend(title=group_col, loc="best")

    if use_dates:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    else:
        plt.xticks(rotation=35, ha="right")

    apply_footnote(fig, footnote)
    plt.savefig(output)
    plt.close()


def plot_heatmap(headers, rows, title, output, value_format, footnote):
    x_col, y_col, value_col = headers[0], headers[1], headers[2]

    x_labels = []
    y_labels = []
    data = {}
    for row in rows:
        x, y, v = row[0], row[1], float(row[2])
        if x not in x_labels:
            x_labels.append(x)
        if y not in y_labels:
            y_labels.append(y)
        data[(x, y)] = v

    grid = [[data.get((x, y), float("nan")) for x in x_labels] for y in y_labels]

    fig, ax = plt.subplots()
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn" if value_format == "pct" else "viridis")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(False)

    for i, y in enumerate(y_labels):
        for j, x in enumerate(x_labels):
            v = data.get((x, y))
            if v is not None:
                ax.text(j, i, format_value(v, value_format), ha="center", va="center",
                        fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_major_formatter(axis_formatter(value_format))

    apply_footnote(fig, footnote)
    plt.savefig(output)
    plt.close()


def plot_facet(headers, rows, title, output, value_format, footnote):
    # Expects: facet_col, x_col, [group_col,] value_col
    # For 4 columns: facet, x, group, value -> one line/bar chart per facet with grouped series
    # For 3 columns: facet, x, value -> one bar chart per facet
    if len(headers) == 4:
        facet_col, x_col, group_col, value_col = headers
        has_groups = True
    elif len(headers) == 3:
        facet_col, x_col, value_col = headers
        group_col = None
        has_groups = False
    else:
        # fall back to line chart with first 3 columns
        plot_line(headers[:3], [r[:3] for r in rows], title, output, value_format, footnote)
        return

    facets = []
    for row in rows:
        if row[0] not in facets:
            facets.append(row[0])

    n = len(facets)
    cols = min(n, 3)
    rows_layout = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_layout, cols, figsize=(5.5 * cols, 4 * rows_layout),
                             squeeze=False, sharey=True)

    for idx, facet in enumerate(facets):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        facet_rows = [row for row in rows if row[0] == facet]

        if has_groups:
            series = {}
            for row in facet_rows:
                x, g, v = row[1], row[2], float(row[3])
                series.setdefault(g, ([], []))
                series[g][0].append(x)
                series[g][1].append(v)
            for i, (g, (xs, ys)) in enumerate(series.items()):
                ax.plot(xs, ys, marker="o", label=g, color=PALETTE[i % len(PALETTE)])
            ax.legend(title=group_col, loc="best", fontsize=8)
        else:
            xs = [row[1] for row in facet_rows]
            ys = [float(row[2]) for row in facet_rows]
            ax.bar(xs, ys, color=PALETTE[idx % len(PALETTE)], width=0.7)

        ax.set_title(f"{facet_col}: {facet}", fontsize=11, fontweight="bold")
        ax.set_xlabel(x_col)
        if c == 0:
            ax.set_ylabel(value_col)
        ax.yaxis.set_major_formatter(axis_formatter(value_format))
        ax.tick_params(axis="x", rotation=35)

    for idx in range(n, rows_layout * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    apply_footnote(fig, footnote)
    plt.savefig(output)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate a chart from a CSV file")
    parser.add_argument("input", help="Path to a CSV file")
    parser.add_argument("-o", "--output", default="chart.png", help="Output image path")
    parser.add_argument("-t", "--title", default=None,
                        help="Chart title (default: derived from input filename)")
    parser.add_argument("--footnote", default="",
                        help="Footnote/caption below the chart (e.g., sample size, date range)")
    parser.add_argument("--type", choices=["bar", "grouped_bar", "line", "heatmap", "facet"],
                        help="Chart type (auto-detected if omitted)")
    parser.add_argument("--format", choices=["auto", "pct", "int", "num"], default="auto",
                        help="Value formatting (default: auto-detect from column name)")
    parser.add_argument("--no-labels", action="store_true",
                        help="Disable data labels on bars")
    parser.add_argument("--sort", action="store_true",
                        help="Sort bar chart categories by value (descending)")
    parser.add_argument("--allow-wide", action="store_true",
                        help="Bypass the wide-format shape check (rare; only for intentional wide CSVs)")
    parser.add_argument("--allow-nulls", action="store_true",
                        help="Skip rows with null/empty numeric cells with a warning, instead of aborting")
    args = parser.parse_args()

    headers, rows = read_csv(args.input)
    if not rows:
        print("CSV has no data rows")
        sys.exit(1)

    if not args.allow_wide and detect_wide_format(headers, rows):
        print(wide_format_error(headers, args.input), file=sys.stderr)
        sys.exit(2)

    title = args.title if args.title is not None else title_from_path(args.input)

    value_format = args.format
    if value_format == "auto":
        last_col = headers[-1].lower()
        if "pct" in last_col or "percent" in last_col or "rate" in last_col:
            value_format = "pct"
        elif any(k in last_col for k in ("users", "count", "rows", "games", "sessions")):
            value_format = "int"
        else:
            value_format = "num"

    plt.rcParams.update(STYLE)

    chart_type = args.type or detect_chart_type(headers)
    rows = validate_data(headers, rows, chart_type, value_format, args.input, args.allow_nulls)
    show_values = not args.no_labels

    if chart_type == "bar":
        plot_bar(headers, rows, title, args.output, value_format, show_values, args.sort, args.footnote)
    elif chart_type == "grouped_bar":
        plot_grouped_bar(headers, rows, title, args.output, value_format, show_values, args.sort, args.footnote)
    elif chart_type == "line":
        plot_line(headers, rows, title, args.output, value_format, args.footnote)
    elif chart_type == "heatmap":
        plot_heatmap(headers, rows, title, args.output, value_format, args.footnote)
    elif chart_type == "facet":
        plot_facet(headers, rows, title, args.output, value_format, args.footnote)

    print(f"Chart saved to {args.output}")


if __name__ == "__main__":
    main()
