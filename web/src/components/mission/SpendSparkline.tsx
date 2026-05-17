import { type FC, useRef, useEffect } from "react";
import * as Plot from "@observablehq/plot";
import type { SpendPoint } from "./types";

interface SpendSparklineProps {
  points: SpendPoint[];
  label: string;
  className?: string;
}

export const SpendSparkline: FC<SpendSparklineProps> = ({
  points,
  label,
  className,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || points.length === 0) return;

    const el = containerRef.current;
    const w = el.clientWidth || 300;
    const h = el.clientHeight || 72;

    const aggregated = new Map<string, number>();
    for (const p of points) {
      aggregated.set(p.date, (aggregated.get(p.date) ?? 0) + p.amountUsd);
    }
    const data = Array.from(aggregated.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, amountUsd]) => ({ date: new Date(date), amountUsd }));

    const chart = Plot.plot({
      width: w,
      height: h,
      margin: 0,
      marginLeft: 28,
      marginBottom: 16,
      marginTop: 4,
      marginRight: 4,
      style: {
        background: "transparent",
        color: "rgba(168,192,214,0.7)",
        fontFamily: "ui-monospace, monospace",
        fontSize: "9px",
        overflow: "visible",
      },
      x: { type: "time", label: null },
      y: { label: null, grid: true },
      marks: [
        Plot.areaY(data, {
          x: "date",
          y: "amountUsd",
          fill: "rgba(0,188,212,0.12)",
          curve: "monotone-x",
        }),
        Plot.lineY(data, {
          x: "date",
          y: "amountUsd",
          stroke: "#00bcd4",
          strokeWidth: 1.5,
          curve: "monotone-x",
        }),
        Plot.dot(data.slice(-1), {
          x: "date",
          y: "amountUsd",
          fill: "#00bcd4",
          r: 3,
        }),
      ],
    });

    el.innerHTML = "";
    el.appendChild(chart);

    return () => {
      chart.remove();
    };
  }, [points]);

  const latest = points.reduce((sum, p) => sum + p.amountUsd, 0);

  return (
    <div
      className={className}
      style={{
        position: "relative",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        gap: 2,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          padding: "0 4px",
          fontSize: 10,
          fontFamily: "ui-monospace, monospace",
          color: "rgba(168,192,214,0.5)",
        }}
      >
        <span>{label}</span>
        {latest > 0 && (
          <span style={{ color: "#00bcd4", fontWeight: 600 }}>
            ${latest.toFixed(2)} est
          </span>
        )}
      </div>
      <div
        ref={containerRef}
        style={{
          flex: 1,
          minHeight: 0,
          overflow: "hidden",
        }}
      />
    </div>
  );
};
