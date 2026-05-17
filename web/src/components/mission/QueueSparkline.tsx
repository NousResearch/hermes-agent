import { type FC, useRef, useEffect } from "react";
import * as Plot from "@observablehq/plot";
import type { QueuePoint } from "./types";

interface QueueSparklineProps {
  points: QueuePoint[];
  openNow: number;
  label: string;
  className?: string;
}

export const QueueSparkline: FC<QueueSparklineProps> = ({
  points,
  openNow,
  label,
  className,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || points.length === 0) return;

    const el = containerRef.current;
    const w = el.clientWidth || 300;
    const h = el.clientHeight || 72;

    const data = points
      .slice()
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((p) => ({ date: new Date(p.date), count: p.count }));

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
      y: { label: null, grid: true, tickFormat: "d" },
      marks: [
        Plot.areaY(data, {
          x: "date",
          y: "count",
          fill: "var(--color-warning, #f5a623)",
          fillOpacity: 0.18,
          curve: "monotone-x",
        }),
        Plot.lineY(data, {
          x: "date",
          y: "count",
          stroke: "var(--color-warning, #f5a623)",
          strokeWidth: 1.2,
          curve: "monotone-x",
        }),
      ],
    });

    el.innerHTML = "";
    el.append(chart);
    return () => {
      chart.remove();
    };
  }, [points]);

  return (
    <div className={className} style={{ position: "relative" }}>
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 28,
          fontSize: 9,
          fontFamily: "ui-monospace, monospace",
          color: "rgba(168,192,214,0.5)",
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          pointerEvents: "none",
          zIndex: 1,
        }}
      >
        {label} <span style={{ color: "var(--color-warning, #f5a623)" }}>· {openNow} open</span>
      </div>
      <div
        ref={containerRef}
        style={{ width: "100%", height: "100%", minHeight: 60 }}
      />
    </div>
  );
};
