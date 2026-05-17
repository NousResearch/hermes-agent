import { mythos } from "../../themes/hermes-mythos";
import { useTheme } from "../../themes/context";

interface Props {
  className?: string;
}

export default function TriadThrones({ className }: Props) {
  const { theme } = useTheme();
  const isDark = theme.palette.background.hex.toLowerCase() < "#888888";
  const labelColor = isDark ? mythos.color.marble : mythos.color.night;

  const Throne = ({ cx, cy, label }: { cx: number; cy: number; label: string }) => (
    <g>
      {/* Throne back */}
      <rect
        x={cx - 14}
        y={cy - 30}
        width={28}
        height={42}
        rx="3"
        fill="none"
        stroke={mythos.color.lapis}
        strokeWidth="2"
      />
      {/* Seat cushion */}
      <rect
        x={cx - 13}
        y={cy + 4}
        width={26}
        height={8}
        rx="2"
        fill={mythos.color.sunGold}
        opacity="0.85"
      />
      {/* Decorative back panel top */}
      <rect
        x={cx - 10}
        y={cy - 26}
        width={20}
        height={26}
        rx="2"
        fill={mythos.color.lapis}
        opacity="0.15"
      />
      {/* Crown finial left */}
      <polygon
        points={`${cx - 10},${cy - 30} ${cx - 7},${cy - 36} ${cx - 4},${cy - 30}`}
        fill={mythos.color.lapis}
      />
      {/* Crown finial center */}
      <polygon
        points={`${cx - 3},${cy - 30} ${cx},${cy - 38} ${cx + 3},${cy - 30}`}
        fill={mythos.color.sunGold}
      />
      {/* Crown finial right */}
      <polygon
        points={`${cx + 4},${cy - 30} ${cx + 7},${cy - 36} ${cx + 10},${cy - 30}`}
        fill={mythos.color.lapis}
      />
      {/* Legs */}
      <line x1={cx - 10} y1={cy + 12} x2={cx - 10} y2={cy + 22} stroke={mythos.color.lapis} strokeWidth="2" />
      <line x1={cx + 10} y1={cy + 12} x2={cx + 10} y2={cy + 22} stroke={mythos.color.lapis} strokeWidth="2" />
      {/* Label */}
      <text
        x={cx}
        y={cy + 34}
        textAnchor="middle"
        fontSize="7"
        fontFamily={mythos.font.body}
        fill={labelColor}
        opacity="0.85"
      >
        {label}
      </text>
    </g>
  );

  return (
    <svg
      className={className}
      viewBox="0 0 140 120"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Triad of thrones: Planner, Executor, Critic"
    >
      {/* Top throne — Planner */}
      <Throne cx={70} cy={32} label="Planner" />
      {/* Bottom-left throne — Executor */}
      <Throne cx={30} cy={82} label="Executor" />
      {/* Bottom-right throne — Critic */}
      <Throne cx={110} cy={82} label="Critic" />
      {/* Triangle connecting lines (subtle) */}
      <line x1={70} y1={46} x2={30} y2={68} stroke={mythos.color.lapis} strokeWidth="1" opacity="0.3" strokeDasharray="3 3" />
      <line x1={70} y1={46} x2={110} y2={68} stroke={mythos.color.lapis} strokeWidth="1" opacity="0.3" strokeDasharray="3 3" />
      <line x1={30} y1={68} x2={110} y2={68} stroke={mythos.color.lapis} strokeWidth="1" opacity="0.3" strokeDasharray="3 3" />
    </svg>
  );
}
