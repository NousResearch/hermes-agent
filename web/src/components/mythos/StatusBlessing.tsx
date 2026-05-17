import { mythos } from "../../themes/hermes-mythos";
import { useTheme } from "../../themes/context";

interface Props {
  className?: string;
}

const OLIVE_GREEN = "#2D7D4F";

export default function StatusBlessing({ className }: Props) {
  // useTheme for potential dark-mode adaptation
  useTheme();

  // Left olive branch leaves — arching from bottom-left to top
  const leftLeaves = [
    { x: 28, y: 72, angle: -20 },
    { x: 23, y: 60, angle: -35 },
    { x: 22, y: 47, angle: -50 },
    { x: 25, y: 35, angle: -65 },
    { x: 33, y: 26, angle: -80 },
    { x: 44, y: 20, angle: -95 },
  ];

  // Right olive branch leaves — arching from bottom-right to top
  const rightLeaves = [
    { x: 72, y: 72, angle: 20 },
    { x: 77, y: 60, angle: 35 },
    { x: 78, y: 47, angle: 50 },
    { x: 75, y: 35, angle: 65 },
    { x: 67, y: 26, angle: 80 },
    { x: 56, y: 20, angle: 95 },
  ];

  return (
    <svg
      className={className}
      viewBox="0 0 100 100"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Olive wreath — success blessing"
    >
      {/* Left branch stem arc */}
      <path
        d="M36 82 C20 70, 14 52, 18 35 C22 20, 34 12, 50 10"
        fill="none"
        stroke={OLIVE_GREEN}
        strokeWidth="2"
        strokeLinecap="round"
      />
      {/* Right branch stem arc */}
      <path
        d="M64 82 C80 70, 86 52, 82 35 C78 20, 66 12, 50 10"
        fill="none"
        stroke={OLIVE_GREEN}
        strokeWidth="2"
        strokeLinecap="round"
      />

      {/* Left olive leaves */}
      {leftLeaves.map(({ x, y, angle }, i) => (
        <ellipse
          key={`l${i}`}
          cx={x}
          cy={y}
          rx="5"
          ry="9"
          fill={OLIVE_GREEN}
          opacity="0.85"
          transform={`rotate(${angle}, ${x}, ${y})`}
        />
      ))}

      {/* Right olive leaves */}
      {rightLeaves.map(({ x, y, angle }, i) => (
        <ellipse
          key={`r${i}`}
          cx={x}
          cy={y}
          rx="5"
          ry="9"
          fill={OLIVE_GREEN}
          opacity="0.85"
          transform={`rotate(${angle}, ${x}, ${y})`}
        />
      ))}

      {/* Small olive berries on left branch */}
      <circle cx="25" cy="54" r="2.5" fill="#1A5C36" opacity="0.7" />
      <circle cx="22" cy="42" r="2" fill="#1A5C36" opacity="0.7" />
      <circle cx="28" cy="30" r="2" fill="#1A5C36" opacity="0.7" />

      {/* Small olive berries on right branch */}
      <circle cx="75" cy="54" r="2.5" fill="#1A5C36" opacity="0.7" />
      <circle cx="78" cy="42" r="2" fill="#1A5C36" opacity="0.7" />
      <circle cx="72" cy="30" r="2" fill="#1A5C36" opacity="0.7" />

      {/* Bottom tie / knot where branches cross */}
      <path
        d="M42 84 C46 88, 50 90, 54 88 C58 86, 60 82, 58 80 C54 84, 46 84, 42 80 C40 82, 40 84, 42 84 Z"
        fill={OLIVE_GREEN}
        opacity="0.7"
      />

      {/* Golden star at the top center */}
      <polygon
        points="50,4 52.4,9.6 58.5,9.6 53.8,13.3 55.6,19 50,15.6 44.4,19 46.2,13.3 41.5,9.6 47.6,9.6"
        fill={mythos.color.sunGold}
      />

      {/* Inner subtle circle to ground the composition */}
      <circle
        cx="50"
        cy="52"
        r="22"
        fill="none"
        stroke={OLIVE_GREEN}
        strokeWidth="0.5"
        opacity="0.2"
      />
    </svg>
  );
}
