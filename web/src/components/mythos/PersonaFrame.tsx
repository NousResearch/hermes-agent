import { mythos } from "../../themes/hermes-mythos";
import { useTheme } from "../../themes/context";

interface Props {
  className?: string;
}

export default function PersonaFrame({ className }: Props) {
  const { theme } = useTheme();
  const isDark = theme.palette.background.hex.toLowerCase() < "#888888";
  const frameBase = isDark ? mythos.color.marble : "#C8C0A8";

  return (
    <svg
      className={className}
      viewBox="0 0 100 120"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Marble cameo persona frame"
    >
      {/* Outer ornate oval — marble colored */}
      <ellipse
        cx="50"
        cy="58"
        rx="46"
        ry="56"
        fill={frameBase}
        opacity="0.92"
      />
      {/* Lapis accent ring just inside the outer oval */}
      <ellipse
        cx="50"
        cy="58"
        rx="46"
        ry="56"
        fill="none"
        stroke={mythos.color.lapis}
        strokeWidth="1.2"
        opacity="0.5"
      />
      {/* Decorative groove ring 1 */}
      <ellipse
        cx="50"
        cy="58"
        rx="42"
        ry="52"
        fill="none"
        stroke={mythos.color.lapis}
        strokeWidth="0.7"
        opacity="0.3"
      />
      {/* Decorative groove ring 2 */}
      <ellipse
        cx="50"
        cy="58"
        rx="39"
        ry="49"
        fill="none"
        stroke={mythos.color.lapis}
        strokeWidth="0.5"
        opacity="0.2"
      />

      {/* Ornamental top laurel accent */}
      <path
        d="M34 8 C38 4, 44 3, 50 2 C56 3, 62 4, 66 8"
        fill="none"
        stroke={mythos.color.sunGold}
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <circle cx="50" cy="2" r="2.5" fill={mythos.color.sunGold} />
      {/* Side leaf motifs top-left */}
      <path d="M28 18 C24 14, 20 16, 22 20 C24 24, 28 22, 28 18 Z" fill={mythos.color.sunGold} opacity="0.7" />
      {/* Side leaf motifs top-right */}
      <path d="M72 18 C76 14, 80 16, 78 20 C76 24, 72 22, 72 18 Z" fill={mythos.color.sunGold} opacity="0.7" />

      {/* Bottom ornamental scroll accent */}
      <path
        d="M34 110 C38 114, 44 115, 50 116 C56 115, 62 114, 66 110"
        fill="none"
        stroke={mythos.color.sunGold}
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <circle cx="50" cy="116" r="2.5" fill={mythos.color.sunGold} />

      {/* Marble veining — subtle diagonal lines */}
      <line x1="18" y1="30" x2="28" y2="50" stroke={mythos.color.lapis} strokeWidth="0.4" opacity="0.12" />
      <line x1="70" y1="70" x2="80" y2="88" stroke={mythos.color.lapis} strokeWidth="0.3" opacity="0.1" />
      <line x1="22" y1="72" x2="35" y2="90" stroke={mythos.color.lapis} strokeWidth="0.3" opacity="0.08" />

      {/* Inner oval — transparent avatar area */}
      <ellipse
        cx="50"
        cy="58"
        rx="32"
        ry="40"
        fill="transparent"
        stroke={mythos.color.lapis}
        strokeWidth="1"
        opacity="0.4"
      />
      {/* Inner oval lapis accent line */}
      <ellipse
        cx="50"
        cy="58"
        rx="30"
        ry="38"
        fill="transparent"
        stroke={mythos.color.lapis}
        strokeWidth="0.5"
        opacity="0.2"
      />
    </svg>
  );
}
