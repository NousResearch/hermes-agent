import { mythos } from "../../themes/hermes-mythos";
import { useTheme } from "../../themes/context";

interface Props {
  className?: string;
}

export default function HermesFlying({ className }: Props) {
  const { theme } = useTheme();
  const isDark = theme.palette.background.hex.toLowerCase() < "#888888";
  const figureColor = isDark ? mythos.color.marble : mythos.color.night;

  return (
    <svg
      className={className}
      viewBox="0 0 120 120"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Hermes flying messenger"
    >
      {/* Left wing — upper arc */}
      <path
        d="M38 58 C20 42, 8 34, 4 22 C12 28, 22 36, 38 48 Z"
        fill={mythos.color.sunGold}
        opacity="0.92"
      />
      {/* Left wing — lower arc */}
      <path
        d="M38 62 C18 52, 6 52, 2 64 C12 58, 24 56, 38 66 Z"
        fill={mythos.color.sunGold}
        opacity="0.75"
      />
      {/* Right wing — upper arc */}
      <path
        d="M82 58 C100 42, 112 34, 116 22 C108 28, 98 36, 82 48 Z"
        fill={mythos.color.sunGold}
        opacity="0.92"
      />
      {/* Right wing — lower arc */}
      <path
        d="M82 62 C102 52, 114 52, 118 64 C108 58, 96 56, 82 66 Z"
        fill={mythos.color.sunGold}
        opacity="0.75"
      />
      {/* Body / torso */}
      <ellipse cx="60" cy="66" rx="8" ry="12" fill={figureColor} />
      {/* Head */}
      <circle cx="60" cy="48" r="9" fill={figureColor} />
      {/* Winged helmet — left wing nub */}
      <path
        d="M52 44 C46 38, 44 34, 48 32 C50 36, 52 40, 54 44 Z"
        fill={mythos.color.sunGold}
      />
      {/* Winged helmet — right wing nub */}
      <path
        d="M68 44 C74 38, 76 34, 72 32 C70 36, 68 40, 66 44 Z"
        fill={mythos.color.sunGold}
      />
      {/* Helmet band */}
      <rect x="51" y="43" width="18" height="4" rx="2" fill={mythos.color.lapis} />
      {/* Outstretched arm holding caduceus */}
      <line x1="68" y1="62" x2="86" y2="54" stroke={figureColor} strokeWidth="3" strokeLinecap="round" />
      {/* Caduceus staff */}
      <line x1="86" y1="54" x2="96" y2="44" stroke={mythos.color.lapis} strokeWidth="2.5" strokeLinecap="round" />
      {/* Caduceus top knob */}
      <circle cx="96" cy="43" r="3" fill={mythos.color.sunGold} />
      {/* Caduceus serpent coil left */}
      <path
        d="M88 52 C84 48, 88 44, 92 48 C96 52, 92 56, 88 52"
        fill="none"
        stroke={mythos.color.lapis}
        strokeWidth="1.5"
      />
      {/* Caduceus serpent coil right */}
      <path
        d="M91 52 C95 48, 99 44, 95 48 C91 52, 95 56, 91 52"
        fill="none"
        stroke={mythos.color.sunGold}
        strokeWidth="1.5"
      />
      {/* Scroll in other hand */}
      <ellipse cx="40" cy="70" rx="6" ry="4" fill={mythos.color.marble} stroke={mythos.color.lapis} strokeWidth="1" />
      <line x1="34" y1="70" x2="46" y2="70" stroke={mythos.color.lapis} strokeWidth="0.8" />
      {/* Other arm */}
      <line x1="52" y1="62" x2="40" y2="70" stroke={figureColor} strokeWidth="3" strokeLinecap="round" />
      {/* Legs trailing back in flight */}
      <line x1="56" y1="78" x2="50" y2="94" stroke={figureColor} strokeWidth="2.5" strokeLinecap="round" />
      <line x1="64" y1="78" x2="70" y2="94" stroke={figureColor} strokeWidth="2.5" strokeLinecap="round" />
      {/* Foot wing nubs */}
      <path d="M46 92 C42 90, 40 94, 44 96 Z" fill={mythos.color.sunGold} />
      <path d="M74 92 C78 90, 80 94, 76 96 Z" fill={mythos.color.sunGold} />
    </svg>
  );
}
