import { mythos } from "../../themes/hermes-mythos";
import { useTheme } from "../../themes/context";

interface Props {
  className?: string;
}

export default function WhileYouSleep({ className }: Props) {
  // useTheme accessed for potential future dark-mode adaptations
  useTheme();

  return (
    <svg
      className={className}
      viewBox="0 0 140 100"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Hermes working while you sleep"
    >
      {/* Night sky background */}
      <rect width="140" height="100" fill={mythos.color.night} rx="4" />

      {/* Stars */}
      {[
        [10, 10], [25, 6], [50, 8], [80, 5], [100, 12], [120, 8], [130, 15],
        [15, 22], [60, 18], [110, 20], [135, 6],
      ].map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r="0.8" fill={mythos.color.marble} opacity="0.7" />
      ))}

      {/* Moon crescent */}
      <circle cx="118" cy="18" r="8" fill="#E8D5A3" />
      <circle cx="122" cy="15" r="7" fill={mythos.color.night} />

      {/* Classical columns — left */}
      <rect x="8" y="46" width="10" height="42" fill={mythos.color.marble} opacity="0.25" />
      <rect x="6" y="44" width="14" height="4" rx="1" fill={mythos.color.marble} opacity="0.35" />
      <rect x="5" y="86" width="16" height="4" rx="1" fill={mythos.color.marble} opacity="0.35" />
      {/* Column fluting lines */}
      <line x1="11" y1="48" x2="11" y2="86" stroke={mythos.color.marble} strokeWidth="0.5" opacity="0.2" />
      <line x1="14" y1="48" x2="14" y2="86" stroke={mythos.color.marble} strokeWidth="0.5" opacity="0.2" />

      {/* Classical columns — right */}
      <rect x="122" y="46" width="10" height="42" fill={mythos.color.marble} opacity="0.25" />
      <rect x="120" y="44" width="14" height="4" rx="1" fill={mythos.color.marble} opacity="0.35" />
      <rect x="119" y="86" width="16" height="4" rx="1" fill={mythos.color.marble} opacity="0.35" />
      <line x1="125" y1="48" x2="125" y2="86" stroke={mythos.color.marble} strokeWidth="0.5" opacity="0.2" />
      <line x1="128" y1="48" x2="128" y2="86" stroke={mythos.color.marble} strokeWidth="0.5" opacity="0.2" />

      {/* Entablature / roof beam connecting columns */}
      <rect x="8" y="42" width="124" height="5" rx="1" fill={mythos.color.marble} opacity="0.2" />

      {/* Ground / floor line */}
      <rect x="8" y="88" width="124" height="3" rx="1" fill={mythos.color.marble} opacity="0.2" />

      {/* Sleeping figure on a couch/bench */}
      {/* Bench */}
      <rect x="28" y="80" width="84" height="8" rx="2" fill={mythos.color.lapis} opacity="0.5" />
      {/* Sleeping body — horizontal */}
      <ellipse cx="70" cy="76" rx="34" ry="6" fill={mythos.color.marble} opacity="0.3" />
      {/* Head */}
      <circle cx="98" cy="74" r="6" fill={mythos.color.marble} opacity="0.4" />
      {/* Pillow */}
      <ellipse cx="98" cy="78" rx="8" ry="3" fill={mythos.color.marble} opacity="0.25" />
      {/* Draped blanket suggestion */}
      <path
        d="M30 76 Q50 70 70 75 Q90 80 112 74"
        fill="none"
        stroke={mythos.color.marble}
        strokeWidth="1.5"
        opacity="0.2"
      />
      {/* Zzz letters */}
      <text x="106" y="68" fontSize="6" fill={mythos.color.marble} opacity="0.45" fontFamily={mythos.font.body}>z</text>
      <text x="112" y="62" fontSize="7" fill={mythos.color.marble} opacity="0.35" fontFamily={mythos.font.body}>z</text>
      <text x="119" y="55" fontSize="8" fill={mythos.color.marble} opacity="0.25" fontFamily={mythos.font.body}>z</text>

      {/* Small Hermes figure flying above — sunGold */}
      {/* Wings */}
      <path d="M52 34 C42 26, 36 22, 32 18 C38 22, 44 28, 52 38 Z" fill={mythos.color.sunGold} opacity="0.9" />
      <path d="M52 38 C40 34, 34 36, 30 44 C36 40, 44 38, 52 42 Z" fill={mythos.color.sunGold} opacity="0.7" />
      <path d="M62 34 C72 26, 78 22, 82 18 C76 22, 70 28, 62 38 Z" fill={mythos.color.sunGold} opacity="0.9" />
      <path d="M62 38 C74 34, 80 36, 84 44 C78 40, 70 38, 62 42 Z" fill={mythos.color.sunGold} opacity="0.7" />
      {/* Body */}
      <ellipse cx="57" cy="42" rx="5" ry="7" fill={mythos.color.sunGold} opacity="0.9" />
      {/* Head */}
      <circle cx="57" cy="33" r="5" fill={mythos.color.sunGold} opacity="0.9" />
      {/* Helmet wings */}
      <path d="M53 31 C49 27, 48 25, 51 24 C52 27, 53 29, 54 31 Z" fill={mythos.color.sunGold} />
      <path d="M61 31 C65 27, 66 25, 63 24 C62 27, 61 29, 60 31 Z" fill={mythos.color.sunGold} />
      {/* Caduceus */}
      <line x1="62" y1="40" x2="70" y2="33" stroke={mythos.color.lapis} strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="70" cy="32" r="2" fill={mythos.color.sunGold} />
    </svg>
  );
}
