import { mythos } from "../../themes/hermes-mythos";
import { useTheme } from "../../themes/context";

interface Props {
  className?: string;
}

export default function StatusOmen({ className }: Props) {
  // useTheme for potential dark-mode adaptation
  useTheme();

  return (
    <svg
      className={className}
      viewBox="0 0 100 100"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Lightning bolt — warning omen"
    >
      {/* Amber glow aura behind the bolt */}
      <radialGradient id="omen-glow" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor={mythos.color.sunGold} stopOpacity="0.35" />
        <stop offset="45%" stopColor={mythos.color.terra} stopOpacity="0.12" />
        <stop offset="100%" stopColor={mythos.color.terra} stopOpacity="0" />
      </radialGradient>
      <circle cx="50" cy="50" r="48" fill="url(#omen-glow)" />

      {/* Outer glow ring — terra amber */}
      <circle
        cx="50"
        cy="50"
        r="42"
        fill="none"
        stroke={mythos.color.terra}
        strokeWidth="1"
        opacity="0.25"
      />
      <circle
        cx="50"
        cy="50"
        r="36"
        fill="none"
        stroke={mythos.color.sunGold}
        strokeWidth="0.6"
        opacity="0.18"
      />

      {/* Main lightning bolt shape */}
      <polygon
        points="58,8 36,48 50,48 42,92 70,44 54,44"
        fill={mythos.color.sunGold}
      />

      {/* Inner highlight on bolt for depth */}
      <polygon
        points="56,14 40,48 50,48 45,80 64,48 54,48"
        fill="#F0C040"
        opacity="0.45"
      />

      {/* Radiating energy lines — short spokes around bolt */}
      {/* Top-left */}
      <line x1="35" y1="22" x2="28" y2="14" stroke={mythos.color.sunGold} strokeWidth="1.5" strokeLinecap="round" opacity="0.6" />
      {/* Left */}
      <line x1="26" y1="42" x2="16" y2="40" stroke={mythos.color.sunGold} strokeWidth="1.5" strokeLinecap="round" opacity="0.5" />
      {/* Bottom-left */}
      <line x1="36" y1="66" x2="26" y2="72" stroke={mythos.color.sunGold} strokeWidth="1.5" strokeLinecap="round" opacity="0.4" />
      {/* Right */}
      <line x1="70" y1="40" x2="82" y2="36" stroke={mythos.color.sunGold} strokeWidth="1.5" strokeLinecap="round" opacity="0.5" />
      {/* Top-right */}
      <line x1="62" y1="24" x2="70" y2="14" stroke={mythos.color.sunGold} strokeWidth="1.5" strokeLinecap="round" opacity="0.5" />
      {/* Bottom-right */}
      <line x1="68" y1="62" x2="78" y2="68" stroke={mythos.color.sunGold} strokeWidth="1.5" strokeLinecap="round" opacity="0.4" />
      {/* Bottom */}
      <line x1="50" y1="82" x2="50" y2="92" stroke={mythos.color.sunGold} strokeWidth="1.5" strokeLinecap="round" opacity="0.35" />

      {/* Small secondary bolt spark — upper right */}
      <polygon
        points="74,18 68,30 73,30 68,42 80,28 74,28"
        fill={mythos.color.sunGold}
        opacity="0.45"
      />

      {/* Small secondary bolt spark — lower left */}
      <polygon
        points="26,58 20,70 25,70 20,82 32,68 26,68"
        fill={mythos.color.sunGold}
        opacity="0.35"
      />
    </svg>
  );
}
