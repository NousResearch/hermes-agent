export const mythos = {
  color: {
    marble:  '#F4F0E6',
    lapis:   '#1E3A8A',
    sunGold: '#D4A017',
    terra:   '#A4421A',
    night:   '#0B1220',
  },
  font: {
    display: '"Fraunces", Georgia, serif',
    body:    '"Inter", system-ui, sans-serif',
  },
  radius: { card: '14px', chip: '999px' },
  shadow: { sanctum: '0 12px 32px -16px rgba(11,18,32,0.45)' },
} as const;

export type MythosColors = typeof mythos.color;
