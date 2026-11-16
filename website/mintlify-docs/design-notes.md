# Hermes Agent — design notes for the Mintlify preview

Reviewed 2026-11-16. Every value below was read from the rendered site (computed styles, CSS custom properties, or asset files), not estimated by eye.

## Sources

| Source | URL | What it gave us |
|---|---|---|
| Marketing site (canonical brand) | https://hermes-agent.nousresearch.com/ | Live colors, typography, header, buttons, hero art |
| Published design system (Storybook, `@nous-research/ui`) | https://design.nousresearch.com/ | Foundations → Typography, Pages → Hermes Landing, Layout → Header/Footer |
| Old docs (Docusaurus) | https://hermes-agent.nousresearch.com/docs/ | Logo link target, system color mode; stock theme with `#FFD700` gold accents (not used: it is not the brand) |
| Marketing stylesheets | `https://web-assets.nousresearch.com/marketing/<build>/_next/static/immutable/chunks/*.css` | Token definitions quoted below |

## Colors

Defined in the marketing CSS as `--hermes-color-*` and derived with `color-mix()`; derived values computed by hand from those formulas.

| Role | Value | Token / where |
|---|---|---|
| Brand blue (primary, hero surface, CTAs) | `#0000F2` | `--hermes-color-blue`, `<meta name="theme-color">` |
| Ink (light-mode headings) | `#000091` | `--hermes-blue-dark-40`; computed color of `h2` "Features" |
| Paper (light background, text on blue) | `#F2F2F2` | `--hermes-color-white`; computed `body` color |
| Neutral (muted text) | `#616191` | `--hermes-color-neutral` |
| Code surface | `#000030` | `--hermes-surface-dark` (= blue-dark-80) |
| Dark-mode page | `#0D0D24` | `--marketing-dark-paper` |
| Dark-mode raised surface | `#19192F` | `--marketing-dark-back` |
| Dark-mode body text | `#E6E6EF` | `--marketing-dark-text` |
| Dark-mode link / emphasis | `#8585F9` | `--marketing-dark-link` |
| Dark-mode hero | `#000091` | `--marketing-dark-blue` |
| Accent palette (unused in docs) | yellow `#F2F200`, cyan `#00F2F2`, green `#00F279`, etc. | `--hermes-color-*` |

Mapping in `docs.json`: `colors.primary #0000F2`, `colors.light #8585F9`, `colors.dark #000091` (all hue 240°); `background.color` light `#F2F2F2` / dark `#0D0D24`. Mintlify's `--gray-*` ramp is re-tinted in `custom.css` to these inks so sidebar, body and borders read as Hermes blue rather than neutral grey.

Contrast (measured in `mint dev`): light body `#222280` on `#F2F2F2` 11.7:1, headings 13.3:1, muted footer links 5.2:1; dark body `#A5A5C8` on `#0D0D24` 8.0:1, headings 17.1:1, links 6.1:1.

## Typography

| Use | Live site (computed) | Design system | Preview substitute (Google Fonts) |
|---|---|---|---|
| Display headings | "Rules Gothic Condensed" 200–400, `-0.02em` tracking (`h1` 108px, `h2` 80px) | "Sigurd Variable" serif (Foundations → Typography) | Sofia Sans Extra Condensed 300/400 |
| Wordmark / small display | "Rules Gothic Compressed" 500, uppercase | — | baked into logo SVG |
| Body | "Rules Variable" 16px | "Rules Variable" | Instrument Sans |
| Buttons / nav | "Rules Condensed" 500, uppercase, `0.3px`–`1.4px` tracking | "Rules Variable" uppercase | Sofia Sans Condensed 600, uppercase |
| Captions / code | "Aeonik Fono Pro" 10–11px uppercase | "Courier Prime" | Geist Mono |

Rules, Sigurd and Aeonik Fono (a TRIAL build) are proprietary and self-hosted by Nous, so they are not loaded here. The design system's serif display face conflicts with the live site's condensed gothic; we followed the live site because it is what visitors see today.

## Logo and favicon

- Marketing nav logo is live text, not an image: "HERMES" over "AGENT", Rules Gothic Compressed 500, 36px, line-height 0.9, centered (`header .hw-teams-gothic`). Rebuilt as outlined SVG paths: `img/brand/logo-light.svg` (`#0000F2`) and `img/brand/logo-dark.svg` (`#F2F2F2`). Navbar height set to 2.5rem so both lines stay legible.
- Favicon: `https://web-assets.nousresearch.com/marketing/<build>/assets/hermes-agent/icon.png` (48×48) → `img/brand/favicon.png`.
- Hero art: `assets/hermes-agent/hero.webp` (950×1024, grey on black) → `img/brand/hero.webp`, blended with `mix-blend-mode: screen` over blue, as on the live hero.
- Logo links to the docs root, matching the old docs' `navbar__brand` (`/docs/`).

## Shape, density, chrome

- Corners: buttons `border-radius: 3px` (computed on "Install Hermes"); badges and cards 3–4px. Mintlify's pills and 16px cards reduced to 3–4px.
- Depth: flat; hairlines `color-mix(primary 20%)`; hover lift `0 2px 8px` blue-dark shadow (`--hpv2-shadow-lift`).
- Buttons: paper-on-blue primary (`bg paper/90`, blue text), ghost secondary (`paper/12`), uppercase condensed labels.
- Header: Nous · Hermes · [wordmark] · Community · Portal · Install Hermes, centered wordmark, no bar background. Mintlify cannot center the logo, so we kept the left logo with uppercase condensed nav labels and a square Download CTA.
- Sections: blue hero, then paper sections with huge light condensed headings and `#N` mono eyebrows over feature titles (mirrored in the landing "Key Features" grid).
- Footer: giant outlined "HERMES" watermark and blue mega-footer. Not reproduced (Mintlify footer is config-driven).

## Not matched

- Proprietary fonts (substituted as above).
- Centered logo and the marketing mega-footer.
- Film-grain / scramble / halftone video effects on the marketing site.
- Callouts keep Mintlify's semantic tints (green tip, blue info, yellow warning).
