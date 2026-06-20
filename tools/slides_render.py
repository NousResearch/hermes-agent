"""Deterministic slide-deck (PPT) HTML renderer for EasyHermes.

Pure-Python, zero-dependency. Turns a slide outline spec into a self-contained
HTML slide deck — keyboard nav (←/→/space), scroll-snap, on-screen controls,
print-to-PDF friendly. No external binaries (no LibreOffice / python-pptx), no
network, no LLM call: EasyHermes' agent writes the outline, this renders it.

Ported from the Kari/Langflow ``kari_slides.py`` renderer (the ``_render``
half — the Copilot/LLM outline step is dropped because EasyHermes is itself
the model that produces the outline).

Spec shape::

    {
      "title": "封面主标题",
      "subtitle": "副标题 / 一句话",
      "slides": [
        {"title": "页标题", "bullets": ["要点1", "要点2", "要点3"], "note": "可选备注"},
        ...
      ]
    }
"""

from __future__ import annotations

import html as _html
from typing import Any

# Re-export the tolerant JSON extractor so callers have one import surface.
from tools.website_render import extract_spec_json  # noqa: F401

# Friendly style name -> inline theme id (matches the deck CSS below).
_THEMES = {"简约": "white", "商务": "league", "科技": "moon", "深色": "black", "优雅": "serif"}


def theme_names() -> list[str]:
    return list(_THEMES)


def resolve_theme(style: str | None) -> str:
    return _THEMES.get(style or "商务", "league")


def render_slides_html(spec: dict[str, Any], images: list[str] | None = None, *, style: str | None = "商务") -> str:
    theme = resolve_theme(style)
    e = _html.escape
    safe_theme = e(theme or "league")
    imgs = [str(u).strip() for u in (images or []) if u and str(u).strip()]
    ci = 0

    def next_img():
        nonlocal ci
        if ci < len(imgs):
            u = imgs[ci]
            ci += 1
            return u
        return ""

    title = e(str(spec.get("title") or "演示"))
    subtitle = e(str(spec.get("subtitle") or ""))
    sections = [
        '<section class="kari-slide kari-slide-cover active" data-index="0" aria-hidden="false">'
        '<div class="kari-slide-inner">'
        f"<h1>{title}</h1>"
        f"<p>{subtitle}</p>"
        "</div></section>"
    ]
    for index, s in enumerate(spec.get("slides", []) or [], start=1):
        slide_title = e(str(s.get("title") or f"第 {index} 页"))
        bullets = "".join(f"<li>{e(str(b))}</li>" for b in (s.get("bullets") or []) if b is not None)
        img = next_img()
        if img:
            body = (
                '<div class="kari-slide-grid">'
                f'<ul class="kari-bullets">{bullets}</ul>'
                f'<img class="kari-slide-image" src="{e(str(img))}" alt="{slide_title}"/>'
                "</div>"
            )
        else:
            body = f'<ul class="kari-bullets">{bullets}</ul>' if bullets else '<p class="kari-empty">本页暂无要点</p>'
        note_text = str(s.get("note") or "").strip()
        note = f'<aside class="notes">{e(note_text)}</aside>' if note_text else ""
        sections.append(
            f'<section class="kari-slide" data-index="{index}" aria-hidden="false">'
            '<div class="kari-slide-inner">'
            f'<p class="kari-kicker">{index:02d}</p>'
            f"<h2>{slide_title}</h2>"
            f"{body}{note}</div></section>"
        )

    slide_count = len(sections)
    return f"""<!doctype html><html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --deck-bg: #f8fafc;
  --slide-bg: #ffffff;
  --slide-fg: #111827;
  --muted-fg: #64748b;
  --accent: #2563eb;
  --border: rgba(15, 23, 42, 0.12);
  --shadow: 0 24px 80px rgba(15, 23, 42, 0.14);
}}
* {{ box-sizing: border-box; }}
html, body {{ width: 100%; min-height: 100%; }}
body {{
  margin: 0;
  overflow: hidden;
  background: var(--deck-bg);
  color: var(--slide-fg);
  font-family: Inter, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif;
}}
body[data-theme="league"] {{
  --deck-bg: #101827;
  --slide-bg: #162033;
  --slide-fg: #f8fafc;
  --muted-fg: #cbd5e1;
  --accent: #38bdf8;
  --border: rgba(226, 232, 240, 0.18);
  --shadow: 0 28px 90px rgba(0, 0, 0, 0.28);
}}
body[data-theme="moon"] {{
  --deck-bg: #111827;
  --slide-bg: #172554;
  --slide-fg: #eff6ff;
  --muted-fg: #bfdbfe;
  --accent: #60a5fa;
  --border: rgba(191, 219, 254, 0.2);
  --shadow: 0 28px 90px rgba(15, 23, 42, 0.34);
}}
body[data-theme="black"] {{
  --deck-bg: #020617;
  --slide-bg: #0f172a;
  --slide-fg: #f8fafc;
  --muted-fg: #94a3b8;
  --accent: #22c55e;
  --border: rgba(148, 163, 184, 0.22);
  --shadow: 0 28px 90px rgba(0, 0, 0, 0.4);
}}
body[data-theme="serif"] {{
  --deck-bg: #f6f2ea;
  --slide-bg: #fffaf0;
  --slide-fg: #1f2937;
  --muted-fg: #6b7280;
  --accent: #b45309;
  --border: rgba(120, 113, 108, 0.2);
}}
.kari-deck {{
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow-y: auto;
  background: var(--deck-bg);
  scroll-behavior: smooth;
  scroll-snap-type: y mandatory;
}}
.kari-slide {{
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  width: 100vw;
  padding: clamp(32px, 6vw, 84px);
  scroll-snap-align: start;
  scroll-snap-stop: always;
}}
.kari-slide-inner {{
  width: min(1120px, 100%);
  min-height: min(680px, calc(100vh - 168px));
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: clamp(18px, 3vw, 34px);
  padding: clamp(32px, 5vw, 72px);
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--slide-bg);
  box-shadow: var(--shadow);
}}
.kari-slide-cover .kari-slide-inner {{
  background: #0b1020;
  color: #f8fafc;
}}
.kari-slide h1, .kari-slide h2, .kari-slide p, .kari-slide ul {{
  margin: 0;
}}
.kari-slide h1 {{
  max-width: 980px;
  font-size: clamp(42px, 6.5vw, 82px);
  line-height: 1.08;
  font-weight: 750;
}}
.kari-slide h2 {{
  font-size: clamp(34px, 4.7vw, 62px);
  line-height: 1.12;
  font-weight: 720;
}}
.kari-slide-cover p {{
  max-width: 860px;
  color: rgba(248, 250, 252, 0.78);
  font-size: clamp(20px, 2.4vw, 31px);
  line-height: 1.5;
}}
.kari-kicker {{
  color: var(--accent);
  font-size: 14px;
  font-weight: 800;
  letter-spacing: 0;
}}
.kari-bullets {{
  display: grid;
  gap: 16px;
  padding-left: 1.25em;
  color: var(--slide-fg);
  font-size: clamp(21px, 2.4vw, 32px);
  line-height: 1.52;
}}
.kari-bullets li::marker {{
  color: var(--accent);
}}
.kari-slide-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 0.92fr);
  gap: clamp(28px, 4vw, 52px);
  align-items: center;
}}
.kari-slide-image {{
  width: 100%;
  max-height: 54vh;
  object-fit: cover;
  border-radius: 14px;
  border: 1px solid var(--border);
}}
.kari-empty {{
  color: var(--muted-fg);
  font-size: 22px;
}}
aside.notes {{
  display: none !important;
}}
.kari-controls {{
  position: fixed;
  right: clamp(16px, 3vw, 36px);
  bottom: clamp(16px, 3vw, 32px);
  z-index: 20;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: color-mix(in srgb, var(--slide-bg) 88%, transparent);
  box-shadow: 0 12px 36px rgba(15, 23, 42, 0.18);
}}
.kari-control-button {{
  width: 40px;
  height: 40px;
  border: 0;
  border-radius: 999px;
  background: var(--accent);
  color: #fff;
  font-size: 22px;
  line-height: 1;
  cursor: pointer;
}}
.kari-control-button:disabled {{
  cursor: default;
  opacity: 0.38;
}}
.kari-counter {{
  min-width: 72px;
  color: var(--muted-fg);
  text-align: center;
  font-size: 14px;
  font-weight: 700;
}}
@media (max-width: 760px) {{
  .kari-slide {{
    min-height: 100svh;
    padding: 18px;
  }}
  .kari-slide-inner {{
    min-height: calc(100svh - 96px);
    padding: 28px;
    border-radius: 14px;
  }}
  .kari-slide-grid {{
    grid-template-columns: 1fr;
  }}
  .kari-slide-image {{
    max-height: 32vh;
  }}
  .kari-controls {{
    right: 12px;
    bottom: 12px;
  }}
}}
@media print {{
  body {{ overflow: visible; background: #fff; }}
  .kari-slide, .kari-slide.active {{
    display: flex;
    page-break-after: always;
  }}
  .kari-controls {{ display: none; }}
}}
</style>
</head><body data-theme="{safe_theme}">
<main class="kari-deck" data-slide-count="{slide_count}">
{"".join(sections)}
</main>
<nav class="kari-controls" aria-label="幻灯片控制">
  <button type="button" class="kari-control-button" aria-label="上一页" data-prev>&lt;</button>
  <span class="kari-counter"><span data-current>1</span>/<span data-total>{slide_count}</span></span>
  <button type="button" class="kari-control-button" aria-label="下一页" data-next>&gt;</button>
</nav>
<script>
(() => {{
  const deck = document.querySelector(".kari-deck");
  const slides = Array.from(document.querySelectorAll(".kari-slide"));
  const prev = document.querySelector("[data-prev]");
  const next = document.querySelector("[data-next]");
  const current = document.querySelector("[data-current]");
  const total = document.querySelector("[data-total]");
  let index = 0;
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  const sync = () => {{
    slides.forEach((slide, slideIndex) => {{
      const isActive = slideIndex === index;
      slide.classList.toggle("active", isActive);
      slide.setAttribute("aria-current", isActive ? "true" : "false");
    }});
    if (current) current.textContent = String(index + 1);
    if (total) total.textContent = String(slides.length);
    if (prev) prev.disabled = index === 0;
    if (next) next.disabled = index === slides.length - 1;
  }};
  const show = (nextIndex, shouldScroll = true) => {{
    if (!slides.length) return;
    index = Math.max(0, Math.min(slides.length - 1, nextIndex));
    sync();
    if (shouldScroll) {{
      slides[index].scrollIntoView({{ behavior: reduceMotion ? "auto" : "smooth", block: "start" }});
    }}
  }};
  let scrollFrame = 0;
  const syncFromScroll = () => {{
    scrollFrame = 0;
    const deckTop = deck?.getBoundingClientRect().top ?? 0;
    let nearestIndex = 0;
    let nearestDistance = Number.POSITIVE_INFINITY;
    slides.forEach((slide, slideIndex) => {{
      const distance = Math.abs(slide.getBoundingClientRect().top - deckTop);
      if (distance < nearestDistance) {{
        nearestDistance = distance;
        nearestIndex = slideIndex;
      }}
    }});
    if (nearestIndex !== index) {{
      index = nearestIndex;
      sync();
    }}
  }};
  deck?.addEventListener("scroll", () => {{
    if (!scrollFrame) scrollFrame = window.requestAnimationFrame(syncFromScroll);
  }});
  prev?.addEventListener("click", () => show(index - 1));
  next?.addEventListener("click", () => show(index + 1));
  document.addEventListener("keydown", (event) => {{
    if (event.key === "ArrowRight" || event.key === "PageDown" || event.key === " ") {{
      event.preventDefault();
      show(index + 1);
    }}
    if (event.key === "ArrowLeft" || event.key === "PageUp") {{
      event.preventDefault();
      show(index - 1);
    }}
  }});
  show(0, false);
}})();
</script>
</body></html>"""
