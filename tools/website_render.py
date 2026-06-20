"""Deterministic website HTML renderer for EasyHermes.

Pure-Python, zero-dependency (stdlib only; the *output* pulls Tailwind from a
CDN). Turns a structured site spec into a complete, responsive single-file
HTML site. No LLM call, no network, no publish — EasyHermes' agent produces the
spec, this renders it deterministically.

Ported from the Kari/Langflow ``lfx.kari_website_render`` module so the website
build capability is native to EasyHermes (Langflow is not involved).

Spec shape::

    {
      "site_title": "...",
      "brand":   {"name": "...", "tagline": "..."},
      "nav":     ["首页", "关于", "联系"],
      "hero":    {"headline": "...", "subhead": "...", "cta": "了解更多"},
      "sections": [
        {"title": "...", "intro": "...", "body": "...",
         "items": [{"title": "...", "desc": "..."}, ...],
         "phone": "...", "email": "...", "address": "..."},
        ...
      ],
      "seo":     {"description": "..."},
      "footer":  "..."
    }
"""

from __future__ import annotations

import html as _html
import json
import re
from typing import Any

STYLE_PALETTES: dict[str, dict[str, str]] = {
    "简约商务": {"bg": "#ffffff", "fg": "#0f172a", "muted": "#475569", "accent": "#2563eb", "soft": "#f1f5f9"},
    "科技感": {"bg": "#0b1020", "fg": "#e5e7eb", "muted": "#94a3b8", "accent": "#22d3ee", "soft": "#111827"},
    "活力多彩": {"bg": "#fffdf7", "fg": "#1f2937", "muted": "#6b7280", "accent": "#f97316", "soft": "#fff7ed"},
    "高端黑金": {"bg": "#0a0a0a", "fg": "#f5f5f4", "muted": "#a8a29e", "accent": "#d4af37", "soft": "#1c1917"},
}


def style_names() -> list[str]:
    return list(STYLE_PALETTES)


def get_palette(style: str | None = None, primary_color: str | None = None) -> dict[str, str]:
    palette = dict(STYLE_PALETTES.get(style or "简约商务", STYLE_PALETTES["简约商务"]))
    color = (primary_color or "").strip()
    if color:
        palette["accent"] = color
    return palette


def extract_spec_json(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from text (tolerates code fences)."""
    value = (text or "").strip()
    value = re.sub(r"^```(?:json)?|```$", "", value, flags=re.MULTILINE).strip()
    start = value.find("{")
    if start < 0:
        msg = "site spec JSON not found"
        raise ValueError(msg)
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(value)):
        char = value[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(value[start : idx + 1])
    msg = "site spec JSON is incomplete"
    raise ValueError(msg)


def render_website_html(
    spec: dict[str, Any],
    images: list[str] | None = None,
    *,
    style: str | None = "简约商务",
    primary_color: str | None = None,
) -> str:
    palette = get_palette(style, primary_color)
    image_urls = [str(url).strip() for url in (images or []) if url and str(url).strip()]
    hero_img = image_urls[0] if image_urls else ""
    card_imgs = image_urls[1:]
    card_img_index = 0

    def esc(value: Any) -> str:
        return _html.escape(str(value or ""))

    def next_img() -> str:
        nonlocal card_img_index
        if card_img_index >= len(card_imgs):
            return ""
        url = card_imgs[card_img_index]
        card_img_index += 1
        return url

    brand = spec.get("brand", {}) or {}
    brand_name_text = brand.get("name") or "公司名称"
    brand_name = esc(brand_name_text)
    tagline = esc(brand.get("tagline") or "")
    hero = spec.get("hero", {}) or {}
    nav = spec.get("nav") or ["首页", "关于", "联系"]
    seo = spec.get("seo", {}) or {}
    accent = palette["accent"]

    nav_html = "".join(
        f'<a href="#sec{idx}" class="px-3 py-2 text-sm transition" style="color:inherit">{esc(label)}</a>'
        for idx, label in enumerate(nav)
    )
    sections_html = _render_sections(spec.get("sections", []) or [], palette, next_img, esc)
    hero_bg = (
        f"background:linear-gradient(rgba(0,0,0,.45),rgba(0,0,0,.45)),url('{esc(hero_img)}') center/cover;"
        if hero_img
        else f"background:{palette['soft']};"
    )
    hero_fg = "#ffffff" if hero_img else palette["fg"]
    font_stack = "ui-sans-serif,system-ui,'PingFang SC','Microsoft YaHei',sans-serif"

    return f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(spec.get("site_title") or brand_name_text)}</title>
<meta name="description" content="{esc(seo.get("description"))}">
<script src="https://cdn.tailwindcss.com"></script>
<style>
html{{scroll-behavior:smooth}}
body{{background:{palette['bg']};color:{palette['fg']};font-family:{font_stack}}}
</style>
</head><body>
<header
  class="sticky top-0 z-20 backdrop-blur"
  style="background:{palette['bg']}cc;border-bottom:1px solid {palette['soft']}"
>
  <div class="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
    <div class="text-lg font-bold" style="color:{accent}">{brand_name}</div>
    <nav class="hidden items-center sm:flex">{nav_html}</nav>
  </div>
</header>
<section class="px-6 py-28 text-center sm:py-36" style="{hero_bg}">
  <div class="mx-auto max-w-4xl" style="color:{hero_fg}">
    <h1 class="text-3xl font-extrabold leading-tight sm:text-5xl">{esc(hero.get("headline") or brand_name_text)}</h1>
    <p class="mt-5 text-lg opacity-90 sm:text-xl">{esc(hero.get("subhead") or tagline)}</p>
    <a
      href="#sec0"
      class="mt-8 inline-block rounded-full px-7 py-3 font-semibold text-white"
      style="background:{accent}"
    >{esc(hero.get("cta") or "了解更多")}</a>
  </div>
</section>
{sections_html}
<footer class="px-6 py-10 text-center text-sm" style="background:{palette['soft']};color:{palette['muted']}">
  {esc(spec.get("footer") or brand_name_text)} · 由 EasyHermes 生成
</footer>
</body></html>"""


def _render_sections(sections: list[dict[str, Any]], palette: dict[str, str], next_img, esc) -> str:
    output: list[str] = []
    for idx, section in enumerate(sections):
        title = esc(section.get("title"))
        inner: list[str] = []
        intro = esc(section.get("intro"))
        body = esc(section.get("body"))
        if intro:
            inner.append(f'<p class="mt-3 text-base" style="color:{palette["muted"]}">{intro}</p>')
        if body:
            inner.append(f'<p class="mt-4 max-w-3xl leading-relaxed" style="color:{palette["muted"]}">{body}</p>')
        items = section.get("items") or []
        if items:
            inner.append(_render_cards(items, palette, next_img, esc))
        contacts = _render_contact(section, palette, esc)
        if contacts:
            inner.append(contacts)
        shade = palette["soft"] if idx % 2 else palette["bg"]
        output.append(
            f'<section id="sec{idx}" class="px-6 py-20" style="background:{shade}">'
            f'<div class="mx-auto max-w-6xl">'
            f'<h2 class="text-2xl font-bold sm:text-3xl">{title}</h2>'
            f'{"".join(inner)}'
            f"</div></section>"
        )
    return "".join(output)


def _render_cards(items: list[dict[str, Any]], palette: dict[str, str], next_img, esc) -> str:
    cards: list[str] = []
    for item in items:
        img = next_img()
        img_html = (
            f'<img src="{esc(img)}" alt="" class="mb-4 h-44 w-full rounded-lg object-cover"/>'
            if img
            else f'<div class="mb-4 h-44 w-full rounded-lg" style="background:{palette["soft"]}"></div>'
        )
        cards.append(
            f'<div class="rounded-xl p-5 shadow-sm" '
            f'style="background:{palette["bg"]};border:1px solid {palette["soft"]}">'
            f"{img_html}"
            f'<h3 class="text-lg font-semibold">{esc(item.get("title"))}</h3>'
            f'<p class="mt-2 text-sm" style="color:{palette["muted"]}">{esc(item.get("desc"))}</p>'
            f"</div>"
        )
    return f'<div class="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">{"".join(cards)}</div>'


def _render_contact(section: dict[str, Any], palette: dict[str, str], esc) -> str:
    bits: list[str] = []
    for label, key in (("电话", "phone"), ("邮箱", "email"), ("地址", "address")):
        if section.get(key):
            bits.append(
                f'<div class="text-sm" style="color:{palette["muted"]}">'
                f'<span class="font-medium" style="color:{palette["fg"]}">{label}:</span> {esc(section.get(key))}'
                f"</div>"
            )
    return f'<div class="mt-6 space-y-2">{"".join(bits)}</div>' if bits else ""
