"""Deterministic "web video" HTML renderer for EasyHermes.

Pure-Python, zero-dependency. Turns a storyboard spec into a single
self-contained HTML page that plays like a video in the browser: full-screen,
auto-advancing scenes with fade/zoom transitions, a progress bar, and a
play/pause button. Not an exported mp4 — a lightweight animated web page (no
render service, no binaries).

Ported from the Kari/Langflow ``kari_webvideo.py`` renderer (the ``_render``
half — the Copilot/LLM storyboard step is dropped because EasyHermes is itself
the model that writes the storyboard).

Spec shape::

    {
      "title": "片名 / 主标题",
      "scenes": [
        {"headline": "这一镜的大字(短、有力)", "sub": "下面的小字(可空)"},
        ...
      ]
    }
"""

from __future__ import annotations

import html as _html
from typing import Any

# Re-export the tolerant JSON extractor so callers have one import surface.
from tools.website_render import extract_spec_json  # noqa: F401

_PALETTES = {
    "科技蓝": {"bg": "#070b1a", "fg": "#eaf2ff", "accent": "#38bdf8"},
    "活力橙": {"bg": "#1a0f07", "fg": "#fff7ed", "accent": "#fb923c"},
    "高端黑金": {"bg": "#0a0a0a", "fg": "#f5f5f4", "accent": "#d4af37"},
    "清新绿": {"bg": "#07140d", "fg": "#ecfdf5", "accent": "#34d399"},
}


def palette_names() -> list[str]:
    return list(_PALETTES)


def get_palette(style: str | None) -> dict[str, str]:
    return _PALETTES.get(style or "科技蓝", _PALETTES["科技蓝"])


def render_webvideo_html(
    spec: dict[str, Any],
    images: list[str] | None = None,
    *,
    style: str | None = "科技蓝",
    seconds: int = 4,
) -> str:
    pal = get_palette(style)
    e = _html.escape
    imgs = [str(u).strip() for u in (images or []) if u and str(u).strip()]
    scenes = spec.get("scenes") or [{"headline": spec.get("title") or "EasyHermes", "sub": ""}]
    try:
        secs = max(2, int(seconds or 4))
    except (TypeError, ValueError):
        secs = 4

    scene_html = []
    for i, sc in enumerate(scenes):
        img = imgs[i % len(imgs)] if imgs else ""
        bg = (
            f'background:linear-gradient(rgba(0,0,0,.45),rgba(0,0,0,.55)),url("{e(img)}") center/cover;'
            if img
            else f"background:radial-gradient(circle at 50% 40%, {pal['accent']}22, {pal['bg']} 70%);"
        )
        scene_html.append(
            f'<div class="scene" style="{bg}">'
            f'<div class="kbg"></div>'
            f'<div class="txt"><div class="hl">{e(sc.get("headline"))}</div>'
            f'<div class="sub">{e(sc.get("sub"))}</div></div></div>'
        )

    return f"""<!doctype html><html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(spec.get("title") or "EasyHermes 网页视频")}</title>
<style>
*{{margin:0;box-sizing:border-box}}
html,body{{height:100%;background:{pal['bg']};color:{pal['fg']};font-family:ui-sans-serif,system-ui,'PingFang SC','Microsoft YaHei',sans-serif;overflow:hidden}}
#stage{{position:fixed;inset:0}}
.scene{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity .8s ease;pointer-events:none}}
.scene.on{{opacity:1}}
.scene .txt{{text-align:center;padding:0 8%;transform:translateY(24px);opacity:0;transition:all .9s cubic-bezier(.2,.7,.2,1)}}
.scene.on .txt{{transform:translateY(0);opacity:1}}
.hl{{font-size:clamp(28px,6vw,72px);font-weight:800;line-height:1.1;text-shadow:0 2px 20px rgba(0,0,0,.4)}}
.sub{{margin-top:18px;font-size:clamp(15px,2.4vw,26px);opacity:.85}}
.scene.on{{animation:kb 6s ease forwards}}
@keyframes kb{{from{{background-size:106% auto}}to{{background-size:116% auto}}}}
#bar{{position:fixed;left:0;top:0;height:3px;background:{pal['accent']};width:0;transition:width .2s linear;z-index:5}}
#play{{position:fixed;right:16px;bottom:16px;z-index:5;background:{pal['accent']};color:#000;border:0;border-radius:999px;padding:8px 16px;font-weight:600;cursor:pointer;opacity:.85}}
</style></head><body>
<div id="stage">{"".join(scene_html)}</div>
<div id="bar"></div>
<button id="play">⏸ 暂停</button>
<script>
const D={secs * 1000}, scenes=[...document.querySelectorAll('.scene')], bar=document.getElementById('bar'), btn=document.getElementById('play');
let i=-1, t0=0, raf, playing=true;
function show(n){{scenes.forEach((s,k)=>s.classList.toggle('on',k===n));}}
function step(){{i=(i+1)%scenes.length;show(i);t0=performance.now();}}
function tick(now){{ if(playing){{ const p=Math.min(1,(now-t0)/D); bar.style.width=(p*100)+'%'; if(p>=1) step(); }} raf=requestAnimationFrame(tick);}}
step(); raf=requestAnimationFrame(tick);
btn.onclick=()=>{{playing=!playing; btn.textContent=playing?'⏸ 暂停':'▶ 播放'; if(playing) t0=performance.now();}};
</script></body></html>"""
