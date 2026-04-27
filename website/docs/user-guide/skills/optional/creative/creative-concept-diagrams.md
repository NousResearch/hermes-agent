---
title: "Concept Diagrams"
sidebar_label: "Concept Diagrams"
description: "Generate flat, minimal light/dark-aware SVG diagrams as standalone HTML files, using a unified educational visual language with 9 semantic color ramps, sente..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Concept Diagrams

Generate flat, minimal light/dark-aware SVG diagrams as standalone HTML files, using a unified educational visual language with 9 semantic color ramps, sentence-case typography, and automatic dark mode. Best suited for educational and non-software visuals — physics setups, chemistry mechanisms, math curves, physical objects (aircraft, turbines, smartphones, mechanical watches), anatomy, floor plans, cross-sections, narrative journeys (lifecycle of X, process of Y), hub-spoke system integrations (smart city, IoT), and exploded layer views. If a more specialized skill exists for the subject (dedicated software/cloud architecture, hand-drawn sketches, animated explainers, etc.), prefer that — otherwise this skill can also serve as a general-purpose SVG diagram fallback with a clean educational look. Ships with 15 example diagrams.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/concept-diagrams` |
| Path | `optional-skills/creative/concept-diagrams` |
| Version | `0.1.0` |
| Author | v1k22 (original PR), ported into hermes-agent |
| License | MIT |
| Tags | `diagrams`, `svg`, `visualization`, `education`, `physics`, `chemistry`, `engineering` |
| Related skills | [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw), `generative-widgets` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Concept Diagrams

Generate production-quality SVG diagrams with a unified flat, minimal design system. Output is a single self-contained HTML file that renders identically in any modern browser, with automatic light/dark mode.

## Scope

**Best suited for:**
- Physics setups, chemistry mechanisms, math curves, biology
- Physical objects (aircraft, turbines, smartphones, mechanical watches, cells)
- Anatomy, cross-sections, exploded layer views
- Floor plans, architectural conversions
- Narrative journeys (lifecycle of X, process of Y)
- Hub-spoke system integrations (smart city, IoT networks, electricity grids)
- Educational / textbook-style visuals in any domain
- Quantitative charts (grouped bars, energy profiles)

**Look elsewhere first for:**
- Dedicated software / cloud infrastructure architecture with a dark tech aesthetic (consider `architecture-diagram` if available)
- Hand-drawn whiteboard sketches (consider `excalidraw` if available)
- Animated explainers or video output (consider an animation skill)

If a more specialized skill is available for the subject, prefer that. If none fits, this skill can serve as a general-purpose SVG diagram fallback — the output will carry the clean educational aesthetic described below, which is a reasonable default for almost any subject.

## Workflow

1. Decide on the diagram type (see Diagram Types below).
2. Lay out components using the Design System rules.
3. Write the full HTML page using `templates/template.html` as the wrapper — paste your SVG where the template says `<!-- PASTE SVG HERE -->`.
4. Save as a standalone `.html` file (for example `~/my-diagram.html` or `./my-diagram.html`).
5. User opens it directly in a browser — no server, no dependencies.

Optional: if the user wants a browsable gallery of multiple diagrams, see "Local Preview Server" at the bottom.

Load the HTML template:
```
skill_view(name="concept-diagrams", file_path="templates/template.html")
```

The template embeds the full CSS design system (`c-*` color classes, text classes, light/dark variables, arrow marker styles). The SVG you generate relies on these classes being present on the hosting page.

---

## Design System

### Philosophy

- **Flat**: no gradients, drop shadows, blur, glow, or neon effects.
- **Minimal**: show the essential. No decorative icons inside boxes.
- **Consistent**: same colors, spacing, typography, and stroke widths across every diagram.
- **Dark-mode ready**: all colors auto-adapt via CSS classes — no per-mode SVG.

### Color Palette

9 color ramps, each with 7 stops. Put the class name on a `<g>` or shape element; the template CSS handles both modes.

| Class      | 50 (lightest) | 100     | 200     | 400     | 600     | 800     | 900 (darkest) |
|------------|---------------|---------|---------|---------|---------|---------|---------------|
| `c-purple` | #EEEDFE | #CECBF6 | #AFA9EC | #7F77DD | #534AB7 | #3C3489 | #26215C |
| `c-teal`   | #E1F5EE | #9FE1CB | #5DCAA5 | #1D9E75 | #0F6E56 | #085041 | #04342C |
| `c-coral`  | #FAECE7 | #F5C4B3 | #F0997B | #D85A30 | #993C1D | #712B13 | #4A1B0C |
| `c-pink`   | #FBEAF0 | #F4C0D1 | #ED93B1 | #D4537E | #993556 | #72243E | #4B1528 |
| `c-gray`   | #F1EFE8 | #D3D1C7 | #B4B2A9 | #888780 | #5F5E5A | #444441 | #2C2C2A |
| `c-blue`   | #E6F1FB | #B5D4F4 | #85B7EB | #378ADD | #185FA5 | #0C447C | #042C53 |
| `c-green`  | #EAF3DE | #C0DD97 | #97C459 | #639922 | #3B6D11 | #27500A | #173404 |
| `c-amber`  | #FAEEDA | #FAC775 | #EF9F27 | #BA7517 | #854F0B | #633806 | #412402 |
| `c-red`    | #FCEBEB | #F7C1C1 | #F09595 | #E24B4A | #A32D2D | #791F1F | #501313 |

#### Color Assignment Rules

Color encodes **meaning**, not sequence. Never cycle through colors like a rainbow.

- Group nodes by **category** — all nodes of the same type share one color.
- Use `c-gray` for neutral/structural nodes (start, end, generic steps, users).
- Use **2-3 colors per diagram**, not 6+.
- Prefer `c-purple`, `c-teal`, `c-coral`, `c-pink` for general categories.
- Reserve `c-blue`, `c-green`, `c-amber`, `c-red` for semantic meaning (info, success, warning, error).

Light/dark stop mapping (handled by the template CSS — just use the class):
- Light mode: 50 fill + 600 stroke + 800 title / 600 subtitle
- Dark mode:  800 fill + 200 stroke + 100 title / 200 subtitle

### Typography

Only two font sizes. No exceptions.

| Class | Size | Weight | Use |
|-------|------|--------|-----|
| `th`  | 14px | 500    | Node titles, region labels |
| `ts`  | 12px | 400    | Subtitles, descriptions, arrow labels |
| `t`   | 14px | 400    | General text |

- **Sentence case always.** Never Title Case, never ALL CAPS.
- Every `<text>` MUST carry a class (`t`, `ts`, or `th`). No unclassed text.
- `dominant-baseline="central"` on all text inside boxes.
- `text-anchor="middle"` for centered text in boxes.

**Width estimation (approx):**
- 14px weight 500: ~8px per character
- 12px weight 400: ~6.5px per character
- Always verify: `box_width >= (char_count × px_per_char) + 48` (24px padding each side)

### Spacing & Layout

- **ViewBox**: `viewBox="0 0 680 H"` where H = content height + 40px buffer.
- **Safe area**: x=40 to x=640, y=40 to y=(H-40).
- **Between boxes**: 60px minimum gap.
- **Inside boxes**: 24px horizontal padding, 12px vertical padding.
- **Arrowhead gap**: 10px between arrowhead and box edge.
- **Single-line box**: 44px height.
- **Two-line box**: 56px height, 18px between title and subtitle baselines.
- **Container padding**: 20px minimum inside every container.
- **Max nesting**: 2-3 levels deep. Deeper gets unreadable at 680px width.

### Stroke & Shape

- **Stroke width**: 0.5px on all node borders. Not 1px, not 2px.
- **Rect rounding**: `rx="8"` for nodes, `rx="12"` for inner containers, `rx="16"` to `rx="20"` for outer containers.
- **Connector paths**: MUST have `fill="none"`. SVG defaults to `fill: black` otherwise.

### Arrow Marker

Include this `<defs>` block at the start of **every** SVG:

```xml
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
```

Use `marker-end="url(#arrow)"` on lines. The arrowhead inherits the line color via `context-stroke`.

### CSS Classes (Provided by the Template)

The template page provides:

- Text: `.t`, `.ts`, `.th`
- Neutral: `.box`, `.arr`, `.leader`, `.node`
- Color ramps: `.c-purple`, `.c-teal`, `.c-coral`, `.c-pink`, `.c-gray`, `.c-blue`, `.c-green`, `.c-amber`, `.c-red` (all with automatic light/dark mode)

You do **not** need to redefine these — just apply them in your SVG. The template file contains the full CSS definitions.

---

## SVG Boilerplate

Every SVG inside the template page starts with this exact structure:

```xml
<svg width="100%" viewBox="0 0 680 {HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Diagram content here -->

</svg>
```

Replace `{HEIGHT}` with the actual computed height (last element bottom + 40px).

### Node Patterns

**Single-line node (44px):**
```xml
<g class="node c-blue">
  <rect x="100" y="20" width="180" height="44" rx="8" stroke-width="0.5"/>
  <text class="th" x="190" y="42" text-anchor="middle" dominant-baseline="central">Service name</text>
</g>
```

**Two-line node (56px):**
```xml
<g class="node c-teal">
  <rect x="100" y="20" width="200" height="56" rx="8" stroke-width="0.5"/>
  <text class="th" x="200" y="38" text-anchor="middle" dominant-baseline="central">Service name</text>
  <text class="ts" x="200" y="56" text-anchor="middle" dominant-baseline="central">Short description</text>
</g>
```

**Connector (no label):**
```xml
<line x1="200" y1="76" x2="200" y2="120" class="arr" marker-end="url(#arrow)"/>
```

**Container (dashed or solid):**
```xml
<g class="c-purple">
  <rect x="40" y="92" width="600" height="300" rx="16" stroke-width="0.5"/>
  <text class="th" x="66" y="116">Container label</text>
  <text class="ts" x="66" y="134">Subtitle info</text>
</g>
```

---

## Diagram Types

Choose the layout that fits the subject:

1. **Flowchart** — CI/CD pipelines, request lifecycles, approval workflows, data processing. Single-direction flow (top-down or left-right). Max 4-5 nodes per row.
2. **Structural / Containment** — Cloud infrastructure nesting, system architecture with layers. Large outer containers with inner regions. Dashed rects for logical groupings.
3. **API / Endpoint Map** — REST routes, GraphQL schemas. Tree from root, branching to resource groups, each containing endpoint nodes.
4. **Microservice Topology** — Service mesh, event-driven systems. Services as nodes, arrows for communication patterns, message queues between.
5. **Data Flow** — ETL pipelines, streaming architectures. Left-to-right flow from sources through processing to sinks.
6. **Physical / Structural** — Vehicles, buildings, hardware, anatomy. Use shapes that match the physical form — `<path>` for curved bodies, `<polygon>` for tapered shapes, `<ellipse>`/`<circle>` for cylindrical parts, nested `<rect>` for compartments. See `references/physical-shape-cookbook.md`.
7. **Infrastructure / Systems Integration** — Smart cities, IoT networks, multi-domain systems. Hub-spoke layout with central platform connecting subsystems. Semantic line styles (`.data-line`, `.power-line`, `.water-pipe`, `.road`). See `references/infrastructure-patterns.md`.
8. **UI / Dashboard Mockups** — Admin panels, monitoring dashboards. Screen frame with nested chart/gauge/indicator elements. See `references/dashboard-patterns.md`.

For physical, infrastructure, and dashboard diagrams, load the matching reference file before generating — each one provides ready-made CSS classes and shape primitives.

---

## Validation Checklist

Before finalizing any SVG, verify ALL of the following:

1. Every `<text>` has class `t`, `ts`, or `th`.
2. Every `<text>` inside a box has `dominant-baseline="central"`.
3. Every connector `<path>` or `<line>` used as arrow has `fill="none"`.
4. No arrow line crosses through an unrelated box.
5. `box_width >= (longest_label_chars × 8) + 48` for 14px text.
6. `box_width >= (longest_label_chars × 6.5) + 48` for 12px text.
7. ViewBox height = bottom-most element + 40px.
8. All content stays within x=40 to x=640.
9. Color classes (`c-*`) are on `<g>` or shape elements, never on `<path>` connectors.
10. Arrow `<defs>` block is present.
11. No gradients, shadows, blur, or glow effects.
12. Stroke width is 0.5px on all node borders.

---

## Output & Preview

### Default: standalone HTML file

Write a single `.html` file the user can open directly. No server, no dependencies, works offline. Pattern:

```python
# 1. Load the template
template = skill_view("concept-diagrams", "templates/template.html")

# 2. Fill in title, subtitle, and paste your SVG
html = template.replace(
    "<!-- DIAGRAM TITLE HERE -->", "SN2 reaction mechanism"
).replace(
    "<!-- OPTIONAL SUBTITLE HERE -->", "Bimolecular nucleophilic substitution"
).replace(
    "<!-- PASTE SVG HERE -->", svg_content
)

# 3. Write to a user-chosen path (or ./ by default)
write_file("./sn2-mechanism.html", html)
```

Tell the user how to open it:

```
# macOS
open ./sn2-mechanism.html
# Linux
xdg-open ./sn2-mechanism.html
```

### Optional: local preview server (multi-diagram gallery)

Only use this when the user explicitly wants a browsable gallery of multiple diagrams.

**Rules:**
- Bind to `127.0.0.1` only. Never `0.0.0.0`. Exposing diagrams on all network interfaces is a security hazard on shared networks.
- Pick a free port (do NOT hard-code one) and tell the user the chosen URL.
- The server is optional and opt-in — prefer the standalone HTML file first.

Recommended pattern (lets the OS pick a free ephemeral port):

```bash
# Put each diagram in its own folder under .diagrams/
mkdir -p .diagrams/sn2-mechanism
# ...write .diagrams/sn2-mechanism/index.html...

# Serve on loopback only, free port
cd .diagrams && python3 -c "
import http.server, socketserver
with socketserver.TCPServer(('127.0.0.1', 0), http.server.SimpleHTTPRequestHandler) as s:
    print(f'Serving at http://127.0.0.1:{s.server_address[1]}/')
    s.serve_forever()
" &
```

If the user insists on a fixed port, use `127.0.0.1:<port>` — still never `0.0.0.0`. Document how to stop the server (`kill %1` or `pkill -f "http.server"`).

---

## Examples Reference

The `examples/` directory ships 15 complete, tested diagrams. Browse them for working patterns before writing a new diagram of a similar type:

| File | Type | Demonstrates |
|------|------|--------------|
| `hospital-emergency-department-flow.md` | Flowchart | Priority routing with semantic colors |
| `feature-film-production-pipeline.md` | Flowchart | Phased workflow, horizontal sub-flows |
| `automated-password-reset-flow.md` | Flowchart | Auth flow with error branches |
| `autonomous-llm-research-agent-flow.md` | Flowchart | Loop-back arrows, decision branches |
| `place-order-uml-sequence.md` | Sequence | UML sequence diagram style |
| `commercial-aircraft-structure.md` | Physical | Paths, polygons, ellipses for realistic shapes |
| `wind-turbine-structure.md` | Physical cross-section | Underground/above-ground separation, color coding |
| `smartphone-layer-anatomy.md` | Exploded view | Alternating left/right labels, layered components |
| `apartment-floor-plan-conversion.md` | Floor plan | Walls, doors, proposed changes in dotted red |
| `banana-journey-tree-to-smoothie.md` | Narrative journey | Winding path, progressive state changes |
| `cpu-ooo-microarchitecture.md` | Hardware pipeline | Fan-out, memory hierarchy sidebar |
| `sn2-reaction-mechanism.md` | Chemistry | Molecules, curved arrows, energy profile |
| `smart-city-infrastructure.md` | Hub-spoke | Semantic line styles per system |
| `electricity-grid-flow.md` | Multi-stage flow | Voltage hierarchy, flow markers |
| `ml-benchmark-grouped-bar-chart.md` | Chart | Grouped bars, dual axis |

Load any example with:
```
skill_view(name="concept-diagrams", file_path="examples/<filename>")
```

---

## Quick Reference: What to Use When

| User says | Diagram type | Suggested colors |
|-----------|--------------|------------------|
| "show the pipeline" | Flowchart | gray start/end, purple steps, red errors, teal deploy |
| "draw the data flow" | Data pipeline (left-right) | gray sources, purple processing, teal sinks |
| "visualize the system" | Structural (containment) | purple container, teal services, coral data |
| "map the endpoints" | API tree | purple root, one ramp per resource group |
| "show the services" | Microservice topology | gray ingress, teal services, purple bus, coral workers |
| "draw the aircraft/vehicle" | Physical | paths, polygons, ellipses for realistic shapes |
| "smart city / IoT" | Hub-spoke integration | semantic line styles per subsystem |
| "show the dashboard" | UI mockup | dark screen, chart colors: teal, purple, coral for alerts |
| "power grid / electricity" | Multi-stage flow | voltage hierarchy (HV/MV/LV line weights) |
| "wind turbine / turbine" | Physical cross-section | foundation + tower cutaway + nacelle color-coded |
| "journey of X / lifecycle" | Narrative journey | winding path, progressive state changes |
| "layers of X / exploded" | Exploded layer view | vertical stack, alternating labels |
| "CPU / pipeline" | Hardware pipeline | vertical stages, fan-out to execution ports |
| "floor plan / apartment" | Floor plan | walls, doors, proposed changes in dotted red |
| "reaction mechanism" | Chemistry | atoms, bonds, curved arrows, transition state, energy profile |
