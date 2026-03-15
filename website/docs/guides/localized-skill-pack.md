# Tutorial: Build a Localized Skill Pack

In this tutorial, you'll build a **localized skill pack** for Hermes Agent — a collection of skills tailored to a specific language, region, or market. We'll use a Turkish locale pack as the working example, but every pattern here applies to any locale.

By the end, you'll have a reusable skill pack that delivers local news, regional market data, styled briefing cards, and automated Telegram delivery — all without a single external API key.

## What We're Building

A locale skill pack bundles four things together:

1. **A regional news skill** — pulls headlines from local sources
2. **A market data skill** — fetches prices in the local currency
3. **A briefing card skill** — generates a styled PNG card
4. **A delivery skill** — sends everything via Telegram on a schedule

The result is a fully automated daily briefing in the user's language, using sources they actually read.

## Why Build a Locale Pack?

Most Hermes skills assume English sources and USD pricing. A locale pack fixes this for a specific market:

- Turkish users read Hürriyet and Bloomberg HT, not TechCrunch
- Brazilian users want BRL prices, not USD
- Japanese users expect NHK and Nikkei, not Reuters

A locale pack is also a great first contribution to the [Skills Hub](https://agentskills.io) — it's high value, easy to test, and immediately useful to a whole community.

## Prerequisites

- Hermes Agent installed — see the [Installation guide](../getting-started/installation)
- Basic familiarity with skills — see the [Skills System](../user-guide/features/skills)
- Telegram configured (optional) — see [Messaging Gateway](../user-guide/messaging/)

## Step 1: Plan Your Locale Pack

Before writing any skill files, map out what your locale needs. For the Turkish pack:

| Need | Source | Notes |
|------|--------|-------|
| News | Hürriyet, Bloomberg HT, NTV | RSS feeds, no API key needed |
| Market data | CoinGecko API | Free tier, no API key needed |
| Local index | BIST100 | Yahoo Finance compatible |
| Delivery | Telegram | Via Hermes gateway |
| Card format | 1200×630px PNG | Pillow, no external service |

Write this down before coding. It prevents scope creep and makes your PR description much easier to write.

## Step 2: Create the Skill Pack Structure

Locale packs live under a named folder in your Hermes skills directory:

```
~/.hermes/skills/
└── turkish-locale/
    ├── SKILL.md               # Main skill entry point
    ├── turkish-news/
    │   └── SKILL.md           # News aggregation skill
    ├── bist100/
    │   └── SKILL.md           # Local market index skill
    ├── turkish-daily-briefing/
    │   └── SKILL.md           # Orchestrator skill
    └── scripts/
        ├── fetch_prices.py    # Market data fetcher
        ├── turkish_brief_card.py  # PNG card generator
        └── telegram_send.py   # Delivery helper
```

The top-level `SKILL.md` is the entry point. It describes the pack and delegates to sub-skills.

## Step 3: Write the News Skill

Create `~/.hermes/skills/turkish-locale/turkish-news/SKILL.md`:

```markdown
# Turkish News Aggregator

Fetches the latest headlines from Turkish news sources.

## Sources
- Hürriyet: https://www.hurriyet.com.tr/rss/anasayfa
- Bloomberg HT: https://www.bloomberght.com/rss
- NTV: https://www.ntv.com.tr/son-dakika.rss

## Usage
Invoke this skill to get the top 5 headlines from each source.
Return results in Turkish with source attribution.

## Output Format
For each headline: title, source name, publication time, and URL.
Group by source. Mark breaking news with 🔴.
```

The agent reads this and knows exactly how to fetch and format Turkish news — no code required for the skill definition itself.

## Step 4: Write the Market Data Skill

For zero-API-key market data, use CoinGecko's public endpoint and Yahoo Finance:

```markdown
# BIST100 & Crypto Prices (TRY)

Fetches current market data in Turkish Lira (TRY).

## Data Sources
- CoinGecko public API: https://api.coingecko.com/api/v3/simple/price
  Params: ids=bitcoin,ethereum&vs_currencies=try
- BIST100 index: fetch via Yahoo Finance (^XU100)

## Output Format
Return a compact table:
| Asset  | Price (TRY) | 24h Change |
|--------|-------------|------------|
| BTC    | ₺1,234,567  | +2.3%      |
| ETH    | ₺67,890     | -0.8%      |
| XU100  | ₺9,876      | +0.4%      |

Use 🟢 for positive change, 🔴 for negative.
```

:::tip Zero API Keys
Both CoinGecko's public tier and Yahoo Finance work without registration. This keeps your skill pack accessible to everyone — no signup friction.
:::

## Step 5: Write the Briefing Card Script

The card generator creates a styled PNG for visual delivery. Save this as `scripts/turkish_brief_card.py`:

```python
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import textwrap

def create_briefing_card(headlines: list[str], prices: dict, output_path: str = "briefing.png"):
    """
    Generate a 1200x630px briefing card with Turkish locale styling.
    
    Args:
        headlines: List of headline strings (max 5)
        prices: Dict of {asset: {price_try, change_pct}}
        output_path: Where to save the PNG
    """
    # Card dimensions (optimized for Telegram)
    WIDTH, HEIGHT = 1200, 630
    
    # Colors — customize for your locale's aesthetic
    BG_COLOR = (15, 15, 25)        # Dark navy
    ACCENT_COLOR = (220, 38, 38)   # Turkish red
    TEXT_COLOR = (240, 240, 240)
    SUBTEXT_COLOR = (160, 160, 180)
    
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Header bar
    draw.rectangle([(0, 0), (WIDTH, 80)], fill=ACCENT_COLOR)
    draw.text((30, 20), "🇹🇷 GÜNLÜK BRİFİNG", fill="white", font_size=36)
    draw.text((WIDTH - 200, 28), datetime.now().strftime("%d %b %Y"), 
              fill="white", font_size=24)
    
    # News section
    draw.text((30, 100), "HABERLER", fill=SUBTEXT_COLOR, font_size=18)
    y = 130
    for i, headline in enumerate(headlines[:4]):
        wrapped = textwrap.fill(headline, width=70)
        draw.text((30, y), f"• {wrapped}", fill=TEXT_COLOR, font_size=20)
        y += 60
    
    # Prices section
    draw.rectangle([(0, HEIGHT - 120), (WIDTH, HEIGHT)], fill=(25, 25, 40))
    x = 30
    for asset, data in prices.items():
        color = (74, 222, 128) if data["change"] >= 0 else (248, 113, 113)
        draw.text((x, HEIGHT - 100), asset, fill=SUBTEXT_COLOR, font_size=16)
        draw.text((x, HEIGHT - 75), f"₺{data['price']:,.0f}", fill=TEXT_COLOR, font_size=22)
        arrow = "▲" if data["change"] >= 0 else "▼"
        draw.text((x, HEIGHT - 48), f"{arrow} {abs(data['change']):.1f}%", fill=color, font_size=18)
        x += 200
    
    img.save(output_path, "PNG", quality=95)
    return output_path
```

## Step 6: Write the Orchestrator Skill

The top-level skill ties everything together:

```markdown
# Turkish Daily Briefing

Generates and delivers a complete Turkish locale daily briefing.

## What it does
1. Invoke `turkish-news` skill to fetch headlines
2. Invoke `bist100` skill to fetch market prices in TRY
3. Run `scripts/turkish_brief_card.py` to generate a 1200x630 PNG card
4. Run `scripts/telegram_send.py` to deliver the card and a text summary

## Schedule
Designed to run daily at 08:00 via cron:
```
/cron add "0 8 * * *" "Run the turkish-daily-briefing skill. Fetch today's top headlines from Hürriyet, Bloomberg HT, and NTV. Fetch BTC, ETH, and BIST100 prices in TRY from CoinGecko and Yahoo Finance. Generate a briefing card and send to Telegram home channel."
```

## Dependencies
- Pillow (`pip install pillow`)
- No external API keys required
```

## Step 7: Test Before Scheduling

Always test the full workflow manually first:

```
hermes
❯ Run the turkish-daily-briefing skill for today
```

Check that:
- Headlines are in Turkish and properly attributed
- Prices show in TRY with correct symbols (₺)
- The PNG card generates without errors
- Telegram delivery succeeds (if configured)

Only after a clean manual run should you schedule with cron.

## Step 8: Package for the Skills Hub

To share your locale pack with the community via [agentskills.io](https://agentskills.io):

1. Move your skill folder to a public GitHub repo
2. Add a top-level `README.md` with installation instructions
3. Test with `hermes skills install github.com/yourname/turkish-locale`
4. Submit to the Skills Hub

### Writing a Good README

Include these sections:

```markdown
# Turkish Locale Skill Pack for Hermes Agent 🇹🇷

## What's inside
- Real-time prices in TRY (CoinGecko, no API key)
- Styled 1200px PNG daily briefing cards
- Turkish news (Hürriyet, Bloomberg HT, NTV)
- Telegram cron automation

## Install
hermes skills install github.com/yourname/turkish-locale

## Requirements
pip install pillow

## Schedule
/cron add "0 8 * * *" "Run turkish-daily-briefing skill..."
```

## Adapting This for Your Locale

Every section of this tutorial is a template. To build a **Brazilian Portuguese pack**, for example:

| Turkish | Brazilian |
|---------|-----------|
| Hürriyet, Bloomberg HT, NTV | Folha de S.Paulo, G1, UOL |
| TRY (₺) | BRL (R$) |
| BIST100 | IBOVESPA |
| 🇹🇷 | 🇧🇷 |

The skill structure, card generator, and cron setup stay identical. Only the data sources and currency symbols change.

## Going Further

- **[Skills System](../user-guide/features/skills)** — Full reference for skill authoring
- **[Scheduled Tasks](../user-guide/features/cron)** — Cron scheduling deep dive  
- **[Messaging Gateway](../user-guide/messaging/)** — Telegram, Discord, Slack, WhatsApp setup
- **[Skills Hub](https://agentskills.io)** — Browse and share community skills
- **[Daily Briefing Bot Tutorial](./daily-briefing-bot)** — The prompt-only approach (no scripts)

:::tip Your locale pack is a contribution
Every locale pack you publish makes Hermes more useful for a whole community of speakers. Turkish, Arabic, Japanese, Portuguese — each one opens the door for users who've never seen their language in an AI agent workflow.
:::
