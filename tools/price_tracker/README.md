# Hermes Ultra: Deal Scout & Anti-Scalper ⚡

Hermes Ultra is a premium price tracking, deal scoring, and market intelligence toolset integrated into [Hermes Agent](https://github.com/NousResearch/hermes-agent). Designed to hunt down the best deals across global e-commerce grids while actively protecting users from scalpers, bot-driven pricing manipulation, and fake discounts.

## 🌟 Key Features

*   **Global Grid Scanning**: Concurrently scans multiple international stores (Amazon, eBay, Best Buy, Newegg, Walmart, PriceSpy, Idealo) to construct a comprehensive market overview.
*   **Stealth Scraper**: Uses Playwright-driven headless browsing with sophisticated anti-bot evasion techniques (randomized human delays, dynamic UA pool rotation, and DOM spoofing) to bypass captchas and shielding, with a lightweight `httpx` fallback.
*   **Deep Intelligence Engine**:
    *   **Deal Scoring**: Evaluates current prices against historical data, MSRP/Original Price, and current stock status to generate a 0-100 `Deal Score`.
    *   **Anti-Scalper Protection**: Detects artificial price inflation and extreme cross-site markups to warn users of high-risk "scalper" pricing.
    *   **Trend Prediction**: Calculates price velocity and market volatility to recommend whether to `BUY NOW` or `WAIT` for a drop.
*   **Persistent Tracking & Portfolio**: Maintains a persistent record of tracked assets and calculates the user's "Lifetime Savings" when a deal is realized using the `buy` command.
*   **Autonomous Watcher**: A background daemon that periodically monitors target prices and broadcasts rich `critical` or `warning` alerts when price targets are hit or scalper behavior is detected. 

## 🚀 Quick Start (Standalone CLI)

Hermes Ultra provides a dedicated standalone CLI (`hermes_ultra.py`) that can be operated independently of the main AI agent loop:

```bash
# General help
python hermes_ultra.py --help

# Search globally for a product and start tracking the best deal
python hermes_ultra.py search "PS5 Slim Console" --target 450

# Track a specific product URL immediately
python hermes_ultra.py track "https://amazon.com/dp/XXXX" --target 500

# View the deep intelligence report for a tracked asset
python hermes_ultra.py check --id 1

# View the price history of an asset
python hermes_ultra.py history --id 1

# View your active tracking pipeline and lifetime savings
python hermes_ultra.py portfolio

# Realize a deal and add to your Lifetime Savings
python hermes_ultra.py buy 1

# Start the background watcher daemon
python hermes_ultra.py watch --interval 30
```

## 🧠 Using with Hermes Agent

The Ultra Toolset exposes six core LLM-callable tools (`price_track`, `price_check`, `price_history`, `price_search_and_track`, `price_alert_config`, and the watcher commands) to the Hermes Agent. 

Simply ask your agent:
> _"Can you search for the best deal on an RTX 4070 Ti and set an alert if it drops below $750?"_

The agent will seamlessly deploy the `price_search_and_track` tool, execute the stealth scraper, parse the resulting DOMs, evaluate the risk/deal algorithms, and return the intelligence report—completely autonomously.

## 🛠️ Installation & Requirements

Hermes Ultra is built to be robust even with minimal dependencies, utilizing standard regex over complex DOM parsing libraries for speed and reliability.

**Required Dependencies:**
*   `httpx`
*   `rich` (For ultra-premium terminal UI alerts and tables)

**Recommended Dependencies:**
*   `playwright` (Strongly recommended for JS-heavy sites like Amazon and Walmart to bypass bot protections)

To install the optional Playwright components via the Hermes Agent project:
```bash
pip install -e ".[price-tracker]"
playwright install chromium
```

## 🏗️ Extensibility

The parser architecture is highly modular. To add support for a new storefront, completely implement the `BaseSiteParser` interface and register it inside `tools/price_tracker/parsers/__init__.py`. 

**Current Native Parsers:**
*   Amazon Global (`amazon_global.py`)
*   eBay Global (`ebay_global.py`)
*   Best Buy (`bestbuy.py`)
*   Newegg (`newegg.py`)
*   Walmart (`walmart.py`)
*   Aggregators: Idealo, PriceSpy, CamelCamelCamel (`price_comparison.py`)
*   Generic Fallback (`generic.py`)

---
*Built for the ultimate shopping advantage. All systems green! ✅*
