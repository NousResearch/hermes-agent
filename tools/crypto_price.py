#!/usr/bin/env python3
"""
Hermes Agent Skill: Crypto Price Fetcher

Bu yetenek CoinGecko API üzerinden kripto para fiyat bilgisi çeker.

Kullanım:
  python crypto_price.py price get bitcoin --currency usd
  python crypto_price.py price history bitcoin --days 7 --currency usd
"""

import argparse
import json
import sys
import requests


API_BASE = "https://api.coingecko.com/api/v3"


# -----------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# -----------------------------------------------------------------------------
def fetch_json(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


# -----------------------------------------------------------------------------
# API Actions
# -----------------------------------------------------------------------------

def price_get(args):
    """
    Tek bir coin'in güncel fiyatını getirir.
    """
    url = f"{API_BASE}/simple/price"
    params = {
        "ids": args.coin,
        "vs_currencies": args.currency
    }
    data = fetch_json(url, params)
    output = {
        "coin": args.coin,
        "currency": args.currency,
        "price": data.get(args.coin, {}).get(args.currency, None)
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def price_history(args):
    """
    Belirli bir coin'in geçmiş fiyat verilerini getirir.
    """
    url = f"{API_BASE}/coins/{args.coin}/market_chart"
    params = {
        "vs_currency": args.currency,
        "days": args.days
    }
    data = fetch_json(url, params)

    output = {
        "coin": args.coin,
        "currency": args.currency,
        "days": args.days,
        "prices": data.get("prices", [])
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


# -----------------------------------------------------------------------------
# CLI Parser
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Crypto Price Skill for Hermes Agent")
    sub = parser.add_subparsers(dest="service", required=True)

    # --- price ---
    price = sub.add_parser("price")
    price_sub = price.add_subparsers(dest="action", required=True)

    # price get
    p = price_sub.add_parser("get")
    p.add_argument("coin", help="örn: bitcoin, ethereum")
    p.add_argument("--currency", default="usd")
    p.set_defaults(func=price_get)

    # price history
    p = price_sub.add_parser("history")
    p.add_argument("coin")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--currency", default="usd")
    p.set_defaults(func=price_history)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
