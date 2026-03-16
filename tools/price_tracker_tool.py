"""Price tracker tool — Hermes Ultra: Deal Scout & Anti-Scalper.

Registers six LLM-callable tools under the ``price_tracker`` toolset:

- ``price_track``       — Add a product URL to tracking
- ``price_check``       — Check current price, deal score, scalper risk, trend
- ``price_history``     — View price history for a tracked product
- ``price_alert_config``— Create/manage alerts on tracked products
- ``price_watcher_start``— Start the background price watcher
- ``price_watcher_stop`` — Stop the background price watcher

All handlers return JSON strings per the hermes-agent convention.
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# Lazy-initialized singletons
_db = None
_scraper = None
_searcher = None
_watcher_thread = None
_watcher_stop_event = threading.Event()


def _get_db():
    global _db
    if _db is None:
        from tools.price_tracker.database import PriceTrackerDB
        _db = PriceTrackerDB()
    return _db


def _get_scraper():
    global _scraper
    if _scraper is None:
        from tools.price_tracker.scraper import StealthScraper
        _scraper = StealthScraper()
    return _scraper


def _get_searcher():
    global _searcher
    if _searcher is None:
        from tools.price_tracker.searcher import ProductSearcher
        _searcher = ProductSearcher(scraper=_get_scraper())
    return _searcher


# ---------------------------------------------------------------------------
# Check requirements
# ---------------------------------------------------------------------------

def _check_requirements() -> bool:
    """Price tracker is always available (only needs sqlite3 + httpx)."""
    return True


# ---------------------------------------------------------------------------
# Handler: price_track
# ---------------------------------------------------------------------------

def _handle_price_track(args: dict, **kw) -> str:
    """Add a product URL to tracking."""
    url = args.get("url", "").strip()
    target_price = args.get("target_price")
    name = args.get("name", "")

    if not url:
        return json.dumps({"error": "URL is required. Please provide a product link."})

    from tools.price_tracker.database import Product
    from tools.price_tracker.parsers import get_parser, list_supported_sites

    db = _get_db()

    # Check if already tracked
    existing = db.get_product_by_url(url)
    if existing:
        return json.dumps({
            "status": "already_tracked",
            "message": f"This product is already being tracked (ID: {existing.id}).",
            "product": existing.to_dict(),
        })

    # Determine site
    parser = get_parser(url)
    site = parser.get_site_name() if parser else "Unknown"

    # Try to scrape and parse
    product_data = None
    try:
        scraper = _get_scraper()
        result = scraper.scrape(url)
        if result.success and parser:
            product_data = parser.parse(result.html, url)
    except Exception as e:
        logger.warning("Initial scrape failed for %s: %s", url, e)

    # Build product
    product = Product(
        url=url,
        name=name or (product_data.name if product_data else ""),
        site=site,
        target_price=float(target_price) if target_price else None,
        current_price=product_data.price if product_data else None,
        original_price=product_data.original_price if product_data else None,
        stock_status=product_data.stock_status if product_data else "unknown",
        seller=product_data.seller if product_data else "",
        image_url=product_data.image_url if product_data else "",
        category=product_data.category if product_data else "",
    )

    product = db.add_product(product)

    # Record initial price if available
    if product_data and product_data.price:
        db.update_product_price(
            product.id,
            product_data.price,
            original_price=product_data.original_price,
            stock_status=product_data.stock_status,
            seller=product_data.seller,
            name=product_data.name,
        )

    # Auto-create a target_price alert if target was specified
    if target_price:
        from tools.price_tracker.database import Alert
        db.add_alert(Alert(
            product_id=product.id,
            alert_type="target_price",
            threshold=float(target_price),
        ))

    response = {
        "status": "success",
        "message": f"Product added to tracking! (ID: {product.id})",
        "product": product.to_dict(),
        "supported_sites": [s["site"] for s in list_supported_sites()],
    }

    # Show alert if we got price data
    if product_data and product_data.price:
        response["current_price"] = product_data.price
        response["price_found"] = True
    else:
        response["price_found"] = False
        response["note"] = "Initial price info could not be retrieved. Will be updated in the next scan."

    return json.dumps(response, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Handler: price_search_and_track
# ---------------------------------------------------------------------------

def _handle_price_search_and_track(args: dict, **kw) -> str:
    """Search for a product across multiple stores and track the best deal."""
    query = args.get("query", "").strip()
    target_price = args.get("target_price")

    if not query:
        return json.dumps({"error": "Search query is required."})

    searcher = _get_searcher()

    # Multi-source search — returns all results sorted by price
    all_results = searcher.search_all_sources(query)

    if not all_results:
        return json.dumps({
            "error": "No product was found in any store, or all sites applied bot protection.",
            "query": query
        })

    # Pick the best (cheapest with valid price) result to track
    priced_results = [r for r in all_results if r.get("price") and r["price"] > 0]
    best = priced_results[0] if priced_results else all_results[0]

    # Track the best deal
    track_args = {
        "url": best["url"],
        "name": best.get("name") or query,
    }
    if target_price:
        track_args["target_price"] = target_price

    track_result_str = _handle_price_track(track_args, **kw)
    track_result = json.loads(track_result_str)

    # Add search context and market prices
    track_result["search_query"] = query
    track_result["found_site"] = best.get("site", "")
    track_result["found_url"] = best.get("url", "")
    track_result["market_prices"] = all_results
    track_result["sources_checked"] = len(all_results)
    track_result["sources_with_price"] = len(priced_results)

    # Product name from best result
    product_name = best.get("name") or query
    track_result["product_name"] = product_name

    # Ensure product_id is available for chaining
    if "product" in track_result and "id" in track_result["product"]:
        track_result["product_id"] = track_result["product"]["id"]

    return json.dumps(track_result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Handler: price_check
# ---------------------------------------------------------------------------

def _handle_price_check(args: dict, **kw) -> str:
    """Check current price, deal score, scalper risk, and trend."""
    product_id = args.get("product_id")
    url = args.get("url", "").strip()
    market_prices = args.get("market_prices")  # From multi-source search

    db = _get_db()

    # Find the product
    product = None
    if product_id:
        product = db.get_product(int(product_id))
    elif url:
        product = db.get_product_by_url(url)

    if not product:
        return json.dumps({
            "error": "Product not found. Please track it first using 'price_track'.",
        })

    # Get price history
    history_records = db.get_price_history(product.id, limit=50)
    history_prices = [r.price for r in history_records]

    # Scrape current price
    from tools.price_tracker.parsers import get_parser
    parser = get_parser(product.url)

    current_data = None
    try:
        scraper = _get_scraper()
        result = scraper.scrape(product.url)
        if result.success and parser:
            current_data = parser.parse(result.html, product.url)
            # Update DB
            if current_data.price:
                db.update_product_price(
                    product.id,
                    current_data.price,
                    original_price=current_data.original_price,
                    stock_status=current_data.stock_status,
                    seller=current_data.seller,
                    name=current_data.name or product.name,
                )
                # Refresh product
                product = db.get_product(product.id)
                history_prices = [current_data.price] + history_prices
    except Exception as e:
        logger.warning("Price check scrape failed: %s", e)

    # If we STILL have no price (e.g. Amazon Bot Protection or item Out of Stock), return early
    if product.current_price is None or product.current_price <= 0:
        return json.dumps({
            "error": "Valid price information could not be found for this product.\n(Price may be hidden due to bot protection or out-of-stock.)",
            "product_id": product.id,
            "url": product.url
        }, ensure_ascii=False)

    # Deal scoring
    from tools.price_tracker.scoring import DealScorer
    scorer = DealScorer()
    deal = scorer.calculate(
        current_price=product.current_price,
        target_price=product.target_price,
        original_price=product.original_price,
        stock_status=product.stock_status,
        price_history=history_prices,
    )

    # Scalper detection
    from tools.price_tracker.scalper_detector import ScalperDetector
    detector = ScalperDetector()
    scalper = detector.check(
        current_price=product.current_price or 0,
        price_history=history_prices,
        original_price=product.original_price,
    )

    # Trend prediction
    from tools.price_tracker.trend_predictor import TrendPredictor
    predictor = TrendPredictor()
    trend = predictor.predict(
        price_history=history_prices,
        current_price=product.current_price,
    )

    # Format alert
    from tools.price_tracker.alerts import format_full_report, _notify_all
    alert_text = format_full_report(
        product_name=product.name or "Unknown Product",
        current_price=product.current_price or 0,
        target_price=product.target_price,
        deal_score=deal.total_score,
        deal_label=deal.label,
        scalper_risk=f"{scalper.risk_emoji} {scalper.risk_level}",
        scalper_text=scalper.analysis_text,
        trend_direction=f"{trend.direction_emoji} {trend.direction}",
        trend_text=trend.analysis_text,
        url=product.url,
        market_prices=market_prices,
        volatility_warning=trend.volatility_warning,
    )

    # Send through notification hooks (Critical for BUY NOW deals)
    level = "critical" if deal.total_score >= 80 else "info"
    _notify_all(f"Hermes Ultra — {deal.label}", alert_text, level=level)

    return json.dumps({
        "product": product.to_dict(),
        "deal_score": {
            "total": deal.total_score,
            "label": deal.label,
            "breakdown": {
                "discount": deal.discount_score,
                "trend": deal.trend_score,
                "stock": deal.stock_score,
                "market": deal.market_score,
            },
            "explanation": deal.explanation,
        },
        "scalper_analysis": {
            "risk_level": scalper.risk_level,
            "deviation_pct": scalper.deviation_pct,
            "avg_price": scalper.avg_price,
            "is_suspicious": scalper.is_suspicious,
            "analysis": scalper.analysis_text,
        },
        "trend_prediction": {
            "direction": trend.direction,
            "predicted_change_pct": trend.predicted_change_pct,
            "predicted_price": trend.predicted_price,
            "confidence": trend.confidence,
            "analysis": trend.analysis_text,
        },
        "alert_rendered": True,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Handler: price_history
# ---------------------------------------------------------------------------

def _handle_price_history(args: dict, **kw) -> str:
    """View price history for a tracked product."""
    product_id = args.get("product_id")
    limit = args.get("limit", 20)

    if not product_id:
        return json.dumps({"error": "product_id is required."})

    db = _get_db()
    product = db.get_product(int(product_id))
    if not product:
        return json.dumps({"error": f"Product not found: {product_id}"})

    records = db.get_price_history(int(product_id), limit=int(limit))

    from tools.price_tracker.alerts import format_price_table

    history_data = []
    for r in records:
        from datetime import datetime
        ts = datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d %H:%M") if r.timestamp else "?"
        history_data.append({
            "price": r.price,
            "original_price": r.original_price,
            "stock_status": r.stock_status,
            "seller": r.seller,
            "timestamp": ts,
        })

    return json.dumps({
        "product_name": product.name,
        "product_url": product.url,
        "current_price": product.current_price,
        "target_price": product.target_price,
        "history_count": len(records),
        "history": history_data,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Handler: price_alert_config
# ---------------------------------------------------------------------------

def _handle_price_alert_config(args: dict, **kw) -> str:
    """Create or manage alerts for products."""
    action = args.get("action", "create")
    product_id = args.get("product_id")

    db = _get_db()

    if action == "create":
        if not product_id:
            return json.dumps({"error": "product_id is required."})

        product = db.get_product(int(product_id))
        if not product:
            return json.dumps({"error": f"Product not found: {product_id}"})

        from tools.price_tracker.database import Alert
        alert_type = args.get("alert_type", "price_drop")
        threshold = args.get("threshold")

        alert = db.add_alert(Alert(
            product_id=int(product_id),
            alert_type=alert_type,
            threshold=float(threshold) if threshold else None,
        ))

        return json.dumps({
            "status": "success",
            "message": f"Alert created (ID: {alert.id}).",
            "alert_type": alert_type,
            "threshold": threshold,
        }, ensure_ascii=False)

    elif action == "list":
        alerts = db.get_active_alerts(
            product_id=int(product_id) if product_id else None
        )
        return json.dumps({
            "active_alerts": [
                {
                    "id": a.id,
                    "product_id": a.product_id,
                    "alert_type": a.alert_type,
                    "threshold": a.threshold,
                }
                for a in alerts
            ],
        })

    elif action == "deactivate":
        alert_id = args.get("alert_id")
        if not alert_id:
            return json.dumps({"error": "alert_id is required."})
        success = db.deactivate_alert(int(alert_id))
        return json.dumps({
            "status": "success" if success else "not_found",
            "message": "Alert deactivated." if success else "Alert not found.",
        })

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# ---------------------------------------------------------------------------
# Handler: price_watcher_start / stop
# ---------------------------------------------------------------------------

def _watcher_loop(interval_minutes: int = 30):
    """Background loop that checks all tracked products."""
    logger.info("Price watcher started (interval: %d min)", interval_minutes)
    while not _watcher_stop_event.is_set():
        try:
            db = _get_db()
            products = db.list_products()
            from tools.price_tracker.parsers import get_parser
            from tools.price_tracker.scoring import DealScorer
            from tools.price_tracker.scalper_detector import ScalperDetector
            from tools.price_tracker.alerts import format_deal_alert, format_scalper_warning, _notify_all

            scraper = _get_scraper()
            scorer = DealScorer()
            detector = ScalperDetector()

            for product in products:
                if _watcher_stop_event.is_set():
                    break

                parser = get_parser(product.url)
                if not parser:
                    continue

                try:
                    result = scraper.scrape(product.url)
                    if not result.success:
                        continue

                    data = parser.parse(result.html, product.url)
                    if not data.price:
                        continue

                    # Update DB
                    old_price = product.current_price
                    db.update_product_price(
                        product.id, data.price,
                        original_price=data.original_price,
                        stock_status=data.stock_status,
                        seller=data.seller,
                        name=data.name or product.name,
                    )

                    history = [r.price for r in db.get_price_history(product.id, limit=50)]

                    # Check alerts
                    alerts = db.get_active_alerts(product.id)
                    for alert in alerts:
                        should_fire = False
                        if alert.alert_type == "target_price" and alert.threshold:
                            if data.price <= alert.threshold:
                                should_fire = True
                        elif alert.alert_type == "price_drop":
                            if old_price and data.price < old_price * 0.95:
                                should_fire = True

                        if should_fire:
                            deal = scorer.calculate(
                                data.price, product.target_price,
                                data.original_price, data.stock_status, history,
                            )
                            alert_text = format_deal_alert(
                                product.name or "Product",
                                data.price, product.target_price,
                                deal.total_score, deal.label, product.url,
                            )
                            _notify_all("Hermes Ultra — Price Alert!", alert_text, "critical")

                    # Scalper check
                    scalper = detector.check(data.price, history, data.original_price)
                    if scalper.is_suspicious:
                        warn_text = format_scalper_warning(
                            product.name or "Product",
                            scalper.risk_level, scalper.risk_emoji,
                            scalper.deviation_pct, scalper.avg_price,
                            data.price, scalper.analysis_text,
                        )
                        _notify_all("Hermes Ultra — Scalper Warning!", warn_text, "warning")

                except Exception as e:
                    logger.warning("Watcher error for product %s: %s", product.id, e)

        except Exception as e:
            logger.error("Watcher loop error: %s", e)

        # Wait for next cycle (check stop event every 10 seconds)
        for _ in range(interval_minutes * 6):
            if _watcher_stop_event.is_set():
                break
            _watcher_stop_event.wait(10)

    logger.info("Price watcher stopped.")


def _handle_watcher_start(args: dict, **kw) -> str:
    """Start the background price watcher."""
    global _watcher_thread
    interval = args.get("interval_minutes", 30)

    if _watcher_thread and _watcher_thread.is_alive():
        return json.dumps({
            "status": "already_running",
            "message": "Price watcher is already running.",
        })

    _watcher_stop_event.clear()
    _watcher_thread = threading.Thread(
        target=_watcher_loop,
        args=(int(interval),),
        daemon=True,
        name="hermes-ultra-watcher",
    )
    _watcher_thread.start()

    db = _get_db()
    product_count = len(db.list_products())

    return json.dumps({
        "status": "started",
        "message": f"Price watcher started! Monitoring {product_count} products.",
        "interval_minutes": interval,
        "tracked_products": product_count,
    }, ensure_ascii=False)


def _handle_watcher_stop(args: dict, **kw) -> str:
    """Stop the background price watcher."""
    global _watcher_thread

    if not _watcher_thread or not _watcher_thread.is_alive():
        return json.dumps({
            "status": "not_running",
            "message": "Price watcher is not running.",
        })

    _watcher_stop_event.set()
    _watcher_thread.join(timeout=15)

    return json.dumps({
        "status": "stopped",
        "message": "Price watcher stopped.",
    })


# ---------------------------------------------------------------------------
# Handler: price_buy
# ---------------------------------------------------------------------------

def _handle_price_buy(args: dict, **kw) -> str:
    """Mark a product as purchased and record lifetime savings."""
    product_id = args.get("product_id")
    if not product_id:
        return json.dumps({"error": "Product ID is required."})

    db = _get_db()
    product = db.get_product(product_id)
    if not product:
        return json.dumps({"error": f"Product {product_id} not found."})

    current = product.current_price or 0.0
    original = product.original_price or 0.0
    
    # Calculate savings vs original/list price
    savings = max(0.0, original - current) if original > 0 else 0.0
    
    if savings > 0:
        db.add_lifetime_savings(savings)
        
    new_total = db.get_lifetime_savings()
    db.delete_product(product_id)
    
    return json.dumps({
        "status": "success",
        "product_id": product_id,
        "product_name": product.name,
        "buy_price": current,
        "savings": savings,
        "lifetime_savings": new_total,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Handler: portfolio
# ---------------------------------------------------------------------------

def _handle_portfolio(args: dict, **kw) -> str:
    """List tracked assets with P/L and Deal Scores."""
    db = _get_db()
    products = db.list_products()
    lifetime_savings = db.get_lifetime_savings()

    results = []
    from tools.price_tracker.scoring import DealScorer
    scorer = DealScorer()

    for p in products:
        current = p.current_price or 0.0
        original = p.original_price or current
        
        # Recalculate deal score for the summary
        history = [r.price for r in db.get_price_history(p.id, limit=20)]
        deal = scorer.calculate(current, p.target_price, original, p.stock_status, history)
        
        results.append({
            "id": p.id,
            "name": p.name,
            "current_price": current,
            "original_price": original,
            "deal_score": deal.total_score,
            "deal_label": deal.label,
        })

    from tools.price_tracker.alerts import format_portfolio_table
    table_text = format_portfolio_table(results, lifetime_savings)
    
    return json.dumps({
        "status": "success",
        "lifetime_savings": lifetime_savings,
        "products": results,
        "table": table_text
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PRICE_TRACK_SCHEMA = {
    "name": "price_track",
    "description": (
        "Add a product URL to tracking. Scrapes initial price and saves to database. "
        "Supports Amazon (Global/TR), eBay, Walmart, BestBuy, Newegg, "
        "and 40+ other global retail sites."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Product page URL.",
            },
            "target_price": {
                "type": "number",
                "description": "Target price. Alerts triggered when price drops below this.",
            },
            "name": {
                "type": "string",
                "description": "Custom name for the product (optional).",
            },
        },
        "required": ["url"],
    },
}

PRICE_SEARCH_AND_TRACK_SCHEMA = {
    "name": "price_search_and_track",
    "description": (
        "Search for a product by name across multiple global stores, "
        "find the best deal, and automatically start tracking it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g. 'PS5 Slim').",
            },
            "target_price": {
                "type": "number",
                "description": "Optional target price.",
            },
        },
        "required": ["query"],
    },
}

PRICE_CHECK_SCHEMA = {
    "name": "price_check",
    "description": (
        "Check current status of a tracked product. Generates a detailed "
        "intelligence report with Deal Score, Scalper Risk, and Trend Prediction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "integer",
                "description": "Product ID.",
            },
            "url": {
                "type": "string",
                "description": "Product URL (can be used instead of product_id).",
            },
        },
        "required": [],
    },
}

PRICE_HISTORY_SCHEMA = {
    "name": "price_history",
    "description": (
        "View price history for a tracked product. "
        "Includes timestamp, price, stock status, and seller info."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "integer",
                "description": "Product ID.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of records to show (default: 20).",
                "default": 20,
            },
        },
        "required": ["product_id"],
    },
}

PRICE_ALERT_CONFIG_SCHEMA = {
    "name": "price_alert_config",
    "description": (
        "Create and manage price alerts for tracked products. "
        "Supports target price, price drop, stock change, and deal score alerts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action type: 'create', 'list', or 'deactivate'.",
                "enum": ["create", "list", "deactivate"],
            },
            "product_id": {
                "type": "integer",
                "description": "Product ID.",
            },
            "alert_type": {
                "type": "string",
                "description": "Alert type: 'price_drop', 'target_price', 'stock_change', 'deal_score'.",
                "default": "price_drop",
            },
            "threshold": {
                "type": "number",
                "description": "Threshold value (price or score).",
            },
            "alert_id": {
                "type": "integer",
                "description": "ID of the alert to deactivate (for action=deactivate).",
            },
        },
        "required": ["action"],
    },
}

PRICE_WATCHER_START_SCHEMA = {
    "name": "price_watcher_start",
    "description": (
        "Start the background price scanner. Periodically checks all tracked products. "
        "Triggers desktop notifications for target hits, stock changes, or scalper risks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "interval_minutes": {
                "type": "integer",
                "description": "Scan interval in minutes. Default: 30.",
                "default": 30,
            },
        },
        "required": [],
    },
}

PRICE_WATCHER_STOP_SCHEMA = {
    "name": "price_watcher_stop",
    "description": "Stop the background price scanner.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

PRICE_PORTFOLIO_SCHEMA = {
    "name": "price_portfolio",
    "description": "List all tracked products with profit/loss, deal scores, and lifetime savings.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

PRICE_BUY_SCHEMA = {
    "name": "price_buy",
    "description": "Mark a product as purchased, record savings to lifetime total, and stop tracking.",
    "parameters": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "integer",
                "description": "The ID of the product purchased.",
            },
        },
        "required": ["product_id"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="price_track",
    toolset="price_tracker",
    schema=PRICE_TRACK_SCHEMA,
    handler=_handle_price_track,
    check_fn=_check_requirements,
)

registry.register(
    name="price_search_and_track",
    toolset="price_tracker",
    schema=PRICE_SEARCH_AND_TRACK_SCHEMA,
    handler=_handle_price_search_and_track,
    check_fn=_check_requirements,
)

registry.register(
    name="price_check",
    toolset="price_tracker",
    schema=PRICE_CHECK_SCHEMA,
    handler=_handle_price_check,
    check_fn=_check_requirements,
)

registry.register(
    name="price_history",
    toolset="price_tracker",
    schema=PRICE_HISTORY_SCHEMA,
    handler=_handle_price_history,
    check_fn=_check_requirements,
)

registry.register(
    name="price_alert_config",
    toolset="price_tracker",
    schema=PRICE_ALERT_CONFIG_SCHEMA,
    handler=_handle_price_alert_config,
    check_fn=_check_requirements,
)

registry.register(
    name="price_watcher_start",
    toolset="price_tracker",
    schema=PRICE_WATCHER_START_SCHEMA,
    handler=_handle_watcher_start,
    check_fn=_check_requirements,
)

registry.register(
    name="price_watcher_stop",
    toolset="price_tracker",
    schema=PRICE_WATCHER_STOP_SCHEMA,
    handler=_handle_watcher_stop,
    check_fn=_check_requirements,
)

registry.register(
    name="price_portfolio",
    toolset="price_tracker",
    schema=PRICE_PORTFOLIO_SCHEMA,
    handler=_handle_portfolio,
    check_fn=_check_requirements,
)

registry.register(
    name="price_buy",
    toolset="price_tracker",
    schema=PRICE_BUY_SCHEMA,
    handler=_handle_price_buy,
    check_fn=_check_requirements,
)

registry.register(
    name="price_watcher_stop",
    toolset="price_tracker",
    schema=PRICE_WATCHER_STOP_SCHEMA,
    handler=_handle_watcher_stop,
    check_fn=_check_requirements,
)
