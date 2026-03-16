"""Scalper / Anti-Scalper detection engine.

Detects artificially inflated prices by comparing current price against
historical averages. Risk levels (calibrated for USD/EUR/GBP):

  HIGH    — price >30% above average
  MEDIUM  — price 15-30% above average
  LOW     — price 5-15% above average
  NONE    — normal price range
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScalperReport:
    """Result of a scalper analysis."""
    risk_level: str = "NONE"  # NONE, LOW, MEDIUM, HIGH
    risk_emoji: str = "✅"
    deviation_pct: float = 0.0
    avg_price: float = 0.0
    current_price: float = 0.0
    analysis_text: str = ""
    is_suspicious: bool = False
    price_spike_detected: bool = False


class ScalperDetector:
    """Detects scalper pricing anomalies."""

    # Deviation thresholds (calibrated for global markets, tighter than TR)
    THRESHOLD_HIGH = 30.0    # >30% above avg = HIGH risk
    THRESHOLD_MEDIUM = 15.0  # >15% above avg = MEDIUM risk
    THRESHOLD_LOW = 5.0      # >5% above avg = LOW risk

    # Spike detection: short-term price jump
    SPIKE_WINDOW = 5         # Look at last N price records
    SPIKE_THRESHOLD = 15.0   # >15% jump in short window

    def check(
        self,
        current_price: float,
        price_history: Optional[List[float]] = None,
        original_price: Optional[float] = None,
    ) -> ScalperReport:
        """Analyze a product's price for scalper patterns.

        Args:
            current_price: The product's current price.
            price_history: List of recent prices (newest first).
            original_price: The product's original/list price.
        """
        report = ScalperReport(current_price=current_price)

        if current_price <= 0:
            report.analysis_text = "Invalid price information."
            return report

        history = price_history or []

        # Calculate average from history
        if len(history) >= 3:
            report.avg_price = sum(history) / len(history)
        elif original_price and original_price > 0:
            report.avg_price = original_price
        else:
            report.analysis_text = (
                "Insufficient price history — cannot run scalper analysis yet. "
                "Accuracy will improve as more data is collected."
            )
            return report

        if report.avg_price <= 0:
            report.analysis_text = "Could not calculate average price."
            return report

        # Calculate deviation
        report.deviation_pct = (
            (current_price - report.avg_price) / report.avg_price
        ) * 100

        # Classify risk
        if report.deviation_pct >= self.THRESHOLD_HIGH:
            report.risk_level = "HIGH"
            report.risk_emoji = "🚨"
            report.is_suspicious = True
            report.analysis_text = (
                f"⚠️ SCALPER RISK! Price is {report.deviation_pct:.1f}% above average. "
                f"Average: ${report.avg_price:,.2f}, "
                f"Current: ${current_price:,.2f}. "
                "This product is highly likely being sold at a scalper price!"
            )
        elif report.deviation_pct >= self.THRESHOLD_MEDIUM:
            report.risk_level = "MEDIUM"
            report.risk_emoji = "⚠️"
            report.is_suspicious = True
            report.analysis_text = (
                f"Caution: Price is {report.deviation_pct:.1f}% above average. "
                f"Average: ${report.avg_price:,.2f}. "
                "Suspiciously high price — compare before buying."
            )
        elif report.deviation_pct >= self.THRESHOLD_LOW:
            report.risk_level = "LOW"
            report.risk_emoji = "🟡"
            report.analysis_text = (
                f"Price is {report.deviation_pct:.1f}% above average. "
                f"Average: ${report.avg_price:,.2f}. "
                "Slight price increase — likely normal fluctuation."
            )
        else:
            report.risk_level = "NONE"
            report.risk_emoji = "✅"
            report.analysis_text = (
                f"Price is in a normal range. "
                f"Average: ${report.avg_price:,.2f}, "
                f"Current: ${current_price:,.2f}."
            )

        # Spike detection: sudden jump in recent history
        if len(history) >= self.SPIKE_WINDOW:
            recent = history[: self.SPIKE_WINDOW]
            recent_avg = sum(recent) / len(recent)
            if recent_avg > 0:
                spike_pct = ((current_price - recent_avg) / recent_avg) * 100
                if spike_pct >= self.SPIKE_THRESHOLD:
                    report.price_spike_detected = True
                    report.analysis_text += (
                        f" A sudden {spike_pct:.1f}% price jump was detected "
                        f"in the last {self.SPIKE_WINDOW} scans!"
                    )

        return report

    def check_cross_site(self, market_prices: list) -> Optional["ScalperReport"]:
        """Analyze price spread across multiple stores for scalper detection.

        Args:
            market_prices: List of dicts with 'site' and 'price' keys.

        Returns:
            ScalperReport if suspicious spread detected, None otherwise.
        """
        priced = [r for r in market_prices if r.get("price") and r["price"] > 0]
        if len(priced) < 2:
            return None

        min_r = min(priced, key=lambda r: r["price"])
        max_r = max(priced, key=lambda r: r["price"])
        min_price = min_r["price"]
        max_price = max_r["price"]

        if min_price <= 0:
            return None

        spread_pct = ((max_price - min_price) / min_price) * 100

        report = ScalperReport(
            current_price=max_price,
            avg_price=min_price,
            deviation_pct=spread_pct,
        )

        if spread_pct >= 50:
            report.risk_level = "HIGH"
            report.risk_emoji = "🚨"
            report.is_suspicious = True
            report.analysis_text = (
                f"EXTREME price spread! {max_r['site']} (${max_price:,.2f}) is "
                f"{spread_pct:.1f}% above {min_r['site']} (${min_price:,.2f}). "
                "This is a strong scalper indicator!"
            )
        elif spread_pct >= 30:
            report.risk_level = "MEDIUM"
            report.risk_emoji = "⚠️"
            report.is_suspicious = True
            report.analysis_text = (
                f"Suspicious spread: {max_r['site']} (${max_price:,.2f}) is "
                f"{spread_pct:.1f}% above {min_r['site']} (${min_price:,.2f}). "
                "Compare prices carefully before buying."
            )
        elif spread_pct >= 15:
            report.risk_level = "LOW"
            report.risk_emoji = "🟡"
            report.analysis_text = (
                f"Price variance: {max_r['site']} is {spread_pct:.1f}% above "
                f"{min_r['site']}. Normal market fluctuation."
            )
        else:
            return None  # No alert needed for small variance

        return report

