"""Rich terminal alerts and notification system.

Provides beautifully formatted terminal output for price alerts,
deal scores, scalper warnings, and trend reports. Uses the ``rich``
library (already a hermes-agent dependency).

Built with a modular notification hook system so Discord/Telegram
integrations can be added later.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Notification hook system (modular, extensible)
# ---------------------------------------------------------------------------

class NotificationHook(ABC):
    """Base class for notification delivery hooks."""

    @abstractmethod
    def send(self, title: str, message: str, level: str = "info") -> bool:
        """Deliver a notification.

        Args:
            title: Alert title.
            message: Full rendered message (may contain ANSI or markdown).
            level: One of 'info', 'warning', 'critical'.

        Returns:
            True if delivery succeeded.
        """


class TerminalHook(NotificationHook):
    """Default hook — prints to stdout via Rich."""

    def send(self, title: str, message: str, level: str = "info") -> bool:
        try:
            from rich.console import Console
            console = Console()
            console.print(message)
            return True
        except ImportError:
            print(message)
            return True


class WindowsNotificationHook(NotificationHook):
    """Sends native Windows toast notifications via PowerShell."""

    def send(self, title: str, message: str, level: str = "info") -> bool:
        if level == "info":
            return True  # Don't spam desktop for info pings
        
        import subprocess
        import os
        import re
        
        # Clean ANSI codes for system toast
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-2]*[amkqtyurHW])')
        clean_msg = ansi_escape.sub('', message).strip()
        # Take first few lines only
        clean_msg = " / ".join(clean_msg.splitlines()[:3]).replace('"', "'")

        ps_script = f"""
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
        $textNodes = $template.GetElementsByTagName("text")
        $textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) | Out-Null
        $textNodes.Item(1).AppendChild($template.CreateTextNode("{clean_msg}")) | Out-Null
        $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("HermesUltra").Show($toast)
        """
        try:
            subprocess.run(["powershell", "-Command", ps_script], 
                           capture_output=True, check=False)
            return True
        except Exception:
            return False


# Global hook list — extensions append to this
_hooks: List[NotificationHook] = [TerminalHook(), WindowsNotificationHook()]


def add_notification_hook(hook: NotificationHook):
    """Register an additional notification hook."""
    _hooks.append(hook)


def _notify_all(title: str, message: str, level: str = "info"):
    """Send a notification through all registered hooks."""
    for hook in _hooks:
        try:
            hook.send(title, message, level)
        except Exception as e:
            logger.warning("Notification hook %s failed: %s", type(hook).__name__, e)


# ---------------------------------------------------------------------------
# Alert formatters (return Rich-renderable strings)
# ---------------------------------------------------------------------------

def format_deal_alert(
    product_name: str,
    current_price: float,
    target_price: Optional[float],
    score: int,
    score_label: str,
    url: str = "",
) -> str:
    """Format a deal score alert as a Rich panel."""
    try:
        from rich.panel import Panel
        from rich.text import Text
        from rich.console import Console
        from io import StringIO

        # Build content
        lines = []
        lines.append(f"[bold cyan]📦 {product_name}[/bold cyan]")
        lines.append("")
        lines.append(f"💰 Price: [bold green]${current_price:,.2f}[/bold green]")
        if target_price:
            if current_price <= target_price:
                diff = target_price - current_price
                lines.append(f"🎯 Target: ${target_price:,.2f} [bold green](✅ ${diff:,.2f} below!)[/bold green]")
            else:
                diff = current_price - target_price
                lines.append(f"🎯 Target: ${target_price:,.2f} [yellow](${diff:,.2f} above)[/yellow]")

        # Score bar
        filled = score // 5
        empty = 20 - filled
        if score >= 80:
            bar_color = "green"
        elif score >= 60:
            bar_color = "cyan"
        elif score >= 30:
            bar_color = "yellow"
        else:
            bar_color = "red"
        bar = f"[{bar_color}]{'█' * filled}{'░' * empty}[/{bar_color}] {score}/100"
        lines.append(f"\n📊 Deal Score: {bar}")
        lines.append(f"   {score_label}")

        if url:
            lines.append(f"\n🔗 {url}")

        content = "\n".join(lines)

        # Determine border color
        if score >= 80:
            border = "bold green"
            title = "🔥 UNMISSABLE DEAL!"
        elif score >= 60:
            border = "bold cyan"
            title = "👍 GOOD DEAL"
        elif score >= 30:
            border = "yellow"
            title = "🤔 FAIR DEAL"
        else:
            border = "red"
            title = "💀 BAD DEAL"

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=70)
        panel = Panel(content, title=title, border_style=border, width=70, padding=(1, 2))
        console.print(panel)
        return buf.getvalue()

    except ImportError:
        # Fallback without Rich
        return _plain_deal_alert(product_name, current_price, target_price, score, score_label, url)


def _plain_deal_alert(name, price, target, score, label, url):
    """Plain-text fallback for deal alerts."""
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        f"║  🔔 DEAL ALERT — Score: {score}/100",
        "╠══════════════════════════════════════════════════════╣",
        f"║  📦 {name}",
        f"║  💰 Price: ${price:,.2f}",
    ]
    if target:
        lines.append(f"║  🎯 Target: ${target:,.2f}")
    lines.append(f"║  📊 {label}")
    if url:
        lines.append(f"║  🔗 {url}")
    lines.append("╚══════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def format_scalper_warning(
    product_name: str,
    risk_level: str,
    risk_emoji: str,
    deviation_pct: float,
    avg_price: float,
    current_price: float,
    analysis_text: str,
) -> str:
    """Format a scalper/karaborsa warning."""
    try:
        from rich.panel import Panel
        from rich.console import Console
        from io import StringIO

        lines = []
        lines.append(f"[bold red]⚠️ SCALPER ANALYSIS[/bold red]")
        lines.append(f"\n📦 {product_name}")
        lines.append(f"💰 Current: [bold]${current_price:,.2f}[/bold]")
        lines.append(f"📊 Average: ${avg_price:,.2f}")
        lines.append(f"📈 Deviation: [bold red]+{deviation_pct:.1f}%[/bold red]")
        lines.append(f"\n{risk_emoji} Risk Level: [bold]{risk_level}[/bold]")
        lines.append(f"\n{analysis_text}")

        content = "\n".join(lines)

        if risk_level == "HIGH":
            border = "bold red"
            title = "🚨 SCALPER WARNING!"
        elif risk_level == "MEDIUM":
            border = "bold yellow"
            title = "⚠️ SUSPICIOUS PRICE"
        else:
            border = "green"
            title = "✅ NORMAL PRICE"

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=70)
        panel = Panel(content, title=title, border_style=border, width=70, padding=(1, 2))
        console.print(panel)
        return buf.getvalue()

    except ImportError:
        return f"[{risk_level}] {product_name}: {analysis_text}"


def format_price_table(products: list) -> str:
    """Format a list of tracked products as a Rich table."""
    try:
        from rich.table import Table
        from rich.console import Console
        from io import StringIO

        table = Table(
            title="📋 Tracked Products",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Product", style="bold", max_width=30)
        table.add_column("Price", justify="right", style="green")
        table.add_column("Target", justify="right")
        table.add_column("Stock", justify="center")
        table.add_column("Site", style="dim")

        stock_icons = {
            "in_stock": "✅",
            "out_of_stock": "❌",
            "limited": "⚠️",
            "unknown": "❓",
        }

        for i, p in enumerate(products, 1):
            price_str = f"${p.get('current_price', 0):,.2f}" if p.get('current_price') else "—"
            target_str = f"${p.get('target_price', 0):,.2f}" if p.get('target_price') else "—"
            stock = stock_icons.get(p.get('stock_status', 'unknown'), '❓')
            name = (p.get('name', 'Unknown'))[:30]
            table.add_row(str(i), name, price_str, target_str, stock, p.get('site', ''))

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=90)
        console.print(table)
        return buf.getvalue()

    except ImportError:
        lines = ["Tracked Products:"]
        for i, p in enumerate(products, 1):
            lines.append(f"  {i}. {p.get('name', '?')} — ${p.get('current_price', '?')}")
        return "\n".join(lines)


def format_market_overview(
    results: list,
    product_name: str = "",
) -> str:
    """Format a multi-source price comparison as a Rich table.

    Only stores with successfully fetched prices are shown.
    If all sources are blocked/captcha'd, returns a clean fallback message.

    Args:
        results: List of dicts with keys: site, price, stock_status, url.
        product_name: The product name to display in the header.
    """
    if not results:
        return ""

    # Filter: only keep results with a valid price
    priced = [r for r in results if r.get("price") and r["price"] > 0]

    # If no store returned a valid price, show a clean single-line message
    if not priced:
        return "🔍 Scanning global markets... All sources currently shielded. Retrying in background."

    best_price = min(priced, key=lambda r: r["price"])["price"]

    try:
        from rich.table import Table
        from rich.panel import Panel
        from rich.console import Console
        from io import StringIO

        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            pad_edge=True,
        )
        table.add_column("Store", style="bold", min_width=12)
        table.add_column("Price", justify="right", min_width=12)
        table.add_column("Stock", justify="center", min_width=8)
        table.add_column("Note", min_width=16)

        stock_icons = {
            "in_stock": "✅",
            "out_of_stock": "❌ OOS",
            "limited": "⚠️",
            "unknown": "❓",
        }

        # Only iterate over stores with valid prices
        for r in priced:
            site = r.get("site", "?")
            price = r["price"]
            stock = r.get("stock_status", "unknown")

            price_str = f"[bold green]${price:,.2f}[/bold green]"
            stock_str = stock_icons.get(stock, "❓")
            note = ""

            if price == best_price:
                note = "[bold green]✅ BEST PRICE[/bold green]"

            table.add_row(site, price_str, stock_str, note)

        # Summary lines
        lines = []
        if product_name:
            lines.append(f"[bold cyan]📦 {product_name}[/bold cyan]\n")

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=72)

        if lines:
            console.print("\n".join(lines))
        console.print(table)

        # Best price and spread summary
        if len(priced) >= 2:
            best_site = next(r["site"] for r in priced if r["price"] == best_price)
            max_price = max(r["price"] for r in priced)
            spread = max_price - best_price
            spread_pct = (spread / best_price) * 100 if best_price > 0 else 0

            console.print(f"\n💰 Best Price: [bold green]${best_price:,.2f}[/bold green] ({best_site})")
            console.print(f"📊 Price Spread: ${spread:,.2f} ({spread_pct:.1f}%)")
        else:
            best_site = next(r["site"] for r in priced if r["price"] == best_price)
            console.print(f"\n💰 Best Price: [bold green]${best_price:,.2f}[/bold green] ({best_site})")

        return buf.getvalue()

    except ImportError:
        # Plain text fallback — only show priced results
        lines = [f"  🏪 MARKET OVERVIEW — {product_name}", ""]
        for r in priced:
            lines.append(f"  {r.get('site', '?'):12s}  ${r['price']:>10,.2f}  {r.get('stock_status', '?')}")
        return "\n".join(lines)


def format_cross_site_scalper_alert(
    product_name: str,
    min_price: float,
    min_site: str,
    max_price: float,
    max_site: str,
    spread_pct: float,
) -> str:
    """Format a cross-site scalper warning when price spread is suspicious."""
    try:
        from rich.panel import Panel
        from rich.console import Console
        from io import StringIO

        lines = [
            f"[bold red]🚨 CROSS-SITE SCALPER ALERT[/bold red]",
            f"\n📦 {product_name}",
            f"💰 Lowest: [bold green]${min_price:,.2f}[/bold green] ({min_site})",
            f"💸 Highest: [bold red]${max_price:,.2f}[/bold red] ({max_site})",
            f"📈 Spread: [bold red]{spread_pct:.1f}%[/bold red]",
            f"\n⚠️ {max_site} is charging {spread_pct:.1f}% more than {min_site}!",
            "   This is a strong indicator of scalper pricing.",
        ]

        if spread_pct >= 50:
            border = "bold red"
            title = "🚨 SCALPER ALERT — EXTREME MARKUP"
        elif spread_pct >= 30:
            border = "bold yellow"
            title = "⚠️ SUSPICIOUS PRICE SPREAD"
        else:
            border = "yellow"
            title = "🟡 PRICE VARIANCE DETECTED"

        content = "\n".join(lines)

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=72)
        panel = Panel(content, title=title, border_style=border, width=72, padding=(1, 2))
        console.print(panel)
        return buf.getvalue()

    except ImportError:
        return f"🚨 SCALPER ALERT: {max_site} (${max_price:,.2f}) is {spread_pct:.1f}% above {min_site} (${min_price:,.2f})"

def format_portfolio_table(products: list, lifetime_savings: float) -> str:
    """Format the user's tracking portfolio with ROI and lifetime savings."""
    try:
        from rich.table import Table
        from rich.console import Console
        from rich.panel import Panel
        from io import StringIO

        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            box=None,
        )
        table.add_column("ID", style="dim")
        table.add_column("Asset", style="bold cyan")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right", style="bold green")
        table.add_column("P/L", justify="right")
        table.add_column("Score", justify="center")

        for p in products:
            curr = p.get('current_price', 0)
            orig = p.get('original_price', curr)
            pl = orig - curr
            pl_str = f"[green]+${pl:,.2f}[/green]" if pl > 0 else f"[red]${pl:,.2f}[/red]"
            
            score = p.get('deal_score', 0)
            if score >= 80:
                score_str = f"[bold green]{score}[/bold green]"
            elif score >= 60:
                score_str = f"[bold cyan]{score}[/bold cyan]"
            else:
                score_str = f"[dim]{score}[/dim]"

            table.add_row(
                str(p['id']),
                p['name'][:35],
                f"${orig:,.2f}",
                f"${curr:,.2f}",
                pl_str,
                score_str
            )

        summary = f"\n💎 [bold gold1]LIFETIME SAVINGS: ${lifetime_savings:,.2f}[/bold gold1]"
        
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=100)
        console.print(Panel(table, title="💼 [bold]INVESTMENT PORTFOLIO[/bold]", border_style="magenta"))
        console.print(summary)
        return buf.getvalue()

    except ImportError:
        return f"Portfolio: {len(products)} assets. Lifetime Savings: ${lifetime_savings:,.2f}"


def format_trend_report(
    product_name: str,
    direction: str,
    direction_emoji: str,
    predicted_change_pct: float,
    predicted_price: Optional[float],
    confidence: str,
    analysis_text: str,
    volatility_warning: bool = False,
) -> str:
    """Format a trend prediction report with volatility support."""
    try:
        from rich.panel import Panel
        from rich.console import Console
        from io import StringIO

        lines = [
            f"[bold]📊 Trend Analysis: {product_name}[/bold]",
            "",
            f"{direction_emoji} Direction: [bold]{direction}[/bold]",
            f"📈 Projected change: {predicted_change_pct:+.1f}%",
        ]
        
        if volatility_warning:
            lines.insert(2, "[bold red]⚠️ HIGH VOLATILITY - USE CAUTION[/bold red]")

        if predicted_price:
            lines.append(f"💰 Projected price: ${predicted_price:,.2f}")
        lines.append(f"🎯 Confidence: {confidence}")
        lines.append(f"\n{analysis_text}")

        content = "\n".join(lines)
        border = "red" if volatility_warning else ("cyan" if direction == "DOWN" else "yellow" if direction == "UP" else "dim")
        title = f"{direction_emoji} TREND PREDICTION"

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=70)
        panel = Panel(content, title=title, border_style=border, width=70, padding=(1, 2))
        console.print(panel)
        return buf.getvalue()

    except ImportError:
        warn = " [HIGH VOLATILITY]" if volatility_warning else ""
        return f"[{direction}]{warn} {product_name}: {analysis_text}"


def format_full_report(
    product_name: str,
    current_price: float,
    target_price: Optional[float],
    deal_score: int,
    deal_label: str,
    scalper_risk: str,
    scalper_text: str,
    trend_direction: str,
    trend_text: str,
    url: str = "",
    market_prices: Optional[list] = None,
    volatility_warning: bool = False,
) -> str:
    """Format a comprehensive investment-advisor-style alarm report.

    Includes: Market Comparison table, Savings Highlight,
    BUY NOW / WAIT recommendation, Scalper & Trend intelligence.
    """
    try:
        from rich.panel import Panel
        from rich.console import Console
        from rich.text import Text
        from io import StringIO

        # ASCII art header
        header = """
   ██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗
   ██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝
   ███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗
   ██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║
   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║
   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝
              ⚡ U L T R A  A L A R M ⚡"""

        lines = [
            f"[bold gold1]{header}[/bold gold1]",
            "",
            f"[bold cyan]📦 {product_name}[/bold cyan]",
            f"💰 Best Price: [bold green]${current_price:,.2f}[/bold green]",
        ]
        if target_price:
            if current_price <= target_price:
                diff = target_price - current_price
                lines.append(f"🎯 Target: ${target_price:,.2f} [bold green](✅ ${diff:,.2f} below!)[/bold green]")
            else:
                diff = current_price - target_price
                lines.append(f"🎯 Target: ${target_price:,.2f} [yellow](${diff:,.2f} above)[/yellow]")

        if volatility_warning:
            lines.append("\n[bold red]⚠️ HIGH VOLATILITY - USE CAUTION[/bold red]")
            lines.append("[red]Price is fluctuating significantly. Trend may be unreliable.[/red]")

        # ── MARKET COMPARISON MINI-TABLE ──
        if market_prices:
            priced = [r for r in market_prices if r.get("price") and r["price"] > 0]
            lines.append(f"\n{'─' * 50}")
            if priced:
                lines.append("[bold]🏪 Market Comparison[/bold]")
                # Only show stores that returned valid prices
                for r in priced:
                    site = r.get("site", "?")
                    price = r["price"]
                    badge = " [bold green]← BEST[/bold green]" if price == current_price else ""
                    lines.append(f"   {site:12s}  [green]${price:>10,.2f}[/green]{badge}")

                # ── SAVINGS HIGHLIGHT ──
                if len(priced) >= 2:
                    avg_price = sum(r["price"] for r in priced) / len(priced)
                    savings_vs_avg = avg_price - current_price
                    savings_pct = (savings_vs_avg / avg_price) * 100 if avg_price > 0 else 0

                    lines.append("")
                    if savings_vs_avg > 0:
                        lines.append(
                            f"💎 [bold green]You save ${savings_vs_avg:,.2f} ({savings_pct:.1f}%) "
                            f"vs market average (${avg_price:,.2f})[/bold green]"
                        )
                    elif savings_vs_avg < 0:
                        lines.append(
                            f"⚠️ [yellow]${abs(savings_vs_avg):,.2f} ({abs(savings_pct):.1f}%) above "
                            f"market average (${avg_price:,.2f})[/yellow]"
                        )
                    else:
                        lines.append(f"📊 At market average (${avg_price:,.2f})")
            else:
                lines.append("[dim]🔍 Scanning global markets... All sources currently shielded. Retrying in background.[/dim]")

        # ── DEAL SCORE ──
        lines.append(f"\n{'─' * 50}")
        filled = deal_score // 5
        empty = 20 - filled
        if deal_score >= 80:
            bar_color = "green"
        elif deal_score >= 60:
            bar_color = "cyan"
        elif deal_score >= 30:
            bar_color = "yellow"
        else:
            bar_color = "red"
        bar = f"[{bar_color}]{'█' * filled}{'░' * empty}[/{bar_color}] {deal_score}/100"
        lines.append(f"📊 Deal Score: {bar}")
        lines.append(f"   {deal_label}")

        # ── BUY / WAIT RECOMMENDATION ──
        lines.append(f"\n{'─' * 50}")
        if deal_score >= 80:
            lines.append("[bold green]🟢 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]")
            lines.append("[bold green]   ✅ RECOMMENDATION: BUY NOW                  [/bold green]")
            lines.append("[bold green]   This is an exceptional deal — act fast!       [/bold green]")
            lines.append("[bold green]🟢 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]")
        elif deal_score >= 60:
            lines.append("[bold cyan]🔵 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            lines.append("[bold cyan]   👍 RECOMMENDATION: GOOD DEAL — Consider It   [/bold cyan]")
            lines.append("[bold cyan]🔵 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        elif deal_score >= 30:
            lines.append("[yellow]🟡 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/yellow]")
            lines.append("[yellow]   ⏳ RECOMMENDATION: WAIT                       [/yellow]")
            lines.append("[yellow]   Price may drop further — monitor for changes.  [/yellow]")
            lines.append("[yellow]🟡 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/yellow]")
        else:
            lines.append("[bold red]🔴 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")
            lines.append("[bold red]   🚫 RECOMMENDATION: DO NOT BUY               [/bold red]")
            lines.append("[bold red]   This is overpriced — wait for a price drop.  [/bold red]")
            lines.append("[bold red]🔴 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")

        # ── SCALPER ANALYSIS ──
        lines.append(f"\n{'─' * 50}")
        lines.append(f"🔍 Scalper Analysis: [bold]{scalper_risk}[/bold]")
        lines.append(f"   {scalper_text}")

        # ── TREND INTELLIGENCE ──
        lines.append(f"\n{'─' * 50}")
        lines.append(f"📈 Trend: [bold]{trend_direction}[/bold]")
        lines.append(f"   {trend_text}")

        if url:
            lines.append(f"\n🔗 {url}")

        content = "\n".join(lines)

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=72)
        panel = Panel(
            content,
            title="[bold gold1]⚡ HERMES ULTRA — INTELLIGENCE REPORT ⚡[/bold gold1]",
            border_style="bold gold1",
            width=72,
            padding=(1, 2),
        )
        console.print(panel)
        return buf.getvalue()

    except ImportError:
        lines = [
            "═" * 55,
            "  ⚡ HERMES ULTRA — INTELLIGENCE REPORT ⚡",
            "═" * 55,
            f"  📦 {product_name}",
            f"  💰 Best Price: ${current_price:,.2f}",
            f"  📊 Score: {deal_score}/100 — {deal_label}",
        ]
        if deal_score >= 80:
            lines.append("  ✅ RECOMMENDATION: BUY NOW")
        else:
            lines.append("  ⏳ RECOMMENDATION: WAIT")
        lines.extend([
            f"  🔍 Scalper: {scalper_risk} — {scalper_text}",
            f"  📈 Trend: {trend_direction} — {trend_text}",
            "═" * 55,
        ])
        return "\n".join(lines)

