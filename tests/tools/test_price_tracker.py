"""Tests for the Hermes Ultra price tracker toolset.

Tests real logic — no network calls, no mocks of external services.
Follows the patterns from test_homeassistant_tool.py and test_registry.py.
"""

import json
import os
import sqlite3
import tempfile
import time

import pytest

# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

from tools.price_tracker.database import (
    PriceTrackerDB,
    Product,
    PriceRecord,
    Alert,
)


@pytest.fixture
def db(tmp_path):
    """Return a fresh PriceTrackerDB using a temp directory."""
    db_path = str(tmp_path / "test_price_tracker.db")
    return PriceTrackerDB(db_path=db_path)


class TestDatabaseProducts:
    def test_add_and_get_product(self, db):
        p = db.add_product(Product(url="https://amazon.com.tr/dp/TEST1", name="Test Ürün"))
        assert p.id is not None
        assert p.id > 0

        fetched = db.get_product(p.id)
        assert fetched is not None
        assert fetched.url == "https://amazon.com.tr/dp/TEST1"
        assert fetched.name == "Test Ürün"

    def test_get_product_by_url(self, db):
        db.add_product(Product(url="https://amazon.com.tr/dp/BY_URL", name="URL Ürün"))
        fetched = db.get_product_by_url("https://amazon.com.tr/dp/BY_URL")
        assert fetched is not None
        assert fetched.name == "URL Ürün"

    def test_get_product_not_found(self, db):
        assert db.get_product(9999) is None

    def test_get_product_by_url_not_found(self, db):
        assert db.get_product_by_url("https://nonexistent.com") is None

    def test_duplicate_url_rejected(self, db):
        db.add_product(Product(url="https://amazon.com.tr/dp/DUP"))
        with pytest.raises(sqlite3.IntegrityError):
            db.add_product(Product(url="https://amazon.com.tr/dp/DUP"))

    def test_list_products_empty(self, db):
        assert db.list_products() == []

    def test_list_products_order(self, db):
        db.add_product(Product(url="https://a.com/1", name="First", created_at=100.0))
        db.add_product(Product(url="https://a.com/2", name="Second", created_at=200.0))
        products = db.list_products()
        assert len(products) == 2
        assert products[0].name == "Second"  # newest first

    def test_update_product_price(self, db):
        p = db.add_product(Product(url="https://amazon.com.tr/dp/UPDATE"))
        db.update_product_price(p.id, 999.99, stock_status="in_stock")
        updated = db.get_product(p.id)
        assert updated.current_price == 999.99
        assert updated.stock_status == "in_stock"
        assert updated.last_checked is not None

    def test_delete_product(self, db):
        p = db.add_product(Product(url="https://amazon.com.tr/dp/DELETE"))
        assert db.delete_product(p.id) is True
        assert db.get_product(p.id) is None

    def test_delete_nonexistent(self, db):
        assert db.delete_product(9999) is False

    def test_product_to_dict(self):
        p = Product(id=1, url="https://x.com", name="Test")
        d = p.to_dict()
        assert d["id"] == 1
        assert d["url"] == "https://x.com"


class TestDatabasePriceHistory:
    def test_price_history_recorded(self, db):
        p = db.add_product(Product(url="https://amazon.com.tr/dp/HIST"))
        db.update_product_price(p.id, 100.0)
        db.update_product_price(p.id, 95.0)
        db.update_product_price(p.id, 90.0)

        history = db.get_price_history(p.id)
        assert len(history) == 3
        # Newest first
        assert history[0].price == 90.0
        assert history[2].price == 100.0

    def test_price_history_limit(self, db):
        p = db.add_product(Product(url="https://amazon.com.tr/dp/LIMIT"))
        for i in range(10):
            db.update_product_price(p.id, float(100 + i))
        history = db.get_price_history(p.id, limit=3)
        assert len(history) == 3

    def test_empty_history(self, db):
        p = db.add_product(Product(url="https://amazon.com.tr/dp/EMPTY"))
        history = db.get_price_history(p.id)
        assert history == []


class TestDatabaseAlerts:
    def test_add_and_get_alert(self, db):
        p = db.add_product(Product(url="https://amazon.com.tr/dp/ALERT"))
        alert = db.add_alert(Alert(product_id=p.id, alert_type="target_price", threshold=500.0))
        assert alert.id is not None

        alerts = db.get_active_alerts(p.id)
        assert len(alerts) == 1
        assert alerts[0].alert_type == "target_price"
        assert alerts[0].threshold == 500.0

    def test_deactivate_alert(self, db):
        p = db.add_product(Product(url="https://amazon.com.tr/dp/DEACT"))
        alert = db.add_alert(Alert(product_id=p.id, alert_type="price_drop"))
        assert db.deactivate_alert(alert.id) is True
        assert db.get_active_alerts(p.id) == []

    def test_deactivate_nonexistent(self, db):
        assert db.deactivate_alert(9999) is False

    def test_get_all_active_alerts(self, db):
        p1 = db.add_product(Product(url="https://a.com/p1"))
        p2 = db.add_product(Product(url="https://a.com/p2"))
        db.add_alert(Alert(product_id=p1.id, alert_type="price_drop"))
        db.add_alert(Alert(product_id=p2.id, alert_type="target_price", threshold=100.0))
        all_alerts = db.get_active_alerts()
        assert len(all_alerts) == 2

    def test_cascade_delete(self, db):
        p = db.add_product(Product(url="https://amazon.com.tr/dp/CASCADE"))
        db.update_product_price(p.id, 100.0)
        db.add_alert(Alert(product_id=p.id, alert_type="price_drop"))
        db.delete_product(p.id)
        assert db.get_price_history(p.id) == []
        assert db.get_active_alerts(p.id) == []


# ---------------------------------------------------------------------------
# Deal scoring tests
# ---------------------------------------------------------------------------

from tools.price_tracker.scoring import DealScorer, DealScore


class TestDealScorer:
    def setup_method(self):
        self.scorer = DealScorer()

    def test_no_price_returns_unknown(self):
        score = self.scorer.calculate(current_price=None)
        assert score.total_score == 0
        assert "Unknown" in score.label

    def test_zero_price_returns_unknown(self):
        score = self.scorer.calculate(current_price=0)
        assert score.total_score == 0

    def test_at_target_price_high_score(self):
        score = self.scorer.calculate(current_price=500, target_price=500)
        assert score.discount_score >= 70

    def test_below_target_price_highest_score(self):
        score = self.scorer.calculate(current_price=400, target_price=500)
        assert score.discount_score >= 80

    def test_above_target_price_lower_score(self):
        score = self.scorer.calculate(current_price=800, target_price=500)
        assert score.discount_score < 50

    def test_no_target_neutral(self):
        score = self.scorer.calculate(current_price=500, target_price=None)
        assert score.discount_score == 50

    def test_price_trend_down_bonus(self):
        history = [200, 180, 160, 150, 140]  # price dropping
        score = self.scorer.calculate(
            current_price=100, price_history=history
        )
        assert score.trend_score >= 70

    def test_price_trend_up_penalty(self):
        history = [100, 120, 140, 160, 180]  # price rising
        score = self.scorer.calculate(
            current_price=200, price_history=history
        )
        assert score.trend_score <= 40

    def test_limited_stock_urgency(self):
        score = self.scorer.calculate(current_price=500, stock_status="limited")
        assert score.stock_score >= 80

    def test_out_of_stock_low(self):
        score = self.scorer.calculate(current_price=500, stock_status="out_of_stock")
        assert score.stock_score <= 20

    def test_big_discount_from_original(self):
        score = self.scorer.calculate(
            current_price=500, original_price=1000
        )
        assert score.market_score >= 80

    def test_no_discount_from_original(self):
        score = self.scorer.calculate(
            current_price=1000, original_price=1000
        )
        assert score.market_score <= 50

    def test_total_score_bounded(self):
        score = self.scorer.calculate(
            current_price=100, target_price=500,
            original_price=2000, stock_status="limited",
            price_history=[500, 400, 300, 200, 150],
        )
        assert 0 <= score.total_score <= 100

    def test_score_has_label(self):
        score = self.scorer.calculate(current_price=500, target_price=1000)
        assert score.label  # Not empty

    def test_score_has_explanation(self):
        score = self.scorer.calculate(current_price=500, target_price=600)
        assert score.explanation  # Not empty


# ---------------------------------------------------------------------------
# Scalper detection tests
# ---------------------------------------------------------------------------

from tools.price_tracker.scalper_detector import ScalperDetector, ScalperReport


class TestScalperDetector:
    def setup_method(self):
        self.detector = ScalperDetector()

    def test_no_history_no_risk(self):
        report = self.detector.check(current_price=100)
        assert report.risk_level == "NONE"
        assert not report.is_suspicious

    def test_normal_price_no_risk(self):
        history = [100, 105, 98, 102, 99]
        report = self.detector.check(current_price=103, price_history=history)
        assert report.risk_level == "NONE"
        assert not report.is_suspicious

    def test_moderate_inflation_medium_risk(self):
        history = [100, 100, 100, 100, 100]
        report = self.detector.check(current_price=120, price_history=history)
        assert report.risk_level == "MEDIUM"
        assert report.is_suspicious

    def test_severe_inflation_high_risk(self):
        history = [100, 100, 100, 100, 100]
        report = self.detector.check(current_price=200, price_history=history)
        assert report.risk_level == "HIGH"
        assert report.is_suspicious
        assert "SCALPER" in report.analysis_text.upper()

    def test_low_inflation_low_risk(self):
        history = [100, 100, 100, 100, 100]
        report = self.detector.check(current_price=110, price_history=history)
        assert report.risk_level == "LOW"

    def test_uses_original_price_when_no_history(self):
        report = self.detector.check(
            current_price=300, original_price=100
        )
        assert report.is_suspicious
        assert report.risk_level == "HIGH"

    def test_invalid_price(self):
        report = self.detector.check(current_price=0)
        assert "invalid" in report.analysis_text.lower()

    def test_spike_detection(self):
        # Stable history then sudden jump
        history = [100, 100, 100, 100, 100, 100, 100]
        report = self.detector.check(current_price=150, price_history=history)
        assert report.price_spike_detected is True

    def test_deviation_calculation(self):
        history = [100, 100, 100]
        report = self.detector.check(current_price=150, price_history=history)
        assert report.deviation_pct == pytest.approx(50.0, abs=0.1)
        assert report.avg_price == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# Trend predictor tests
# ---------------------------------------------------------------------------

from tools.price_tracker.trend_predictor import TrendPredictor, TrendReport


class TestTrendPredictor:
    def setup_method(self):
        self.predictor = TrendPredictor()

    def test_insufficient_data(self):
        report = self.predictor.predict([100, 200])
        assert "data points" in report.analysis_text.lower()

    def test_falling_prices(self):
        # Newest first: prices have been dropping
        history = [80, 90, 100, 110, 120, 130, 140, 150]
        report = self.predictor.predict(history, current_price=80)
        assert report.direction == "DOWN"
        assert report.predicted_change_pct < 0

    def test_rising_prices(self):
        # Newest first: prices have been rising
        history = [150, 140, 130, 120, 110, 100, 90, 80]
        report = self.predictor.predict(history, current_price=150)
        assert report.direction == "UP"
        assert report.predicted_change_pct > 0

    def test_stable_prices(self):
        history = [100, 101, 99, 100, 101, 99, 100, 101]
        report = self.predictor.predict(history, current_price=100)
        assert report.direction == "STABLE"

    def test_predicted_price_non_negative(self):
        history = [10, 20, 30, 40, 50]
        report = self.predictor.predict(history, current_price=10)
        assert report.predicted_price is not None
        assert report.predicted_price >= 0

    def test_data_points_counted(self):
        history = [100, 200, 300, 400, 500]
        report = self.predictor.predict(history)
        assert report.data_points == 5

    def test_moving_avg_calculated(self):
        history = [100] * 10
        report = self.predictor.predict(history)
        assert report.moving_avg_7 is not None
        assert report.moving_avg_7 == pytest.approx(100.0)

    def test_confidence_with_few_points(self):
        history = [100, 200, 300]
        report = self.predictor.predict(history)
        assert report.confidence == "low"

    def test_analysis_text_not_empty(self):
        history = [100, 200, 300, 400]
        report = self.predictor.predict(history)
        assert report.analysis_text


# ---------------------------------------------------------------------------
# Price utilities tests
# ---------------------------------------------------------------------------

from tools.price_tracker.parsers.price_utils import parse_price, detect_currency, extract_json_ld, extract_meta
from tools.price_tracker.parsers.amazon_global import AmazonGlobalParser
from tools.price_tracker.parsers.generic import GenericParser
from tools.price_tracker.parsers.ebay_global import EbayGlobalParser
from tools.price_tracker.parsers.price_comparison import IdealoParser, PriceSpyParser, CamelParser
import pytest
from unittest.mock import patch


class TestPriceUtils:
    def test_parse_turkish_price(self):
        assert parse_price("1.299,00 TL", "TRY") == 1299.0

    def test_parse_us_price(self):
        assert parse_price("$1,299.00", "USD") == 1299.0

    def test_parse_euro_price(self):
        assert parse_price("1.299,00 \u20ac", "EUR") == 1299.0

    def test_parse_gbp_price(self):
        assert parse_price("\u00a31,299.00", "GBP") == 1299.0

    def test_parse_auto_turkish(self):
        assert parse_price("1.299,00") == 1299.0

    def test_parse_auto_english(self):
        assert parse_price("1,299.00") == 1299.0

    def test_parse_none(self):
        assert parse_price("") is None
        assert parse_price(None) is None

    def test_detect_currency_amazon_tr(self):
        assert detect_currency("https://www.amazon.com.tr/dp/B123") == "USD"

    def test_detect_currency_amazon_de(self):
        assert detect_currency("https://www.amazon.de/dp/B123") == "EUR"

    def test_detect_currency_amazon_uk(self):
        assert detect_currency("https://www.amazon.co.uk/dp/B123") == "GBP"

    def test_detect_currency_amazon_us(self):
        assert detect_currency("https://www.amazon.com/dp/B123") == "USD"

    def test_extract_json_ld(self):
        html = '<script type="application/ld+json">{"@type": "Product", "name": "Test"}</script>'
        blocks = extract_json_ld(html)
        assert len(blocks) == 1
        assert blocks[0]["name"] == "Test"

    def test_extract_meta(self):
        html = '<meta property="og:title" content="My Product">'
        assert extract_meta(html, "title") == "My Product"


class TestAmazonGlobalParser:
    def setup_method(self):
        self.parser = AmazonGlobalParser()

    def test_site_name(self):
        assert self.parser.get_site_name() == "Amazon (Global)"

    def test_domains(self):
        domains = self.parser.get_domains()
        assert "amazon.com" in domains

    def test_can_handle_amazon_global(self):
        assert self.parser.can_handle("https://www.amazon.com/dp/B09XXXX")
        assert self.parser.can_handle("https://www.amazon.co.uk/dp/B09XXXX")
        assert self.parser.can_handle("https://www.amazon.de/dp/B09XXXX")

    def test_cannot_handle_other(self):
        assert not self.parser.can_handle("https://trendyol.com/p-12345")

    def test_parse_product_title(self):
        html = '<span id="productTitle"> Samsung Galaxy S24 Ultra </span>'
        data = self.parser.parse(html)
        assert "Samsung Galaxy S24 Ultra" in data.name

    def test_parse_price_whole_fraction(self):
        html = '''
            <span class="a-price-whole">1.299</span>
            <span class="a-price-fraction">00</span>
        '''
        data = self.parser.parse(html)
        assert data.price == 1299.0

    def test_parse_in_stock(self):
        html = '<div id="availability"><span>Stokta var.</span></div>'
        data = self.parser.parse(html)
        assert data.stock_status == "in_stock"

    def test_parse_out_of_stock(self):
        html = '<div id="availability"><span>Stokta yok</span></div>'
        data = self.parser.parse(html)
        assert data.stock_status == "out_of_stock"

    def test_parse_empty_html(self):
        data = self.parser.parse("")
        assert data.name == ""
        assert data.price is None

    def test_parse_seller_amazon(self):
        html = '<div>Ships from and sold by Amazon.com</div>'
        data = self.parser.parse(html, url="https://www.amazon.com/dp/B123")
        assert "Amazon" in data.seller

    def test_parse_seller_various(self):
        parser = AmazonGlobalParser()
        html = '<div><a id="sellerProfileTriggerId">Other Sellers</a></div>'
        data = parser.parse(html)
        assert "Other Sellers" in data.seller





# ---------------------------------------------------------------------------
# Searcher Tests
# ---------------------------------------------------------------------------

class TestProductSearcher:
    @patch("tools.price_tracker.searcher.StealthScraper")
    def test_search_amazon_success(self, mock_scraper_cls):
        from tools.price_tracker.searcher import ProductSearcher
        
        mock_instance = mock_scraper_cls.return_value
        mock_instance.scrape.return_value.success = True
        mock_instance.scrape.return_value.html = '''
            <div>
                <a href="/Apple-iPhone-15-Pro-Max/dp/B000000000">Product</a>
            </div>
        '''
        
        searcher = ProductSearcher(scraper=mock_instance)
        result = searcher._search_amazon("iphone 15 pro max")
        
        assert result is not None
        assert result["success"] is True
        assert result["site"] == "Amazon"
        assert result["url"] == "https://www.amazon.com/Apple-iPhone-15-Pro-Max/dp/B000000000"

    @patch("tools.price_tracker.searcher.StealthScraper")
    def test_search_fallback_to_ebay(self, mock_scraper_cls):
        from tools.price_tracker.searcher import ProductSearcher
        
        mock_instance = mock_scraper_cls.return_value
        
        def side_effect(url):
            from tools.price_tracker.scraper import ScrapeResult
            res = ScrapeResult(success=True, html="", status_code=200)
            if "amazon.com" in url:
                res.html = "captcha detected"
            else:
                res.html = '<a href="https://www.ebay.com/itm/123456789">Link</a>'
            return res
            
        mock_instance.scrape.side_effect = side_effect
        
        searcher = ProductSearcher(scraper=mock_instance)
        result = searcher.search("iphone 15")
        
        assert result["success"] is True
        assert result["site"] == "eBay"
        assert result["url"] == "https://www.ebay.com/itm/123456789"


# ---------------------------------------------------------------------------
# GenericParser tests
# ---------------------------------------------------------------------------

class TestGenericParser:
    def setup_method(self):
        self.parser = GenericParser()

    def test_always_can_handle(self):
        assert self.parser.can_handle("https://any-site.com/product/123")
        assert self.parser.can_handle("https://unknown.org")

    def test_parse_json_ld_product(self):
        html = '''<script type="application/ld+json">
        {"@type": "Product", "name": "Test Product",
         "offers": {"price": "99.99", "priceCurrency": "USD",
                    "availability": "https://schema.org/InStock"},
         "brand": {"name": "TestBrand"}}
        </script>'''
        data = self.parser.parse(html, url="https://www.walmart.com/ip/123")
        assert data.name == "Test Product"
        assert data.price == 99.99
        assert data.currency == "USD"
        assert data.stock_status == "in_stock"
        assert data.seller == "TestBrand"

    def test_parse_og_meta_tags(self):
        html = '''<meta property="og:title" content="OG Product">
        <meta property="product:price:amount" content="49.99">
        <meta property="product:price:currency" content="EUR">'''
        data = self.parser.parse(html)
        assert data.name == "OG Product"
        assert data.price == 49.99
        assert data.currency == "EUR"

    def test_parse_microdata(self):
        html = '<span itemprop="name">Micro Product</span><span itemprop="price" content="29.99">29.99</span>'
        data = self.parser.parse(html)
        assert data.name == "Micro Product"
        assert data.price == 29.99

    def test_parse_empty_html(self):
        data = self.parser.parse("")
        assert data.name == ""
        assert data.price is None


# ---------------------------------------------------------------------------
# eBay Global Parser tests
# ---------------------------------------------------------------------------

class TestEbayGlobalParser:
    def setup_method(self):
        self.parser = EbayGlobalParser()

    def test_can_handle_ebay_domains(self):
        assert self.parser.can_handle("https://www.ebay.com/itm/123")
        assert self.parser.can_handle("https://www.ebay.co.uk/itm/456")
        assert self.parser.can_handle("https://www.ebay.de/itm/789")
        assert not self.parser.can_handle("https://www.amazon.com/dp/B123")

    def test_parse_json_ld(self):
        html = '''<script type="application/ld+json">
        {"@type": "Product", "name": "eBay Item",
         "offers": {"price": "199.99", "priceCurrency": "USD",
                    "availability": "https://schema.org/InStock"}}
        </script>'''
        data = self.parser.parse(html, url="https://www.ebay.com/itm/123")
        assert data.name == "eBay Item"
        assert data.price == 199.99


# ---------------------------------------------------------------------------
# Turkish Sites Parser tests
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Price Comparison Parser tests
# ---------------------------------------------------------------------------

class TestPriceComparisonParser:
    def test_idealo_can_handle(self):
        assert IdealoParser().can_handle("https://www.idealo.de/product/123")
        assert not IdealoParser().can_handle("https://www.amazon.de/dp/B123")

    def test_pricespy_can_handle(self):
        assert PriceSpyParser().can_handle("https://pricespy.co.uk/product/123")
        assert PriceSpyParser().can_handle("https://www.pricerunner.com/product/123")

    def test_camel_can_handle(self):
        assert CamelParser().can_handle("https://camelcamelcamel.com/product/B123")
        assert not CamelParser().can_handle("https://www.amazon.com/dp/B123")


# ---------------------------------------------------------------------------
# Alert formatting tests (no Rich dependency required)
# ---------------------------------------------------------------------------

from tools.price_tracker.alerts import (
    _plain_deal_alert,
    format_deal_alert,
    format_scalper_warning,
    format_price_table,
    format_trend_report,
)


class TestAlertFormatting:
    def test_plain_deal_alert_contains_info(self):
        text = _plain_deal_alert(
            "Test Ürün", 999.99, 1500.0, 85, "🔥 Kaçırılmaz", "https://x.com"
        )
        assert "Test Ürün" in text
        assert "999" in text
        assert "85" in text

    def test_format_deal_alert_renders(self):
        # Should not raise
        text = format_deal_alert("Test", 100.0, 200.0, 90, "🔥 Kaçırılmaz", "https://x.com")
        assert text  # Non-empty
        assert "Test" in text

    def test_format_scalper_warning_renders(self):
        text = format_scalper_warning(
            "Test", "YÜKSEK", "🚨", 60.0, 100.0, 160.0, "Karaborsa riski!"
        )
        assert text
        assert "Test" in text

    def test_format_price_table_renders(self):
        products = [
            {"name": "P1", "current_price": 100.0, "target_price": 150.0,
             "stock_status": "in_stock", "site": "Amazon TR"},
        ]
        text = format_price_table(products)
        assert text
        assert "P1" in text

    def test_format_trend_report_renders(self):
        text = format_trend_report(
            "Test", "DÜŞÜŞ", "📉", -10.5, 900.0, "orta", "Fiyat düşecek."
        )
        assert text
        assert "Test" in text


# ---------------------------------------------------------------------------
# Parser registry tests
# ---------------------------------------------------------------------------

from tools.price_tracker.parsers import get_parser, list_supported_sites


class TestParserRegistry:
    def test_amazon_tr_parser_found(self):
        parser = get_parser("https://www.amazon.com.tr/dp/B09XXXX")
        assert parser is not None
        assert parser.get_site_name() == "Amazon (Global)"

    def test_amazon_us_parser_found(self):
        parser = get_parser("https://www.amazon.com/dp/B09XXXX")
        assert parser is not None
        assert parser.get_site_name() == "Amazon (Global)"

    def test_ebay_parser_found(self):
        parser = get_parser("https://www.ebay.com/itm/123456")
        assert parser is not None
        assert parser.get_site_name() == "eBay (Global)"

    def test_list_supported_sites(self):
        sites = list_supported_sites()
        assert len(sites) >= 5  # specific parsers
        names = [s["site"] for s in sites]
        assert "Amazon (Global)" in names
        assert "eBay (Global)" in names
        assert "Idealo" in names


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------

class TestToolRegistration:
    @classmethod
    def setup_class(cls):
        """Ensure tool registration has happened."""
        import tools.price_tracker_tool  # noqa: F401 — triggers registry.register() calls

    def test_tools_registered_in_registry(self):
        from tools.registry import registry
        names = registry.get_all_tool_names()
        for tool in (
            "price_track", "price_check", "price_history",
            "price_alert_config", "price_watcher_start", "price_watcher_stop",
            "price_search_and_track", # New tool
        ):
            assert tool in names, f"{tool} not found in registry"

    def test_tools_in_price_tracker_toolset(self):
        from tools.registry import registry
        toolset_map = registry.get_tool_to_toolset_map()
        for tool in (
            "price_track", "price_check", "price_history",
            "price_alert_config", "price_watcher_start", "price_watcher_stop",
            "price_search_and_track", # New tool
        ):
            assert toolset_map[tool] == "price_tracker"

    def test_check_fn_always_available(self):
        from tools.registry import registry
        defs = registry.get_definitions({
            "price_track", "price_check", "price_history",
            "price_alert_config", "price_watcher_start", "price_watcher_stop",
            "price_search_and_track", # New tool
        })
        assert len(defs) == 7


# ---------------------------------------------------------------------------
# Handler validation tests
# ---------------------------------------------------------------------------

import json
import tools.price_tracker_tool


class TestHandlerValidation:
    def test_price_track_missing_url(self):
        from tools.price_tracker_tool import _handle_price_track
        result = json.loads(_handle_price_track({}))
        assert "error" in result

    def test_price_track_empty_url(self):
        from tools.price_tracker_tool import _handle_price_track
        result = json.loads(_handle_price_track({"url": ""}))
        assert "error" in result

    def test_price_search_missing_query(self):
        result = tools.price_tracker_tool._handle_price_search_and_track({})
        data = json.loads(result)
        assert "error" in data
        assert "query" in data["error"].lower()

    def test_price_history_missing_id(self):
        from tools.price_tracker_tool import _handle_price_history
        result = json.loads(_handle_price_history({}))
        assert "error" in result

    def test_price_check_no_product(self):
        from tools.price_tracker_tool import _handle_price_check
        result = json.loads(_handle_price_check({"product_id": 99999}))
        assert "error" in result

    def test_alert_config_unknown_action(self):
        from tools.price_tracker_tool import _handle_price_alert_config
        result = json.loads(_handle_price_alert_config({"action": "destroy"}))
        assert "error" in result

    def test_watcher_stop_when_not_running(self):
        from tools.price_tracker_tool import _handle_watcher_stop
        result = json.loads(_handle_watcher_stop({}))
        assert result["status"] == "not_running"
