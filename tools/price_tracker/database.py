"""SQLite database for product tracking, price history, and alerts.

Thread-safe via check_same_thread=False. Database stored at
``~/.hermes/price_tracker.db`` (or HERMES_HOME override).
"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Product:
    id: Optional[int] = None
    url: str = ""
    name: str = ""
    site: str = ""
    target_price: Optional[float] = None
    current_price: Optional[float] = None
    original_price: Optional[float] = None
    last_checked: Optional[float] = None
    image_url: str = ""
    category: str = ""
    seller: str = ""
    stock_status: str = "unknown"  # in_stock, out_of_stock, limited, unknown
    created_at: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PriceRecord:
    id: Optional[int] = None
    product_id: int = 0
    price: float = 0.0
    original_price: Optional[float] = None
    stock_status: str = "unknown"
    seller: str = ""
    timestamp: Optional[float] = None


@dataclass
class Alert:
    id: Optional[int] = None
    product_id: int = 0
    alert_type: str = "price_drop"  # price_drop, target_price, stock_change, deal_score
    threshold: Optional[float] = None
    active: bool = True
    created_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

def _get_db_path() -> str:
    """Return the database file path, respecting HERMES_HOME."""
    hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return os.path.join(hermes_home, "price_tracker.db")


class PriceTrackerDB:
    """Thread-safe SQLite store for price tracking data."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _get_db_path()
        self._lock = threading.Lock()
        self._ensure_directory()
        self._init_db()

    def _ensure_directory(self):
        dirpath = os.path.dirname(self.db_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS products (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT NOT NULL UNIQUE,
                        name TEXT DEFAULT '',
                        site TEXT DEFAULT '',
                        target_price REAL,
                        current_price REAL,
                        original_price REAL,
                        last_checked REAL,
                        image_url TEXT DEFAULT '',
                        category TEXT DEFAULT '',
                        seller TEXT DEFAULT '',
                        stock_status TEXT DEFAULT 'unknown',
                        created_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS price_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        product_id INTEGER NOT NULL,
                        price REAL NOT NULL,
                        original_price REAL,
                        stock_status TEXT DEFAULT 'unknown',
                        seller TEXT DEFAULT '',
                        timestamp REAL NOT NULL,
                        FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        product_id INTEGER NOT NULL,
                        alert_type TEXT NOT NULL DEFAULT 'price_drop',
                        threshold REAL,
                        active INTEGER DEFAULT 1,
                        created_at REAL NOT NULL,
                        FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_price_history_product
                        ON price_history(product_id, timestamp);
                    CREATE INDEX IF NOT EXISTS idx_alerts_product
                        ON alerts(product_id, active);
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Settings / User Stats
    # ------------------------------------------------------------------

    def get_setting(self, key: str, default: str = "") -> str:
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
                return row["value"] if row else default
            finally:
                conn.close()

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, str(value)),
                )
                conn.commit()
            finally:
                conn.close()

    def get_lifetime_savings(self) -> float:
        val = self.get_setting("lifetime_savings", "0.0")
        try:
            return float(val)
        except ValueError:
            return 0.0

    def add_lifetime_savings(self, amount: float) -> float:
        current = self.get_lifetime_savings()
        new_val = current + amount
        self.set_setting("lifetime_savings", str(new_val))
        return new_val

    # ------------------------------------------------------------------
    # Product CRUD
    # ------------------------------------------------------------------

    def add_product(self, product: Product) -> Product:
        """Insert a new product. Returns the product with its new id."""
        now = time.time()
        product.created_at = product.created_at or now
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    """INSERT INTO products
                       (url, name, site, target_price, current_price, original_price,
                        last_checked, image_url, category, seller, stock_status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (product.url, product.name, product.site, product.target_price,
                     product.current_price, product.original_price, product.last_checked,
                     product.image_url, product.category, product.seller,
                     product.stock_status, product.created_at),
                )
                conn.commit()
                product.id = cur.lastrowid
                return product
            finally:
                conn.close()

    def get_product(self, product_id: int) -> Optional[Product]:
        """Fetch a single product by id."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM products WHERE id = ?", (product_id,)
                ).fetchone()
                return self._row_to_product(row) if row else None
            finally:
                conn.close()

    def get_product_by_url(self, url: str) -> Optional[Product]:
        """Fetch a product by its URL."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM products WHERE url = ?", (url,)
                ).fetchone()
                return self._row_to_product(row) if row else None
            finally:
                conn.close()

    def list_products(self) -> List[Product]:
        """Return all tracked products."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM products ORDER BY created_at DESC"
                ).fetchall()
                return [self._row_to_product(r) for r in rows]
            finally:
                conn.close()

    def update_product_price(
        self, product_id: int, price: float,
        original_price: Optional[float] = None,
        stock_status: str = "unknown",
        seller: str = "",
        name: str = "",
    ) -> None:
        """Update current price and record history entry."""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            try:
                updates = {
                    "current_price": price,
                    "last_checked": now,
                    "stock_status": stock_status,
                }
                if original_price is not None:
                    updates["original_price"] = original_price
                if seller:
                    updates["seller"] = seller
                if name:
                    updates["name"] = name

                set_clause = ", ".join(f"{k} = ?" for k in updates)
                values = list(updates.values()) + [product_id]
                conn.execute(
                    f"UPDATE products SET {set_clause} WHERE id = ?", values,
                )

                conn.execute(
                    """INSERT INTO price_history
                       (product_id, price, original_price, stock_status, seller, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (product_id, price, original_price, stock_status, seller, now),
                )
                conn.commit()
            finally:
                conn.close()

    def delete_product(self, product_id: int) -> bool:
        """Remove a product and its history/alerts (CASCADE)."""
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    def get_price_history(
        self, product_id: int, limit: int = 100
    ) -> List[PriceRecord]:
        """Return price history for a product, newest first."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM price_history
                       WHERE product_id = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (product_id, limit),
                ).fetchall()
                return [self._row_to_price_record(r) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def add_alert(self, alert: Alert) -> Alert:
        now = time.time()
        alert.created_at = alert.created_at or now
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    """INSERT INTO alerts (product_id, alert_type, threshold, active, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (alert.product_id, alert.alert_type, alert.threshold,
                     1 if alert.active else 0, alert.created_at),
                )
                conn.commit()
                alert.id = cur.lastrowid
                return alert
            finally:
                conn.close()

    def get_active_alerts(self, product_id: Optional[int] = None) -> List[Alert]:
        with self._lock:
            conn = self._get_conn()
            try:
                if product_id is not None:
                    rows = conn.execute(
                        "SELECT * FROM alerts WHERE product_id = ? AND active = 1",
                        (product_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM alerts WHERE active = 1"
                    ).fetchall()
                return [self._row_to_alert(r) for r in rows]
            finally:
                conn.close()

    def deactivate_alert(self, alert_id: int) -> bool:
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    "UPDATE alerts SET active = 0 WHERE id = ?", (alert_id,)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Row converters
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_product(row: sqlite3.Row) -> Product:
        return Product(
            id=row["id"],
            url=row["url"],
            name=row["name"],
            site=row["site"],
            target_price=row["target_price"],
            current_price=row["current_price"],
            original_price=row["original_price"],
            last_checked=row["last_checked"],
            image_url=row["image_url"],
            category=row["category"],
            seller=row["seller"],
            stock_status=row["stock_status"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_price_record(row: sqlite3.Row) -> PriceRecord:
        return PriceRecord(
            id=row["id"],
            product_id=row["product_id"],
            price=row["price"],
            original_price=row["original_price"],
            stock_status=row["stock_status"],
            seller=row["seller"],
            timestamp=row["timestamp"],
        )

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> Alert:
        return Alert(
            id=row["id"],
            product_id=row["product_id"],
            alert_type=row["alert_type"],
            threshold=row["threshold"],
            active=bool(row["active"]),
            created_at=row["created_at"],
        )
