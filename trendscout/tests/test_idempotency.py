"""
Tests for trendscout term frequency idempotency.

Verifies that running ingest→record→recompute multiple times on the same
data produces identical term_frequency results (no double-counting).
"""
import sqlite3
import tempfile
from pathlib import Path

from trendscout import db, scoring


def test_recompute_idempotent():
    """Run record_post_terms + record_social_terms + recompute_term_frequency
    3× on the same fixture posts + one social term; assert term_frequency
    is identical across runs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'test.db'
        conn = db.connect(db_path)
        today = '2026-07-15'
        
        # Fixture: 3 posts with overlapping terms
        posts = [
            {'id': 'post1', 'title': 'AI safety research', 'selftext': 'New AI safety research published'},
            {'id': 'post2', 'title': 'Machine learning advances', 'selftext': 'ML advances in safety'},
            {'id': 'post3', 'title': 'AI regulation', 'selftext': 'New regulations for AI systems'},
        ]
        social_terms = ['#aisafety']
        
        # Run 1
        scoring.record_post_terms(conn, posts, today)
        scoring.record_social_terms(conn, social_terms, today)
        scoring.recompute_term_frequency(conn, today)
        run1_freq = dict(conn.execute(
            'SELECT term, frequency FROM term_frequency WHERE date=? ORDER BY term',
            (today,)
        ).fetchall())
        
        # Run 2 (same data)
        scoring.record_post_terms(conn, posts, today)
        scoring.record_social_terms(conn, social_terms, today)
        scoring.recompute_term_frequency(conn, today)
        run2_freq = dict(conn.execute(
            'SELECT term, frequency FROM term_frequency WHERE date=? ORDER BY term',
            (today,)
        ).fetchall())
        
        # Run 3 (same data)
        scoring.record_post_terms(conn, posts, today)
        scoring.record_social_terms(conn, social_terms, today)
        scoring.recompute_term_frequency(conn, today)
        run3_freq = dict(conn.execute(
            'SELECT term, frequency FROM term_frequency WHERE date=? ORDER BY term',
            (today,)
        ).fetchall())
        
        # Assert idempotency
        assert run1_freq == run2_freq, f"Run 1 and 2 differ: {run1_freq} vs {run2_freq}"
        assert run2_freq == run3_freq, f"Run 2 and 3 differ: {run2_freq} vs {run3_freq}"
        
        # Spot-check: '#aisafety' should appear once, and we should have multiple terms
        assert run1_freq.get('#aisafety') == 1, f"Expected '#aisafety' frequency=1, got {run1_freq.get('#aisafety')}"
        assert len(run1_freq) > 1, f"Expected multiple terms, got {len(run1_freq)}"
        
        conn.close()
        print("✓ Idempotency test passed: term_frequency identical across 3 runs")


def test_post_terms_dedup():
    """Verify post_terms table prevents duplicate (post_id, term, date) rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'test.db'
        conn = db.connect(db_path)
        today = '2026-07-15'
        
        post = {'id': 'post1', 'title': 'AI safety', 'selftext': 'AI safety matters'}
        
        # Call twice
        scoring.record_post_terms(conn, [post], today)
        scoring.record_post_terms(conn, [post], today)
        
        # Should have exactly 2 rows (one per unique term), not 4
        rows = conn.execute(
            'SELECT COUNT(*) FROM post_terms WHERE date=?', (today,)
        ).fetchone()[0]
        
        # Terms extracted: 'ai', 'ai safety' (or similar) — should be deduped
        assert rows == 2, f"Expected 2 post_terms rows (deduped), got {rows}"
        
        conn.close()
        print("✓ post_terms dedup test passed")


if __name__ == '__main__':
    test_recompute_idempotent()
    test_post_terms_dedup()
    print("\n✅ All idempotency tests passed")
