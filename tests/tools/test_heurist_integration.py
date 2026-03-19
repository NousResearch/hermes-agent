#!/usr/bin/env python3
"""
Integration test: validate HeuristSource works against the live Heurist Marketplace API.
Verifies that at least 3 skills can be searched, inspected, and fetched successfully.

Run: python tests/tools/test_heurist_integration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.skills_hub import HeuristSource


def main():
    src = HeuristSource()
    print(f"Source ID: {src.source_id()}")
    print(f"Base URL: {src.BASE_URL}")
    print()

    # 1. Search
    print("=== Search (empty query — featured skills) ===")
    results = src.search("", limit=10)
    print(f"Found {len(results)} skills")
    for r in results[:5]:
        print(f"  - {r.identifier}: {r.name} (risk={r.extra.get('risk_tier')}, category={r.extra.get('category')})")
    assert len(results) >= 3, f"Expected at least 3 skills, got {len(results)}"
    print()

    # 2. Search with query
    print("=== Search (query='crypto') ===")
    crypto_results = src.search("crypto", limit=5)
    print(f"Found {len(crypto_results)} skills for 'crypto'")
    for r in crypto_results:
        print(f"  - {r.identifier}: {r.name}")
    print()

    # 3. Validate 3 skills: inspect + fetch
    test_slugs = [r.identifier for r in results[:3]]
    print(f"=== Validating 3 skills: {test_slugs} ===")
    print()

    success_count = 0
    for identifier in test_slugs:
        slug = identifier.split(":", 1)[-1] if ":" in identifier else identifier
        print(f"--- {identifier} ---")

        # Inspect
        meta = src.inspect(identifier)
        if meta is None:
            print(f"  FAIL: inspect returned None")
            continue
        print(f"  Inspect OK: {meta.name}")
        print(f"    Description: {meta.description[:80]}...")
        print(f"    Risk tier: {meta.extra.get('risk_tier')}")
        print(f"    Capabilities: {meta.extra.get('capabilities')}")
        print(f"    Is folder: {meta.extra.get('is_folder')}")
        print(f"    SHA256: {meta.extra.get('approved_sha256', 'N/A')[:16]}...")

        # Fetch
        bundle = src.fetch(identifier)
        if bundle is None:
            print(f"  FAIL: fetch returned None")
            continue
        print(f"  Fetch OK: {len(bundle.files)} file(s)")
        for fname in sorted(bundle.files.keys()):
            content = bundle.files[fname]
            size = len(content) if content else 0
            print(f"    {fname} ({size} bytes)")
        assert "SKILL.md" in bundle.files, "Missing SKILL.md"
        print(f"  Bundle metadata: risk_tier={bundle.metadata.get('risk_tier')}, sha256={bundle.metadata.get('approved_sha256', 'N/A')[:16]}...")
        print(f"  PASS")
        success_count += 1
        print()

    print(f"=== Results: {success_count}/3 skills validated successfully ===")
    assert success_count >= 3, f"Only {success_count}/3 skills passed validation"
    print("ALL INTEGRATION TESTS PASSED")


if __name__ == "__main__":
    main()
