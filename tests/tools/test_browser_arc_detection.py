"""Tests for Arc browser resolver and detection support."""
import posixpath
from pathlib import Path
from unittest.mock import patch
import pytest

class TestArcBrowserDetection:
    def test_arc_registered_in_browsers(self):
        import hermes_cli.browser_connect as bc
        assert "arc" in bc._BROWSER_BY_KEY
        b = bc._BROWSER_BY_KEY["arc"]
        assert b.mac_app == "/Applications/Arc.app/Contents/MacOS/Arc"
        assert b.mac_support == ("Arc", "User Data")

    def test_arc_data_dir_darwin(self):
        import hermes_cli.browser_connect as bc
        got = bc.real_profile_data_dir("arc", "Darwin")
        expected = str(Path.home() / "Library" / "Application Support" / "Arc" / "User Data")
        assert got == expected

    def test_arc_darwin_bundle_map(self):
        import hermes_cli.browser_connect as bc
        m = dict(bc._DARWIN_BUNDLE_MAP)
        assert m.get("company.thebrowser.browser") == "arc"

    def test_arc_launchservices_classification(self):
        import hermes_cli.browser_connect as bc
        sample_dump = '{\n    LSHandlerRoleAll = "company.thebrowser.browser";\n    LSHandlerURLScheme = https;\n}'
        bundle = bc._launchservices_https_handler(sample_dump)
        assert bundle == "company.thebrowser.browser"
        assert bc._classify_default(bundle.lower(), bc._DARWIN_CHANNEL_BUNDLES, bc._DARWIN_BUNDLE_MAP, str.__eq__) == "arc"
