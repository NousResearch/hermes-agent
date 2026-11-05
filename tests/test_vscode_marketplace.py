"""Unit tests for hermes_cli.vscode_marketplace and its web router."""

from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from hermes_cli import vscode_marketplace
from hermes_cli.web_routers import themes


def test_looks_like_icon_theme():
    assert vscode_marketplace._looks_like_icon_theme({"tags": ["icon-theme"]})
    assert vscode_marketplace._looks_like_icon_theme({"tags": ["product-icon-theme"]})
    assert vscode_marketplace._looks_like_icon_theme({"displayName": "Material Icon Theme"})
    assert vscode_marketplace._looks_like_icon_theme({"shortDescription": "Awesome file icons for VS Code"})
    assert not vscode_marketplace._looks_like_icon_theme({"displayName": "Tokyo Night", "shortDescription": "A clean dark theme"})


def test_search_marketplace_themes():
    mock_gallery_res = {
        "results": [
            {
                "extensions": [
                    {
                        "extensionName": "tokyo-night",
                        "displayName": "Tokyo Night",
                        "shortDescription": "A clean dark theme",
                        "publisher": {"publisherName": "enkia", "displayName": "Enkia"},
                        "statistics": [{"statisticName": "install", "value": 1500000}],
                    },
                    {
                        "extensionName": "icons",
                        "displayName": "File Icons",
                        "shortDescription": "File icons",
                        "tags": ["icon-theme"],
                        "publisher": {"publisherName": "someone", "displayName": "Someone"},
                    },
                ]
            }
        ]
    }

    with patch("hermes_cli.vscode_marketplace._query_gallery", return_value=mock_gallery_res):
        results = vscode_marketplace.search_marketplace_themes("tokyo")
        assert len(results) == 1
        assert results[0]["extensionId"] == "enkia.tokyo-night"
        assert results[0]["displayName"] == "Tokyo Night"
        assert results[0]["publisher"] == "Enkia"
        assert results[0]["installs"] == 1500000


def test_fetch_marketplace_themes():
    # Build a dummy in-memory VSIX package
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        pkg = {
            "name": "dummy-theme",
            "displayName": "Dummy Theme",
            "contributes": {
                "themes": [
                    {
                        "label": "Dummy Dark",
                        "uiTheme": "vs-dark",
                        "path": "./themes/dark.json",
                    }
                ]
            },
        }
        zf.writestr("extension/package.json", json.dumps(pkg))
        zf.writestr("extension/themes/dark.json", '{"name": "Dummy Dark", "colors": {}}')

    vsix_bytes = buf.getvalue()

    mock_gallery_res = {
        "results": [
            {
                "extensions": [
                    {
                        "extensionName": "dummy-theme",
                        "displayName": "Dummy Theme",
                        "publisher": {"publisherName": "tester"},
                        "versions": [
                            {
                                "files": [
                                    {
                                        "assetType": "Microsoft.VisualStudio.Services.VSIXPackage",
                                        "source": "https://example.com/dummy.vsix",
                                    }
                                ]
                            }
                        ],
                    }
                ]
            }
        ]
    }

    with patch("hermes_cli.vscode_marketplace._query_gallery", return_value=mock_gallery_res), patch(
        "hermes_cli.vscode_marketplace._request", return_value=vsix_bytes
    ):
        result = vscode_marketplace.fetch_marketplace_themes("tester.dummy-theme")
        assert result["extensionId"] == "tester.dummy-theme"
        assert result["displayName"] == "Dummy Theme"
        assert len(result["themes"]) == 1
        assert result["themes"][0]["label"] == "Dummy Dark"
        assert result["themes"][0]["uiTheme"] == "vs-dark"
        assert "colors" in result["themes"][0]["contents"]


def test_fetch_marketplace_themes_invalid_id():
    with pytest.raises(vscode_marketplace.MarketplaceError):
        vscode_marketplace.fetch_marketplace_themes("invalid-id-without-dot")


def test_themes_router():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(themes.router)
    client = TestClient(app)

    with patch(
        "hermes_cli.vscode_marketplace.search_marketplace_themes",
        return_value=[{"extensionId": "test.theme", "displayName": "Test Theme"}],
    ):
        resp = client.get("/api/themes/marketplace/search?q=test")
        assert resp.status_code == 200
        assert resp.json() == [{"extensionId": "test.theme", "displayName": "Test Theme"}]

    with patch(
        "hermes_cli.vscode_marketplace.fetch_marketplace_themes",
        return_value={"extensionId": "test.theme", "displayName": "Test Theme", "themes": []},
    ):
        resp = client.get("/api/themes/marketplace/fetch?id=test.theme")
        assert resp.status_code == 200
        assert resp.json()["extensionId"] == "test.theme"
