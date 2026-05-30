"""Tests for marketplace listing extraction tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from tools import marketplace_tools


def _mock_response(payload, status_code=200):
    response = Mock()
    response.ok = status_code < 400
    response.status_code = status_code
    response.text = "mock response"
    response.json.return_value = payload
    return response


def test_detect_marketplace_and_parse_listing_ids():
    assert marketplace_tools.detect_marketplace("https://www.amazon.com/dp/B0DZZWMB2L") == "amazon"
    assert marketplace_tools.parse_listing_id("https://www.amazon.com/dp/B0DZZWMB2L", "amazon") == "B0DZZWMB2L"
    assert marketplace_tools.parse_listing_id("https://www.amazon.com/gp/product/b0dzzwmb2l", "amazon") == "B0DZZWMB2L"

    assert marketplace_tools.detect_marketplace("https://www.ebay.com/itm/277922572500") == "ebay"
    assert marketplace_tools.parse_listing_id("https://www.ebay.com/itm/277922572500", "ebay") == "277922572500"
    assert marketplace_tools.parse_listing_id("https://www.ebay.com/itm/lg-gram/277922572500", "ebay") == "277922572500"


def test_detect_marketplace_rejects_unsupported_hosts():
    assert marketplace_tools.detect_marketplace("https://example.com/dp/B0DZZWMB2L") is None


@pytest.mark.asyncio
async def test_marketplace_listing_extract_returns_config_error_without_backend():
    with patch.object(marketplace_tools, "_has_serpapi_config", return_value=False), patch.object(
        marketplace_tools, "_has_ebay_browse_config", return_value=False
    ), patch.object(
        marketplace_tools, "check_firecrawl_api_key", return_value=False
    ):
        result = await marketplace_tools.marketplace_listing_extract_tool(
            ["https://www.ebay.com/itm/277922572500"]
        )

    assert result["success"] is False
    assert result["configuration_required"]["supported_now"] == [
        "serpapi_amazon",
        "serpapi_ebay",
        "ebay_browse_api",
        "firecrawl_structured_extract",
    ]
    assert result["results"][0]["marketplace"] == "ebay"
    assert result["results"][0]["listing_id"] == "277922572500"


@pytest.mark.asyncio
async def test_marketplace_listing_extract_prefers_ebay_browse_api_for_ebay_urls():
    api_result = {
        "url": "https://www.ebay.com/itm/277922572500",
        "title": "LG Gram",
        "marketplace": "ebay",
        "listing_id": "277922572500",
        "source": "ebay_browse_api",
        "structured_data": {"title": "LG Gram", "price": "499.99", "currency": "USD"},
    }

    with patch.object(marketplace_tools, "_has_serpapi_config", return_value=False), patch.object(
        marketplace_tools, "_has_ebay_browse_config", return_value=True
    ), patch.object(
        marketplace_tools, "_fetch_ebay_item_by_legacy_id", return_value=api_result
    ) as fetch_item, patch.object(marketplace_tools, "check_firecrawl_api_key", return_value=False):
        result = await marketplace_tools.marketplace_listing_extract_tool(
            ["https://www.ebay.com/itm/277922572500"]
        )

    assert result["success"] is True
    fetch_item.assert_called_once_with("https://www.ebay.com/itm/277922572500", "277922572500")
    assert result["results"] == [api_result]


@pytest.mark.asyncio
async def test_marketplace_listing_extract_uses_firecrawl_structured_schema():
    provider = AsyncMock()
    provider.extract.return_value = [
        {
            "url": "https://www.amazon.com/dp/B0DZZWMB2L",
            "title": "LG Gram",
            "structured_data": {"title": "LG Gram", "price": "$799"},
        }
    ]

    with patch.object(marketplace_tools, "_has_serpapi_config", return_value=False), patch.object(
        marketplace_tools, "_has_ebay_browse_config", return_value=False
    ), patch.object(
        marketplace_tools, "check_firecrawl_api_key", return_value=True
    ), patch.object(
        marketplace_tools, "FirecrawlWebSearchProvider", return_value=provider
    ):
        result = await marketplace_tools.marketplace_listing_extract_tool(
            ["https://www.amazon.com/dp/B0DZZWMB2L"]
        )

    assert result["success"] is True
    provider.extract.assert_awaited_once_with(
        ["https://www.amazon.com/dp/B0DZZWMB2L"],
        json_schema=marketplace_tools.MARKETPLACE_LISTING_SCHEMA,
        include_markdown=True,
    )
    assert result["results"][0]["marketplace"] == "amazon"
    assert result["results"][0]["listing_id"] == "B0DZZWMB2L"
    assert result["results"][0]["structured_data"]["price"] == "$799"


@pytest.mark.asyncio
async def test_marketplace_listing_extract_rejects_marketplace_mismatch():
    result = await marketplace_tools.marketplace_listing_extract_tool(
        ["https://www.amazon.com/dp/B0DZZWMB2L"], marketplace="ebay"
    )

    assert result["success"] is False
    assert "appears to be amazon" in result["results"][0]["error"]


@pytest.mark.asyncio
async def test_marketplace_listing_extract_prefers_serpapi_for_ebay_urls():
    payload = {
        "product_results": {
            "product_id": "277922572500",
            "title": "LG Gram 17 Laptop",
            "link": "https://www.ebay.com/itm/277922572500",
            "buy": {
                "buy_it_now": {"price": {"amount": 499.99, "currency": "USD"}},
            },
            "shipping": {"options": [{"via": "USPS Ground Advantage", "free": True}]},
            "returns": [[{"title": "Returns", "snippets": [{"text": "Accepted within 30 days"}]}]],
            "condition": "Open box",
            "seller_results": {"username": "top-seller", "feedback_percentage": "99.8%"},
            "specifications": {
                "groups": [
                    {
                        "sections": [
                            {
                                "fields": [
                                    {"title": "Brand", "value": "LG"},
                                    {"title": "Screen Size", "value": "17 in"},
                                ]
                            }
                        ]
                    }
                ]
            },
            "media": [
                {
                    "image": [
                        {"link": "https://i.ebayimg.com/images/g/a/s-l140.jpg", "size": {"width": 140}},
                        {"link": "https://i.ebayimg.com/images/g/a/s-l1600.jpg", "size": {"width": 1600}},
                    ]
                }
            ],
            "watch_count": "14 watchers",
        }
    }

    with patch.dict(marketplace_tools.os.environ, {"SERPAPI_API_KEY": "test-key"}, clear=False), patch.object(
        marketplace_tools.requests, "get", return_value=_mock_response(payload)
    ) as get, patch.object(marketplace_tools, "_fetch_ebay_item_by_legacy_id") as browse_fetch, patch.object(
        marketplace_tools, "check_firecrawl_api_key", return_value=False
    ):
        result = await marketplace_tools.marketplace_listing_extract_tool(
            ["https://www.ebay.com/itm/277922572500"]
        )

    assert result["success"] is True
    browse_fetch.assert_not_called()
    get.assert_called_once()
    assert get.call_args.kwargs["params"]["engine"] == "ebay_product"
    assert get.call_args.kwargs["params"]["product_id"] == "277922572500"
    listing = result["results"][0]
    assert listing["source"] == "serpapi_ebay"
    assert listing["structured_data"]["title"] == "LG Gram 17 Laptop"
    assert listing["structured_data"]["price"] == "499.99"
    assert listing["structured_data"]["currency"] == "USD"
    assert listing["structured_data"]["shipping"] == "USPS Ground Advantage"
    assert listing["structured_data"]["returns"] == "Accepted within 30 days"
    assert listing["structured_data"]["seller"] == "top-seller"
    assert listing["structured_data"]["product_specs"]["Brand"] == "LG"
    assert listing["metadata"]["image_links"] == ["https://i.ebayimg.com/images/g/a/s-l1600.jpg"]


@pytest.mark.asyncio
async def test_marketplace_listing_extract_prefers_serpapi_for_amazon_urls():
    payload = {
        "product_results": {
            "asin": "B0DZZWMB2L",
            "title": "LG Gram 17 Laptop",
            "link": "https://www.amazon.com/dp/B0DZZWMB2L",
            "price": "$799.99",
            "availability": "In Stock",
            "brand": "LG",
            "rating": "4.4 out of 5",
            "reviews": "123 ratings",
            "feature_bullets": ["17 inch display", "Lightweight"],
            "images": ["https://m.media-amazon.com/images/I/example.jpg"],
        }
    }

    with patch.dict(marketplace_tools.os.environ, {"SERPAPI_API_KEY": "test-key"}, clear=False), patch.object(
        marketplace_tools.requests, "get", return_value=_mock_response(payload)
    ) as get, patch.object(marketplace_tools, "check_firecrawl_api_key", return_value=False):
        result = await marketplace_tools.marketplace_listing_extract_tool(
            ["https://www.amazon.com/dp/B0DZZWMB2L"]
        )

    assert result["success"] is True
    get.assert_called_once()
    assert get.call_args.kwargs["params"] == {
        "engine": "amazon_product",
        "asin": "B0DZZWMB2L",
        "amazon_domain": "amazon.com",
        "api_key": "test-key",
    }
    listing = result["results"][0]
    assert listing["source"] == "serpapi_amazon"
    assert listing["marketplace"] == "amazon"
    assert listing["listing_id"] == "B0DZZWMB2L"
    assert listing["structured_data"]["title"] == "LG Gram 17 Laptop"
    assert listing["structured_data"]["price"] == "$799.99"
    assert listing["structured_data"]["currency"] == "USD"
    assert listing["structured_data"]["availability"] == "In Stock"
    assert listing["structured_data"]["product_specs"]["feature_bullets"] == "17 inch display; Lightweight"
    assert listing["metadata"]["image_links"] == ["https://m.media-amazon.com/images/I/example.jpg"]


@pytest.mark.asyncio
async def test_marketplace_listing_extract_falls_back_to_browse_after_serpapi_failure():
    api_result = {
        "url": "https://www.ebay.com/itm/277922572500",
        "title": "LG Gram",
        "marketplace": "ebay",
        "listing_id": "277922572500",
        "source": "ebay_browse_api",
        "structured_data": {"title": "LG Gram"},
    }

    with patch.object(marketplace_tools, "_has_serpapi_config", return_value=True), patch.object(
        marketplace_tools, "_fetch_ebay_product_with_serpapi", side_effect=RuntimeError("serpapi down")
    ) as serp_fetch, patch.object(
        marketplace_tools, "_has_ebay_browse_config", return_value=True
    ), patch.object(
        marketplace_tools, "_fetch_ebay_item_by_legacy_id", return_value=api_result
    ) as browse_fetch, patch.object(
        marketplace_tools, "check_firecrawl_api_key", return_value=False
    ):
        result = await marketplace_tools.marketplace_listing_extract_tool(
            ["https://www.ebay.com/itm/277922572500"]
        )

    assert result["success"] is True
    serp_fetch.assert_called_once()
    browse_fetch.assert_called_once_with("https://www.ebay.com/itm/277922572500", "277922572500")
    assert result["results"] == [api_result]


@pytest.mark.asyncio
async def test_marketplace_listing_search_uses_serpapi_ebay_engine():
    payload = {
        "organic_results": [
            {
                "product_id": "277922572500",
                "title": "LG Gram 17",
                "link": "https://www.ebay.com/itm/277922572500",
                "price": "$499.99",
                "shipping": "Free shipping",
                "condition": "Used",
                "seller": "top-seller",
            }
        ]
    }

    with patch.dict(marketplace_tools.os.environ, {"SERPAPI_API_KEY": "test-key"}, clear=False), patch.object(
        marketplace_tools.requests, "get", return_value=_mock_response(payload)
    ) as get:
        result = await marketplace_tools.marketplace_listing_search_tool("LG Gram 17", max_results=5)

    assert result["success"] is True
    assert result["source"] == "serpapi_ebay"
    assert get.call_args.kwargs["params"] == {
        "engine": "ebay",
        "_nkw": "LG Gram 17",
        "api_key": "test-key",
    }
    assert result["results"][0]["product_id"] == "277922572500"
    assert result["results"][0]["title"] == "LG Gram 17"
    assert result["results"][0]["currency"] == "USD"
    assert result["results"][0]["seller"] == "top-seller"


@pytest.mark.asyncio
async def test_marketplace_listing_search_uses_serpapi_amazon_engine():
    payload = {
        "organic_results": [
            {
                "asin": "B0DZZWMB2L",
                "title": "LG Gram 17",
                "link": "https://www.amazon.com/dp/B0DZZWMB2L",
                "price": "$799.99",
                "rating": "4.4",
                "reviews": "123",
            }
        ]
    }

    with patch.dict(marketplace_tools.os.environ, {"SERPAPI_API_KEY": "test-key"}, clear=False), patch.object(
        marketplace_tools.requests, "get", return_value=_mock_response(payload)
    ) as get:
        result = await marketplace_tools.marketplace_listing_search_tool("LG Gram 17", marketplace="amazon", max_results=5)

    assert result["success"] is True
    assert result["source"] == "serpapi_amazon"
    assert get.call_args.kwargs["params"] == {
        "engine": "amazon",
        "k": "LG Gram 17",
        "amazon_domain": "amazon.com",
        "api_key": "test-key",
    }
    assert result["results"][0]["product_id"] == "B0DZZWMB2L"
    assert result["results"][0]["source"] == "serpapi_amazon"
    assert result["results"][0]["currency"] == "USD"
