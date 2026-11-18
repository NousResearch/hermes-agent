#!/usr/bin/env python3
"""
x402 Client — Autonomous HTTP 402 Payment Handler
Detects x402 payment requirements and pays with USDC on Arc Testnet
"""

import json
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from x402_payment import x402_send_usdc_eip3009

def x402_request(url: str, method: str = "GET", max_payment_usd: float = 1.0, **kwargs):
    """
    Make an HTTP request. If 402 is returned, pay and retry automatically.
    
    Args:
        url: Target URL
        method: HTTP method
        max_payment_usd: Max allowed payment in USD
        **kwargs: Additional requests arguments
    
    Returns:
        dict with response data and payment info
    """
    print(f"[x402] Requesting: {method} {url}")
    
    # First attempt
    response = requests.request(method, url, **kwargs)
    
    if response.status_code != 402:
        return {
            "ok": True,
            "status": response.status_code,
            "payment_required": False,
            "body": response.text[:500]
        }
    
    print(f"[x402] Got 402 Payment Required")
    
    # Parse payment requirements
    try:
        payment_info = response.json()
        print(f"[x402] Payment info: {json.dumps(payment_info, indent=2)}")
    except Exception:
        payment_info = {}
    
    # Extract amount and recipient
    amount = payment_info.get("amount", 0.001)
    recipient = payment_info.get("payTo") or payment_info.get("recipient") or os.getenv("X402_PAYMENT_RECIPIENT")
    
    if not recipient:
        return {"ok": False, "error": "No payment recipient found in 402 response"}
    
    # Safety check
    if float(amount) > max_payment_usd:
        return {"ok": False, "error": f"Payment ${amount} exceeds max allowed ${max_payment_usd}"}
    
    print(f"[x402] Paying ${amount} USDC to {recipient}")
    
    # Make payment
    raw = x402_send_usdc_eip3009(recipient, float(amount))
    payment_result = json.loads(raw) if isinstance(raw, str) else raw
    
    if not payment_result.get("ok"):
        return {"ok": False, "error": "Payment failed", "payment": payment_result}
    
    print(f"[x402] Payment confirmed: {payment_result.get('tx_hash')}")
    
    # Retry with payment proof
    headers = kwargs.pop("headers", {})
    headers["X-PAYMENT"] = payment_result.get("tx_hash", "")
    headers["X-PAYMENT-CHAIN"] = "arc_testnet"
    
    retry = requests.request(method, url, headers=headers, **kwargs)
    
    return {
        "ok": retry.status_code == 200,
        "status": retry.status_code,
        "payment_required": True,
        "payment": payment_result,
        "body": retry.text[:500]
    }


if __name__ == "__main__":
    # Test with x402.org demo endpoint
    url = sys.argv[1] if len(sys.argv) > 1 else "https://x402.org/demo"
    result = x402_request(url)
    print(json.dumps(result, indent=2))
