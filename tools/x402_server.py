#!/usr/bin/env python3
"""
x402 Demo Server — Arc Testnet
A simple HTTP server that requires x402 USDC payment to access premium data
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv

load_dotenv()

PAYMENT_RECIPIENT = os.getenv("X402_ADDRESS", "")
PAYMENT_AMOUNT = 0.001  # $0.001 USDC per request
PAYMENT_CHAIN = "arc_testnet"

class X402Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "server": "x402-arc-testnet"})
            return

        if self.path == "/premium/data":
            # Check for payment proof
            payment_header = self.headers.get("X-PAYMENT", "")
            
            if not payment_header:
                # Return 402 with payment requirements
                self.send_response(402)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Payment-Required", "true")
                self.end_headers()
                body = {
                    "error": "Payment Required",
                    "payTo": PAYMENT_RECIPIENT,
                    "amount": PAYMENT_AMOUNT,
                    "chain": PAYMENT_CHAIN,
                    "token": "USDC",
                    "description": "Premium Arc Testnet data — $0.001 USDC"
                }
                self.wfile.write(json.dumps(body).encode())
                print(f"[x402 server] 402 sent — payment required")
                return

            # Payment proof received
            print(f"[x402 server] Payment proof: {payment_header}")
            self._respond(200, {
                "ok": True,
                "data": {
                    "message": "Premium data unlocked via x402 on Arc Testnet!",
                    "tx_hash": payment_header,
                    "timestamp": __import__("time").time(),
                    "arc_block": "latest"
                }
            })
            return

        self._respond(404, {"error": "Not found"})

    def _respond(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body, indent=2).encode())

    def log_message(self, format, *args):
        print(f"[x402 server] {format % args}")


if __name__ == "__main__":
    port = int(os.getenv("X402_SERVER_PORT", 8402))
    print(f"[x402 server] Starting on http://localhost:{port}")
    print(f"[x402 server] Payment recipient: {PAYMENT_RECIPIENT}")
    print(f"[x402 server] Price: ${PAYMENT_AMOUNT} USDC per request")
    HTTPServer(("", port), X402Handler).serve_forever()
