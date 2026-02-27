"""
Async Solana RPC client for querying the Solana blockchain.
Uses public RPC endpoints by default, configurable via SOLANA_RPC_URL env var.
"""

import os
import httpx
import base58
from typing import Any, Optional

DEFAULT_RPC_URL = "https://api.mainnet-beta.solana.com"
HELIUS_BASE = "https://api.helius.xyz/v0"

# Known token mints for labeling
KNOWN_TOKENS = {
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": ("USDC", 6),
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": ("USDT", 6),
    "So11111111111111111111111111111111111111112": ("Wrapped SOL", 9),
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": ("BONK", 5),
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN": ("JUP", 6),
    "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs": ("WETH", 8),
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": ("mSOL", 9),
    "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj": ("stSOL", 9),
    "rndrizKT3MK1iimdxRdWabcF7Zg7AR5T4nud4EkHBof": ("RNDR", 8),
    "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3": ("PYTH", 6),
}

# Known program IDs
KNOWN_PROGRAMS = {
    "11111111111111111111111111111111": "System Program",
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA": "Token Program",
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL": "Associated Token",
    "ComputeBudget111111111111111111111111111111": "Compute Budget",
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "Jupiter v6",
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "Orca Whirlpool",
    "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin": "Serum DEX v3",
    "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s": "Metaplex Metadata",
    "BGUMAp9Gq7iTEuizy4pqaxsTyUCBK68MDfK752saRPUY": "Bubblegum (cNFTs)",
    "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr": "Memo Program",
}


class SolanaClient:
    """Async client for Solana RPC and DAS API calls."""

    def __init__(self):
        self.rpc_url = os.environ.get("SOLANA_RPC_URL", DEFAULT_RPC_URL)
        self.helius_key = os.environ.get("HELIUS_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def _rpc_call(self, method: str, params: list = None) -> dict:
        """Make a JSON-RPC call to Solana."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or [],
        }
        resp = await self.client.post(self.rpc_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise Exception(f"RPC Error: {data['error']}")
        return data.get("result", {})

    async def _helius_call(self, endpoint: str, payload: dict = None) -> Any:
        """Make a call to Helius DAS API (if key available)."""
        if not self.helius_key:
            return None
        url = f"{self.rpc_url}"  # Helius RPC supports DAS
        if payload:
            resp = await self.client.post(url, json=payload)
        else:
            resp = await self.client.get(f"{HELIUS_BASE}/{endpoint}?api-key={self.helius_key}")
        resp.raise_for_status()
        return resp.json()

    # ── Wallet Info ──────────────────────────────────────────

    async def get_wallet_info(self, address: str) -> dict:
        """Get comprehensive wallet information."""
        # Validate address
        try:
            decoded = base58.b58decode(address)
            if len(decoded) != 32:
                return {"error": f"Invalid Solana address: must be 32 bytes, got {len(decoded)}"}
        except Exception:
            return {"error": f"Invalid base58 address: {address}"}

        # Get SOL balance
        balance_result = await self._rpc_call("getBalance", [address])
        sol_balance = balance_result.get("value", 0) / 1e9

        # Get token accounts
        token_accounts = await self._rpc_call(
            "getTokenAccountsByOwner",
            [
                address,
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                {"encoding": "jsonParsed"},
            ],
        )

        tokens = []
        for account in token_accounts.get("value", []):
            parsed = account["account"]["data"]["parsed"]["info"]
            mint = parsed["mint"]
            amount_raw = int(parsed["tokenAmount"]["amount"])
            decimals = parsed["tokenAmount"]["decimals"]
            amount = amount_raw / (10 ** decimals) if decimals > 0 else amount_raw

            if amount == 0:
                continue

            token_name, _ = KNOWN_TOKENS.get(mint, (None, None))
            tokens.append({
                "mint": mint,
                "symbol": token_name or "Unknown",
                "balance": round(amount, 6),
                "decimals": decimals,
            })

        # Sort by balance descending
        tokens.sort(key=lambda x: x["balance"], reverse=True)

        # Get account info for more details
        account_info = await self._rpc_call(
            "getAccountInfo", [address, {"encoding": "jsonParsed"}]
        )

        is_executable = False
        owner_program = "System Program"
        if account_info and account_info.get("value"):
            is_executable = account_info["value"].get("executable", False)
            owner_program = KNOWN_PROGRAMS.get(
                account_info["value"].get("owner", ""),
                account_info["value"].get("owner", "Unknown"),
            )

        return {
            "address": address,
            "sol_balance": round(sol_balance, 9),
            "sol_balance_usd_approx": None,  # Would need price feed
            "token_count": len(tokens),
            "tokens": tokens[:20],  # Top 20 tokens
            "is_executable": is_executable,
            "owner_program": owner_program,
        }

    # ── Transaction Lookup ───────────────────────────────────

    async def get_transaction(self, signature: str) -> dict:
        """Get transaction details by signature."""
        result = await self._rpc_call(
            "getTransaction",
            [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
        )

        if not result:
            return {"error": f"Transaction not found: {signature}"}

        meta = result.get("meta", {})
        tx = result.get("transaction", {})
        message = tx.get("message", {})

        # Extract account keys
        account_keys = []
        for key_info in message.get("accountKeys", []):
            if isinstance(key_info, dict):
                account_keys.append(key_info.get("pubkey", ""))
            else:
                account_keys.append(str(key_info))

        # Extract instructions
        instructions = []
        for ix in message.get("instructions", []):
            program_id = ix.get("programId", "")
            program_name = KNOWN_PROGRAMS.get(program_id, program_id[:16] + "...")

            parsed = ix.get("parsed")
            if parsed:
                if isinstance(parsed, dict):
                    instructions.append({
                        "program": program_name,
                        "type": parsed.get("type", "unknown"),
                        "info": _summarize_parsed_info(parsed.get("info", {})),
                    })
                else:
                    instructions.append({
                        "program": program_name,
                        "type": str(parsed),
                    })
            else:
                instructions.append({
                    "program": program_name,
                    "type": "raw instruction",
                })

        # Fee
        fee = meta.get("fee", 0) / 1e9

        # Balance changes
        pre_balances = meta.get("preBalances", [])
        post_balances = meta.get("postBalances", [])
        balance_changes = []
        for i, (pre, post) in enumerate(zip(pre_balances, post_balances)):
            diff = (post - pre) / 1e9
            if abs(diff) > 0.000001 and i < len(account_keys):
                balance_changes.append({
                    "account": account_keys[i][:16] + "...",
                    "change_sol": round(diff, 9),
                })

        # Token balance changes
        token_changes = []
        pre_token = meta.get("preTokenBalances", [])
        post_token = meta.get("postTokenBalances", [])
        post_map = {tb["accountIndex"]: tb for tb in post_token}
        pre_map = {tb["accountIndex"]: tb for tb in pre_token}

        all_indices = set(list(post_map.keys()) + list(pre_map.keys()))
        for idx in all_indices:
            pre_tb = pre_map.get(idx)
            post_tb = post_map.get(idx)
            pre_amount = float(pre_tb["uiTokenAmount"]["uiAmountString"]) if pre_tb else 0
            post_amount = float(post_tb["uiTokenAmount"]["uiAmountString"]) if post_tb else 0
            diff = post_amount - pre_amount
            if abs(diff) > 0:
                mint = (post_tb or pre_tb).get("mint", "unknown")
                symbol = KNOWN_TOKENS.get(mint, (mint[:12] + "...",))[0]
                token_changes.append({
                    "token": symbol,
                    "change": round(diff, 6),
                })

        return {
            "signature": signature,
            "slot": result.get("slot"),
            "block_time": result.get("blockTime"),
            "success": meta.get("err") is None,
            "fee_sol": round(fee, 9),
            "instructions_count": len(instructions),
            "instructions": instructions[:10],
            "sol_balance_changes": balance_changes[:5],
            "token_balance_changes": token_changes[:10],
            "log_messages_count": len(meta.get("logMessages", [])),
        }

    # ── Recent Activity ──────────────────────────────────────

    async def get_recent_activity(self, address: str, limit: int = 10) -> dict:
        """Get recent transaction signatures for an address."""
        try:
            decoded = base58.b58decode(address)
            if len(decoded) != 32:
                return {"error": "Invalid address"}
        except Exception:
            return {"error": f"Invalid base58 address: {address}"}

        sigs = await self._rpc_call(
            "getSignaturesForAddress",
            [address, {"limit": min(limit, 25)}],
        )

        transactions = []
        for sig_info in sigs:
            transactions.append({
                "signature": sig_info["signature"][:32] + "...",
                "full_signature": sig_info["signature"],
                "slot": sig_info.get("slot"),
                "block_time": sig_info.get("blockTime"),
                "success": sig_info.get("err") is None,
                "memo": sig_info.get("memo"),
            })

        return {
            "address": address,
            "transaction_count": len(transactions),
            "transactions": transactions,
        }

    # ── Token Info ───────────────────────────────────────────

    async def get_token_info(self, mint_address: str) -> dict:
        """Get token metadata and supply info."""
        try:
            decoded = base58.b58decode(mint_address)
            if len(decoded) != 32:
                return {"error": "Invalid mint address"}
        except Exception:
            return {"error": f"Invalid base58 address: {mint_address}"}

        # Get supply
        supply_result = await self._rpc_call("getTokenSupply", [mint_address])
        supply = supply_result.get("value", {})

        # Get largest holders
        holders_result = await self._rpc_call(
            "getTokenLargestAccounts", [mint_address]
        )
        largest_holders = []
        for holder in holders_result.get("value", [])[:10]:
            largest_holders.append({
                "address": holder["address"][:16] + "...",
                "full_address": holder["address"],
                "amount": holder.get("uiAmountString", "0"),
            })

        # Get account info for metadata hint
        account_info = await self._rpc_call(
            "getAccountInfo", [mint_address, {"encoding": "jsonParsed"}]
        )

        mint_authority = None
        freeze_authority = None
        if account_info and account_info.get("value"):
            parsed = account_info["value"].get("data", {})
            if isinstance(parsed, dict) and "parsed" in parsed:
                info = parsed["parsed"].get("info", {})
                mint_authority = info.get("mintAuthority")
                freeze_authority = info.get("freezeAuthority")

        known_name = KNOWN_TOKENS.get(mint_address, (None, None))[0]

        return {
            "mint_address": mint_address,
            "known_symbol": known_name,
            "total_supply": supply.get("uiAmountString", "0"),
            "decimals": supply.get("decimals", 0),
            "mint_authority": mint_authority,
            "freeze_authority": freeze_authority,
            "top_holders": largest_holders,
        }

    # ── NFT Portfolio ────────────────────────────────────────

    async def get_nft_portfolio(self, address: str) -> dict:
        """Get NFTs owned by a wallet using token accounts."""
        try:
            decoded = base58.b58decode(address)
            if len(decoded) != 32:
                return {"error": "Invalid address"}
        except Exception:
            return {"error": f"Invalid base58 address: {address}"}

        # NFTs are tokens with amount=1 and decimals=0
        token_accounts = await self._rpc_call(
            "getTokenAccountsByOwner",
            [
                address,
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                {"encoding": "jsonParsed"},
            ],
        )

        nfts = []
        for account in token_accounts.get("value", []):
            parsed = account["account"]["data"]["parsed"]["info"]
            amount = int(parsed["tokenAmount"]["amount"])
            decimals = parsed["tokenAmount"]["decimals"]

            # NFT heuristic: amount == 1 and decimals == 0
            if amount == 1 and decimals == 0:
                mint = parsed["mint"]
                nfts.append({
                    "mint": mint,
                    "token_account": account["pubkey"],
                })

        return {
            "address": address,
            "nft_count": len(nfts),
            "nfts": nfts[:50],  # Limit to 50
            "note": "For full metadata, use a DAS-compatible RPC (e.g., Helius) via SOLANA_RPC_URL",
        }

    # ── Whale Detector ───────────────────────────────────────

    async def detect_whales(self, min_sol: float = 1000) -> dict:
        """Check recent large SOL transfers by scanning recent blocks."""
        # Get recent block
        slot = await self._rpc_call("getSlot")

        # Get block with transactions
        block = await self._rpc_call(
            "getBlock",
            [
                slot - 2,  # Slightly behind to ensure finalization
                {
                    "encoding": "jsonParsed",
                    "transactionDetails": "full",
                    "maxSupportedTransactionVersion": 0,
                },
            ],
        )

        if not block:
            return {"error": "Could not fetch recent block"}

        whale_txs = []
        for tx_wrapper in block.get("transactions", []):
            meta = tx_wrapper.get("meta", {})
            if meta.get("err") is not None:
                continue

            tx = tx_wrapper.get("transaction", {})
            message = tx.get("message", {})

            # Check for large SOL movements
            pre = meta.get("preBalances", [])
            post = meta.get("postBalances", [])
            account_keys = []
            for k in message.get("accountKeys", []):
                if isinstance(k, dict):
                    account_keys.append(k.get("pubkey", ""))
                else:
                    account_keys.append(str(k))

            for i, (p, q) in enumerate(zip(pre, post)):
                diff_sol = abs(q - p) / 1e9
                if diff_sol >= min_sol and i < len(account_keys):
                    sig = tx_wrapper.get("transaction", {}).get("signatures", [""])[0]
                    whale_txs.append({
                        "signature": sig[:32] + "..." if sig else "unknown",
                        "full_signature": sig,
                        "account": account_keys[i][:16] + "...",
                        "full_account": account_keys[i],
                        "sol_movement": round(diff_sol, 4),
                        "direction": "inflow" if (q - p) > 0 else "outflow",
                    })

        # Deduplicate by signature
        seen = set()
        unique_whales = []
        for w in whale_txs:
            if w["full_signature"] not in seen:
                seen.add(w["full_signature"])
                unique_whales.append(w)

        unique_whales.sort(key=lambda x: x["sol_movement"], reverse=True)

        return {
            "slot_scanned": slot - 2,
            "min_sol_threshold": min_sol,
            "whale_transactions_found": len(unique_whales),
            "whales": unique_whales[:20],
        }

    # ── Network Stats ────────────────────────────────────────

    async def get_network_stats(self) -> dict:
        """Get current Solana network statistics."""
        # Parallel calls for speed
        slot = await self._rpc_call("getSlot")
        epoch_info = await self._rpc_call("getEpochInfo")
        perf = await self._rpc_call("getRecentPerformanceSamples", [5])
        health = await self._rpc_call("getHealth")
        supply = await self._rpc_call("getSupply")
        version = await self._rpc_call("getVersion")

        # Calculate TPS from performance samples
        avg_tps = 0
        if perf:
            total_txs = sum(s.get("numTransactions", 0) for s in perf)
            total_secs = sum(s.get("samplePeriodSecs", 1) for s in perf)
            avg_tps = round(total_txs / total_secs, 2) if total_secs > 0 else 0

        supply_val = supply.get("value", {})

        return {
            "current_slot": slot,
            "epoch": epoch_info.get("epoch"),
            "epoch_progress": f"{round(epoch_info.get('slotIndex', 0) / max(epoch_info.get('slotsInEpoch', 1), 1) * 100, 1)}%",
            "slots_in_epoch": epoch_info.get("slotsInEpoch"),
            "average_tps": avg_tps,
            "health": health if isinstance(health, str) else "ok",
            "total_supply_sol": round(supply_val.get("total", 0) / 1e9, 0),
            "circulating_supply_sol": round(supply_val.get("circulating", 0) / 1e9, 0),
            "solana_version": version.get("solana-core", "unknown"),
        }

    async def close(self):
        await self.client.aclose()


def _summarize_parsed_info(info: dict) -> dict:
    """Summarize parsed instruction info for readability."""
    summary = {}
    for key, val in info.items():
        if isinstance(val, str) and len(val) > 20:
            # Probably an address, truncate
            known = KNOWN_PROGRAMS.get(val) or KNOWN_TOKENS.get(val, (None,))[0]
            summary[key] = known or (val[:16] + "...")
        elif isinstance(val, dict):
            summary[key] = {k: str(v)[:30] for k, v in val.items()}
        else:
            summary[key] = val
    return summary
