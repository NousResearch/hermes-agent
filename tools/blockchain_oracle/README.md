# 🔮 Blockchain Oracle — Solana MCP Toolset

> **Giving Hermes Agent superpowers on the Solana blockchain.**

A Model Context Protocol (MCP) server that connects Hermes Agent to the Solana blockchain, enabling natural language queries for wallets, tokens, NFTs, transactions, whale movements, and network health.

## Tools

| # | Tool | Description |
|---|------|-------------|
| 🏦 | `solana_wallet_info` | Query any wallet's SOL balance, SPL token holdings (top 20), account type, and owner program |
| 🔍 | `solana_transaction` | Look up full transaction details by signature — parsed instructions, fees, balance changes, status |
| 🪙 | `solana_token_info` | Get token metadata: total supply, decimals, mint/freeze authorities, top 10 holders |
| 📜 | `solana_recent_activity` | Fetch recent transactions for any wallet (up to 25) with timestamps and status |
| 🎨 | `solana_nft_portfolio` | List NFTs in a wallet — identifies tokens with amount=1 and decimals=0 |
| 🐋 | `whale_detector` | Scan recent blocks for large SOL transfers — configurable threshold (default: 1000 SOL) |
| 📊 | `solana_network_stats` | Current slot, epoch progress, average TPS, supply info, node version, health |

## Architecture

```
User → Hermes Agent → MCP Protocol → Blockchain Oracle → Solana RPC (Mainnet-Beta)
```

The oracle is **stateless**, **composable**, and **independently deployable**:
- `server.py` — MCP server layer: tool registration, schema definitions, request routing
- `solana_client.py` — Blockchain layer: async Solana RPC client with JSON-RPC 2.0

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch with Hermes Agent
hermes-agent --mcp blockchain=hermes-blockchain-oracle
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `SOLANA_RPC_URL` | `https://api.mainnet-beta.solana.com` | Solana RPC endpoint |
| `HELIUS_API_KEY` | *(none)* | Optional Helius DAS API key for enhanced NFT data |

> 💡 For production use, consider a dedicated RPC provider like [Helius](https://helius.dev), [QuickNode](https://quicknode.com), or [Triton](https://triton.one) for higher rate limits.

## Key Features

- **Pure async** — Non-blocking httpx client for high throughput
- **Smart labeling** — Auto-resolves 10+ known tokens (USDC, BONK, JUP, mSOL…) and 10+ known programs (Jupiter, Orca, Metaplex…)
- **Address validation** — Base58 decode + 32-byte length check before any RPC call
- **Error resilience** — Every tool returns structured JSON with `error` field on failure, never throws
- **Whale detection** — Scans actual block data, deduplicates by signature, sorts by magnitude
- **Zero configuration** — Works out of the box with public Solana mainnet RPC

## Usage Examples

```
You: "Check the SOL balance of GsBd49...2kMp"
Hermes: That wallet holds 1,247.83 SOL along with 12 token holdings
        including 50,000 BONK and 2.4 JTO...

You: "Are there any whale movements happening right now?"
Hermes: 🐋 Detected 3 large transfers in the last block:
        • 50,000 SOL moved from Binance hot wallet → unknown wallet
        • 25,000 SOL transferred between two whale wallets...

You: "How's the Solana network doing?"
Hermes: Solana is healthy. TPS: 3,847 | Slot: 248,392,105 | Epoch 578 (63%)
```

## Author

**Deniz Alagoz** — [@gizdusum](https://github.com/gizdusum) · [Discord: gizdusum](https://discord.com) · [𝕏 @gizdusumandnode](https://x.com/gizdusumandnode)

## Links

- 📦 **Standalone repo**: [github.com/gizdusum/hermes-blockchain-oracle](https://github.com/gizdusum/hermes-blockchain-oracle)
- 🏠 **Hermes Agent**: [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
- 🧠 **Nous Research**: [nousresearch.com](https://nousresearch.com)

## License

MIT — see [LICENSE](https://github.com/gizdusum/hermes-blockchain-oracle/blob/main/LICENSE)
