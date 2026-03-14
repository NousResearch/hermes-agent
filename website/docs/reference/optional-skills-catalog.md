---
sidebar_position: 6
title: "Official Optional Skills Catalog"
description: "Catalog of official optional skills available from the repository"
---

# Official Optional Skills Catalog

Official optional skills live in the repository under `optional-skills/`. Install them with `hermes skills install official/<category>/<skill>` or browse them with `hermes skills browse --source official`.

## autonomous-ai-agents

| Skill | Description | Path |
|-------|-------------|------|
| `blackbox` | Delegate coding tasks to Blackbox AI CLI agent. Multi-model agent with built-in judge that runs tasks through multiple LLMs and picks the best result. Requires the blackbox CLI and a Blackbox AI API key. | `autonomous-ai-agents/blackbox` |

## blockchain

| Skill | Description | Path |
|-------|-------------|------|
| `solana` | Query Solana blockchain data with USD pricing — wallet balances, token portfolios with values, transaction details, NFTs, whale detection, and live network stats. Uses Solana RPC + CoinGecko. No API key required. | `blockchain/solana` |

## email

| Skill | Description | Path |
|-------|-------------|------|
| `agentmail` | Give the agent its own dedicated email inbox via AgentMail. Send, receive, and manage email autonomously using agent-owned email addresses (e.g. hermes-agent@agentmail.to). | `email/agentmail` |

## health

| Skill | Description | Path |
|-------|-------------|------|
| `neuroskill-bci` | Connect to a running NeuroSkill instance and incorporate the user's real-time cognitive and emotional state (focus, relaxation, mood, cognitive load, drowsiness, heart rate, HRV, sleep staging, and 40+ derived EXG scores) into responses. Requires a BCI wearable (Muse 2/S or Open… | `health/neuroskill-bci` |

## migration

| Skill | Description | Path |
|-------|-------------|------|
| `openclaw-migration` | Migrate a user's OpenClaw customization footprint into Hermes Agent. Imports Hermes-compatible memories, SOUL.md, command allowlists, user skills, and selected workspace assets from ~/.openclaw, then reports exactly what could not be migrated and why. | `migration/openclaw-migration` |

## research

| Skill | Description | Path |
|-------|-------------|------|
| `qmd` | Search personal knowledge bases, notes, docs, and meeting transcripts locally using qmd — a hybrid retrieval engine with BM25, vector search, and LLM reranking. Supports CLI and MCP integration. | `research/qmd` |

## security

| Skill | Description | Path |
|-------|-------------|------|
| `1password` | Set up and use 1Password CLI (op). Use when installing the CLI, enabling desktop app integration, signing in, and reading/injecting secrets for commands. | `security/1password` |
