---
title: "Shop — Shop catalog search, carts, checkout, and order tracking"
sidebar_label: "Shop"
description: "Shop catalog search, carts, checkout, and order tracking"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Shop

Shop catalog search, carts, checkout, and order tracking.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/shop` |
| Path | `optional-skills/productivity/shop` |
| Version | `1.1.0` |
| Author | Shopify |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Shopping`, `E-commerce`, `Shop`, `Products`, `Carts`, `Checkout`, `Orders` |
| Related skills | [`shopify`](../../optional/productivity/productivity-shopify.md), [`maps`](../../bundled/productivity/productivity-maps.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Shop (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [Shopify/shop](https://github.com/Shopify/shop): the project ships the skill
> directory (`skills/shop/`) with its catalog, transaction and CLI references,
> and embeds the same files in the `shop` command (`shop skill` prints them).
> `hermes skills install official/productivity/shop` pulls the current
> directory live from that repo (quarantined and scanned like any hub install)
> — this directory holds only the catalog metadata, so the vendored copy can
> never go stale.

`shop` lets the agent shop across millions of Shopify stores: search the Shop
catalog by text, image or similar item; read product options, stock and price;
build carts and checkouts; and, once the user connects their Shop account, use
saved addresses, Shop Pay and order history. A purchase needs the user's
explicit approval of merchant, items, quantities and total; steps the agent
cannot complete hand off to a checkout link. The skill drives the `shop` CLI,
which the agent runs through the `terminal` tool.

## Prerequisites

- Node.js 22.21+ or 24.5+ with the `shop` command installed:

  ```sh
  npm install --global @shopify/shop@latest
  ```

  Installing this skill does not install the command, and installing the
  command does not install this skill. Keep them in step: `shop --version`
  for the command, `hermes skills update shop` for the skill. On a Hermes host
  do not also run `shop skill --install`; the hub copy is the one Hermes loads.
- Signing in is optional. The skill offers to connect the user's Shop account
  with a device code (`shop account login`); searching and checkout links work
  without it.

Full documentation: https://github.com/Shopify/shop#readme
