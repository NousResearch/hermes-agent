---
title: "Vitest — Vitest fast unit testing framework powered by Vite with Jest-compatible API"
sidebar_label: "Vitest"
description: "Vitest fast unit testing framework powered by Vite with Jest-compatible API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Vitest

Vitest fast unit testing framework powered by Vite with Jest-compatible API. Use when writing tests, mocking, configuring coverage, or working with test filtering and fixtures.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/web-development/vitest` |
| Path | `optional-skills/web-development/vitest` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

Vitest is a next-generation testing framework powered by Vite. It provides a Jest-compatible API with native ESM, TypeScript, and JSX support out of the box. Vitest shares the same config, transformers, resolvers, and plugins with your Vite app.

**Key Features:**
- Vite-native: Uses Vite's transformation pipeline for fast HMR-like test updates
- Jest-compatible: Drop-in replacement for most Jest test suites
- Smart watch mode: Only reruns affected tests based on module graph
- Native ESM, TypeScript, JSX support without configuration
- Multi-threaded workers for parallel test execution
- Built-in coverage via V8 or Istanbul
- Snapshot testing, mocking, and spy utilities

> The skill is based on Vitest 5.x (beta), generated at 2026-06-22.

## Core

| Topic | Description | Reference |
|-------|-------------|-----------|
| Configuration | Vitest and Vite config integration, defineConfig usage | [core-config](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/core-config.md) |
| CLI | Command line interface, commands and options | [core-cli](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/core-cli.md) |
| Test API | test/it function, modifiers like skip, only, concurrent | [core-test-api](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/core-test-api.md) |
| Describe API | describe/suite for grouping tests and nested suites | [core-describe](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/core-describe.md) |
| Expect API | Assertions with toBe, toEqual, matchers and asymmetric matchers | [core-expect](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/core-expect.md) |
| Hooks | beforeEach, afterEach, beforeAll, afterAll, aroundEach | [core-hooks](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/core-hooks.md) |

## Features

| Topic | Description | Reference |
|-------|-------------|-----------|
| Mocking | Mock functions, modules, timers, dates with vi utilities | [features-mocking](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/features-mocking.md) |
| Snapshots | Snapshot testing with toMatchSnapshot and inline snapshots | [features-snapshots](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/features-snapshots.md) |
| Coverage | Code coverage with V8 or Istanbul providers | [features-coverage](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/features-coverage.md) |
| Test Context | Test fixtures, context.expect, test.extend for custom fixtures | [features-context](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/features-context.md) |
| Concurrency | Concurrent tests, parallel execution, sharding | [features-concurrency](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/features-concurrency.md) |
| Filtering | Filter tests by name, file patterns, tags | [features-filtering](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/features-filtering.md) |
| Test Tags | Label tests with tags to filter runs and apply shared options | [features-test-tags](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/features-test-tags.md) |
| Reporters | Built-in reporters, default selection, CI/output config | [features-reporters](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/features-reporters.md) |
| Benchmarking | Write benchmarks with the bench fixture (Tinybench) | [features-benchmarking](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/features-benchmarking.md) |

## Advanced

| Topic | Description | Reference |
|-------|-------------|-----------|
| Vi Utilities | vi helper: mock, spyOn, fake timers, hoisted, waitFor | [advanced-vi](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/advanced-vi.md) |
| Environments | Test environments: node, jsdom, happy-dom, custom | [advanced-environments](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/advanced-environments.md) |
| Type Testing | Type-level testing with expectTypeOf and assertType | [advanced-type-testing](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/advanced-type-testing.md) |
| Projects | Multi-project workspaces, different configs per project | [advanced-projects](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/web-development/vitest/references/advanced-projects.md) |
