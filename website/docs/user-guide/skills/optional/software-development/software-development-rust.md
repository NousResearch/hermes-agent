---
title: "Rust — Ownership, error types, cargo gates, unsafe/FFI UB checks"
sidebar_label: "Rust"
description: "Ownership, error types, cargo gates, unsafe/FFI UB checks"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Rust

Ownership, error types, cargo gates, unsafe/FFI UB checks.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/software-development/rust` |
| Path | `optional-skills/software-development/rust` |
| Version | `0.1.0` |
| Author | hope (rlaope), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `rust`, `cargo`, `clippy`, `borrow-checker`, `unsafe`, `ffi`, `miri`, `sanitizers`, `loom` |
| Related skills | [`systematic-debugging`](../../bundled/software-development/software-development-systematic-debugging.md), [`test-driven-development`](../../bundled/software-development/software-development-test-driven-development.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Rust Skill

Make Rust changes with the ownership shape, error type and gate list settled before the code,
and escalate deterministically to undefined-behaviour checks (Miri, sanitizers, loom) whenever a
change touches `unsafe`, raw pointers, FFI or lock-free code. It does not cover debugging a
stripped native binary or reviewing someone else's finished crate.

## When to Use

- Writing or changing Rust code: a new module, a port into Rust, a refactor.
- Borrow-checker, lifetime, `Send`/`Sync` or trait-bound errors that won't go away.
- Any change that adds or edits `unsafe`, `extern "C"`, `MaybeUninit`, `transmute`, atomics.
- Deciding whether a crate's `cargo` gates are enough to call a change done.

Don't use for: a Rust vocabulary question with no change behind it (answer directly), or a
service/API design that merely names Rust as the stack.

## Prerequisites

- A Rust toolchain (`rustup`, `cargo`). Check with `terminal(command="cargo --version && rustup toolchain list")`.
- Escalated changes also need nightly + Miri: `terminal(command="rustup +nightly component add miri")`.
  Sanitizers need nightly and a supported target (Linux/macOS x86_64 or aarch64; not MSVC for most).
- `loom` is a dev-dependency the crate opts into; don't add it without telling the user.

## How to Run

Run every gate through `terminal` from the crate root and report each one's real exit status.
Read the crate first with `read_file` / `search_files`: `Cargo.toml` (edition, MSRV
`rust-version`, features), the existing error type, and the CI workflow so you run the gates the
repository actually runs, not just the generic list below.

## Quick Reference

| Gate | Command | Proves | Does not prove |
| --- | --- | --- | --- |
| format | `cargo fmt --check` | formatting | behaviour |
| lint | `cargo clippy --all-targets -- -D warnings` | lint-clean at the crate's level | runtime behaviour |
| test | `cargo test` | the existing tests and doc tests pass | paths without tests |
| release | `cargo test --release` | behaviour under optimisation | debug assertions (they stop running) |
| docs | `RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --no-deps` | intra-doc links resolve | doc tests (`cargo test` runs them), API quality |
| Miri (escalated) | `cargo +nightly miri test` | no Miri-detectable UB on executed paths | paths no test reaches |
| ASan (escalated) | `RUSTFLAGS=-Zsanitizer=address cargo +nightly test --target <host-triple>` | no detected memory fault in that run | other inputs/interleavings |
| loom (escalated) | `RUSTFLAGS="--cfg loom" cargo test --release <loom tests>` | modelled interleavings are sound | interleavings outside the model |

## Procedure

1. **Escalation check first.** Before any code, decide whether the change is *escalated* using the
   trigger list in `references/ub-escalation.md` (`unsafe`, raw pointers, FFI, `MaybeUninit`,
   `transmute`, `unsafe impl Send/Sync`, weaker-than-`SeqCst` atomics, hand-written `Pin`
   projection, `#[repr]` on an FFI/transmute type). If you can't inspect enough to answer, the
   verdict is escalated (say it is conservative). Done when the verdict and the trigger that
   decided it are stated in your first message about the change.
2. **Ownership shape.** Name who owns each value; which borrows cross a function return, struct
   field, `.await` or thread spawn (a borrow held across `.await` is the usual reason a future is
   not `Send`); every deliberate `clone`, `Arc`, `Rc<RefCell<_>>`/`Mutex` and the reason. When
   the checker refuses, climb this ladder in order: restructure ownership → narrow the borrow's
   scope → split the type → clone deliberately → interior mutability. `unsafe` is not a rung;
   reaching for it re-runs step 1. Done when every clone and interior-mutability wrapper the
   change adds has a one-line reason.
3. **Errors and API.** Name the error type and where conversion happens (libraries: their own
   error enum, `thiserror`-style; binaries may collapse to one boxed/`anyhow` error at `main`; don't
   mix the two inside one crate). List every surviving `unwrap`/`expect`/`panic!` with its
   justification — "guaranteed by the constructor" is one, silence is not. Prefer types that make
   illegal states unrepresentable (newtypes, enums over strings, exhaustive `match`). State the
   semver impact of any public API change now, not at release.
4. **Async and concurrency.** Name the runtime; never add a blocking call inside an async task
   (`spawn_blocking` or an async API instead). Shared mutable state names its primitive and its
   lock order. State what happens when a future is dropped mid-operation (cancellation safety).
5. **Implement**, then **run the gates** from Quick Reference plus the repo's own CI gates.
   Report each gate separately with its real result. Done when every gate has an observed pass,
   or a failing gate is reported with its output.
6. **Escalated changes only:** load `references/ub-escalation.md` and satisfy all four blocking
   items — a `// SAFETY:` invariant comment on every `unsafe` block, Miri on tests that reach the
   `unsafe` path, a sanitizer where Miri can't reach (FFI, real I/O), and loom-style exhaustive
   interleaving checks for lock-free/atomic code. Done when each item is observed green or the
   change is reported as blocked with the smallest substitute proof named.

## Pitfalls

- **"It compiles" is one gate of five.** The compiler does not check the invariant an `unsafe`
  block asserts; never call an `unsafe` block safe because it builds.
- **Clones until it compiled** means ownership was never designed. Name the owner and re-derive
  the borrows.
- **`Rc<RefCell<_>>` everywhere** moves a compile-time error to a runtime panic or deadlock.
- **`unwrap` in a library path** assumes a contract instead of encoding it.
- **Green Miri on a test that never reaches the `unsafe` path proves nothing.** Write the test that
  reaches it first.
- **A stress loop is not a concurrency proof.** Ten thousand passing runs sample the interleaving
  space; loom explores it.
- **Toolchain can't run Miri/sanitizers?** The change stays blocked. Never downgrade the verdict
  because the tool is missing.
- Edition 2024 requires `unsafe extern` blocks and `#[unsafe(no_mangle)]`; check the crate's
  edition before suggesting FFI syntax.

## Verification

- The escalation verdict and its deciding trigger were stated before code.
- Every gate in Quick Reference (plus the repo's CI gates) has a real result from `terminal`.
- Escalated: every `unsafe` block has a `// SAFETY:` comment, and Miri/sanitizer/loom results are
  observed, not assumed.
- Every new clone, interior-mutability wrapper and surviving `unwrap` has a stated reason.

Ported from rlaope/oh-my-hermes (MIT), `skills/omh-rust`.
