# Pinned Input/Q/core composition replay

This **source-only** recipe reconstructs the declared dependency composition used for the nine-case Input proof. It does not merge the lower owners into `main`, change production behavior, grant protected authorization, or replace the standalone Input owner's tests. The Input contribution remains attributable to its published source commits; the replay's two local merge commits are synthetic and are not for publication.

From a checkout containing `scripts/replay_input_declared_composition.py`, with Python 3, Git, and access to public GitHub, choose a **new, absent directory** whose parent already exists:

```sh
python3 scripts/replay_input_declared_composition.py "$PWD/../input-q-core-replay"
git -C "$PWD/../input-q-core-replay" rev-parse 'HEAD^{tree}'
git -C "$PWD/../input-q-core-replay" status --porcelain
```

The expected tree is `5ff1604d08aa31d9286a79c443af94a92a7c22b1`; status is empty. The script initializes a fresh SHA-1 Git repository, fetches exact commit objects from `https://github.com/dokterdok/hermes-agent.git` with shallow depth 1 and then Q/core at depth 10, and checks the Q/core merge base. It does not clone a local checkout, install alternates, use a promisor pack or local remote, or accept ambient Git object/config overrides. Each fetch is bounded to 110 seconds. A fetch or geometry failure stops rather than substituting private objects. Re-run with another absent destination if needed; it never overwrites an existing directory.

| Source | Pinned commit | Role |
| --- | --- | --- |
| Historical Input base (#106742) | `485d5f6848c2598ca00fa7704e50e8ae66d0983a` | Full published Input owner delta starts here |
| Q (#99107) | `43b9f02183444de0d48dc72170341346d58e09dc` | Passive/post-preparation refusal provider |
| Core (#111216) | `2fb90a347b5a0d7864367b221544dc058c6cf70c` | New-admission transaction guard |
| Q/core common base | `b36398f71929d6069905ccca59af8d1516c5ee3a` | Verified by `git merge-base` in the shallow fetch |
| Input (#111362) | `c061ac2cad7e699823b1f5166be795298e64b994` | Parent `ccbd43704e4ab12ae32586075831d7c45338bca6`; tree `2639c4f767717442f9b207afe0f8cb0b7f0c884e` |
| Core public forward tip | `03ab73504dae640e8cfef0b19841c7053aa31cbe` | Fetched for identity, not imported; this shallow replay does **not** independently prove its thousands-of-commits ancestry from the pinned core owner |

The script derives the complete `git diff --binary --abbrev=11` between Input base and tip from fetched public objects, checks SHA-256 `6b9dd749ce20bea533427225bdc5b0b3d389b6c4a85177b74e34ce4d94a5300d`, merges core into Q once, and applies the owner delta once. It rejects any conflict paths other than `gateway/run_runtime.py`, `gateway/session_hosted_rpc.py`, and `hermes_state_runtime.py`. The fixed reconciliations retain Q's bootstrap and post-preparation authorizer refusal, core's write authorizer, and Input's runtime initialization, prepared payload, and custody. It checks the final complete tree and clean worktree. The replay does not fetch or execute private provider implementations.

The previously accepted focused evidence is **nine passing cases**: five direct-submit refusal controls and four native-input retention regressions. Tree equality connects this public-only replay to that exact source snapshot; it is **not a new test run**. To reproduce the focused selection after provisioning project dev dependencies, run from the replay checkout (the canonical runner isolates files and credentials; constrain resources appropriately on your host):

```sh
cd "$PWD/../input-q-core-replay"
scripts/run_tests.sh -j 1 --file-retries 0 --file-timeout 110 \
  tests/gateway/test_input_custody_lifetimes.py \
  tests/gateway/test_input_native_publication.py \
  -k '(test_refused_mixed_preparation_native_bytes_are_collectible or test_interrupted_consumed_native_image_waits_for_exact_raw_retirement or test_consumed_native_image_rejects_live_or_foreign_admission or (test_competing_winner_survives_failed_handoff_then_reclaims and refusal) or (test_cold_equal_images_preserve_order_with_one_physical_owner and True))'
```

The selection covers the two `refused_mixed_preparation` parameters, two `competing_winner` refusal parameters, one `cold_equal_images[True]`, two `interrupted_consumed` parameters and two `consumed_native_image` parameters. Do not add this independent run to the already accepted count. This recipe does not claim standalone Input green or authorize held F1 hosted-work-record, secondary Output permission provider, legacy atomic Stop/quarantine, or Disband work.
