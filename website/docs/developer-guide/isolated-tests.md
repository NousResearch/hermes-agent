---
title: Isolated test runner
---

# Isolated test runner (fail-closed, Docker)

`scripts/isolated_tests/main.py` runs an allowlisted test suite inside an offline,
immutable local Docker image. It refuses whenever containment cannot be shown.

Launch only as `python3 -I -S scripts/isolated_tests/main.py <plan|prove-human|run> ...`.
It is stdlib only and never imports Hermes.

## Inputs (all required, no passthrough)

| Flag | Rule |
|---|---|
| `--runtime docker` | anything else exits 4 (`sandbox-exec` is rejected) |
| `--docker` | absolute path, resolved, executable regular file |
| `--socket` | absolute local Unix socket (root- or self-owned); no TCP, no context, no `DOCKER_HOST` |
| `--image sha256:<64hex>` | local image ID; `--pull=never` |
| `--python` | offline test interpreter baked into the image, with pytest |
| `--snapshot` | clean source snapshot + `ISOLATION-MANIFEST.json` (tree must equal manifest) |
| `--suite` | key of `policy.SUITES` |
| `--inputs` | parent-owned trusted prepare inputs (`hermes-isolated-prepare-inputs/1`), outside the snapshot |
| `--receipt` | `prepare.py` receipt (`hermes-isolated-prepare-receipt/1`) bound to `--inputs`; malformed or mismatched → exit 8 |
| `--out` | new directory, disjoint from the snapshot; never mounted into the container |

## Containment

`--network=none`, `--read-only`, uid 65534, `--cap-drop=ALL`, `no-new-privileges`, `--init`,
CPU 1, memory 2 GiB with no extra swap, pids 128, nofile 256, fsize 64 MiB, `--log-driver=none`,
`--shm-size` 64 MiB, tmpfs `/scratch` 256 + `/var/tmp` 192 + `/tmp` 64 = 512 MiB (charged to the
memory cgroup), `HERMES_TEST_WORKERS=1` (HOME/TMP/cache/HERMES_HOME/HERMES_RUNTIME_DIR live there), wall 600 s,
stdout 4 MiB / stderr 1 MiB caps. The entrypoint is `/usr/bin/env -i` with a fixed
allowlist: image ENV is not inherited by the test shell, but `/usr/bin/env` itself starts
with it, so images with `Config.Volumes` or loader-risk ENV (`LD_*`, `GCONV_PATH`, `PYTHON*`, ...)
are refused by an offline `docker image inspect` before any payload. The only mount is the snapshot,
read-only. No Docker socket, host HOME or service socket is mounted. The docker CLI runs
with an allowlisted env and an empty scratch `DOCKER_CONFIG`. After a normal exit, a timeout, a
launcher exception, or a SIGINT/SIGTERM/SIGHUP received while the docker CLI run and its cleanup
hold the signal gate, the launcher tries to remove the container: only after its run label
matches, then confirms its absence by Docker's "No such container", and kills the CLI group.
The cleanup docker calls run in their own session, so a terminal Ctrl-C or hangup is not
delivered to them. If ownership, removal or absence cannot be confirmed, the result is exit 8
INVALID (`valid: false`) and the container may be left behind. A Ctrl-C outside the gate ends
the launcher with exit 8 without writing `result.json` or an attestation, and attempts no
further removal. A SIGKILLed launcher or host power loss cannot run cleanup. The three tmpfs
mounts must be present and writable. Residual Docker-managed mounts (`/etc/hosts`,
`/etc/hostname`, `/etc/resolv.conf`, `/dev/shm`, `/dev/mqueue`) are an upper bound: each may be
writable, read-only or absent (Docker Desktop mounts the three `/etc` files read-only under
`--read-only`), the probe records which are writable, and none is claimed scratch-only. Any
other writable mount, or a missing or read-only tmpfs mount, refuses the proof. The writable-mount inventory leaves
out the trees `/proc`, `/sys` and `/dev/pts` and the single point `/dev` (so `/dev/shm` and
`/dev/mqueue` stay in it), and records each one it skipped. `/dev` is not assumed safe: the
required `dev_write_denied` check must show uid 65534 cannot create a new file there. A
writable bind mount below an excluded tree is not seen by the inventory; this is a coverage
limit (Docker's default configuration adds none).

`network_denied` requires all of: a connect to a public address is refused, `lo` exists,
every other interface in `/sys/class/net` (directories only, so the `bonding_masters` file is
skipped) has `IFF_UP` clear, every flags read succeeds, and neither `/proc/net/route` nor
`/proc/net/ipv6_route` (a missing file passes) has a row for an interface other than `lo`.
Docker Desktop's LinuxKit kernel adds fallback tunnel devices (`gre0`, `sit0`, `tunl0`, ...)
to every network namespace; with `--cap-drop=ALL` the container has no `CAP_NET_ADMIN`, so it
cannot bring them up or add a route. The probe records each interface's flags.
`env_exact` requires the non-`ISOTEST_` env keys to equal the runner's allowlist, plus `PWD`
only when its value is exactly `/src`: `/bin/sh -c` exports the container workdir as `PWD`.
Any other extra or missing key fails.

The image is trusted as pinned: the runner cannot audit its contents.

## Snapshot (prepared by `prepare.py` from trusted inputs)

Provenance comes from a parent-owned trusted inputs file kept outside the snapshot, never from
the snapshot's own manifest. Schema `hermes-isolated-prepare-inputs/1`: `base_commit`,
`candidate_patches` (zero or more `{path, sha256, manifest: {path: sha256}}`),
`runner_allowlist` (`{path: sha256}` of new runner files) and `exception_policy`
(`{allowed: {path: sha256}, max_count}`). `.git`, `.env` (except `.env.example`), private keys,
`auth.json`, `.netrc`, `.pypirc`, `.docker` and databases can never be listed as exceptions.
Denied-name matching ignores case (`.ENV`, `ID_RSA` are denied); the `.env.example` allowance is
an exact-case match. All values are parameters.

`scripts/isolated_tests/prepare.py` then, in order: creates a new 0700 directory under the
given scratch; lists the base with `git ls-tree` under an allowlisted env (no system/global
config, hooks, credential helper, network or repository discovery), refusing absolute paths, `..`,
symlinks (120000), submodules (160000) and any mode other than 100644/100755; reads every blob
through one `git cat-file --batch` stream with bounded reads, checking each `<oid> blob <size>`
header against the requested object, refusing missing objects and anything over the
`policy.PREPARE_LIMITS` bounds (file count, file size, total bytes, wall time) before the body is
read, and re-hashing every body to its blob id; writes each file new (exec bit from the tree
mode); re-walks the written base against the ls-tree list and blob ids; applies each patch only
if its bytes match the pinned sha256 and it touches exactly its manifest paths with the pinned results; copies exactly the
allowlisted runner files; accepts denied names only as approved policy entries with matching
hashes; re-walks the final tree, writes the full `ISOLATION-MANIFEST.json`, makes it read-only,
re-verifies it, and writes the receipt to a new parent-owned path outside the snapshot. The
receipt links the inputs digest, patch digests, runner allowlist digest, exception policy
digest, limits digest and manifest digest.

Launch checks the trusted inputs first, then the receipt against them, and only then walks the
snapshot with the receipt's policy-checked `applied_exceptions`. The read-only bits are not an
OS boundary against a same-account writer. Content-level secret scanning is not performed
(needs_validation).

Trust contract (what launch does and does not prove):

- Base-file content is compared with git (`ls-tree` blob ids) only at prepare time. At launch,
  base content and any file not named by a patch manifest or the runner allowlist is bound only
  by the receipt's `manifest_sha256`.
- The receipt is unsigned and the trusted inputs file is an ordinary owner-only file. Neither is
  a boundary against a writer running as the same account: such a writer can rewrite the
  snapshot, the receipt and the inputs together.
- The parent (orchestrator) therefore records the trusted inputs file's sha256 and the receipt's
  `manifest_sha256` in a separate channel it controls (its own report or state record) when
  prepare finishes, and compares both before any launch. A mismatch means the run is not used.
- The base snapshot holds the raw blob bytes. In-tree `.gitattributes`, `info/attributes`,
  `export-ignore`/`export-subst` and clean/smudge filters are not applied, so a file can differ
  from a checkout (for example `*.ps1 text eol=crlf` files are LF in the snapshot).
- The post-write check compares the files on disk with the same object store the bytes came from
  (ls-tree list and blob ids). It proves faithful extraction, not an independent second source.
  Repo-local git config of the source repository stays inside the trust envelope.

## Phases

1. `plan`: validates inputs and prints the binding and exact argv. Starts nothing.
2. `prove-human`: a human at a real terminal retypes a random code read from `/dev/tty`,
   the canary container runs a probe (no target code), the results are shown, and a second
   code confirms them. Only if every check passes is `attestation.json` written to `--out`.
   No flag, env var or CI mode skips this.
3. `run`: exit 5 unless the attestation binding (runner, probe, image, policy, scope)
   matches exactly. Then runs `bash scripts/run_tests.sh <suite paths>` with
   `HERMES_PYTHON` set; a missing interpreter/pytest/bash exits 3 instead of provisioning.

The attestation is tamper-evident for honest operators, not a security boundary: anyone
who can write the repo or the attestation can forge it.

## Offline image

`docker/isolated-tests.Dockerfile` defines the test image. Building it is the only step that
uses the network, and it is a one-time step a human approves; the runner never builds or pulls.
The build context holds only that Dockerfile, `pyproject.toml` and `uv.lock`:

```bash
ctx="$(mktemp -d)"
cp docker/isolated-tests.Dockerfile "$ctx/Dockerfile"
cp pyproject.toml uv.lock "$ctx/"
DOCKER_CONFIG="$(mktemp -d)" docker build --pull=false --iidfile "$ctx/image-id" "$ctx"
cat "$ctx/image-id"   # sha256:<64hex>
```

Record the resulting image ID and pass exactly that value as `--image`; a rebuilt image is a new
ID and needs a new proof. The image pins its base by the repository `Dockerfile`'s Debian digest,
takes Python 3.14 and uv by the URL and sha256 in `pm/lock.json`, and installs the `uv.lock` test
set (CI's extras and `dev` group) into `/opt/isotest-venv` without installing Hermes itself, so
the read-only snapshot is the only Hermes code on `sys.path`. Its final stage has no `VOLUME`,
no ENV but `PATH`, and no compiler, `curl`, `uv`, `git` or SSH. The interpreter for `--python`
is `/opt/isotest-venv/bin/python`.

## Exit codes

0 ok · 1 target failed · 2 refused input · 3 prerequisite missing · 4 unsupported runtime ·
5 attestation missing/stale · 6 proof failed · 124 wall timeout.

8 INVALID (cleanup unverified, interrupted, output cap exceeded, snapshot changed, docker exit 125) ·
7 internal launcher error.

## Limits and prerequisites (review pending — not ready to run)

- Status: **Security static re-review, human proof and an offline image are required** before
  any use. No human command is published here; this page is a review-pending template.
- Linux containers only (cgroup v2 required); a Docker Desktop VM run proves nothing about macOS
  native behaviour. `sandbox-exec` is rejected.
- "Human-only" proof is an operating rule: a PTY-owning agent cannot be told apart from a human.
  Confirmation expires after 120 s (HOLD). Agents never run `prove-human`.
- The attestation and the snapshot receipt are tamper-evident for honest operators, not a
  boundary against a same-account writer. The snapshot is re-verified after the run.
- Without `.git` in the snapshot `run_tests.sh`'s `compileall -j 0` walks `sys.path`
  (wall/pids cost, needs_validation). Missing image/toolchain/deps fail closed; nothing is
  installed. `scripts/run_tests.sh` is not modified.
- After a clean pass `scripts/run_tests_parallel.py` writes `test_durations.json` into the repo
  root, which is the read-only snapshot; the durations cache write fails there and is reported as
  a stderr warning; the exit code reflects only test results (needs_validation until a run).
