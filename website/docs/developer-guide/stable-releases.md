# Stable release admission and promotion

`Stable Release` is the release gate. A successful builder alone is not a
stable release. Stable releases use attempt refs as locks and final tags as
publish receipts; neither tag is a workflow trigger. Canary builds have their
own scheduled workflow.

The committed project version is always `0.0.0`. Release jobs derive the payload
version from the admitted ref and stamp isolated build trees. Do not bump version
files on `main`.

## Order

1. Refresh `origin/main` and the remote attempt and marker refs (`rc.*` and
   `abandoned-rc.*`). Derive the next SemVer from the published release family
   seeded at `0.21.4` alone: the newer of the protected R2 stable head and the
   newest published non-prerelease GitHub release with a `vX.Y.Z` tag. A
   release that skipped bundles moves only the second. Attempts do not move the
   version line. A cut whose next version already has a final `vX.Y.Z` tag is
   refused until that publication finishes.
2. Push an annotated `rc.<N>-vX.Y.Z` attempt ref atomically, create one
   non-prerelease GitHub draft on it, and dispatch `Stable Release` on that exact
   ref. The attempt number comes from the existing attempt refs of that version.
   The claim message binds its commit, attempt number, autopublish policy,
   `skipBundles` and `skipTests` flags, and
   one monotonically allocated epoch. That epoch is the release date and native
   packaging clock for every matrix leg and retry. One outstanding attempt, of
   any version, blocks a new `release`.
3. Admit the exact remote annotated tag-object SHA, peeled commit, `GITHUB_REF`,
   `GITHUB_SHA`, checked-out `HEAD`, and ancestry on `origin/main`.
4. Run the whole source, Docker, Nix, PM bundle, install/update, Termux, Windows,
   signed-package, and native-upgrade acceptance graph.
5. Push immutable versioned Docker and R2 artifacts from the tested bytes. The
   Docker image is pushed as soon as its own tests pass, tagged by the attempt
   ref. Do not move `stable` or `latest` aliases yet and do not rebuild for
   publication.
6. Create the annotated final `vMAJOR.MINOR.PATCH` receipt at publish, not at
   green. It binds the winning attempt's ref, object, commit, archive prefix,
   candidate-manifest SHA256 (`null` when the claim skipped bundles), Docker
   manifest digest, and autopublish policy.
7. The stable publication controller resolves releases oldest first. It creates
   `vX.Y.Z`, retargets the still-draft release onto it, strips the warning
   blocks, makes the release public as the last call, then verifies and promotes
   the receipt-bound Docker digest and advances App Installer, macOS, APT,
   downloads-page, and protected R2 heads.

A sole green draft waits unless its claim selected autopublish. A later green
claim flushes all contiguous earlier green drafts in order. An older running claim
blocks newer publication. A burned attempt is skipped, and its version is cut
again after `abandon`. Failed, cancelled, missing, and unexpectedly skipped
requirements remain red.

Docker Hub, R2, APT, GitHub, and the Store do not support one cross-service
transaction. The controller is therefore idempotent: after a partial failure,
run it again and let every mutation verify its current state before continuing.
The final tag is the custody receipt. Never rebuild or replace accepted bytes to
repair a pointer.

Promotion replaces the stable downloads page at `releases/stable/index.html` on
the R2 public origin. Canary tag builds own `releases/canary/index.html`, and
commit builds own `releases/commit/<sha>/index.html`. Pages list only staged,
receipt-backed objects; an older run cannot regress a protected channel.

## Run, publish, or abandon a stable release

Start from an exact commit already on remote `main`:

```sh
python scripts/release.py release --commit "$(git rev-parse origin/main)" --bump patch --remote origin
```

`--bump` defaults to `patch`; pass `minor` or `major` only when that change is
intentional. The selected commit must descend from the newest published
`vX.Y.Z`. With no published tag, any commit already on remote `main` is
accepted. An abandoned attempt puts no constraint on the next cut; it was often
abandoned because its commit is bad. `release` also refuses while any attempt,
of any version, is outstanding. The refusal prints the attempt's workflow run
URL, the `abandon` command, and the rerun command before it raises.

The attempt ref is a public lock, not a SemVer prerelease. It reads
`rc.1-v0.21.5`, not `v0.21.5-rc.1`. The package version stays plain `0.21.5`.
`parse_attempt_ref` in `scripts/releases/versioning.py` is the one grammar for
the shape.

The draft body carries a fenced warning block at the top and at the bottom: do
not publish the release from the GitHub UI. Publishing it by hand skips the
`vX.Y.Z` receipt tag, the update feeds, the Docker aliases, and the Store
check. Published releases are immutable, so a hand-published release cannot
be fixed afterwards, and it blocks the pipeline: the attempt then has neither a
marker ref nor a final tag, so `release` refuses it as outstanding and
`abandon` refuses it because it is published. The publication pass strips both
fenced blocks while the release is still a draft.

Add `--autopublish` to publish immediately when the claim becomes the oldest
green release. Without it, the release stays a draft until an explicit publish
or a later green claim forces ordered resolution.

### Skip bundles or tests

Two `release` flags remove parts of the pipeline. They can be used together,
and they combine with `--autopublish`.

```sh
# Tag, GitHub release and Docker image only
python scripts/release.py release --commit "$(git rev-parse origin/main)" --skip-bundles --remote origin
# Emergency release: build and publish everything, run no tests
python scripts/release.py release --commit "$(git rev-parse origin/main)" --skip-tests --remote origin
```

| | `--skip-bundles` | `--skip-tests` |
|---|---|---|
| Source CI (`ci.yaml`), Nix, Termux, Windows live, install/update E2E, bootstrap identity | run | skipped |
| Docker image | built, tested, published | built and published, `tests/docker` skipped |
| Native PM bundle check | skipped | skipped |
| Desktop and Termux candidates | skipped | built, signed and staged, with no native smokes or in-build test suites |
| Signed-package upgrade acceptance (`transitions-*`, `*-packaged`) | skipped | skipped |
| Publication | final tag, GitHub release, Docker `stable`/`latest` aliases | everything, as a normal release |

The flags are written into the claim message, never passed as workflow
inputs. `admit` reads them from the claim and emits `skip-bundles` and
`skip-tests`. Every job condition and every gate reads those outputs. A
recovery rerun cannot change them. To change a flag, `abandon` the attempt and
cut again.

The gates stay strict. `scripts.releases.stable gate` reads the flags and
requires each job the flags remove to report `skipped`, and every other gated
job to report `success`. A job that ran although a flag removes it also
blocks the release. `SKIPPED_BY` in `scripts/releases/stable.py` is the one
table of which flag removes which job.

**`--skip-bundles`.** The draft, Docker image, final tag and GitHub release are
the whole release. No candidate manifest exists, so the final tag records
`candidateManifestSha256: null`. Publication moves only the Docker aliases. The
protected R2 stable head, App Installer and macOS feeds, APT channel,
downloads page, `releases/stable/release-candidates.json`, signed-package
baseline, and Store submission all stay on the previous bundle release. Source
checkouts on the official repository follow
`releases/stable/release-candidates.json`, so they also stay on the previous
bundle release. Such a release is complete when the Docker `stable` alias
carries the digest its final tag binds. The sequencer uses that alias, next to
the R2 head, to decide which releases still need their publication pass.

**`--skip-tests`.** Every artifact is built, signed, staged and published the
same way as a normal release. The candidate manifest records each native smoke
as `skipped`, never as passed. The next release uses that manifest as its
upgrade baseline like any other. Every reader that knows the claim (`complete`
and the protected R2 advance) refuses a manifest whose smoke results disagree
with the claim's `skipTests` flag. Use it only for an emergency fix, and cut a
normal release after it.

The claim push is the atomic version lock. Two callers may derive the same next
version, but only one push wins; the loser reports the winning tagger, time, and
commit. A rejected or abandoned claim remains spent. To publish or abandon:

```sh
python scripts/release.py publish --version 0.21.5 --remote origin
python scripts/release.py abandon --version 0.21.5 --remote origin
```

`publish` performs a synchronous supersession preflight, then dispatches the same
ordered controller used by automatic recovery. It refuses a known burned version
below a newer published release. `abandon` deletes the draft when one exists,
writes an `abandoned-rc.<N>-vX.Y.Z` marker ref, and keeps the attempt ref. The
marker is the record of abandonment; the attempt ref is never deleted. The
version is not spent, so the next cut is `rc.<N+1>-vX.Y.Z`.

Do not manually dispatch `Stable Release` from a final tag. Recovery keeps the
original claim ref, object SHA, commit, draft database ID, autopublish policy,
and skip flags.

## Failure and recovery

When a stable run fails, its completion event starts `Stable Release
Publication`, which at once reruns only the failed jobs in the same GitHub
Actions run. There is no backoff and no schedule. The pass applies only the
oldest unresolved retry, because GitHub keeps only one pending run in the shared
stable-release concurrency group. The rerun waits in that group until the pass
ends. At most two retries are admitted (run attempts 2 and 3). After attempt 3
fails, the claim is burned and the sequencer may resolve later claims. A lost
retry request, or a crash between publication mutations, is recovered by the
next failure event or by dispatching `Stable Release Publication` by hand.

A newly pushed claim with no observed workflow run remains unresolved for one
hour. This grace window covers the non-atomic draft and dispatch steps. After the
hour, a still-unstarted claim is derived as burned. No claim is burned during the
normal creation window.

Desktop and Termux handoffs live in the immutable R2 tag archive. The archive
key is the attempt ref: `releases/tag/rc.<N>-vX.Y.Z/`. `releases/tag/vX.Y.Z/`
is never written. Each producer writes a `handoff-<target>.json` receipt
containing the tag, commit, paths, sizes, and SHA256 digests. Per-arch receipts
live beside the candidate manifest under that prefix. Candidate assembly emits
one pinned `release-candidates.json`; consumers verify its digest and do not
re-upload the packages. Its digest is recorded nowhere before publish: the
publication pass hashes the archive copy and writes that hash into the final
tag. Docker publication pushes one immutable versioned manifest tagged by the
attempt ref and records its registry digest in the final tag. Delayed
publication promotes that digest registry-side, so it does not depend on
expiring Actions artifacts.

Windows, macOS, and Termux candidate jobs stage packages and metadata under
`releases/tag/<attempt ref>/` before acceptance. No stable feed is written at
this stage. The stable APT pool path also carries the attempt ref:
`releases/termux/stable/pool/<attempt ref>/<c>/<name>.deb`. The pool upload is
immutable with a one-year cache header, so a recut of the same version writes a
different pool key.
Immutable uploads accept an existing object only when its bytes match. If an
archive object, final-tag digest, versioned Docker tag, or read-back differs, stop
recovery rather than replacing the accepted candidate.

The candidate manifest binds the admitted claim epoch. Admission recomputes the
stable Windows quad from that epoch and rejects a merely well-formed but incorrect
MSIX, executable VERSIONINFO, or App Installer version. Native admission also
reads the Electron artifact filename and macOS plist from the built packages. The
acceptance graph stamps an isolated bootstrap-installer tree, asks Cargo to read
the resulting Tauri package version, builds and inspects Python wheel/sdist
metadata and filenames, and checks the Nix and Docker runtime identities. Any
consumer-facing `0.0.0` or mismatched version fails the gate.

The draft stays on the attempt ref until publish. Deleting it is `abandon`, and
it does not burn the version. The annotated final tag binds the GitHub release
database ID admitted with the claim. Publish edits the release only while it is
still a draft: it creates `vX.Y.Z`, retargets the release onto it, strips the
warning blocks from the body, and reads the tag and body back. Making the
release public is the last call. Published releases are immutable, so nothing
can retarget or re-tag one afterwards.

Nothing points a client at an attempt until publish. The updater compares
versions, not URLs, and that is safe only because of this. The diagnostic page
at `releases/tag/<attempt ref>/index.html` is not an update feed, and an
attempt installed from it is not upgrade-safe: every attempt of a version has
the same package version, so the published build never replaces it. The Docker
image is pushed early under the attempt ref; the `stable` and `latest` aliases
move with the feed in the publication pass. The Store submission is held: the
green run submits with auto-publish off. The submission API cannot release a
held submission, so the publication pass only checks it and prints a GitHub
warning; once certification passes, click Publish now for that submission in
Partner Center. A failed submission leaves the publication run red. Stable channel
records carry an optional `archiveRef` naming the attempt ref; `releaseTag`
stays `vX.Y.Z` and the protected prefix falls back to it when `archiveRef` is
absent.

The desktop workflow's optional `termux_upgrade_from_tag` input names an exact
published release with a Termux R2 handoff. Explicit non-publishing desktop builds
retain no downloadable job artifacts.

The desktop workflow takes a `jobs` input with the groups `darwin-arm64`,
`darwin-x64`, `win32-arm64`, `win32-x64`, `win32-bundle`, `linux-x64`,
`linux-arm64`, and `termux`. Stable calls it once per group, except the linux
groups, which wait for a real Linux build. Each Mac arch and the Windows bundle
stage a receipt, and each install arm starts from its own receipt as soon as
its own bytes are staged. `acceptance` is the one join that blocks publication.
The `smoke-win32-universal` job is gone; the per-arch MSIX smokes cover each
arch, and the Windows install arms install the `.msixbundle` on both arches.

## Tag namespaces and receipts

Before relying on attempt refs as locks, apply a repository ruleset for
`refs/tags/v*`, `refs/tags/rc.*`, and `refs/tags/abandoned-rc.*` that restricts
creation to organization administrators and the release integration and blocks
update and deletion. The ruleset is not applied yet; until an admin applies it,
these locks are honor-system — anyone with push access can delete a marker ref
or move a receipt tag. Attempt refs, marker refs, and final tags are immutable.
Channel and commit builds write annotated post-build receipts such as
`v0.0.7+channel.<YYYYMMDDTHHMMSSZ>.<run-id>` and
`v0.0.0+commit.<YYYYMMDDTHHMMSSZ>.<run-id>` only after publication succeeds.
Receipt tags do not create GitHub releases and do not trigger workflows.

Canary source identity is only
`v<stable>+canary.<YYYYMMDDTHHMMSSZ>`. Build metadata intentionally makes it
SemVer-equal to its stable core; the protected channel record and embedded UTC
timestamp decide progression. Desktop clients treat that validated channel
sequence as the update authority rather than asking SemVer to order build
metadata. `Canary Release` runs once a day at 06:41 UTC from the default
branch, and a manual dispatch starts one at any time. Runs queue with
`cancel-in-progress: false`. If `main` has no new commits, the run resumes an
unpublished canary or does nothing. Historical `-canary.` identities are
unsupported.
If a process stops after pushing the canary tag, rerunning the command verifies
the exact remote tag object, repairs the missing draft, and redispatches until the
protected canary head receipts that tag.
The protected publication controller verifies that exact annotated tag object
before it flips the GitHub prerelease from draft to public, reads both states back,
and only then advances the R2 head.

## Canary and one-off desktop identities

`release.py --canary` builds the separate canary application. Its package
identity and CLI command (`hermes-canary`) differ from stable; the existing
canary feed updates that application only. One-off builds use
`release.py --build-commit REV --remote REMOTE` (add `--publish` to dispatch).
Their application identity and CLI command (`hermes-<7-character-sha>`) include
the pinned commit. Two different commit builds do not replace each other.

Branding is selected from those build inputs, not from runtime settings:
canary uses yellow/dark-yellow icons; one-off builds use red icons bearing
the short SHA. All desktop icon formats derive from the same artwork.

One-off stamps use `source: commit-build`. No app update feed or App Installer
subscription is published for them, and both the GUI and bundled CLI refuse
update requests. They direct the recipient to ask the developer for a new
build. Source checkout channels are separate: `hermes update --set-channel`
remains available there and selects the published release's source commit.

`--build-commit` prints its deterministic downloads-page URL before dispatch,
including in dry runs:
`https://hermes-assets.nousresearch.com/releases/commit/<full-sha>/index.html`.
`CLOUDFLARE_R2_PUBLIC_URL` overrides the public origin. After admission, the
commit summary runs even when a build or assembly job fails; it lists only
receipt-backed existing downloads and marks missing binaries as not built.
Missing binaries link to the workflow run under **View build run**, not to
nonexistent downloads. Disabled platforms have no download or failure link.
Page publication still requires working R2 access. The commit links to its source
on GitHub; tag and channel pages link to the corresponding GitHub release tag.
Commit pages also list explicit non-secret `--bundle-env` defaults and
`--bundle-unset` clears passed to the desktop bundles, not the CI environment.
Values are shown as JSON strings (including `""` for an empty value); clears are
labeled **Unset**. The section is omitted when no overrides were supplied.

Tagged builds also publish a per-tag diagnostic page at
`releases/tag/<tag>/index.html` after build or feed failures, including when no
artifacts were uploaded. An incomplete build does not advance the channel page
or pass the release-success gate.

Store submission retains its fixed official stable identity. Nonstable
packages must not be submitted under that identity.

## Dynamic channels in R2

Channel names are R2 objects, not a repository registry. A preview channel owns
one native application identity across exact-commit builds. The immutable build
request records its source commit, bundle defaults, channel sequence and package
versions separately. The existing native build and smoke jobs must all pass
before the channel head advances.

Preview a custom build, then explicitly dispatch it. Repository identity does
not select behavior: the same direct dispatch runs from any GitHub remote, but
`--channel` performs no R2 access locally — it resolves the exact pushed commit
and dispatches the default-branch workflow, whose privileged allocation step
creates the channel and mints the immutable build request in CI. The local
command needs only a `gh` token with write, maintain or admin permission on the
selected repository; no R2 credentials are required.

```sh
python scripts/release.py --channel pm-preview --build-commit my-branch --remote origin
python scripts/release.py --channel pm-preview --build-commit my-branch --remote origin --publish
python scripts/release.py --channels --remote origin
```

Disposable R2 scoping is opt-in, for test runs only. Dispatch the desktop
workflow with `disposable_channel` and `build_commit` to allocate a namespace
under `ci-disposable/<repository-id>/<run-id>/`, then use the exact scoped build
command from its summary. For local administration of that allocation, carry
its `R2_DISPOSABLE_RUN` and `GITHUB_REPOSITORY_ID` in the command environment,
with the configured public URL still at the unscoped root. These are test-run
inputs, not persistent application settings; the repository ID is checked
against the selected remote before credentials are read, so a scoped namespace
still belongs to exactly one repository.

The first publishing invocation creates the channel (during CI allocation), and
later invocations retain its identity. Requests and artifacts live under
`releases/channel-builds/BUILD_ID/`; the mutable pointer is
`releases/channels/NAME.json`. A failed build leaves its previous head intact.
Conditional writes reject stale publication and permanently retired channels.
Retrying re-dispatches `--channel NAME --build-commit SHA --publish`, which
allocates a fresh sequence slot in CI; there is no separate resume command.

Retirement pins the current official stable build as the first receiver:

```text
python scripts/release.py --retire-channel pm-preview --to stable --minimum-version VERSION --remote origin
```

Add `--publish` only after reviewing the dry run and the exact native acceptance
evidence. This does not rebuild stable under the preview identity or silently
uninstall clients. Protocol-aware clients offer a consented cross-application
handoff; the destination must confirm readiness before preview removal. Keep
the retirement object and pinned artifacts available for offline clients.
Existing one-off builds have no retirement reader and require replacement.

The destination manifest must declare receiver support read from its packaged
stamp. The existing protected release workflow owns acceptance; there is no
separate public certification document or dependency on expiring Actions artifacts.
Native signature and identity checks, recipient consent, preserved-state preflight,
and destination readiness remain mandatory. An offline preview reaches its pinned
first receiver even after stable advances; that app then updates through stable.

Before shipping R2-only source readers, seed the existing `main` source-branch
record and published stable/canary records through the explicit protected
bootstrap operation. Review actual accepted manifests; do not invent a native
manifest for `main`. Production seeding, CDN cache/CAS verification and native
signed-package qualification are release operations, not implied by a passing
local helper suite. Never store the bootstrap output as a repo channel list.

## Signed-package baseline

The last successful stable release that shipped bundles records
`releases/stable/release-candidates.json` on the configured R2 public origin.
A release that skipped bundles does not replace it.
It identifies actual Windows universal MSIX bundles, macOS ZIPs and package
provenance. The next run combines those records with its candidate manifest
and uses the existing native bundled-update drivers.

For an existing stable release that predates this metadata, supply
`baseline-manifest` as an HTTPS URL on the configured R2 public origin to a
current schema-2 manifest of its actual published packages and successful native
smoke results. Schema-1 candidates are rejected; there is no legacy admission
path. Manifest redirects must stay on the same origin. The baseline tag must be
a published stable release, package identities must agree, and versions must
increase. Stable Windows
packages use electron-builder's `storePackageVersionAt` policy over the admitted
claim tag object's immutable tagger timestamp (`year.hourOfYear.secondOfHour.0`),
independently of the SemVer payload tag.
Missing baseline
artifacts are a blocker, not permission to fabricate or skip acceptance.
See [the bundled update contract](https://github.com/NousResearch/hermes-agent/blob/main/tests/install/BUNDLED_UPDATES.md).

## Explicit exclusions and policy

- Desktop Playwright E2E (`e2e-desktop.yml`) is deferred at the owner's request
  because it is flaky. It is reported as deferred, not passed. Stabilize it
  and prove repeatable CI runs before adding it to this gate.
- Install/update E2E and native signed-package acceptance are **not** deferred.
  Only a claim cut with `--skip-tests` removes them, and the gate then requires
  them to be skipped.
- PR-only history, label and diff review checks do not apply to a stable tag.
  All applicable source CI jobs still run, regardless of changed paths.
- OSV vulnerability findings retain their existing advisory policy. Required
  scanner execution failures are failures, not advisory findings.
- Disabled Linux desktop packaging is not claimed as shipped. Native Linux
  PM bundles, Docker, Nix and install/update checks remain required.
- Housekeeping, autofix, comment and skills-index/deploy workflows are not
  release acceptance suites.

## Implementation ownership

Actions owns job ordering, runner selection, permissions and environments.
Python owns shared release admission, manifests, artifact hashes, publication
and channel promotion under `scripts/releases/` and `scripts/bundles/`.
Electron-builder configuration/hooks and native Windows/macOS adapters remain
in JavaScript or PowerShell. These adapters consume release facts rather than
reimplementing the release gate. Gate jobs use only Python's standard library;
they do not install the application or the JS workspace to report a verdict.

`scripts.bundles.release_artifacts` owns App Installer XML and feed publication.
Its serializer takes explicit package identity, publisher, version, subscription
URI and artifact URI. Stable promotion uses the accepted candidate metadata;
canary publication verifies the native bundle manifest against the adapter's
expected identity before uploading the bundle, then the descriptor. Native SDK
bundling/signing stays in `stage-msixbundle.mjs`. Store and commit builds stop
before that feed handoff; `stable-store` submits the verified candidate without
rebuilding it. Native acceptance uses the same Python serializer, including its
12-hour on-launch check policy.

Signing and publication credentials stay in their protected job environments.
The source CI call does not inherit deployment secrets. Configure the existing
release-signing and container-publish environments before running this pipeline.
