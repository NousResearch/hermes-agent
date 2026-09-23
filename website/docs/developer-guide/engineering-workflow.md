# Engineering workflow routing

`hermes engineering` runs a bounded planner → worker → host verifier → reviewer
sequence in one Hermes process. The operator chooses a provider and model for
all three model stages from the existing model options catalogue and chooser.
The route is checked again before every model call. There is no MoA dispatch,
persona resolver, delegated child agent, or provider promotion.

## Run it

Create a JSON file containing the checks that will decide success. The operator
owns this file; model output cannot add, remove, or change a check.

```json
[
  {"id": "unit", "argv": ["python", "-m", "pytest", "tests/unit", "-q"], "timeout": 120}
]
```

Run from a Git worktree whose `.git` is a file, with a preinstalled Docker
image that contains the project dependencies:

```sh
hermes engineering --objective 'Repair the specified defect' \
  --workspace /absolute/path/to/worktree \
  --checks-file /absolute/path/to/checks.json \
  --backend docker --image local-engineering:1
```

The CLI asks for planner, worker and reviewer routes through the same
provider/model catalogue used by Hermes's existing model picker. It refuses
noninteractive input rather than silently selecting its first item. The
`--backend native` option is explicit and has weaker isolation. A missing
Docker daemon or image does not switch to native execution. The CLI returns
machine-readable status and reason codes; a non-DONE result exits nonzero.
The current default is three attempts, two reviewer replans and 24 model
calls in total. A worker can return `BLOCKED` with a required decision.

## Security and verification

Model inference runs in the parent Hermes process. Child processes receive a
new allow-listed environment with a temporary home and no inherited provider
key, OAuth token or `.env` values. A worker sends only one bounded JSON action
at a time. Commands use argv and `shell=False`; command output is capped and
known parent credential values are redacted before being sent to a model.

The Docker backend uses an ephemeral container with no network, dropped Linux
capabilities, a read-only container filesystem and no host home mount. It
refuses a normal clone's `.git` directory, links and Windows reparse points,
local dependency directories, and common credential filenames in the mounted
worktree. The worktree itself is writable
because the worker must edit source. Keep credentials and private data outside
that worktree. Docker isolation is unavailable when Docker cannot start. On interruption or
timeout, the host removes the named container and blocks if cleanup cannot be
confirmed.

The host runs each operator check and issues receipts tied to run, worktree,
attempt, plan revision, source digest and check ID. Missing, duplicate, stale,
foreign, incomplete and timed-out results are rejected. A zero exit status
from every check is necessary for DONE. The source digest is recomputed after
each check and before DONE. These receipts attest to the chosen checks only;
they do not prove that the checks cover every requirement.

## Isolation limits

Native execution limits inherited environment variables and temporary home
but does not isolate filesystem visibility, OS credential stores or keychains,
process handles and descriptors, network, shared memory, or process inspection.
Child process inheritance differs by platform. Docker narrows these surfaces
but the writable worktree remains visible, and a privileged host or Docker
daemon is outside this boundary. An operator should inspect the worktree for
unrecognised secrets before using it; filename screening is deliberately
conservative and cannot detect every credential format. Neither backend
turns arbitrary model-generated commands into trusted code.

## Related routing work

PRs #103346 and #87179 concern operator-defined routes for delegated children.
This workflow controls the stage sequence, verification and termination. It
uses the current model catalogue and the existing parent-owned auxiliary
inference path; it does not copy their profile or persona resolution.
