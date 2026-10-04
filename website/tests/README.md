# Website behavior tests

Using the pinned Node/npm toolchain, run `npm ci --workspace tests-js --ignore-scripts`
from the repository root, then `npm ci --prefix website` and `npm test --prefix website`.
Tests reuse root Vitest/jsdom with the site's React. `npm run typecheck --prefix website`
checks the page and tests together.

The catalog test mounts the real marketplace page with React DOM; only the
Docusaurus navigation shell and feed transport are replaced. It exercises
filters, sort controls, Featured placement, ownership badges, and star links.
The extraction test runs the real Python publishing script against temporary
catalog files and reads both generated feeds. It requires `python3` with
`ruamel.yaml`; set `HERMES_PYTHON` to an existing development interpreter when
that is not the interpreter on PATH.

The tests do not install plugins, access credentials, or fetch external data.
