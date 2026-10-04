# Website behavior tests

Using the pinned Node/npm toolchain, run `npm ci --workspace tests-js --ignore-scripts`
from the repository root, then `npm ci --prefix website` and `npm test --prefix website`.
Tests reuse root Vitest/jsdom with the site's React. `npm run typecheck --prefix website`
checks the page and tests together.

The catalog test mounts the real marketplace page with React DOM; only the
Docusaurus navigation shell and feed transport are replaced. It exercises
category deep links, unknown categories, preserved URLs, and picker mode.

The tests do not install plugins, access credentials, or fetch external data.
