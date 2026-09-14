# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Installation

```bash
yarn
```

## Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Documentation search

DocSearch is enabled only when `ALGOLIA_SEARCH_API_KEY` is supplied at build time.
Local builds and source archives contain no default key; the site builds without search when the variable is absent or blank.

For the hosted site, create an Algolia key with only the `search` permission, restricted to the documentation index.
Store it in the `ALGOLIA_SEARCH_API_KEY` secret for the repository or its `github-pages` environment before merging this configuration change.
The deployment workflow requires this secret before building, so missing configuration leaves the previous deployment in place.

The generated browser bundle contains the configured key, as required by DocSearch.
Never supply an admin key or a key with unnecessary `browse`, write, or delete permissions.
Moving a key out of source does not revoke an older key; the Algolia application owner must replace or revoke it separately.

Run the configuration checks after installing website dependencies:

```bash
node --test docusaurus.config.test.mjs
```

## Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.

## Diagram Linting

CI runs `ascii-guard` to lint docs for ASCII box diagrams. Use Mermaid (````mermaid`) or plain lists/tables instead of ASCII boxes to avoid CI failures.
