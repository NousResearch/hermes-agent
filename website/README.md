# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

> **Reading the docs on GitHub?** The Markdown under `docs/` is authored for the rendered site at
> <https://hermes-agent.nousresearch.com/docs/>. Cross-page links are relative Markdown paths, so they
> follow through on GitHub's file viewer too. Every page on the site has an **Edit this page** link
> that opens the source file here.

## Authoring links in `docs/`

- Link to another page with a relative Markdown path, anchors included:
  `[Profiles](../user-guide/profiles.md)`, `[Bundles](../user-guide/features/skills.md#skill-bundles)`.
  Docusaurus turns the file path into the page route; GitHub follows the same path. Site routes
  (`/user-guide/profiles`, `/docs/user-guide/profiles`) only work on the rendered site — GitHub
  resolves them as repository paths and 404s, and the `/docs/` form also emits
  `/docs/zh-Hans/docs/...` 404s in the zh-Hans build because `baseUrl` is already `/docs/`.
- `python3 website/scripts/check_doc_links.py` fails on any route-style link in hand-authored pages
  (EN and the zh-Hans mirror); `--fix` rewrites them. It runs in the `Docs Site Checks` workflow.
  Generated pages (`user-guide/skills/{bundled,optional}`, `reference/*skills-catalog.md`) are
  produced by `scripts/generate-skill-docs.py`, which emits the same relative form.
- Pin `{#anchor}` on cross-linked headings so the zh-Hans mirror keeps the same id.

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
