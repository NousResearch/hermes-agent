// Docusaurus plugin: static pages for every plugin catalog entry and every author.
//
//   /docs/plugins/<name>          one page per entry in website/static/api/plugins.json
//   /docs/plugins/by/<slug>       one page per maintainer (slug from extract-plugins.py)
//
// Both are generated at build time from the same plugins.json the catalog grid fetches, so
// there is no second data source: a merged catalog PR is the only way a page appears,
// changes or disappears. Entries that opt in with `readme: true` get their README fetched
// from the pinned commit and rendered through the allowlist in ./readme.js.
//
// The site degrades, never fails: a missing plugins.json (extract-plugins.py did not run)
// generates zero pages with a warning, and a README that cannot be fetched leaves that
// section off the page.

const fs = require("node:fs");
const path = require("node:path");
const { fetchReadme, renderReadme } = require("./readme.js");

const PLUGINS_JSON = path.join("static", "api", "plugins.json");
const META_JSON = path.join("static", "api", "plugins-meta.json");
const README_CONCURRENCY = 6;

function log(msg) {
  console.warn(`[plugin-catalog-pages] ${msg}`);
}

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return fallback;
  }
}

async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

/** Group entries by maintainerSlug, preserving the catalog's (stars-first) order inside a group. */
function groupAuthors(entries) {
  const authors = new Map();
  for (const entry of entries) {
    const slug = entry.maintainerSlug || "unknown";
    const author = authors.get(slug) || { slug, name: entry.maintainer || slug, plugins: [] };
    author.plugins.push(entry);
    authors.set(slug, author);
  }
  return [...authors.values()];
}

module.exports = function pluginCatalogPages(context) {
  const { siteDir, baseUrl } = context;
  const cacheDir = path.join(siteDir, ".cache", "plugin-readmes");

  return {
    name: "plugin-catalog-pages",

    getPathsToWatch() {
      return [path.join(siteDir, PLUGINS_JSON)];
    },

    async loadContent() {
      const entries = readJson(path.join(siteDir, PLUGINS_JSON), null);
      if (!Array.isArray(entries)) {
        log(`${PLUGINS_JSON} missing or invalid; generating no plugin pages`);
        return { entries: [], meta: {} };
      }
      const meta = readJson(path.join(siteDir, META_JSON), {});
      const wantReadme = entries.filter((e) => e.readme && e.readmeUrl);
      const rendered = await mapLimit(wantReadme, README_CONCURRENCY, async (entry) => {
        const fetched = await fetchReadme(entry.readmeUrl, cacheDir, log);
        if (fetched == null) return null;
        try {
          return { html: await renderReadme(fetched.markdown, fetched.url), url: fetched.url };
        } catch (err) {
          log(`README for ${entry.name} failed to render: ${err && err.message ? err.message : err}`);
          return null;
        }
      });
      const readmes = Object.fromEntries(wantReadme.map((entry, i) => [entry.name, rendered[i]]));
      if (wantReadme.length) {
        log(`rendered ${rendered.filter(Boolean).length}/${wantReadme.length} opted-in READMEs from pinned commits`);
      }
      return { entries, meta, readmes };
    },

    async contentLoaded({ content, actions }) {
      const { addRoute, createData } = actions;
      const { entries, meta, readmes = {} } = content;
      if (!entries.length) return;
      const authors = groupAuthors(entries);
      const bySlug = new Map(authors.map((a) => [a.slug, a]));
      const prefix = `${baseUrl.replace(/\/$/, "")}/plugins`;

      // Compact cards for shelves ("More by this author", author page grid) — the full entry
      // minus nothing heavy; keeping the shape identical to plugins.json keeps one card type.
      for (const entry of entries) {
        const author = bySlug.get(entry.maintainerSlug || "unknown");
        const siblings = author ? author.plugins.filter((p) => p.name !== entry.name) : [];
        const data = await createData(
          `plugin-${entry.name}.json`,
          JSON.stringify({
            plugin: entry,
            readmeHtml: readmes[entry.name]?.html || null,
            readmeSourceUrl: readmes[entry.name]?.url || null,
            author: author ? { slug: author.slug, name: author.name, count: author.plugins.length } : null,
            moreByAuthor: siblings,
            generatedAt: meta.generatedAt || null,
            starsFetchedAt: meta.starsFetchedAt || null,
          }),
        );
        addRoute({
          path: `${prefix}/${entry.name}`,
          component: "@site/src/components/PluginCatalog/PluginPage",
          exact: true,
          modules: { data },
        });
      }

      for (const author of authors) {
        const data = await createData(
          `author-${author.slug}.json`,
          JSON.stringify({
            author: { slug: author.slug, name: author.name, count: author.plugins.length },
            plugins: author.plugins,
            generatedAt: meta.generatedAt || null,
          }),
        );
        addRoute({
          path: `${prefix}/by/${author.slug}`,
          component: "@site/src/components/PluginCatalog/AuthorPage",
          exact: true,
          modules: { data },
        });
      }
      log(`generated ${entries.length} plugin pages and ${authors.length} author pages under ${prefix}/`);
    },
  };
};
