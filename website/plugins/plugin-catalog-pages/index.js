// Docusaurus marketplace route generator.
//
// Builds detail and creator pages for the reviewed Plugin and Bot catalogs.
// Both catalogs are extracted before Docusaurus loads this plugin, so the
// generated pages and browse grids always share one data source.
// Plugin READMEs come from pinned commits through the allowlist in ./readme.js,
// unless an entry sets `readme: false`. Failed README fetches omit that section.

const fs = require("node:fs");
const path = require("node:path");
const { fetchReadme, renderReadme } = require("./readme.js");

const PLUGINS_JSON = path.join("static", "api", "plugins.json");
const PLUGIN_META_JSON = path.join("static", "api", "plugins-meta.json");
const BOTS_JSON = path.join("static", "api", "bots.json");
const README_CONCURRENCY = 12;

function log(message) {
  console.warn(`[marketplace-pages] ${message}`);
}

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return fallback;
  }
}

async function mapLimit(items, limit, fn) {
  const output = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const index = next++;
      output[index] = await fn(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return output;
}

function groupCreators(entries) {
  const creators = new Map();
  for (const entry of entries) {
    const slug = entry.maintainerSlug || "unknown";
    const creator = creators.get(slug) || { slug, name: entry.maintainer || slug, entries: [] };
    creator.entries.push(entry);
    creators.set(slug, creator);
  }
  return [...creators.values()];
}

module.exports = function marketplacePages(context) {
  const { siteDir, baseUrl } = context;
  const cacheDir = path.join(siteDir, ".cache", "plugin-readmes");
  return {
    name: "marketplace-pages",

    getPathsToWatch() {
      return [path.join(siteDir, PLUGINS_JSON), path.join(siteDir, BOTS_JSON)];
    },

    async loadContent() {
      const plugins = readJson(path.join(siteDir, PLUGINS_JSON), []);
      const bots = readJson(path.join(siteDir, BOTS_JSON), []);
      const safePlugins = Array.isArray(plugins) ? plugins : [];
      const safeBots = Array.isArray(bots) ? bots : [];
      const wantReadme = safePlugins.filter((entry) => entry.readme && entry.readmeUrl);
      const rendered = await mapLimit(wantReadme, README_CONCURRENCY, async (entry) => {
        const fetched = await fetchReadme(entry.readmeUrl, cacheDir, log);
        if (fetched == null) return null;
        try {
          return { html: await renderReadme(fetched.markdown, fetched.url), url: fetched.url };
        } catch (error) {
          log(`README for ${entry.name} failed to render: ${error && error.message ? error.message : error}`);
          return null;
        }
      });
      if (wantReadme.length) {
        log(`rendered ${rendered.filter(Boolean).length}/${wantReadme.length} READMEs from pinned commits`);
      }
      return {
        plugins: safePlugins,
        pluginMeta: readJson(path.join(siteDir, PLUGIN_META_JSON), {}),
        readmes: Object.fromEntries(wantReadme.map((entry, index) => [entry.name, rendered[index]])),
        bots: safeBots,
      };
    },

    async contentLoaded({ content, actions }) {
      const { addRoute, createData } = actions;
      const { plugins, pluginMeta, readmes, bots } = content;
      const root = baseUrl.replace(/\/$/, "");

      const pluginCreators = groupCreators(plugins);
      const pluginsByCreator = new Map(pluginCreators.map((creator) => [creator.slug, creator]));
      for (const entry of plugins) {
        const creator = pluginsByCreator.get(entry.maintainerSlug || "unknown");
        const siblings = creator ? creator.entries.filter((item) => item.name !== entry.name) : [];
        const data = await createData(`plugin-${entry.name}.json`, JSON.stringify({
          plugin: entry,
          readmeHtml: readmes[entry.name]?.html || null,
          readmeSourceUrl: readmes[entry.name]?.url || null,
          author: creator ? { slug: creator.slug, name: creator.name, count: creator.entries.length } : null,
          moreByAuthor: siblings,
          generatedAt: pluginMeta.generatedAt || null,
          starsFetchedAt: pluginMeta.starsFetchedAt || null,
        }));
        addRoute({ path: `${root}/plugins/${encodeURIComponent(entry.name)}`, component: "@site/src/components/PluginCatalog/PluginPage", exact: true, modules: { data } });
      }
      for (const creator of pluginCreators) {
        const data = await createData(`plugin-author-${creator.slug}.json`, JSON.stringify({
          author: { slug: creator.slug, name: creator.name, count: creator.entries.length },
          plugins: creator.entries,
          generatedAt: pluginMeta.generatedAt || null,
        }));
        addRoute({ path: `${root}/plugins/by/${encodeURIComponent(creator.slug)}`, component: "@site/src/components/PluginCatalog/AuthorPage", exact: true, modules: { data } });
      }

      const botCreators = groupCreators(bots);
      const botsByCreator = new Map(botCreators.map((creator) => [creator.slug, creator]));
      for (const bot of bots) {
        const creator = botsByCreator.get(bot.maintainerSlug || "unknown");
        const data = await createData(`bot-${bot.name}.json`, JSON.stringify({
          bot,
          creator: creator ? { slug: creator.slug, name: creator.name, count: creator.entries.length } : null,
          moreByCreator: creator ? creator.entries.filter((item) => item.name !== bot.name) : [],
        }));
        addRoute({ path: `${root}/bots/${encodeURIComponent(bot.name)}`, component: "@site/src/components/BotCatalog/BotPage", exact: true, modules: { data } });
      }
      for (const creator of botCreators) {
        const data = await createData(`bot-creator-${creator.slug}.json`, JSON.stringify({
          creator: { slug: creator.slug, name: creator.name, count: creator.entries.length },
          bots: creator.entries,
        }));
        addRoute({ path: `${root}/bots/by/${encodeURIComponent(creator.slug)}`, component: "@site/src/components/BotCatalog/CreatorPage", exact: true, modules: { data } });
      }
      log(`generated ${plugins.length} plugin pages, ${pluginCreators.length} plugin creator pages, ${bots.length} bot pages, and ${botCreators.length} bot creator pages`);
    },
  };
};
