import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import { type CatalogPlugin, CATEGORY_ORDER, categoryOf, formatDate, formatStars, repoUrl } from "./catalog";
import MiniCard from "./MiniCard";
import styles from "./pages.module.css";

interface AuthorPageData {
  author: { slug: string; name: string; count: number };
  plugins: CatalogPlugin[];
  generatedAt: string | null;
}

/** GitHub/GitLab profile when every listed repo belongs to one owner; null when they differ. */
function profileUrl(plugins: CatalogPlugin[]): string | null {
  const owners = new Set<string>();
  for (const p of plugins) {
    const m = repoUrl(p).match(/^https:\/\/(github\.com|gitlab\.com)\/([^/]+)\//);
    if (!m) return null;
    owners.add(`https://${m[1]}/${m[2]}`);
  }
  return owners.size === 1 ? [...owners][0] : null;
}

export default function AuthorPage({ data }: { data: AuthorPageData }) {
  const { author, plugins } = data;
  const stars = plugins.reduce((sum, p) => sum + (typeof p.stars === "number" ? p.stars : 0), 0);
  const hasStars = plugins.some((p) => typeof p.stars === "number");
  const profile = profileUrl(plugins);
  const firstListed = plugins
    .map((p) => p.addedAt)
    .filter((d): d is string => Boolean(d))
    .sort()[0];
  const categories = CATEGORY_ORDER.filter((c) => plugins.some((p) => categoryOf(p) === categoryOf({ category: c })));
  const official = plugins.filter((p) => p.tier === "official").length;

  return (
    <Layout
      title={`Plugins by ${author.name} · Plugin Catalog`}
      description={`${plugins.length} Hermes Agent plugin${plugins.length === 1 ? "" : "s"} by ${author.name} in the reviewed catalog.`}
    >
      <div className={styles.page}>
        <nav className={styles.crumbs} aria-label="Breadcrumb">
          <Link to="/plugins">Plugin Catalog</Link>
          <span aria-hidden="true">/</span>
          <span>Authors</span>
          <span aria-hidden="true">/</span>
          <span className={styles.crumbCurrent}>{author.name}</span>
        </nav>

        <header className={`${styles.hero} ${styles.authorHero}`}>
          <div className={styles.heroBody}>
            <p className={styles.eyebrow}>Plugin author</p>
            <h1 className={styles.title}>{author.name}</h1>
            <p className={styles.byline}>
              {plugins.length} plugin{plugins.length === 1 ? "" : "s"} in the catalog
              {official > 0 && <> · {official} official</>}
              {hasStars && <> · {"\u2605"} {formatStars(stars)} GitHub stars across them</>}
              {firstListed && <> · first listed {formatDate(firstListed)}</>}
              {categories.length > 0 && (
                <> · {categories.map((c) => categoryOf({ category: c }).label).join(", ")}</>
              )}
            </p>
            {profile && (
              <div className={styles.links}>
                <a className={styles.link} href={profile} target="_blank" rel="noopener noreferrer">
                  {profile.replace(/^https:\/\//, "")} ↗
                </a>
              </div>
            )}
          </div>
        </header>

        <main className={styles.main}>
          <div className={styles.shelf}>
            {plugins.map((p) => <MiniCard key={p.name} plugin={p} />)}
          </div>
          <p className={styles.footerMeta}>
            <Link to="/plugins">← Back to the catalog</Link>
          </p>
        </main>
      </div>
    </Layout>
  );
}
