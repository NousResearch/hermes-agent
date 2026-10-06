import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import { type CatalogBot, botInstallLink, botPagePath, categoryLabel } from "./catalog";
import styles from "../PluginCatalog/pages.module.css";
import botStyles from "./pages.module.css";

interface CreatorPageData {
  creator: { slug: string; name: string; count: number };
  bots: CatalogBot[];
}

export default function CreatorPage({ data }: { data: CreatorPageData }) {
  const { creator, bots } = data;
  return (
    <Layout title={`Bots by ${creator.name} · Bot Marketplace`} description={`${bots.length} reviewed Hermes bots by ${creator.name}.`}>
      <div className={styles.page}>
        <nav className={styles.crumbs} aria-label="Breadcrumb"><Link to="/bots">Bot Marketplace</Link><span>/</span><span>Maintainers</span><span>/</span><span className={styles.crumbCurrent}>{creator.name}</span></nav>
        <header className={`${styles.hero} ${styles.authorHero}`}><div className={styles.heroBody}><p className={styles.eyebrow}>Bot maintainer</p><h1 className={styles.title}>{creator.name}</h1><p className={styles.byline}>{bots.length} bot{bots.length === 1 ? "" : "s"} in the reviewed catalog</p></div></header>
        <main className={styles.main}><div className={styles.shelf}>{bots.map((bot) => {
          const color = bot.presentation?.color || "#6f65e8";
          return <article key={bot.name} className={styles.miniCard}><span className={styles.miniAccent} style={{ background: color }} /><div className={styles.miniTop}><span className={styles.miniIcon}>{bot.presentation?.emoji || "🤖"}</span><Link className={styles.miniName} to={botPagePath(bot.name)}>{bot.title}</Link></div><p className={styles.miniDesc}>{bot.summary}</p><span className={styles.miniCategory}>{categoryLabel(bot.category)}</span><a className={botStyles.miniAdd} href={botInstallLink(bot.name)}>Add Bot</a></article>;
        })}</div><p className={styles.footerMeta}><Link to="/bots">← Back to the Bot Marketplace</Link></p></main>
      </div>
    </Layout>
  );
}
