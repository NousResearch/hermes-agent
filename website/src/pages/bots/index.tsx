import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import CatalogTabs from "../../components/Marketplace/CatalogTabs";
import usePickerMode from "../../components/Marketplace/usePickerMode";
import {
  type BotCatalogMeta,
  type CatalogBot,
  botInstallLink,
  botPagePath,
  botSearchText,
  categoryBlurb,
  categoryLabel,
} from "../../components/BotCatalog/catalog";
import market from "../plugins/styles.module.css";
import styles from "./styles.module.css";

const BOTS_URL = "/docs/api/bots.json";
const META_URL = "/docs/api/bots-meta.json";

function BotCard({ bot, picker, featured = false, onPick }: {
  bot: CatalogBot;
  picker: boolean;
  featured?: boolean;
  onPick: (bot: CatalogBot) => void;
}) {
  const color = bot.presentation?.color || "#6f65e8";
  const detail = botPagePath(bot.name);
  return (
    <article className={`${market.card} ${styles.botCard} ${featured ? styles.featuredCard : ""}`}>
      <span className={market.cardAccent} style={{ background: color }} />
      <div className={market.cardInner}>
        <div className={styles.cardHeading}>
          <span className={styles.avatar} style={{ background: `${color}20`, borderColor: `${color}66` }} aria-hidden="true">
            {bot.presentation?.emoji || "🤖"}
          </span>
          <div className={styles.cardIdentity}>
            <h3 className={market.cardTitle}><Link className={market.cardTitleLink} to={detail}>{bot.title}</Link></h3>
            <span className={styles.catalogName}>{bot.name}</span>
          </div>
          <span className={market.tierPill} style={{ color, borderColor: `${color}66`, background: `${color}15` }}>
            {bot.tier === "official" ? "✓ Official" : "Community"}
          </span>
        </div>
        <p className={styles.outcome}>{bot.summary}</p>
        <p className={market.cardDesc}>{bot.profile?.description}</p>
        <div className={market.cardMeta}>
          <span className={market.categoryChip}>{categoryLabel(bot.category)}</span>
          {(bot.tags || []).slice(0, 3).map((tag) => <span key={tag} className={market.capChip}>{tag}</span>)}
        </div>
        <div className={styles.creator}>By <Link to={bot.authorPath}>{bot.maintainer}</Link></div>
        <div className={styles.cardActions}>
          {picker ? (
            <button className={`${market.pickBtn} ${styles.cardPick}`} type="button" onClick={() => onPick(bot)}>Add Bot</button>
          ) : (
            <a className={`${market.pickBtn} ${styles.cardPick}`} href={botInstallLink(bot.name)}>Add Bot</a>
          )}
          <Link className={styles.detailsLink} to={detail}>View details →</Link>
        </div>
      </div>
    </article>
  );
}

export default function BotMarketplacePage() {
  const picker = usePickerMode();
  const [bots, setBots] = useState<CatalogBot[] | null>(null);
  const [meta, setMeta] = useState<BotCatalogMeta>({});
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch(BOTS_URL).then((response) => {
        if (!response.ok) throw new Error(`bots.json HTTP ${response.status}`);
        return response.json();
      }),
      fetch(META_URL).then((response) => response.ok ? response.json() : {}).catch(() => ({})),
    ]).then(([rows, sidecar]) => {
      if (!cancelled) {
        setBots(Array.isArray(rows) ? rows : []);
        setMeta(sidecar || {});
      }
    }).catch((reason) => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "/" && document.activeElement?.tagName !== "INPUT") {
        event.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const categories = useMemo(() => {
    const counts = new Map<string, number>();
    for (const bot of bots || []) counts.set(bot.category, (counts.get(bot.category) || 0) + 1);
    return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [bots]);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return (bots || []).filter((bot) =>
      (category === "all" || bot.category === category) && (!query || botSearchText(bot).includes(query)),
    );
  }, [bots, search, category]);

  const featured = useMemo(() => (bots || []).filter((bot) => bot.tier === "official").slice(0, 4), [bots]);
  const shelves = useMemo(() => {
    if (search.trim() || category !== "all") return [];
    return categories.map(([key]) => [key, (bots || []).filter((bot) => bot.category === key)] as const);
  }, [bots, categories, search, category]);

  const pickBot = useCallback((bot: CatalogBot) => {
    if (typeof window === "undefined" || window.parent === window) return;
    window.parent.postMessage({ type: "hermes-bot-pick", catalog: bot.name }, "*");
  }, []);

  return (
    <Layout title="Bot Marketplace" description="Add purpose-built Hermes bots for research, inbox triage, and focused work.">
      <div className={`${market.page} ${styles.page} ${picker ? market.pickerMode : ""}`}>
        <header className={market.hero}>
          <div className={market.heroGlow} />
          <div className={market.heroContent}>
            <p className={market.heroEyebrow}>Hermes Agent</p>
            <h1 className={market.heroTitle}>Bot Marketplace</h1>
            <CatalogTabs active="bots" />
            <p className={market.heroSub}>Start with a focused teammate, not a blank chat. Each bot is a reviewed, isolated Hermes profile you control.</p>
            {bots && <p className={market.heroMeta}>{bots.length} bots across {categories.length} categories</p>}
            {error && <p className={styles.error}>Could not load the catalog: {error}</p>}
          </div>
        </header>

        {(bots?.length || 0) > 0 && (
          <div className={market.controlsBar}>
            <div className={market.searchWrap}>
              <span className={market.searchIcon} aria-hidden="true">⌕</span>
              <input ref={searchRef} className={market.searchInput} value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search by the outcome you want" aria-label="Search bots" />
              {search && <button type="button" className={market.clearBtn} onClick={() => setSearch("")} aria-label="Clear search">×</button>}
            </div>
            <div className={market.categoryPills} role="tablist" aria-label="Bot categories">
              <button type="button" className={`${market.categoryBtn} ${category === "all" ? market.categoryBtnActive : ""}`} onClick={() => setCategory("all")}>All bots</button>
              {categories.map(([key, count]) => (
                <button key={key} type="button" className={`${market.categoryBtn} ${category === key ? market.categoryBtnActive : ""}`} onClick={() => setCategory(key)}>
                  {categoryLabel(key)} <span className={market.tierCount}>{count}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        <main className={market.main}>
          {!bots && !error ? (
            <div className={market.empty}><div className={market.loadingSpinner} /><h2 className={market.emptyTitle}>Loading bots…</h2></div>
          ) : bots?.length === 0 ? (
            <div className={market.empty}><div className={market.emptyIcon}>🤖</div><h2 className={market.emptyTitle}>Bots are on their way</h2><p className={market.emptyDesc}>The marketplace publishes reviewed bot blueprints from the Hermes catalog.</p></div>
          ) : search.trim() || category !== "all" ? (
            <section aria-labelledby="results-heading">
              <header className={market.categoryHeader}><h2 id="results-heading" className={market.categoryTitle}>{filtered.length} result{filtered.length === 1 ? "" : "s"}</h2></header>
              <div className={market.grid}>{filtered.map((bot) => <BotCard key={bot.name} bot={bot} picker={picker} onPick={pickBot} />)}</div>
              {!filtered.length && <div className={market.empty}><h2 className={market.emptyTitle}>No bots found</h2><button className={market.emptyReset} onClick={() => { setSearch(""); setCategory("all"); }}>Clear filters</button></div>}
            </section>
          ) : (
            <>
              {featured.length > 0 && (
                <section className={market.categorySection} aria-labelledby="featured-bots">
                  <header className={market.categoryHeader}><h2 id="featured-bots" className={market.categoryTitle}>✦ Featured bots</h2><p className={market.categoryBlurb}>Official starting points for common, high-value work.</p></header>
                  <div className={market.grid}>{featured.map((bot) => <BotCard key={bot.name} bot={bot} featured picker={picker} onPick={pickBot} />)}</div>
                </section>
              )}
              {shelves.map(([key, rows]) => (
                <section key={key} className={market.categorySection} aria-labelledby={`bots-${key}`}>
                  <header className={market.categoryHeader}><h2 id={`bots-${key}`} className={market.categoryTitle}>{categoryLabel(key)} <span className={market.tierCount}>{rows.length}</span></h2><p className={market.categoryBlurb}>{categoryBlurb(key)}</p><button className={market.categoryViewAll} onClick={() => setCategory(key)}>View all →</button></header>
                  <div className={market.grid}>{rows.map((bot) => <BotCard key={bot.name} bot={bot} picker={picker} onPick={pickBot} />)}</div>
                </section>
              ))}
            </>
          )}
          {meta.removedCount ? <p className={styles.catalogNote}>{meta.removedCount} retired listing{meta.removedCount === 1 ? " is" : "s are"} excluded from installation.</p> : null}
        </main>
      </div>
    </Layout>
  );
}
