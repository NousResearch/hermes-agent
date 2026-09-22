import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import Head from "@docusaurus/Head";
import {
  type BotSetupRequirement,
  type CatalogBot,
  botAuthorPath,
  botInstallLink,
  botPagePath,
  categoryLabel,
} from "./catalog";
import styles from "../PluginCatalog/pages.module.css";
import botStyles from "./pages.module.css";

interface BotPageData {
  bot: CatalogBot;
  creator: { slug: string; name: string; count: number } | null;
  moreByCreator: CatalogBot[];
}

function Chips({ label, values }: { label: string; values: string[] }) {
  if (!values?.length) return null;
  return <div className={styles.capRow}><h3 className={styles.capLabel}>{label} <span className={styles.count}>{values.length}</span></h3><div className={styles.chipWrap}>{values.map((value) => <code key={value} className={styles.chip}>{value}</code>)}</div></div>;
}

function RequirementList({ title, requirements }: { title: string; requirements: BotSetupRequirement[] }) {
  return (
    <div className={botStyles.requirementGroup}>
      <h3 className={botStyles.subheading}>{title}</h3>
      {requirements.length === 0 ? (
        <p className={botStyles.noRequirements}>None</p>
      ) : (
        <ul className={botStyles.requirementList}>
          {requirements.map((requirement) => (
            <li key={`${requirement.kind}:${requirement.id}`} className={botStyles.requirement}>
              <div><code>{requirement.id}</code> <span className={botStyles.requirementKind}>{requirement.kind}</span></div>
              <p>{requirement.purpose}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function BotMiniCard({ bot }: { bot: CatalogBot }) {
  const color = bot.presentation?.color || "#6f65e8";
  return (
    <Link className={styles.miniCard} to={botPagePath(bot.name)}>
      <span className={styles.miniAccent} style={{ background: color }} />
      <div className={styles.miniTop}><span className={styles.miniIcon}>{bot.presentation?.emoji || "🤖"}</span><span className={styles.miniName}>{bot.title}</span></div>
      <p className={styles.miniDesc}>{bot.summary}</p>
      <span className={styles.miniCategory}>{categoryLabel(bot.category)}</span>
    </Link>
  );
}

export default function BotPage({ data }: { data: BotPageData }) {
  const { bot, creator, moreByCreator } = data;
  const color = bot.presentation?.color || "#6f65e8";
  const requirements = bot.setup?.requirements || [];
  const requiredSetup = requirements.filter((requirement) => requirement.required);
  const optionalSetup = requirements.filter((requirement) => !requirement.required);
  return (
    <Layout title={`${bot.title} · Bot Marketplace`} description={bot.summary}>
      <Head><meta property="og:type" content="website" /></Head>
      <div className={styles.page}>
        <nav className={styles.crumbs} aria-label="Breadcrumb"><Link to="/bots">Bot Marketplace</Link><span>/</span><span>{categoryLabel(bot.category)}</span><span>/</span><span className={styles.crumbCurrent}>{bot.title}</span></nav>
        <header className={`${styles.hero} ${botStyles.hero}`} style={{ "--bot-color": color } as React.CSSProperties}>
          <div className={styles.heroBody}>
            <div className={styles.titleRow}>
              <span className={botStyles.heroAvatar} style={{ borderColor: color, background: `${color}20` }}>{bot.presentation?.emoji || "🤖"}</span>
              <h1 className={styles.title}>{bot.title}</h1>
              <span className={styles.tierPill} style={{ color, borderColor: `${color}66`, background: `${color}15` }}>{bot.tier === "official" ? "✓ Official" : "Community"}</span>
              {bot.version && <span className={styles.versionPill}>v{bot.version.replace(/^v/i, "")}</span>}
            </div>
            <p className={styles.byline}><code>{bot.name}</code> · {categoryLabel(bot.category)} · by {creator ? <Link to={botAuthorPath(creator.slug)}>{bot.maintainer}</Link> : bot.maintainer}</p>
            <p className={styles.lede}>{bot.summary}</p>
            <div className={styles.actions}><a className={styles.installBtn} href={botInstallLink(bot.name)}>Add Bot</a><span className={botStyles.installNote}>Opens a review in Hermes Desktop before creating an isolated profile.</span></div>
          </div>
        </header>

        <main className={styles.main}>
          <div className={styles.columns}>
            <div className={styles.primary}>
              <section className={styles.section} aria-labelledby="outcomes"><h2 id="outcomes" className={styles.sectionTitle}>What this bot owns</h2><p className={botStyles.bodyCopy}>{bot.profile.description}</p></section>
              <section className={styles.section} aria-labelledby="start"><h2 id="start" className={styles.sectionTitle}>Start here</h2><blockquote className={botStyles.starter}>{bot.starterExample}</blockquote></section>
              <section className={styles.section} aria-labelledby="capabilities"><h2 id="capabilities" className={styles.sectionTitle}>Included capabilities</h2><Chips label="Skills" values={bot.capabilities?.skills || []} /><Chips label="Toolsets" values={bot.capabilities?.toolsets || []} /></section>
              {requirements.length > 0 && (
                <section className={styles.section} aria-labelledby="setup">
                  <h2 id="setup" className={styles.sectionTitle}>Setup</h2>
                  <p className={botStyles.sectionIntro}>Review these dependencies before installation. Optional dependencies expand the workflow but are not needed for its baseline task.</p>
                  <RequirementList title="Required" requirements={requiredSetup} />
                  <RequirementList title="Optional" requirements={optionalSetup} />
                </section>
              )}
              {(bot.routines || []).length > 0 && (
                <section className={styles.section} aria-labelledby="routines">
                  <h2 id="routines" className={styles.sectionTitle}>Routine proposals</h2>
                  <p className={botStyles.routineNotice}><strong>Disabled by default.</strong> Routines are suggestions only. Choose the local time, timezone, and delivery destination after the first successful task, then activate one explicitly.</p>
                  <div className={botStyles.routineList}>
                    {bot.routines.map((routine) => (
                      <article key={routine.id} className={botStyles.routine}>
                        <h3>{routine.name}</h3>
                        <p><strong>Suggested cadence:</strong> {routine.schedule}</p>
                        <details><summary>Preview prompt</summary><pre><code>{routine.prompt}</code></pre></details>
                      </article>
                    ))}
                  </div>
                </section>
              )}
              {bot.workflowPackage && (
                <section className={styles.section} aria-labelledby="examples">
                  <h2 id="examples" className={styles.sectionTitle}>Try an example</h2>
                  <p className={botStyles.sectionIntro}>These previews come from the shipped <code>{bot.workflowPackage.skill}</code> workflow package.</p>
                  {bot.workflowPackage.exampleTasks.length > 0 && (
                    <ul className={botStyles.exampleTasks}>{bot.workflowPackage.exampleTasks.map((task) => <li key={task}>{task}</li>)}</ul>
                  )}
                  <div className={botStyles.sampleList}>
                    {bot.workflowPackage.samples.map((sample) => (
                      <details key={sample.path} className={botStyles.sample} open={sample.kind === "input"}>
                        <summary>{sample.title} <code>{sample.path}</code></summary>
                        <pre><code>{sample.preview}{sample.truncated ? "\n\n… preview truncated" : ""}</code></pre>
                      </details>
                    ))}
                  </div>
                </section>
              )}
              <section className={styles.section} aria-labelledby="contract"><h2 id="contract" className={styles.sectionTitle}>Operating contract</h2><p className={botStyles.contractIntro}>This durable SOUL is copied into your new profile. You can review it here and edit your installed copy later.</p><pre className={botStyles.soul}><code>{bot.profile.soul}</code></pre></section>
            </div>
            <aside className={styles.side}>
              <dl className={styles.facts}>
                <div className={styles.fact}><dt className={styles.factLabel}>Maintainer</dt><dd className={styles.factValue}>{creator ? <Link to={botAuthorPath(creator.slug)}>{bot.maintainer}</Link> : bot.maintainer}</dd></div>
                <div className={styles.fact}><dt className={styles.factLabel}>Catalog ID</dt><dd className={styles.factValue}><code>{bot.name}</code></dd></div>
                <div className={styles.fact}><dt className={styles.factLabel}>Suggested profile</dt><dd className={styles.factValue}><code>{bot.profile.suggested_name}</code></dd></div>
                <div className={styles.fact}><dt className={styles.factLabel}>Category</dt><dd className={styles.factValue}>{categoryLabel(bot.category)}</dd></div>
                <div className={styles.fact}><dt className={styles.factLabel}>Tier</dt><dd className={styles.factValue}>{bot.tier === "official" ? "Official" : "Community"}</dd></div>
              </dl>
              <p className={styles.sideNote}>Installing copies this reviewed blueprint. It does not subscribe your profile to future catalog changes or copy another user's data.</p>
            </aside>
          </div>
          {creator && moreByCreator.length > 0 && <section className={styles.section}><div className={styles.shelfHead}><h2 className={styles.sectionTitle}>More by {creator.name}</h2><Link className={styles.shelfAll} to={botAuthorPath(creator.slug)}>All {creator.count} bots →</Link></div><div className={styles.shelf}>{moreByCreator.map((item) => <BotMiniCard key={item.name} bot={item} />)}</div></section>}
          <p className={styles.footerMeta}><Link to="/bots">← Back to the Bot Marketplace</Link></p>
        </main>
      </div>
    </Layout>
  );
}
