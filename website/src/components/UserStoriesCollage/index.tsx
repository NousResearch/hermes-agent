import React, { useMemo, useState } from 'react';
import stories from '@site/src/data/userStories.json';
import styles from './styles.module.css';

interface Story {
  id: string;
  source: string;
  author: string;
  url: string;
  date: string;
  category: string;
  headline: string;
  quote: string;
  size: 'sm' | 'md' | 'lg';
}

const allStories = stories as Story[];

// Category → pretty label + accent colors (solid + soft fill + gradient top-strip)
const CATEGORIES: Record<
  string,
  { label: string; solid: string; soft: string; strip: string }
> = {
  'dev-workflow': {
    label: 'Dev Workflow',
    solid: '#60a5fa',
    soft: 'rgba(96, 165, 250, 0.14)',
    strip: 'linear-gradient(90deg, #3b82f6, #60a5fa, #a78bfa)',
  },
  'personal-assistant': {
    label: 'Personal Assistant',
    solid: '#34d399',
    soft: 'rgba(52, 211, 153, 0.14)',
    strip: 'linear-gradient(90deg, #10b981, #34d399, #a7f3d0)',
  },
  'content-creation': {
    label: 'Content Creation',
    solid: '#f472b6',
    soft: 'rgba(244, 114, 182, 0.14)',
    strip: 'linear-gradient(90deg, #ec4899, #f472b6, #fda4af)',
  },
  'business-ops': {
    label: 'Business Ops',
    solid: '#fb923c',
    soft: 'rgba(251, 146, 60, 0.14)',
    strip: 'linear-gradient(90deg, #f97316, #fb923c, #fcd34d)',
  },
  trading: {
    label: 'Trading & Markets',
    solid: '#facc15',
    soft: 'rgba(250, 204, 21, 0.16)',
    strip: 'linear-gradient(90deg, #eab308, #facc15, #fde047)',
  },
  research: {
    label: 'Research',
    solid: '#a78bfa',
    soft: 'rgba(167, 139, 250, 0.14)',
    strip: 'linear-gradient(90deg, #8b5cf6, #a78bfa, #c4b5fd)',
  },
  creative: {
    label: 'Creative',
    solid: '#f87171',
    soft: 'rgba(248, 113, 113, 0.14)',
    strip: 'linear-gradient(90deg, #ef4444, #f87171, #fca5a5)',
  },
  marketing: {
    label: 'Marketing',
    solid: '#e879f9',
    soft: 'rgba(232, 121, 249, 0.14)',
    strip: 'linear-gradient(90deg, #d946ef, #e879f9, #f0abfc)',
  },
  integrations: {
    label: 'Integrations',
    solid: '#38bdf8',
    soft: 'rgba(56, 189, 248, 0.14)',
    strip: 'linear-gradient(90deg, #0ea5e9, #38bdf8, #7dd3fc)',
  },
  enterprise: {
    label: 'Enterprise',
    solid: '#94a3b8',
    soft: 'rgba(148, 163, 184, 0.16)',
    strip: 'linear-gradient(90deg, #64748b, #94a3b8, #cbd5e1)',
  },
  messaging: {
    label: 'Messaging',
    solid: '#22d3ee',
    soft: 'rgba(34, 211, 238, 0.14)',
    strip: 'linear-gradient(90deg, #06b6d4, #22d3ee, #67e8f9)',
  },
  privacy: {
    label: 'Privacy & Self-Hosted',
    solid: '#4ade80',
    soft: 'rgba(74, 222, 128, 0.14)',
    strip: 'linear-gradient(90deg, #16a34a, #4ade80, #86efac)',
  },
  'cost-optimization': {
    label: 'Cost Optimization',
    solid: '#fbbf24',
    soft: 'rgba(251, 191, 36, 0.16)',
    strip: 'linear-gradient(90deg, #f59e0b, #fbbf24, #fde68a)',
  },
  meta: {
    label: 'Meta & Ecosystem',
    solid: '#c084fc',
    soft: 'rgba(192, 132, 252, 0.14)',
    strip: 'linear-gradient(90deg, #a855f7, #c084fc, #d8b4fe)',
  },
  general: {
    label: 'General',
    solid: '#9ca3af',
    soft: 'rgba(156, 163, 175, 0.16)',
    strip: 'linear-gradient(90deg, #6b7280, #9ca3af, #d1d5db)',
  },
};

// Source → compact label shown in the badge row
const SOURCE_LABELS: Record<string, string> = {
  x: 'X · Twitter',
  hn: 'Hacker News',
  reddit: 'Reddit',
  github: 'GitHub',
  youtube: 'YouTube',
  blog: 'Blog',
  podcast: 'Podcast',
  linkedin: 'LinkedIn',
  gist: 'GitHub Gist',
  producthunt: 'Product Hunt',
  discord: 'Discord',
};

function sourceColor(source: string): string {
  switch (source) {
    case 'x': return '#1d9bf0';
    case 'hn': return '#ff6600';
    case 'reddit': return '#ff4500';
    case 'github': return '#8b949e';
    case 'youtube': return '#ff0033';
    case 'blog': return '#a78bfa';
    case 'podcast': return '#8b5cf6';
    case 'linkedin': return '#0a66c2';
    case 'gist': return '#8b949e';
    case 'producthunt': return '#da552f';
    case 'discord': return '#5865f2';
    default: return '#64748b';
  }
}

export default function UserStoriesCollage(): JSX.Element {
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [activeSource, setActiveSource] = useState<string>('all');

  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const s of allStories) counts[s.category] = (counts[s.category] ?? 0) + 1;
    return counts;
  }, []);

  const sourceCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const s of allStories) counts[s.source] = (counts[s.source] ?? 0) + 1;
    return counts;
  }, []);

  const visible = useMemo(() => {
    return allStories.filter((s) => {
      if (activeCategory !== 'all' && s.category !== activeCategory) return false;
      if (activeSource !== 'all' && s.source !== activeSource) return false;
      return true;
    });
  }, [activeCategory, activeSource]);

  return (
    <div className={styles.wrap}>
      <div className={styles.hero}>
        <h1>User Stories &amp; Use Cases</h1>
        <p>
          What the Hermes Agent community is actually building. Every tile
          below links to a real post, issue, video, or gist where someone
          describes how they use Hermes &mdash; scraped from X, GitHub, Reddit,
          Hacker News, YouTube, blogs, and podcasts.
        </p>
        <div className={styles.meta}>
          <span><strong>{allStories.length}</strong> stories</span>
          <span><strong>{Object.keys(categoryCounts).length}</strong> categories</span>
          <span><strong>{Object.keys(sourceCounts).length}</strong> sources</span>
        </div>
      </div>

      {/* Category filters */}
      <div className={styles.filters}>
        <button
          type="button"
          className={`${styles.filterBtn} ${activeCategory === 'all' ? styles.filterActive : ''}`}
          onClick={() => setActiveCategory('all')}
        >
          All<span className={styles.filterCount}>{allStories.length}</span>
        </button>
        {Object.entries(CATEGORIES)
          .filter(([key]) => categoryCounts[key])
          .sort((a, b) => (categoryCounts[b[0]] ?? 0) - (categoryCounts[a[0]] ?? 0))
          .map(([key, meta]) => (
            <button
              key={key}
              type="button"
              className={`${styles.filterBtn} ${activeCategory === key ? styles.filterActive : ''}`}
              onClick={() => setActiveCategory(key)}
              style={
                activeCategory === key
                  ? { background: meta.solid, borderColor: meta.solid, color: '#0f172a' }
                  : undefined
              }
            >
              {meta.label}
              <span className={styles.filterCount}>{categoryCounts[key]}</span>
            </button>
          ))}
      </div>

      {/* Source filters — smaller, secondary row */}
      <div className={styles.filters} style={{ marginTop: '-0.75rem' }}>
        <button
          type="button"
          className={`${styles.filterBtn} ${activeSource === 'all' ? styles.filterActive : ''}`}
          onClick={() => setActiveSource('all')}
          style={{ fontSize: '0.72rem' }}
        >
          All sources
        </button>
        {Object.entries(SOURCE_LABELS)
          .filter(([key]) => sourceCounts[key])
          .map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={`${styles.filterBtn} ${activeSource === key ? styles.filterActive : ''}`}
              onClick={() => setActiveSource(key)}
              style={{
                fontSize: '0.72rem',
                ...(activeSource === key
                  ? { background: sourceColor(key), borderColor: sourceColor(key), color: '#fff' }
                  : {}),
              }}
            >
              {label}
              <span className={styles.filterCount}>{sourceCounts[key]}</span>
            </button>
          ))}
      </div>

      {/* Collage grid */}
      {visible.length === 0 ? (
        <div className={styles.empty}>No stories match that filter.</div>
      ) : (
        <div className={styles.grid}>
          {visible.map((s) => {
            const cat = CATEGORIES[s.category] ?? CATEGORIES.general;
            const sizeClass =
              s.size === 'lg' ? styles.tileLg : s.size === 'sm' ? styles.tileSm : styles.tileMd;
            const srcColor = sourceColor(s.source);
            return (
              <a
                key={s.id}
                className={`${styles.tile} ${sizeClass}`}
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                style={
                  {
                    '--tile-accent': cat.strip,
                    '--tile-accent-solid': cat.solid,
                    '--tile-accent-soft': cat.soft,
                  } as React.CSSProperties
                }
              >
                <div className={styles.badgeRow}>
                  <span className={styles.sourceBadge}>
                    <span className={styles.sourceIcon} style={{ background: srcColor }} />
                    {SOURCE_LABELS[s.source] ?? s.source}
                  </span>
                  <span className={styles.catTag}>{cat.label}</span>
                </div>
                <h3 className={styles.headline}>{s.headline}</h3>
                <p className={styles.quote}>&ldquo;{s.quote}&rdquo;</p>
                <span className={styles.author}>
                  {s.author}
                  {s.date ? <> &middot; {s.date}</> : null}
                </span>
                <span className={styles.external} aria-hidden="true">↗</span>
              </a>
            );
          })}
        </div>
      )}

      <div className={styles.footer}>
        Built something with Hermes?{' '}
        <a
          href="https://github.com/NousResearch/hermes-agent/edit/main/website/src/data/userStories.json"
          target="_blank"
          rel="noopener noreferrer"
        >
          Add your story to this page
        </a>{' '}
        by editing <code>userStories.json</code>, or post it in the{' '}
        <a href="https://discord.gg/NousResearch" target="_blank" rel="noopener noreferrer">
          Nous Research Discord
        </a>{' '}
        and we&apos;ll pick it up.
      </div>
    </div>
  );
}
