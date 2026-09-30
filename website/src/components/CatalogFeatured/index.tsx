import React from "react";
import Link from "@docusaurus/Link";

import styles from "./styles.module.css";

/** The catalog hero both /skills and /plugins show, laid out like the desktop
 *  app's featured card (author, title, one-line description, tags; action
 *  bottom-right; no artwork). The pick comes from apps/shared `pickFeatured`,
 *  so the site and the app always feature the same entry. */
export default function CatalogFeatured({
  label,
  author,
  title,
  description,
  tags,
  href,
  cta,
}: {
  /** Section label, e.g. "Featured skill"; names the region for screen readers too. */
  label: string;
  author?: string | null;
  title: string;
  description: string;
  /** Falsy entries are dropped, so callers can pass optional fields inline. */
  tags: Array<string | false | null | undefined>;
  /** Internal docs route (Link) or an absolute URL (new tab). */
  href: string;
  cta: string;
}) {
  const external = /^https?:\/\//.test(href);
  const chips = tags.filter((tag): tag is string => Boolean(tag));

  return (
    <section aria-label={label} className={styles.featured}>
      <p className={styles.label}>{label}</p>
      <div className={styles.card}>
        <div className={styles.copy}>
          {author && <p className={styles.author}>{author}</p>}
          <h2 className={styles.title}>{title}</h2>
          {description && <p className={styles.description}>{description}</p>}
          {chips.length > 0 && (
            <div className={styles.tags}>
              {chips.map((tag) => (
                <span className={styles.tag} key={tag}>
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
        {external ? (
          <a className={styles.cta} href={href} rel="noopener noreferrer" target="_blank">
            {cta} ↗
          </a>
        ) : (
          <Link className={styles.cta} to={href}>
            {cta} →
          </Link>
        )}
      </div>
    </section>
  );
}
