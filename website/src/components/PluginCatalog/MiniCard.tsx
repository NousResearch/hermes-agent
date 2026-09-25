import React from "react";
import Link from "@docusaurus/Link";
import {
  type CatalogPlugin,
  categoryOf,
  formatStars,
  pluginPagePath,
  repoUrl,
  tierOf,
} from "./catalog";
import styles from "./pages.module.css";

/** Compact card for shelves: "More by this author" on a plugin page and the author page grid. */
export default function MiniCard({ plugin }: { plugin: CatalogPlugin }) {
  const tier = tierOf(plugin);
  const category = categoryOf(plugin);
  return (
    <Link className={styles.miniCard} to={pluginPagePath(plugin.name)}>
      <span className={styles.miniAccent} style={{ background: tier.color }} />
      <div className={styles.miniTop}>
        <span className={styles.miniIcon} aria-hidden="true" title={category.label}>
          {category.icon}
        </span>
        <span className={styles.miniName}>{plugin.name}</span>
        <span
          className={styles.tierPill}
          style={{ color: tier.color, background: tier.bg, borderColor: tier.border }}
        >
          {tier.icon} {tier.label}
        </span>
        {typeof plugin.stars === "number" && (
          <span className={styles.miniStars} title={`${plugin.stars.toLocaleString()} GitHub stars on ${repoUrl(plugin)}`}>
            {"\u2605"} {formatStars(plugin.stars)}
          </span>
        )}
      </div>
      <p className={styles.miniDesc}>{plugin.description || "No description available."}</p>
      <span className={styles.miniCategory}>{category.label}</span>
    </Link>
  );
}
