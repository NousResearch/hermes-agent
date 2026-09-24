import React from "react";
import Link from "@docusaurus/Link";
import styles from "./catalog-tabs.module.css";

export type MarketplaceKind = "skills" | "plugins" | "bots";

const ITEMS: { kind: MarketplaceKind; label: string; to: string }[] = [
  { kind: "skills", label: "Skills", to: "/skills" },
  { kind: "plugins", label: "Plugins", to: "/plugins" },
  { kind: "bots", label: "Bots", to: "/bots" },
];

export default function CatalogTabs({ active }: { active: MarketplaceKind }) {
  return (
    <nav className={styles.tabs} aria-label="Marketplace pages">
      {ITEMS.map((item) =>
        item.kind === active ? (
          <span key={item.kind} className={`${styles.tab} ${styles.active}`} aria-current="page">
            {item.label}
          </span>
        ) : (
          <Link key={item.kind} className={styles.tab} to={item.to}>
            {item.label}
          </Link>
        ),
      )}
    </nav>
  );
}
