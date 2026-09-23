import React, { useEffect, useState } from "react";
import styles from "./styles.module.css";

interface BlueprintField {
  name: string;
  type: string;
  label: string;
  default: string | null;
  options: string[];
  optional: boolean;
  help: string;
}

interface Blueprint {
  key: string;
  title: string;
  description: string;
  category: string;
  tags: string[];
  fields: BlueprintField[];
  scheduleHuman: string;
  command: string;
  appUrl: string;
}

const INDEX_URL = "/docs/api/automation-blueprints-index.json";

function CopyButton({ text }: { text: string }): JSX.Element {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className={styles.copyBtn}
      onClick={() => {
        navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
      aria-label="Copy command"
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function BlueprintCard({ blueprint }: { blueprint: Blueprint }): JSX.Element {
  return (
    <div className={styles.card}>
      <div className={styles.cardHead}>
        <h3 className={styles.title}>{blueprint.title}</h3>
        <span className={styles.schedule}>{blueprint.scheduleHuman}</span>
      </div>
      <p className={styles.desc}>{blueprint.description}</p>

      <div className={styles.tags}>
        {blueprint.tags.map((t) => (
          <span key={t} className={styles.tag}>
            {t}
          </span>
        ))}
      </div>

      <div className={styles.cmdRow}>
        <code className={styles.cmd}>{blueprint.command}</code>
        <CopyButton text={blueprint.command} />
      </div>

      <div className={styles.actions}>
        <a className={styles.appBtn} href={blueprint.appUrl}>
          Send to App ↗
        </a>
        <span className={styles.hint}>
          or paste the command into the CLI, TUI, or any messenger
        </span>
      </div>
    </div>
  );
}

export default function AutomationBlueprintsCatalog(): JSX.Element {
  const [blueprints, setBlueprints] = useState<Blueprint[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(INDEX_URL)
      .then((r) => r.json())
      .then((data: Blueprint[]) => {
        if (!cancelled) setBlueprints(data);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return <p>Couldn't load the blueprint catalog: {error}</p>;
  }
  if (blueprints === null) {
    return <p>Loading blueprints…</p>;
  }
  if (blueprints.length === 0) {
    return <p>No automation blueprints are available.</p>;
  }

  return (
    <div className={styles.grid}>
      {blueprints.map((r) => (
        <BlueprintCard key={r.key} blueprint={r} />
      ))}
    </div>
  );
}
