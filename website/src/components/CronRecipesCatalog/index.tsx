import React, { useEffect, useState } from "react";
import styles from "./styles.module.css";

interface RecipeField {
  name: string;
  type: string;
  label: string;
  default: string | null;
  options: string[];
  optional: boolean;
  help: string;
}

interface Recipe {
  key: string;
  title: string;
  description: string;
  category: string;
  tags: string[];
  fields: RecipeField[];
  scheduleHuman: string;
  command: string;
  appUrl: string;
}

const INDEX_URL = "/docs/api/cron-recipes-index.json";

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

function RecipeCard({ recipe }: { recipe: Recipe }): JSX.Element {
  return (
    <div className={styles.card}>
      <div className={styles.cardHead}>
        <h3 className={styles.title}>{recipe.title}</h3>
        <span className={styles.schedule}>{recipe.scheduleHuman}</span>
      </div>
      <p className={styles.desc}>{recipe.description}</p>

      <div className={styles.tags}>
        {recipe.tags.map((t) => (
          <span key={t} className={styles.tag}>
            {t}
          </span>
        ))}
      </div>

      <div className={styles.cmdRow}>
        <code className={styles.cmd}>{recipe.command}</code>
        <CopyButton text={recipe.command} />
      </div>

      <div className={styles.actions}>
        <a className={styles.appBtn} href={recipe.appUrl}>
          Send to App ↗
        </a>
        <span className={styles.hint}>
          or paste the command into the CLI, TUI, or any messenger
        </span>
      </div>
    </div>
  );
}

export default function CronRecipesCatalog(): JSX.Element {
  const [recipes, setRecipes] = useState<Recipe[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(INDEX_URL)
      .then((r) => r.json())
      .then((data: Recipe[]) => {
        if (!cancelled) setRecipes(data);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return <p>Couldn't load the recipe catalog: {error}</p>;
  }
  if (recipes === null) {
    return <p>Loading recipes…</p>;
  }
  if (recipes.length === 0) {
    return <p>No cron recipes are available.</p>;
  }

  return (
    <div className={styles.grid}>
      {recipes.map((r) => (
        <RecipeCard key={r.key} recipe={r} />
      ))}
    </div>
  );
}
