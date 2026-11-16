import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { I18nProvider, LOCALE_META, useI18n } from "./context";
import { en } from "./en";
import type { Locale } from "./types";

function CronLabels() {
  const { locale, t } = useI18n();
  return (
    <div lang={locale}>
      <span>{t.cron.scheduledJobs}</span>
      {Object.entries(t.cron.analytics).map(([key, label]) => (
        <span key={key} data-label={key}>{label}</span>
      ))}
    </div>
  );
}

function renderLabels(locale: Locale) {
  vi.stubGlobal("localStorage", { getItem: () => locale });
  return renderToStaticMarkup(<I18nProvider><CronLabels /></I18nProvider>);
}

afterEach(() => vi.unstubAllGlobals());

describe("cron analytics labels through the real I18nProvider", () => {
  it("preserves Turkish labels and falls back to English for new analytics labels", () => {
    const html = renderLabels("tr");
    expect(html).toContain('lang="tr"');
    expect(html).toContain("Zamanlanmış Görevler");
    expect(html).toContain(en.cron.analytics.period);
    expect(html).toContain(en.cron.analytics.costBasis);
  });

  it.each(Object.keys(LOCALE_META) as Locale[])("renders every analytics label for %s", (locale) => {
    const html = renderLabels(locale);
    for (const key of Object.keys(en.cron.analytics)) {
      expect(html).toContain(`data-label="${key}"`);
    }
  });
});
