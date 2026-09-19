import { describe, expect, it } from "vitest";

import { SOURCE_TRANSLATIONS } from "./context";
import { en } from "./en";

const SHARED_MISSING_KEYS = [
  "app.currentProfileOption", "app.diskCriticalBanner", "app.diskElevatedBanner", "app.dismiss",
  "app.managingProfile", "app.managingProfileBanner", "app.memoryCriticalBanner", "app.memoryElevatedBanner",
  "app.memoryOomRestartBanner", "common.gateway", "common.gatewayHint", "cron.delivery.needsHomeChannel",
  "cron.delivery.noneConfigured", "kanban.assigneeLabel", "kanban.assigneeLabelHint", "kanban.boardSettings",
  "kanban.boardSettingsTitle", "kanban.boardSettingsTitleFor", "kanban.commentHint", "kanban.commentHintTitle",
  "kanban.confirmScheduled", "kanban.create", "kanban.needsAssignee", "kanban.needsAssigneeHint",
  "kanban.newTaskTitle", "kanban.parentLabel", "kanban.parentLabelHint", "kanban.projectDirectoryOverrideHint",
  "kanban.saving", "kanban.skillsLabel", "kanban.skillsLabelHint", "kanban.taskTitleLabel",
  "kanban.trash.confirmManyTitle", "kanban.trash.confirmTitle", "pluginsPage.catalogConfirmInstallNote", "pluginsPage.catalogConfirmTitle",
  "pluginsPage.catalogEmpty", "pluginsPage.catalogEmptyDocsLink", "pluginsPage.catalogHeading", "pluginsPage.catalogHint",
  "pluginsPage.catalogInstallBtn", "pluginsPage.catalogInstalledBadge", "pluginsPage.catalogRemovedBadge", "pluginsPage.catalogRequiresEnv",
  "pluginsPage.catalogSearchPlaceholder", "pluginsPage.catalogUpdateBtn", "pluginsPage.removedFromCatalog", "profiles.actions",
  "profiles.activeBadge", "profiles.activeProfile", "profiles.activeSet", "profiles.advancedOptions",
  "profiles.aliasBadge", "profiles.autoGenerate", "profiles.cloneAll", "profiles.describeFailed",
  "profiles.description", "profiles.descriptionOptional", "profiles.descriptionPlaceholder", "profiles.descriptionSaved",
  "profiles.distribution", "profiles.editDescription", "profiles.editModel", "profiles.gatewayRunning",
  "profiles.gatewayRunningWarning", "profiles.gatewayStopped", "profiles.generating", "profiles.modelInherit",
  "profiles.modelLoading", "profiles.modelNone", "profiles.modelOptional", "profiles.modelSaved",
  "profiles.modelSelect", "profiles.noDescription", "profiles.noSkillsOption", "profiles.reviewBadge",
  "profiles.setActive", "skills.currentProfile", "skills.managingProfile", "skills.profileSelector",
  "status.disabled", "status.restartGatewayConfirmMessage", "status.restartGatewayConfirmTitle", "status.updateHermesConfirmMessage",
  "status.updateHermesConfirmNow", "status.updateHermesConfirmTitle", "theme.fontDefault", "theme.fontDefaultHint",
  "theme.fontMono", "theme.fontSans", "theme.fontSerif", "theme.fontTitle",
] as const;

const ARABIC_ONLY_MISSING_KEYS = [
  "cron.scheduleDescribe.dailyAt", "cron.scheduleDescribe.everyDays", "cron.scheduleDescribe.everyHours", "cron.scheduleDescribe.everyMinutes",
  "cron.scheduleDescribe.monthlyAt", "cron.scheduleDescribe.none", "cron.scheduleDescribe.onceAt", "cron.scheduleDescribe.weeklyAt",
  "cron.scheduleMode", "cron.scheduleModes.custom", "cron.scheduleModes.customHint", "cron.scheduleModes.customLabel",
  "cron.scheduleModes.customPlaceholder", "cron.scheduleModes.daily", "cron.scheduleModes.dayOfMonth", "cron.scheduleModes.interval",
  "cron.scheduleModes.intervalEvery", "cron.scheduleModes.intervalUnit", "cron.scheduleModes.monthly", "cron.scheduleModes.once",
  "cron.scheduleModes.onceAt", "cron.scheduleModes.preview", "cron.scheduleModes.previewEmpty", "cron.scheduleModes.timeOfDay",
  "cron.scheduleModes.unitDays", "cron.scheduleModes.unitHours", "cron.scheduleModes.unitMinutes", "cron.scheduleModes.weekdays",
  "cron.scheduleModes.weekdaysShort", "cron.scheduleModes.weekly", "env.add", "env.addCustomKey",
  "env.customConfigured", "env.customHint", "env.customKeyName", "env.customKeyNamePlaceholder",
  "env.customTitle", "env.invalidKeyName", "env.showLess", "env.showMore",
  "kanban.columnHelp.scheduled", "kanban.columnLabels.scheduled", "oauth.copyCode", "oauth.copyFailed",
  "oauth.sessionExpiredNoError", "profiles.cloneFrom", "profiles.cloneFromNone", "sessions.clearSelection",
  "sessions.deleteEmpty", "sessions.deleteEmptyConfirmMessage", "sessions.deleteEmptyConfirmTitle", "sessions.deleteSelected",
  "sessions.deleteSelectedConfirmMessage", "sessions.deleteSelectedConfirmTitle", "sessions.emptySessionsDeleted", "sessions.failedToDeleteEmpty",
  "sessions.failedToDeleteSelected", "sessions.history", "sessions.newChat", "sessions.overview",
  "sessions.selectAllOnPage", "sessions.selectSession", "sessions.selectedCount", "sessions.selectedSessionsDeleted",
] as const;

const ARABIC_MISSING_KEYS = [
  ...SHARED_MISSING_KEYS.filter(
    (key) => key !== "kanban.needsAssignee" && key !== "kanban.needsAssigneeHint",
  ),
  ...ARABIC_ONLY_MISSING_KEYS,
];

const EXPECTED_MISSING_KEYS: Record<string, readonly string[]> = {
  en: [],
  zh: SHARED_MISSING_KEYS,
  "zh-hant": SHARED_MISSING_KEYS,
  ja: SHARED_MISSING_KEYS,
  de: SHARED_MISSING_KEYS,
  es: SHARED_MISSING_KEYS,
  fr: SHARED_MISSING_KEYS,
  tr: SHARED_MISSING_KEYS,
  uk: SHARED_MISSING_KEYS,
  af: SHARED_MISSING_KEYS,
  ko: SHARED_MISSING_KEYS,
  it: SHARED_MISSING_KEYS,
  ga: SHARED_MISSING_KEYS,
  pt: SHARED_MISSING_KEYS,
  ru: SHARED_MISSING_KEYS,
  hu: SHARED_MISSING_KEYS,
  ar: ARABIC_MISSING_KEYS,
};

function collectKeyPaths(value: unknown, prefix = ""): string[] {
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    return [prefix];
  }

  return Object.entries(value).flatMap(([key, child]) =>
    collectKeyPaths(child, prefix ? `${prefix}.${key}` : key),
  );
}

describe("web locale catalog key coverage", () => {
  it.each(Object.entries(SOURCE_TRANSLATIONS))(
    "%s does not add untranslated English fallback keys",
    (locale, catalog) => {
      const localeKeyPaths = collectKeyPaths(catalog);
      const missingKeys = collectKeyPaths(en).filter(
        (key) => !localeKeyPaths.includes(key),
      );
      const expectedMissingKeys = EXPECTED_MISSING_KEYS[locale];

      expect(localeKeyPaths.filter((key) => !collectKeyPaths(en).includes(key))).toEqual([]);
      expect(missingKeys.filter((key) => !expectedMissingKeys.includes(key))).toEqual([]);
      expect(expectedMissingKeys.filter((key) => !missingKeys.includes(key))).toEqual([]);
    },
  );
});
