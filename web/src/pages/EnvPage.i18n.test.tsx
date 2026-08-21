// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import type { ComponentType, Dispatch, SetStateAction } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { EnvVarInfo } from "@/lib/api";
import type { Translations } from "@/i18n/types";
import { en } from "@/i18n/en";
import { pl } from "@/i18n/pl";

import { EnvCategoryCard, ProviderGroupCard } from "./EnvPage";

const i18nState = vi.hoisted(() => ({ t: undefined as unknown }));

vi.mock("@/i18n", () => ({
  useI18n: () => ({ t: i18nState.t as Translations }),
}));

const setEdits = vi.fn() as unknown as Dispatch<
  SetStateAction<Record<string, string>>
>;
const noop = vi.fn();
const rowProps = {
  edits: {},
  setEdits,
  revealed: {},
  saving: null,
  onSave: noop,
  onClear: noop,
  onReveal: noop,
  onCancelEdit: noop,
};

function envInfo(isSet: boolean): EnvVarInfo {
  return {
    is_set: isSet,
    redacted_value: isSet ? "***" : null,
    description: "Test key",
    url: null,
    category: "provider",
    is_password: true,
    tools: [],
    advanced: false,
  };
}

const entries: [string, EnvVarInfo][] = [
  ["TEST_API_KEY", envInfo(true)],
  ["TEST_BASE_URL", envInfo(false)],
  ["TEST_REGION", envInfo(false)],
];

const TestIcon: ComponentType<{ className?: string }> = () => null;

describe("EnvPage localized count composition", () => {
  beforeEach(() => {
    i18nState.t = pl;
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("renders complete Polish provider-group labels from catalog callbacks", () => {
    render(
      <ProviderGroupCard
        group={{ name: "Test", priority: 1, entries, hasAnySet: true }}
        {...rowProps}
      />,
    );

    expect(screen.getByText("Skonfigurowano: 1")).not.toBeNull();
    expect(screen.getByText("Liczba kluczy: 3")).not.toBeNull();
    expect(screen.queryByText("1 ustaw")).toBeNull();
  });

  it("renders a complete Polish category summary and preserves English copy", () => {
    const section = {
      category: "provider",
      icon: TestIcon,
      label: "Providers",
      setEntries: entries.slice(0, 1),
      totalEntries: 2,
      unsetEntries: entries.slice(1, 2),
    };

    const { rerender } = render(
      <EnvCategoryCard section={section} {...rowProps} />,
    );

    expect(screen.getByText("Skonfigurowano: 1/2")).not.toBeNull();
    expect(screen.queryByText("1 z 2 skonfigurowane")).toBeNull();

    i18nState.t = en;
    rerender(<EnvCategoryCard section={section} {...rowProps} />);

    expect(screen.getByText("1 of 2 configured")).not.toBeNull();
  });
});
