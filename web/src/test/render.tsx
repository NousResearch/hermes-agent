import type { ReactElement, ReactNode } from "react";
import { render } from "@testing-library/react";
import { MemoryRouter, type MemoryRouterProps } from "react-router-dom";
import { I18nProvider } from "@/i18n";

interface RenderOptions {
  routerProps?: MemoryRouterProps;
}

export function renderWithAppProviders(
  ui: ReactElement,
  { routerProps }: RenderOptions = {},
) {
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <MemoryRouter {...routerProps}>
      <I18nProvider>{children}</I18nProvider>
    </MemoryRouter>
  );

  return render(ui, { wrapper: Wrapper });
}
