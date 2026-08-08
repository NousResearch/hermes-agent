export function configSectionLabel(section: string, category: string): string | null {
  if (section === "gateway" && category === "buzz") return null;
  return section.replace(/_/g, " ");
}
