export const providerDisplayNames = (providers: readonly { name: string; slug: string }[]): string[] => {
  const counts = new Map<string, number>()

  for (const p of providers) {
    counts.set(p.name, (counts.get(p.name) ?? 0) + 1)
  }

  return providers.map(p =>
    (counts.get(p.name) ?? 0) > 1 && p.slug && p.slug !== p.name ? `${p.name} (${p.slug})` : p.name
  )
}
