export const providerDisplayNames = (providers) => {
    const counts = new Map();
    for (const p of providers) {
        counts.set(p.name, (counts.get(p.name) ?? 0) + 1);
    }
    return providers.map(p => (counts.get(p.name) ?? 0) > 1 && p.slug && p.slug !== p.name ? `${p.name} (${p.slug})` : p.name);
};
