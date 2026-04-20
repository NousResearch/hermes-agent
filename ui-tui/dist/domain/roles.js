export const ROLE = {
    assistant: t => ({ body: t.color.cornsilk, glyph: t.brand.tool, prefix: t.color.bronze }),
    system: t => ({ body: '', glyph: '·', prefix: t.color.dim }),
    tool: t => ({ body: t.color.dim, glyph: '⚡', prefix: t.color.dim }),
    user: t => ({ body: t.color.label, glyph: t.brand.prompt, prefix: t.color.label })
};
