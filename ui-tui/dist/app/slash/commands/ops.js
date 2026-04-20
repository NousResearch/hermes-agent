import { patchOverlayState } from '../../overlayStore.js';
export const opsCommands = [
    {
        help: 'browse, inspect, install skills',
        name: 'skills',
        run: (arg, ctx) => {
            const text = arg.trim();
            if (!text) {
                return patchOverlayState({ skillsHub: true });
            }
            const [sub, ...rest] = text.split(/\s+/);
            const query = rest.join(' ').trim();
            const { rpc } = ctx.gateway;
            const { page, panel, sys } = ctx.transcript;
            if (sub === 'list') {
                rpc('skills.manage', { action: 'list' })
                    .then(ctx.guarded(r => {
                    const cats = Object.entries(r.skills ?? {}).sort();
                    if (!cats.length) {
                        return sys('no skills available');
                    }
                    panel('Skills', cats.map(([title, items]) => ({ items, title })));
                }))
                    .catch(ctx.guardedErr);
                return;
            }
            if (sub === 'inspect') {
                if (!query) {
                    return sys('usage: /skills inspect <name>');
                }
                rpc('skills.manage', { action: 'inspect', query })
                    .then(ctx.guarded(r => {
                    const info = r.info ?? {};
                    if (!info.name) {
                        return sys(`unknown skill: ${query}`);
                    }
                    const rows = [
                        ['Name', String(info.name)],
                        ['Category', String(info.category ?? '')],
                        ['Path', String(info.path ?? '')]
                    ];
                    const sections = [{ rows }];
                    if (info.description) {
                        sections.push({ text: String(info.description) });
                    }
                    panel('Skill', sections);
                }))
                    .catch(ctx.guardedErr);
                return;
            }
            if (sub === 'search') {
                if (!query) {
                    return sys('usage: /skills search <query>');
                }
                rpc('skills.manage', { action: 'search', query })
                    .then(ctx.guarded(r => {
                    const results = r.results ?? [];
                    if (!results.length) {
                        return sys(`no results for: ${query}`);
                    }
                    panel(`Search: ${query}`, [{ rows: results.map(s => [s.name, s.description ?? '']) }]);
                }))
                    .catch(ctx.guardedErr);
                return;
            }
            if (sub === 'install') {
                if (!query) {
                    return sys('usage: /skills install <name or url>');
                }
                sys(`installing ${query}…`);
                rpc('skills.manage', { action: 'install', query })
                    .then(ctx.guarded(r => sys(r.installed ? `installed ${r.name ?? query}` : 'install failed')))
                    .catch(ctx.guardedErr);
                return;
            }
            if (sub === 'browse') {
                const pageNum = query ? parseInt(query, 10) : 1;
                if (Number.isNaN(pageNum) || pageNum < 1) {
                    return sys('usage: /skills browse [page]  (page must be a positive number)');
                }
                sys('fetching community skills (scans 6 sources, may take ~15s)…');
                rpc('skills.manage', { action: 'browse', page: pageNum })
                    .then(ctx.guarded(r => {
                    const items = r.items ?? [];
                    if (!items.length) {
                        return sys(`no skills on page ${pageNum}${r.total ? ` (total ${r.total})` : ''}`);
                    }
                    const rows = items.map(s => [
                        s.trust ? `${s.name} · ${s.trust}` : s.name,
                        String(s.description ?? '').slice(0, 160)
                    ]);
                    const footer = [];
                    if (r.page && r.total_pages) {
                        footer.push(`page ${r.page} of ${r.total_pages}`);
                    }
                    if (r.total) {
                        footer.push(`${r.total} skills total`);
                    }
                    if (r.page && r.total_pages && r.page < r.total_pages) {
                        footer.push(`/skills browse ${r.page + 1} for more`);
                    }
                    panel(`Browse Skills${pageNum > 1 ? ` — p${pageNum}` : ''}`, [
                        { rows },
                        ...(footer.length ? [{ text: footer.join(' · ') }] : [])
                    ]);
                }))
                    .catch(ctx.guardedErr);
                return;
            }
            sys('usage: /skills [list | inspect <n> | install <n> | search <q> | browse [page]]');
        }
    },
    {
        help: 'enable or disable tools (client-side history reset on change)',
        name: 'tools',
        run: (arg, ctx) => {
            const [subcommand, ...names] = arg.trim().split(/\s+/).filter(Boolean);
            if (subcommand !== 'disable' && subcommand !== 'enable') {
                return;
            }
            if (!names.length) {
                ctx.transcript.sys(`usage: /tools ${subcommand} <name> [name ...]`);
                ctx.transcript.sys(`built-in toolset: /tools ${subcommand} web`);
                ctx.transcript.sys(`MCP tool: /tools ${subcommand} github:create_issue`);
                return;
            }
            ctx.gateway
                .rpc('tools.configure', { action: subcommand, names, session_id: ctx.sid })
                .then(ctx.guarded(r => {
                if (r.info) {
                    ctx.session.setSessionStartedAt(Date.now());
                    ctx.session.resetVisibleHistory(r.info);
                }
                if (r.changed?.length) {
                    ctx.transcript.sys(`${subcommand === 'disable' ? 'disabled' : 'enabled'}: ${r.changed.join(', ')}`);
                }
                if (r.unknown?.length) {
                    ctx.transcript.sys(`unknown toolsets: ${r.unknown.join(', ')}`);
                }
                if (r.missing_servers?.length) {
                    ctx.transcript.sys(`missing MCP servers: ${r.missing_servers.join(', ')}`);
                }
                if (r.reset) {
                    ctx.transcript.sys('session reset. new tool configuration is active.');
                }
            }))
                .catch(ctx.guardedErr);
        }
    }
];
