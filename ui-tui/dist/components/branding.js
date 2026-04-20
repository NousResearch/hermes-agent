import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, Text, useStdout } from '@hermes/ink';
import { artWidth, caduceus, CADUCEUS_WIDTH, logo, LOGO_WIDTH } from '../banner.js';
import { flat } from '../lib/text.js';
export function ArtLines({ lines }) {
    return (_jsx(_Fragment, { children: lines.map(([c, text], i) => (_jsx(Text, { color: c, children: text }, i))) }));
}
export function Banner({ t }) {
    const cols = useStdout().stdout?.columns ?? 80;
    const logoLines = logo(t.color, t.bannerLogo || undefined);
    return (_jsxs(Box, { flexDirection: "column", marginBottom: 1, children: [cols >= (t.bannerLogo ? artWidth(logoLines) : LOGO_WIDTH) ? (_jsx(ArtLines, { lines: logoLines })) : (_jsxs(Text, { bold: true, color: t.color.gold, children: [t.brand.icon, " NOUS HERMES"] })), _jsxs(Text, { color: t.color.dim, children: [t.brand.icon, " Nous Research \u00B7 Messenger of the Digital Gods"] })] }));
}
export function SessionPanel({ info, sid, t }) {
    const cols = useStdout().stdout?.columns ?? 100;
    const heroLines = caduceus(t.color, t.bannerHero || undefined);
    const leftW = Math.min((artWidth(heroLines) || CADUCEUS_WIDTH) + 4, Math.floor(cols * 0.4));
    const wide = cols >= 90 && leftW + 40 < cols;
    const w = Math.max(20, wide ? cols - leftW - 14 : cols - 12);
    const lineBudget = Math.max(12, w - 2);
    const strip = (s) => (s.endsWith('_tools') ? s.slice(0, -6) : s);
    const truncLine = (pfx, items) => {
        let line = '';
        let shown = 0;
        for (const item of [...items].sort()) {
            const next = line ? `${line}, ${item}` : item;
            if (pfx.length + next.length > lineBudget) {
                return line ? `${line}, …+${items.length - shown}` : `${item}, …`;
            }
            line = next;
            shown++;
        }
        return line;
    };
    const section = (title, data, max = 8, overflowLabel = 'more…') => {
        const entries = Object.entries(data).sort();
        const shown = entries.slice(0, max);
        const overflow = entries.length - max;
        return (_jsxs(Box, { flexDirection: "column", marginTop: 1, children: [_jsxs(Text, { bold: true, color: t.color.amber, children: ["Available ", title] }), shown.map(([k, vs]) => (_jsxs(Text, { wrap: "truncate", children: [_jsxs(Text, { color: t.color.dim, children: [strip(k), ": "] }), _jsx(Text, { color: t.color.cornsilk, children: truncLine(strip(k) + ': ', vs) })] }, k))), overflow > 0 && (_jsxs(Text, { color: t.color.dim, children: ["(and ", overflow, " ", overflowLabel, ")"] }))] }));
    };
    return (_jsxs(Box, { borderColor: t.color.bronze, borderStyle: "round", marginBottom: 1, paddingX: 2, paddingY: 1, children: [wide && (_jsxs(Box, { flexDirection: "column", marginRight: 2, width: leftW, children: [_jsx(ArtLines, { lines: heroLines }), _jsx(Text, {}), _jsxs(Text, { color: t.color.amber, children: [info.model.split('/').pop(), _jsx(Text, { color: t.color.dim, children: " \u00B7 Nous Research" })] }), _jsx(Text, { color: t.color.dim, wrap: "truncate-end", children: info.cwd || process.cwd() }), sid && (_jsxs(Text, { children: [_jsx(Text, { color: t.color.sessionLabel, children: "Session: " }), _jsx(Text, { color: t.color.sessionBorder, children: sid })] }))] })), _jsxs(Box, { flexDirection: "column", width: w, children: [_jsx(Box, { justifyContent: "center", marginBottom: 1, children: _jsxs(Text, { bold: true, color: t.color.gold, children: [t.brand.name, info.version ? ` v${info.version}` : '', info.release_date ? ` (${info.release_date})` : ''] }) }), section('Tools', info.tools, 8, 'more toolsets…'), section('Skills', info.skills), info.mcp_servers && info.mcp_servers.length > 0 && (_jsxs(Box, { flexDirection: "column", marginTop: 1, children: [_jsx(Text, { bold: true, color: t.color.amber, children: "MCP Servers" }), info.mcp_servers.map(s => (_jsxs(Text, { wrap: "truncate", children: [_jsx(Text, { color: t.color.dim, children: `  ${s.name} ` }), _jsx(Text, { color: t.color.dim, children: `[${s.transport}]` }), _jsx(Text, { color: t.color.dim, children: ": " }), s.connected ? (_jsxs(Text, { color: t.color.cornsilk, children: [s.tools, " tool", s.tools === 1 ? '' : 's'] })) : (_jsx(Text, { color: t.color.error, children: "failed" }))] }, s.name)))] })), _jsx(Text, {}), _jsxs(Text, { color: t.color.cornsilk, children: [flat(info.tools).length, " tools", ' · ', flat(info.skills).length, " skills", info.mcp_servers?.length ? ` · ${info.mcp_servers.length} MCP` : '', ' · ', _jsx(Text, { color: t.color.dim, children: "/help for commands" })] }), typeof info.update_behind === 'number' && info.update_behind > 0 && (_jsxs(Text, { bold: true, color: "yellow", children: ["! ", info.update_behind, " ", info.update_behind === 1 ? 'commit' : 'commits', " behind", _jsxs(Text, { bold: false, color: "yellow", dimColor: true, children: [' ', "- run", ' '] }), _jsx(Text, { bold: true, color: "yellow", children: info.update_command || 'hermes update' }), _jsxs(Text, { bold: false, color: "yellow", dimColor: true, children: [' ', "to update"] })] }))] })] }));
}
export function Panel({ sections, t, title }) {
    return (_jsxs(Box, { borderColor: t.color.bronze, borderStyle: "round", flexDirection: "column", paddingX: 2, paddingY: 1, children: [_jsx(Box, { justifyContent: "center", marginBottom: 1, children: _jsx(Text, { bold: true, color: t.color.gold, children: title }) }), sections.map((sec, si) => (_jsxs(Box, { flexDirection: "column", marginTop: si > 0 ? 1 : 0, children: [sec.title && (_jsx(Text, { bold: true, color: t.color.amber, children: sec.title })), sec.rows?.map(([k, v], ri) => (_jsxs(Text, { wrap: "truncate", children: [_jsx(Text, { color: t.color.dim, children: k.padEnd(20) }), _jsx(Text, { color: t.color.cornsilk, children: v })] }, ri))), sec.items?.map((item, ii) => (_jsx(Text, { color: t.color.cornsilk, wrap: "truncate", children: item }, ii))), sec.text && _jsx(Text, { color: t.color.dim, children: sec.text })] }, si)))] }));
}
