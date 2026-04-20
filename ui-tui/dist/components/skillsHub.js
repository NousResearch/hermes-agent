import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, Text, useInput } from '@hermes/ink';
import { useEffect, useState } from 'react';
import { rpcErrorMessage } from '../lib/rpc.js';
const VISIBLE = 12;
const pageOffset = (count, sel) => Math.max(0, Math.min(sel - Math.floor(VISIBLE / 2), count - VISIBLE));
const visibleItems = (items, sel) => {
    const off = pageOffset(items.length, sel);
    return { items: items.slice(off, off + VISIBLE), off };
};
export function SkillsHub({ gw, onClose, t }) {
    const [skillsByCat, setSkillsByCat] = useState({});
    const [selectedCat, setSelectedCat] = useState('');
    const [catIdx, setCatIdx] = useState(0);
    const [skillIdx, setSkillIdx] = useState(0);
    const [stage, setStage] = useState('category');
    const [info, setInfo] = useState(null);
    const [installing, setInstalling] = useState(false);
    const [err, setErr] = useState('');
    const [loading, setLoading] = useState(true);
    useEffect(() => {
        gw.request('skills.manage', { action: 'list' })
            .then(r => {
            setSkillsByCat(r?.skills ?? {});
            setErr('');
            setLoading(false);
        })
            .catch((e) => {
            setErr(rpcErrorMessage(e));
            setLoading(false);
        });
    }, [gw]);
    const cats = Object.keys(skillsByCat).sort();
    const skills = selectedCat ? (skillsByCat[selectedCat] ?? []) : [];
    const skillName = skills[skillIdx] ?? '';
    const inspect = (name) => {
        setInfo(null);
        setErr('');
        gw.request('skills.manage', { action: 'inspect', query: name })
            .then(r => setInfo(r?.info ?? { name }))
            .catch((e) => setErr(rpcErrorMessage(e)));
    };
    const install = (name) => {
        setInstalling(true);
        setErr('');
        gw.request('skills.manage', { action: 'install', query: name })
            .then(() => onClose())
            .catch((e) => setErr(rpcErrorMessage(e)))
            .finally(() => setInstalling(false));
    };
    useInput((ch, key) => {
        if (installing) {
            return;
        }
        if (key.escape) {
            if (stage === 'actions') {
                setStage('skill');
                setInfo(null);
                setErr('');
                return;
            }
            if (stage === 'skill') {
                setStage('category');
                setSkillIdx(0);
                return;
            }
            onClose();
            return;
        }
        if (stage === 'actions') {
            if (key.return) {
                setStage('skill');
                setInfo(null);
                setErr('');
                return;
            }
            if (ch.toLowerCase() === 'x' && skillName) {
                install(skillName);
                return;
            }
            if (ch.toLowerCase() === 'i' && skillName) {
                inspect(skillName);
            }
            return;
        }
        const count = stage === 'category' ? cats.length : skills.length;
        const sel = stage === 'category' ? catIdx : skillIdx;
        const setSel = stage === 'category' ? setCatIdx : setSkillIdx;
        if (key.upArrow && sel > 0) {
            setSel(v => v - 1);
            return;
        }
        if (key.downArrow && sel < count - 1) {
            setSel(v => v + 1);
            return;
        }
        if (key.return) {
            if (stage === 'category') {
                const cat = cats[catIdx];
                if (!cat) {
                    return;
                }
                setSelectedCat(cat);
                setSkillIdx(0);
                setStage('skill');
                return;
            }
            const name = skills[skillIdx];
            if (name) {
                setStage('actions');
                inspect(name);
            }
            return;
        }
        const n = ch === '0' ? 10 : parseInt(ch, 10);
        if (!Number.isNaN(n) && n >= 1 && n <= Math.min(10, count)) {
            const off = pageOffset(count, sel);
            const next = off + n - 1;
            if (stage === 'category') {
                const cat = cats[next];
                if (cat) {
                    setSelectedCat(cat);
                    setCatIdx(next);
                    setSkillIdx(0);
                    setStage('skill');
                }
                return;
            }
            const name = skills[next];
            if (name) {
                setSkillIdx(next);
                setStage('actions');
                inspect(name);
            }
        }
    });
    if (loading) {
        return _jsx(Text, { color: t.color.dim, children: "loading skills\u2026" });
    }
    if (err && stage === 'category') {
        return (_jsxs(Box, { flexDirection: "column", children: [_jsxs(Text, { color: t.color.label, children: ["error: ", err] }), _jsx(Text, { color: t.color.dim, children: "Esc to cancel" })] }));
    }
    if (!cats.length) {
        return (_jsxs(Box, { flexDirection: "column", children: [_jsx(Text, { color: t.color.dim, children: "no skills available" }), _jsx(Text, { color: t.color.dim, children: "Esc to cancel" })] }));
    }
    if (stage === 'category') {
        const rows = cats.map(c => `${c} · ${skillsByCat[c]?.length ?? 0} skills`);
        const { items, off } = visibleItems(rows, catIdx);
        return (_jsxs(Box, { flexDirection: "column", children: [_jsx(Text, { bold: true, color: t.color.amber, children: "Skills Hub" }), _jsx(Text, { color: t.color.dim, children: "select a category" }), off > 0 && _jsxs(Text, { color: t.color.dim, children: [" \u2191 ", off, " more"] }), items.map((row, i) => {
                    const idx = off + i;
                    return (_jsxs(Text, { color: catIdx === idx ? t.color.cornsilk : t.color.dim, children: [catIdx === idx ? '▸ ' : '  ', i + 1, ". ", row] }, row));
                }), off + VISIBLE < rows.length && _jsxs(Text, { color: t.color.dim, children: [" \u2193 ", rows.length - off - VISIBLE, " more"] }), _jsx(Text, { color: t.color.dim, children: "\u2191/\u2193 select \u00B7 Enter open \u00B7 1-9,0 quick \u00B7 Esc cancel" })] }));
    }
    if (stage === 'skill') {
        const { items, off } = visibleItems(skills, skillIdx);
        return (_jsxs(Box, { flexDirection: "column", children: [_jsx(Text, { bold: true, color: t.color.amber, children: selectedCat }), _jsxs(Text, { color: t.color.dim, children: [skills.length, " skill(s)"] }), !skills.length ? _jsx(Text, { color: t.color.dim, children: "no skills in this category" }) : null, off > 0 && _jsxs(Text, { color: t.color.dim, children: [" \u2191 ", off, " more"] }), items.map((row, i) => {
                    const idx = off + i;
                    return (_jsxs(Text, { color: skillIdx === idx ? t.color.cornsilk : t.color.dim, children: [skillIdx === idx ? '▸ ' : '  ', i + 1, ". ", row] }, row));
                }), off + VISIBLE < skills.length && _jsxs(Text, { color: t.color.dim, children: [" \u2193 ", skills.length - off - VISIBLE, " more"] }), _jsx(Text, { color: t.color.dim, children: skills.length ? '↑/↓ select · Enter open · 1-9,0 quick · Esc back' : 'Esc back' })] }));
    }
    return (_jsxs(Box, { flexDirection: "column", children: [_jsx(Text, { bold: true, color: t.color.amber, children: info?.name ?? skillName }), _jsx(Text, { color: t.color.dim, children: info?.category ?? selectedCat }), info?.description ? _jsx(Text, { color: t.color.cornsilk, children: info.description }) : null, info?.path ? _jsxs(Text, { color: t.color.dim, children: ["path: ", info.path] }) : null, !info && !err ? _jsx(Text, { color: t.color.dim, children: "loading\u2026" }) : null, err ? _jsxs(Text, { color: t.color.label, children: ["error: ", err] }) : null, installing ? _jsx(Text, { color: t.color.amber, children: "installing\u2026" }) : null, _jsx(Text, { color: t.color.dim, children: "i reinspect \u00B7 x reinstall \u00B7 Enter/Esc back" })] }));
}
