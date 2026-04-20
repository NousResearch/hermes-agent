import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, Text } from '@hermes/ink';
import { useStore } from '@nanostores/react';
import { useGateway } from '../app/gatewayContext.js';
import { $overlayState, patchOverlayState } from '../app/overlayStore.js';
import { $uiState } from '../app/uiStore.js';
import { FloatBox } from './appChrome.js';
import { MaskedPrompt } from './maskedPrompt.js';
import { ModelPicker } from './modelPicker.js';
import { ApprovalPrompt, ClarifyPrompt, ConfirmPrompt } from './prompts.js';
import { SessionPicker } from './sessionPicker.js';
import { SkillsHub } from './skillsHub.js';
export function PromptZone({ cols, onApprovalChoice, onClarifyAnswer, onSecretSubmit, onSudoSubmit }) {
    const overlay = useStore($overlayState);
    const ui = useStore($uiState);
    if (overlay.approval) {
        return (_jsx(Box, { flexDirection: "column", flexShrink: 0, paddingX: 1, paddingY: 1, children: _jsx(ApprovalPrompt, { onChoice: onApprovalChoice, req: overlay.approval, t: ui.theme }) }));
    }
    if (overlay.confirm) {
        const req = overlay.confirm;
        const onConfirm = () => {
            patchOverlayState({ confirm: null });
            req.onConfirm();
        };
        const onCancel = () => patchOverlayState({ confirm: null });
        return (_jsx(Box, { flexDirection: "column", flexShrink: 0, paddingX: 1, paddingY: 1, children: _jsx(ConfirmPrompt, { onCancel: onCancel, onConfirm: onConfirm, req: req, t: ui.theme }) }));
    }
    if (overlay.clarify) {
        return (_jsx(Box, { flexDirection: "column", flexShrink: 0, paddingX: 1, paddingY: 1, children: _jsx(ClarifyPrompt, { cols: cols, onAnswer: onClarifyAnswer, onCancel: () => onClarifyAnswer(''), req: overlay.clarify, t: ui.theme }) }));
    }
    if (overlay.sudo) {
        return (_jsx(Box, { flexDirection: "column", flexShrink: 0, paddingX: 1, paddingY: 1, children: _jsx(MaskedPrompt, { cols: cols, icon: "\uD83D\uDD10", label: "sudo password required", onSubmit: onSudoSubmit, t: ui.theme }) }));
    }
    if (overlay.secret) {
        return (_jsx(Box, { flexDirection: "column", flexShrink: 0, paddingX: 1, paddingY: 1, children: _jsx(MaskedPrompt, { cols: cols, icon: "\uD83D\uDD11", label: overlay.secret.prompt, onSubmit: onSecretSubmit, sub: `for ${overlay.secret.envVar}`, t: ui.theme }) }));
    }
    return null;
}
export function FloatingOverlays({ cols, compIdx, completions, onModelSelect, onPickerSelect, pagerPageSize }) {
    const { gw } = useGateway();
    const overlay = useStore($overlayState);
    const ui = useStore($uiState);
    const hasAny = overlay.modelPicker || overlay.pager || overlay.picker || overlay.skillsHub || completions.length;
    if (!hasAny) {
        return null;
    }
    const start = Math.max(0, compIdx - 8);
    return (_jsxs(Box, { alignItems: "flex-start", bottom: "100%", flexDirection: "column", left: 0, position: "absolute", right: 0, children: [overlay.picker && (_jsx(FloatBox, { color: ui.theme.color.bronze, children: _jsx(SessionPicker, { gw: gw, onCancel: () => patchOverlayState({ picker: false }), onSelect: onPickerSelect, t: ui.theme }) })), overlay.modelPicker && (_jsx(FloatBox, { color: ui.theme.color.bronze, children: _jsx(ModelPicker, { gw: gw, onCancel: () => patchOverlayState({ modelPicker: false }), onSelect: onModelSelect, sessionId: ui.sid, t: ui.theme }) })), overlay.skillsHub && (_jsx(FloatBox, { color: ui.theme.color.bronze, children: _jsx(SkillsHub, { gw: gw, onClose: () => patchOverlayState({ skillsHub: false }), t: ui.theme }) })), overlay.pager && (_jsx(FloatBox, { color: ui.theme.color.bronze, children: _jsxs(Box, { flexDirection: "column", paddingX: 1, paddingY: 1, children: [overlay.pager.title && (_jsx(Box, { justifyContent: "center", marginBottom: 1, children: _jsx(Text, { bold: true, color: ui.theme.color.gold, children: overlay.pager.title }) })), overlay.pager.lines.slice(overlay.pager.offset, overlay.pager.offset + pagerPageSize).map((line, i) => (_jsx(Text, { children: line }, i))), _jsx(Box, { marginTop: 1, children: _jsx(Text, { color: ui.theme.color.dim, children: overlay.pager.offset + pagerPageSize < overlay.pager.lines.length
                                    ? `Enter/Space for more · q to close (${Math.min(overlay.pager.offset + pagerPageSize, overlay.pager.lines.length)}/${overlay.pager.lines.length})`
                                    : `end · q to close (${overlay.pager.lines.length} lines)` }) })] }) })), !!completions.length && (_jsx(FloatBox, { color: ui.theme.color.gold, children: _jsx(Box, { flexDirection: "column", width: Math.max(28, cols - 6), children: completions.slice(start, compIdx + 8).map((item, i) => {
                        const active = start + i === compIdx;
                        return (_jsxs(Box, { backgroundColor: active ? ui.theme.color.completionCurrentBg : undefined, flexDirection: "row", width: "100%", children: [_jsxs(Text, { bold: true, color: ui.theme.color.label, children: [' ', item.display] }), item.meta ? _jsxs(Text, { color: ui.theme.color.dim, children: [" ", item.meta] }) : null] }, `${start + i}:${item.text}:${item.display}:${item.meta ?? ''}`));
                    }) }) }))] }));
}
