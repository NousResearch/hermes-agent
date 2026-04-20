import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { Ansi, Box, NoSelect, Text } from '@hermes/ink';
import { memo } from 'react';
import { LONG_MSG, MAX_MSG_CHARS, MAX_MSG_LINES } from '../config/limits.js';
import { userDisplay } from '../domain/messages.js';
import { ROLE } from '../domain/roles.js';
import { compactPreview, hasAnsi, isPasteBackedText, stripAnsi } from '../lib/text.js';
import { Md } from './markdown.js';
import { ToolTrail } from './thinking.js';
// Truncate message text to prevent memory issues
const truncateText = (text, maxChars, maxLines) => {
    if (text.length <= maxChars) {
        const lines = text.split('\n');
        if (lines.length <= maxLines) {
            return { text, truncated: false };
        }
        return {
            text: lines.slice(0, maxLines).join('\n') + '\n... [truncated]',
            truncated: true
        };
    }
    const lines = text.split('\n');
    if (lines.length > maxLines) {
        return {
            text: text.slice(0, maxChars) + '\n... [truncated]',
            truncated: true
        };
    }
    return {
        text: text.slice(0, maxChars) + '... [truncated]',
        truncated: true
    };
};
export const MessageLine = memo(function MessageLine({ cols, compact, detailsMode = 'collapsed', isStreaming = false, msg, t }) {
    if (msg.kind === 'trail' && msg.tools?.length) {
        return detailsMode === 'hidden' ? null : (_jsx(Box, { flexDirection: "column", marginTop: 1, children: _jsx(ToolTrail, { detailsMode: detailsMode, t: t, trail: msg.tools }) }));
    }
    if (msg.role === 'tool') {
        const maxChars = Math.max(24, cols - 14);
        const stripped = hasAnsi(msg.text) ? stripAnsi(msg.text) : msg.text;
        const preview = compactPreview(stripped, maxChars) || '(empty tool result)';
        return (_jsx(Box, { alignSelf: "flex-start", borderColor: t.color.dim, borderStyle: "round", marginLeft: 3, paddingX: 1, children: hasAnsi(msg.text) ? (_jsx(Text, { wrap: "truncate-end", children: _jsx(Ansi, { children: msg.text }) })) : (_jsx(Text, { color: t.color.dim, wrap: "truncate-end", children: preview })) }));
    }
    const { body, glyph, prefix } = ROLE[msg.role](t);
    const thinking = msg.thinking?.trim() ?? '';
    const showDetails = detailsMode !== 'hidden' && (Boolean(msg.tools?.length) || Boolean(thinking));
    // Apply truncation for large messages to prevent memory issues
    const { text: displayText, truncated } = truncateText(msg.text, MAX_MSG_CHARS, MAX_MSG_LINES);
    const content = (() => {
        if (msg.kind === 'slash') {
            return _jsx(Text, { color: t.color.dim, children: displayText });
        }
        if (msg.role !== 'user' && hasAnsi(displayText)) {
            return _jsx(Ansi, { children: displayText });
        }
        if (msg.role === 'assistant') {
            const text = isStreaming ? displayText : displayText;
            const result = isStreaming ? _jsx(Text, { color: body, children: text }) : _jsx(Md, { compact: compact, t: t, text: text });
            return truncated ? (_jsxs(_Fragment, { children: [result, _jsx(Text, { color: t.color.dim, children: "... [message truncated to prevent memory issues]" })] })) : result;
        }
        if (msg.role === 'user' && displayText.length > LONG_MSG && isPasteBackedText(displayText)) {
            const [head, ...rest] = userDisplay(displayText).split('[long message]');
            return (_jsxs(Text, { color: body, children: [head, _jsx(Text, { color: t.color.dim, dimColor: true, children: "[long message]" }), rest.join('')] }));
        }
        return _jsx(Text, { ...(body ? { color: body } : {}), children: displayText });
    })();
    return (_jsxs(Box, { flexDirection: "column", marginBottom: msg.role === 'user' ? 1 : 0, marginTop: msg.role === 'user' || msg.kind === 'slash' ? 1 : 0, children: [showDetails && (_jsx(Box, { flexDirection: "column", marginBottom: 1, children: _jsx(ToolTrail, { detailsMode: detailsMode, reasoning: thinking, reasoningTokens: msg.thinkingTokens, t: t, toolTokens: msg.toolTokens, trail: msg.tools }) })), _jsxs(Box, { children: [_jsx(NoSelect, { flexShrink: 0, fromLeftEdge: true, width: 3, children: _jsxs(Text, { bold: msg.role === 'user', color: prefix, children: [glyph, ' '] }) }), _jsx(Box, { width: Math.max(20, cols - 5), children: content })] })] }));
});
