import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import { Box, Text } from '@hermes/ink';
import { useState } from 'react';
import { TextInput } from './textInput.js';
export function MaskedPrompt({ cols = 80, icon, label, onSubmit, sub, t }) {
    const [value, setValue] = useState('');
    return (_jsxs(Box, { flexDirection: "column", children: [_jsxs(Text, { bold: true, color: t.color.warn, children: [icon, " ", label] }), sub && _jsxs(Text, { color: t.color.dim, children: [" ", sub] }), _jsxs(Box, { children: [_jsx(Text, { color: t.color.label, children: '> ' }), _jsx(TextInput, { columns: Math.max(20, cols - 6), mask: "*", onChange: setValue, onSubmit: onSubmit, value: value })] })] }));
}
