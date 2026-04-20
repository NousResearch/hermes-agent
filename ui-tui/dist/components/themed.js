import { jsx as _jsx } from "react/jsx-runtime";
import { Text } from '@hermes/ink';
import { useStore } from '@nanostores/react';
import { $uiState } from '../app/uiStore.js';
export function Fg({ bold, c, children, dim, italic, literal, strikethrough, underline, wrap }) {
    const { theme } = useStore($uiState);
    return (_jsx(Text, { color: literal ?? (c && theme.color[c]), dimColor: dim, bold, italic, strikethrough, underline, wrap, children: children }));
}
