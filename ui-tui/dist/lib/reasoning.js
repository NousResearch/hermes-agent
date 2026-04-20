const TAGS = ['think', 'reasoning', 'thinking', 'thought', 'REASONING_SCRATCHPAD'];
export function splitReasoning(input) {
    let text = input;
    const reasoning = [];
    for (const tag of TAGS) {
        const paired = new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>\\s*`, 'gi');
        text = text.replace(paired, (_m, inner) => {
            const trimmed = inner.trim();
            if (trimmed) {
                reasoning.push(trimmed);
            }
            return '';
        });
        const unclosed = new RegExp(`<${tag}>([\\s\\S]*)$`, 'i');
        text = text.replace(unclosed, (_m, inner) => {
            const trimmed = inner.trim();
            if (trimmed) {
                reasoning.push(trimmed);
            }
            return '';
        });
    }
    return {
        reasoning: reasoning.join('\n\n').trim(),
        text: text.trim()
    };
}
export const hasReasoningTag = (input) => {
    for (const tag of TAGS) {
        if (input.includes(`<${tag}>`)) {
            return true;
        }
    }
    return false;
};
