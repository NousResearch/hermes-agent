export const looksLikeSlashCommand = (text) => /^\/[^\s/]*(?:\s|$)/.test(text);
export const parseSlashCommand = (cmd) => {
    const [name = '', ...rest] = cmd.slice(1).split(/\s+/);
    return { arg: rest.join(' '), cmd, name: name.toLowerCase() };
};
