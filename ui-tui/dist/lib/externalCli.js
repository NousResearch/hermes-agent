import { spawn } from 'node:child_process';
const resolveHermesBin = () => process.env.HERMES_BIN?.trim() || 'hermes';
export const launchHermesCommand = (args) => new Promise(resolve => {
    const child = spawn(resolveHermesBin(), args, { stdio: 'inherit' });
    child.on('error', err => resolve({ code: null, error: err.message }));
    child.on('exit', code => resolve({ code }));
});
