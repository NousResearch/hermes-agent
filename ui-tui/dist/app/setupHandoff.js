import { patchUiState } from './uiStore.js';
export async function runExternalSetup({ args, ctx, done, launcher, suspend }) {
    const { gateway, session, transcript } = ctx;
    transcript.sys(`launching \`hermes ${args.join(' ')}\`…`);
    patchUiState({ status: 'setup running…' });
    let result = { code: null };
    await suspend(async () => {
        result = await launcher(args);
    });
    if (result.error) {
        transcript.sys(`error launching hermes: ${result.error}`);
        patchUiState({ status: 'setup required' });
        return;
    }
    if (result.code !== 0) {
        transcript.sys(`hermes ${args[0]} exited with code ${result.code}`);
        patchUiState({ status: 'setup required' });
        return;
    }
    const setup = await gateway.rpc('setup.status', {});
    if (setup?.provider_configured === false) {
        transcript.sys('still no provider configured');
        patchUiState({ status: 'setup required' });
        return;
    }
    transcript.sys(done);
    session.newSession();
}
