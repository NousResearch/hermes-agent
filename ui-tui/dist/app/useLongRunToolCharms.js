import { useEffect, useRef } from 'react';
import { LONG_RUN_CHARMS } from '../content/charms.js';
import { pick, toolTrailLabel } from '../lib/text.js';
import { turnController } from './turnController.js';
const DELAY_MS = 8_000;
const INTERVAL_MS = 10_000;
const MAX_CHARMS_PER_TOOL = 2;
export function useLongRunToolCharms(busy, tools) {
    const slots = useRef(new Map());
    useEffect(() => {
        if (!busy || !tools.length) {
            slots.current.clear();
            return;
        }
        const tick = () => {
            const now = Date.now();
            const liveIds = new Set(tools.map(t => t.id));
            for (const key of [...slots.current.keys()]) {
                if (!liveIds.has(key)) {
                    slots.current.delete(key);
                }
            }
            for (const tool of tools) {
                if (!tool.startedAt || now - tool.startedAt < DELAY_MS) {
                    continue;
                }
                const slot = slots.current.get(tool.id) ?? { count: 0, lastAt: 0 };
                if (slot.count >= MAX_CHARMS_PER_TOOL || now - slot.lastAt < INTERVAL_MS) {
                    continue;
                }
                slots.current.set(tool.id, { count: slot.count + 1, lastAt: now });
                turnController.pushActivity(`${pick(LONG_RUN_CHARMS)} (${toolTrailLabel(tool.name)} · ${Math.round((now - tool.startedAt) / 1000)}s)`);
            }
        };
        tick();
        const id = setInterval(tick, 1000);
        return () => clearInterval(id);
    }, [busy, tools]);
}
