/// <reference types="vite/client" />
import { Terminal } from '@xterm/xterm';
import '@xterm/xterm/css/xterm.css';
import { installPtyBrowserInput } from '../src/lib/pty-browser-input';

const term = new Terminal();
term.open(document.querySelector('#terminal')!);
const input = installPtyBrowserInput(term);
const frames: string[] = [];
term.onData(data => frames.push(data));
term.focus();
Object.assign(window, { terminal: term, frames, input });
