"use strict";
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
var _a;
Object.defineProperty(exports, "__esModule", { value: true });
var react_1 = require("react");
var ink_1 = require("ink");
var ink_text_input_1 = require("ink-text-input");
function App() {
    var _a = (0, react_1.useState)(''), input = _a[0], setInput = _a[1];
    var _b = (0, react_1.useState)([]), messages = _b[0], setMessages = _b[1];
    var handleSubmit = function (value) {
        if (!value.trim())
            return;
        setMessages(function (prev) { return __spreadArray(__spreadArray([], prev, true), ["> ".concat(value), "[echo] ".concat(value)], false); });
        setInput('');
    };
    return (<ink_1.Box flexDirection="column" padding={1}>
      <ink_1.Box marginBottom={1}>
        <ink_1.Text bold color="yellow">hermes</ink_1.Text>
        <ink_1.Text dimColor> (ink proof-of-concept)</ink_1.Text>
      </ink_1.Box>

      <ink_1.Box flexDirection="column" marginBottom={1}>
        {messages.map(function (msg, i) { return (<ink_1.Text key={i}>{msg}</ink_1.Text>); })}
      </ink_1.Box>

      <ink_1.Box>
        <ink_1.Text bold color="cyan">{'> '}</ink_1.Text>
        <ink_text_input_1.default value={input} onChange={setInput} onSubmit={handleSubmit}/>
      </ink_1.Box>
    </ink_1.Box>);
}
var isTTY = (_a = process.stdin.isTTY) !== null && _a !== void 0 ? _a : false;
if (!isTTY) {
    console.log('hermes-tui: ink loaded, no TTY attached (run in a real terminal)');
    process.exit(0);
}
(0, ink_1.render)(<App />);
