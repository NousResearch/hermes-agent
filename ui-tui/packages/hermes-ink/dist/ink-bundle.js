var __getOwnPropNames = Object.getOwnPropertyNames;
var __esm = (fn, res) => function __init() {
  return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
};

// src/ink/devtools.ts
var devtools_exports = {};
var init_devtools = __esm({
  "src/ink/devtools.ts"() {
    "use strict";
  }
});

// src/hooks/use-stderr.ts
import { useMemo } from "react";
function useStderr() {
  return useMemo(
    () => ({
      stderr: process.stderr,
      write: (data) => process.stderr.write(data)
    }),
    []
  );
}

// src/hooks/use-stdout.ts
import { useMemo as useMemo2 } from "react";
function useStdout() {
  return useMemo2(
    () => ({
      stdout: process.stdout,
      write: (data) => process.stdout.write(data)
    }),
    []
  );
}

// src/ink/Ansi.tsx
import React from "react";
import { c as _c3 } from "react/compiler-runtime";

// src/ink/components/Link.tsx
import { c as _c2 } from "react/compiler-runtime";

// src/ink/supports-hyperlinks.ts
import supportsHyperlinksLib from "supports-hyperlinks";
var ADDITIONAL_HYPERLINK_TERMINALS = ["ghostty", "Hyper", "kitty", "alacritty", "iTerm.app", "iTerm2"];
function supportsHyperlinks(options) {
  const stdoutSupported = options?.stdoutSupported ?? supportsHyperlinksLib.stdout;
  if (stdoutSupported) {
    return true;
  }
  const env2 = options?.env ?? process.env;
  const termProgram = env2["TERM_PROGRAM"];
  if (termProgram && ADDITIONAL_HYPERLINK_TERMINALS.includes(termProgram)) {
    return true;
  }
  const lcTerminal = env2["LC_TERMINAL"];
  if (lcTerminal && ADDITIONAL_HYPERLINK_TERMINALS.includes(lcTerminal)) {
    return true;
  }
  const term = env2["TERM"];
  if (term?.includes("kitty")) {
    return true;
  }
  return false;
}

// src/ink/components/Text.tsx
import { c as _c } from "react/compiler-runtime";
import { jsx } from "react/jsx-runtime";
var memoizedStylesForWrap = {
  wrap: {
    flexGrow: 0,
    flexShrink: 1,
    flexDirection: "row",
    textWrap: "wrap"
  },
  "wrap-trim": {
    flexGrow: 0,
    flexShrink: 1,
    flexDirection: "row",
    textWrap: "wrap-trim"
  },
  end: {
    flexGrow: 0,
    flexShrink: 1,
    flexDirection: "row",
    textWrap: "end"
  },
  middle: {
    flexGrow: 0,
    flexShrink: 1,
    flexDirection: "row",
    textWrap: "middle"
  },
  "truncate-end": {
    flexGrow: 0,
    flexShrink: 1,
    flexDirection: "row",
    textWrap: "truncate-end"
  },
  truncate: {
    flexGrow: 0,
    flexShrink: 1,
    flexDirection: "row",
    textWrap: "truncate"
  },
  "truncate-middle": {
    flexGrow: 0,
    flexShrink: 1,
    flexDirection: "row",
    textWrap: "truncate-middle"
  },
  "truncate-start": {
    flexGrow: 0,
    flexShrink: 1,
    flexDirection: "row",
    textWrap: "truncate-start"
  }
};
function Text(t0) {
  const $ = _c(29);
  const {
    color,
    backgroundColor,
    bold,
    dim,
    italic: t1,
    underline: t2,
    strikethrough: t3,
    inverse: t4,
    wrap: t5,
    children
  } = t0;
  const italic = t1 === void 0 ? false : t1;
  const underline = t2 === void 0 ? false : t2;
  const strikethrough = t3 === void 0 ? false : t3;
  const inverse = t4 === void 0 ? false : t4;
  const wrap = t5 === void 0 ? "wrap" : t5;
  if (children === void 0 || children === null) {
    return null;
  }
  let t6;
  if ($[0] !== color) {
    t6 = color && {
      color
    };
    $[0] = color;
    $[1] = t6;
  } else {
    t6 = $[1];
  }
  let t7;
  if ($[2] !== backgroundColor) {
    t7 = backgroundColor && {
      backgroundColor
    };
    $[2] = backgroundColor;
    $[3] = t7;
  } else {
    t7 = $[3];
  }
  let t8;
  if ($[4] !== dim) {
    t8 = dim && {
      dim
    };
    $[4] = dim;
    $[5] = t8;
  } else {
    t8 = $[5];
  }
  let t9;
  if ($[6] !== bold) {
    t9 = bold && {
      bold
    };
    $[6] = bold;
    $[7] = t9;
  } else {
    t9 = $[7];
  }
  let t10;
  if ($[8] !== italic) {
    t10 = italic && {
      italic
    };
    $[8] = italic;
    $[9] = t10;
  } else {
    t10 = $[9];
  }
  let t11;
  if ($[10] !== underline) {
    t11 = underline && {
      underline
    };
    $[10] = underline;
    $[11] = t11;
  } else {
    t11 = $[11];
  }
  let t12;
  if ($[12] !== strikethrough) {
    t12 = strikethrough && {
      strikethrough
    };
    $[12] = strikethrough;
    $[13] = t12;
  } else {
    t12 = $[13];
  }
  let t13;
  if ($[14] !== inverse) {
    t13 = inverse && {
      inverse
    };
    $[14] = inverse;
    $[15] = t13;
  } else {
    t13 = $[15];
  }
  let t14;
  if ($[16] !== t10 || $[17] !== t11 || $[18] !== t12 || $[19] !== t13 || $[20] !== t6 || $[21] !== t7 || $[22] !== t8 || $[23] !== t9) {
    t14 = {
      ...t6,
      ...t7,
      ...t8,
      ...t9,
      ...t10,
      ...t11,
      ...t12,
      ...t13
    };
    $[16] = t10;
    $[17] = t11;
    $[18] = t12;
    $[19] = t13;
    $[20] = t6;
    $[21] = t7;
    $[22] = t8;
    $[23] = t9;
    $[24] = t14;
  } else {
    t14 = $[24];
  }
  const textStyles = t14;
  const t15 = memoizedStylesForWrap[wrap];
  let t16;
  if ($[25] !== children || $[26] !== t15 || $[27] !== textStyles) {
    t16 = /* @__PURE__ */ jsx("ink-text", { style: t15, textStyles, children });
    $[25] = children;
    $[26] = t15;
    $[27] = textStyles;
    $[28] = t16;
  } else {
    t16 = $[28];
  }
  return t16;
}

// src/ink/components/Link.tsx
import { jsx as jsx2 } from "react/jsx-runtime";
function Link(t0) {
  const $ = _c2(5);
  const { children, url, fallback } = t0;
  const content = children ?? url;
  if (supportsHyperlinks()) {
    let t12;
    if ($[0] !== content || $[1] !== url) {
      t12 = /* @__PURE__ */ jsx2(Text, { children: /* @__PURE__ */ jsx2("ink-link", { href: url, children: content }) });
      $[0] = content;
      $[1] = url;
      $[2] = t12;
    } else {
      t12 = $[2];
    }
    return t12;
  }
  const t1 = fallback ?? content;
  let t2;
  if ($[3] !== t1) {
    t2 = /* @__PURE__ */ jsx2(Text, { children: t1 });
    $[3] = t1;
    $[4] = t2;
  } else {
    t2 = $[4];
  }
  return t2;
}

// src/utils/intl.ts
var graphemeSegmenter = null;
function getGraphemeSegmenter() {
  if (!graphemeSegmenter) {
    graphemeSegmenter = new Intl.Segmenter(void 0, {
      granularity: "grapheme"
    });
  }
  return graphemeSegmenter;
}

// src/ink/termio/ansi.ts
var C0 = {
  NUL: 0,
  SOH: 1,
  STX: 2,
  ETX: 3,
  EOT: 4,
  ENQ: 5,
  ACK: 6,
  BEL: 7,
  BS: 8,
  HT: 9,
  LF: 10,
  VT: 11,
  FF: 12,
  CR: 13,
  SO: 14,
  SI: 15,
  DLE: 16,
  DC1: 17,
  DC2: 18,
  DC3: 19,
  DC4: 20,
  NAK: 21,
  SYN: 22,
  ETB: 23,
  CAN: 24,
  EM: 25,
  SUB: 26,
  ESC: 27,
  FS: 28,
  GS: 29,
  RS: 30,
  US: 31,
  DEL: 127
};
var ESC = "\x1B";
var BEL = "\x07";
var SEP = ";";
var ESC_TYPE = {
  CSI: 91,
  // [ - Control Sequence Introducer
  OSC: 93,
  // ] - Operating System Command
  DCS: 80,
  // P - Device Control String
  APC: 95,
  // _ - Application Program Command
  PM: 94,
  // ^ - Privacy Message
  SOS: 88,
  // X - Start of String
  ST: 92
  // \ - String Terminator
};
function isEscFinal(byte) {
  return byte >= 48 && byte <= 126;
}

// src/ink/termio/csi.ts
var CSI_PREFIX = ESC + String.fromCharCode(ESC_TYPE.CSI);
var CSI_RANGE = {
  PARAM_START: 48,
  PARAM_END: 63,
  INTERMEDIATE_START: 32,
  INTERMEDIATE_END: 47,
  FINAL_START: 64,
  FINAL_END: 126
};
function isCSIParam(byte) {
  return byte >= CSI_RANGE.PARAM_START && byte <= CSI_RANGE.PARAM_END;
}
function isCSIIntermediate(byte) {
  return byte >= CSI_RANGE.INTERMEDIATE_START && byte <= CSI_RANGE.INTERMEDIATE_END;
}
function isCSIFinal(byte) {
  return byte >= CSI_RANGE.FINAL_START && byte <= CSI_RANGE.FINAL_END;
}
function csi(...args) {
  if (args.length === 0) {
    return CSI_PREFIX;
  }
  if (args.length === 1) {
    return `${CSI_PREFIX}${args[0]}`;
  }
  const params = args.slice(0, -1);
  const final = args[args.length - 1];
  return `${CSI_PREFIX}${params.join(SEP)}${final}`;
}
var CSI = {
  // Cursor movement
  CUU: 65,
  // A - Cursor Up
  CUD: 66,
  // B - Cursor Down
  CUF: 67,
  // C - Cursor Forward
  CUB: 68,
  // D - Cursor Back
  CNL: 69,
  // E - Cursor Next Line
  CPL: 70,
  // F - Cursor Previous Line
  CHA: 71,
  // G - Cursor Horizontal Absolute
  CUP: 72,
  // H - Cursor Position
  CHT: 73,
  // I - Cursor Horizontal Tab
  VPA: 100,
  // d - Vertical Position Absolute
  HVP: 102,
  // f - Horizontal Vertical Position
  // Erase
  ED: 74,
  // J - Erase in Display
  EL: 75,
  // K - Erase in Line
  ECH: 88,
  // X - Erase Character
  // Insert/Delete
  IL: 76,
  // L - Insert Lines
  DL: 77,
  // M - Delete Lines
  ICH: 64,
  // @ - Insert Characters
  DCH: 80,
  // P - Delete Characters
  // Scroll
  SU: 83,
  // S - Scroll Up
  SD: 84,
  // T - Scroll Down
  // Modes
  SM: 104,
  // h - Set Mode
  RM: 108,
  // l - Reset Mode
  // SGR
  SGR: 109,
  // m - Select Graphic Rendition
  // Other
  DSR: 110,
  // n - Device Status Report
  DECSCUSR: 113,
  // q - Set Cursor Style (with space intermediate)
  DECSTBM: 114,
  // r - Set Top and Bottom Margins
  SCOSC: 115,
  // s - Save Cursor Position
  SCORC: 117,
  // u - Restore Cursor Position
  CBT: 90
  // Z - Cursor Backward Tabulation
};
var ERASE_DISPLAY = ["toEnd", "toStart", "all", "scrollback"];
var ERASE_LINE_REGION = ["toEnd", "toStart", "all"];
var CURSOR_STYLES = [
  { style: "block", blinking: true },
  // 0 - default
  { style: "block", blinking: true },
  // 1
  { style: "block", blinking: false },
  // 2
  { style: "underline", blinking: true },
  // 3
  { style: "underline", blinking: false },
  // 4
  { style: "bar", blinking: true },
  // 5
  { style: "bar", blinking: false }
  // 6
];
function cursorUp(n = 1) {
  return n === 0 ? "" : csi(n, "A");
}
function cursorDown(n = 1) {
  return n === 0 ? "" : csi(n, "B");
}
function cursorForward(n = 1) {
  return n === 0 ? "" : csi(n, "C");
}
function cursorBack(n = 1) {
  return n === 0 ? "" : csi(n, "D");
}
function cursorTo(col) {
  return csi(col, "G");
}
var CURSOR_LEFT = csi("G");
function cursorPosition(row, col) {
  return csi(row, col, "H");
}
var CURSOR_HOME = csi("H");
function cursorMove(x, y) {
  let result = "";
  if (x < 0) {
    result += cursorBack(-x);
  } else if (x > 0) {
    result += cursorForward(x);
  }
  if (y < 0) {
    result += cursorUp(-y);
  } else if (y > 0) {
    result += cursorDown(y);
  }
  return result;
}
var CURSOR_SAVE = csi("s");
var CURSOR_RESTORE = csi("u");
var ERASE_LINE = csi(2, "K");
var ERASE_SCREEN = csi(2, "J");
var ERASE_SCROLLBACK = csi(3, "J");
function eraseLines(n) {
  if (n <= 0) {
    return "";
  }
  let result = "";
  for (let i = 0; i < n; i++) {
    result += ERASE_LINE;
    if (i < n - 1) {
      result += cursorUp(1);
    }
  }
  result += CURSOR_LEFT;
  return result;
}
function scrollUp(n = 1) {
  return n === 0 ? "" : csi(n, "S");
}
function scrollDown(n = 1) {
  return n === 0 ? "" : csi(n, "T");
}
function setScrollRegion(top, bottom) {
  return csi(top, bottom, "r");
}
var RESET_SCROLL_REGION = csi("r");
var PASTE_START = csi("200~");
var PASTE_END = csi("201~");
var FOCUS_IN = csi("I");
var FOCUS_OUT = csi("O");
var ENABLE_KITTY_KEYBOARD = csi(">1u");
var DISABLE_KITTY_KEYBOARD = csi("<u");
var ENABLE_MODIFY_OTHER_KEYS = csi(">4;2m");
var DISABLE_MODIFY_OTHER_KEYS = csi(">4m");

// src/ink/termio/dec.ts
var DEC = {
  CURSOR_VISIBLE: 25,
  ALT_SCREEN: 47,
  ALT_SCREEN_CLEAR: 1049,
  MOUSE_NORMAL: 1e3,
  MOUSE_BUTTON: 1002,
  MOUSE_ANY: 1003,
  MOUSE_SGR: 1006,
  FOCUS_EVENTS: 1004,
  BRACKETED_PASTE: 2004,
  SYNCHRONIZED_UPDATE: 2026
};
function decset(mode) {
  return csi(`?${mode}h`);
}
function decreset(mode) {
  return csi(`?${mode}l`);
}
var BSU = decset(DEC.SYNCHRONIZED_UPDATE);
var ESU = decreset(DEC.SYNCHRONIZED_UPDATE);
var EBP = decset(DEC.BRACKETED_PASTE);
var DBP = decreset(DEC.BRACKETED_PASTE);
var EFE = decset(DEC.FOCUS_EVENTS);
var DFE = decreset(DEC.FOCUS_EVENTS);
var SHOW_CURSOR = decset(DEC.CURSOR_VISIBLE);
var HIDE_CURSOR = decreset(DEC.CURSOR_VISIBLE);
var ENTER_ALT_SCREEN = decset(DEC.ALT_SCREEN_CLEAR);
var EXIT_ALT_SCREEN = decreset(DEC.ALT_SCREEN_CLEAR);
var ENABLE_MOUSE_TRACKING = decset(DEC.MOUSE_NORMAL) + decset(DEC.MOUSE_BUTTON) + decset(DEC.MOUSE_ANY) + decset(DEC.MOUSE_SGR);
var DISABLE_MOUSE_TRACKING = decreset(DEC.MOUSE_SGR) + decreset(DEC.MOUSE_ANY) + decreset(DEC.MOUSE_BUTTON) + decreset(DEC.MOUSE_NORMAL);

// src/ink/termio/esc.ts
function parseEsc(chars) {
  if (chars.length === 0) {
    return null;
  }
  const first = chars[0];
  if (first === "c") {
    return { type: "reset" };
  }
  if (first === "7") {
    return { type: "cursor", action: { type: "save" } };
  }
  if (first === "8") {
    return { type: "cursor", action: { type: "restore" } };
  }
  if (first === "D") {
    return {
      type: "cursor",
      action: { type: "move", direction: "down", count: 1 }
    };
  }
  if (first === "M") {
    return {
      type: "cursor",
      action: { type: "move", direction: "up", count: 1 }
    };
  }
  if (first === "E") {
    return { type: "cursor", action: { type: "nextLine", count: 1 } };
  }
  if (first === "H") {
    return null;
  }
  if ("()".includes(first) && chars.length >= 2) {
    return null;
  }
  return { type: "unknown", sequence: `\x1B${chars}` };
}

// src/ink/termio/osc.ts
import { Buffer as Buffer2 } from "buffer";

// src/utils/env.ts
function detectTerminal() {
  if (process.env.CURSOR_TRACE_ID) {
    return "cursor";
  }
  if (process.env.TERM === "xterm-ghostty") {
    return "ghostty";
  }
  if (process.env.TERM?.includes("kitty")) {
    return "kitty";
  }
  if (process.env.TERM_PROGRAM) {
    return process.env.TERM_PROGRAM;
  }
  if (process.env.TMUX) {
    return "tmux";
  }
  if (process.env.STY) {
    return "screen";
  }
  if (process.env.KITTY_WINDOW_ID) {
    return "kitty";
  }
  if (process.env.WT_SESSION) {
    return "windows-terminal";
  }
  return process.env.TERM ?? null;
}
var env = {
  terminal: detectTerminal()
};

// src/utils/execFileNoThrow.ts
import { spawn } from "child_process";
function execFileNoThrow(file, args, options = {}) {
  return new Promise((resolve) => {
    const child = spawn(file, args, {
      cwd: options.useCwd ? process.cwd() : void 0,
      env: options.env,
      stdio: "pipe"
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = options.timeout ? setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, options.timeout) : null;
    child.stdout?.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr?.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", (error) => {
      if (timer) {
        clearTimeout(timer);
      }
      resolve({ stdout, stderr, code: 1, error: String(error) });
    });
    child.on("close", (code) => {
      if (timer) {
        clearTimeout(timer);
      }
      resolve({ stdout, stderr, code: timedOut ? 124 : code ?? 0 });
    });
    if (options.input) {
      child.stdin?.write(options.input);
    }
    child.stdin?.end();
  });
}

// src/ink/termio/osc.ts
var OSC_PREFIX = ESC + String.fromCharCode(ESC_TYPE.OSC);
var ST = ESC + "\\";
function osc(...parts) {
  const terminator = env.terminal === "kitty" ? ST : BEL;
  return `${OSC_PREFIX}${parts.join(SEP)}${terminator}`;
}
function wrapForMultiplexer(sequence) {
  if (process.env["TMUX"]) {
    const escaped = sequence.replaceAll("\x1B", "\x1B\x1B");
    return `\x1BPtmux;${escaped}\x1B\\`;
  }
  if (process.env["STY"]) {
    return `\x1BP${sequence}\x1B\\`;
  }
  return sequence;
}
function tmuxPassthrough(payload) {
  return `${ESC}Ptmux;${payload.replaceAll(ESC, ESC + ESC)}${ST}`;
}
async function tmuxLoadBuffer(text) {
  if (!process.env["TMUX"]) {
    return false;
  }
  const args = process.env["LC_TERMINAL"] === "iTerm2" ? ["load-buffer", "-"] : ["load-buffer", "-w", "-"];
  const { code } = await execFileNoThrow("tmux", args, {
    input: text,
    useCwd: false,
    timeout: 2e3
  });
  return code === 0;
}
async function setClipboard(text) {
  const b64 = Buffer2.from(text, "utf8").toString("base64");
  const raw = osc(OSC.CLIPBOARD, "c", b64);
  if (!process.env["SSH_CONNECTION"]) {
    copyNative(text);
  }
  const tmuxBufferLoaded = await tmuxLoadBuffer(text);
  if (tmuxBufferLoaded) {
    return tmuxPassthrough(`${ESC}]52;c;${b64}${BEL}`);
  }
  return raw;
}
var linuxCopy;
function copyNative(text) {
  const opts = { input: text, useCwd: false, timeout: 2e3 };
  switch (process.platform) {
    case "darwin":
      void execFileNoThrow("pbcopy", [], opts);
      return;
    case "linux": {
      if (linuxCopy === null) {
        return;
      }
      if (linuxCopy === "wl-copy") {
        void execFileNoThrow("wl-copy", [], opts);
        return;
      }
      if (linuxCopy === "xclip") {
        void execFileNoThrow("xclip", ["-selection", "clipboard"], opts);
        return;
      }
      if (linuxCopy === "xsel") {
        void execFileNoThrow("xsel", ["--clipboard", "--input"], opts);
        return;
      }
      void execFileNoThrow("wl-copy", [], opts).then((r) => {
        if (r.code === 0) {
          linuxCopy = "wl-copy";
          return;
        }
        void execFileNoThrow("xclip", ["-selection", "clipboard"], opts).then((r2) => {
          if (r2.code === 0) {
            linuxCopy = "xclip";
            return;
          }
          void execFileNoThrow("xsel", ["--clipboard", "--input"], opts).then((r3) => {
            linuxCopy = r3.code === 0 ? "xsel" : null;
          });
        });
      });
      return;
    }
    case "win32":
      void execFileNoThrow("clip", [], opts);
      return;
  }
}
var OSC = {
  SET_TITLE_AND_ICON: 0,
  SET_ICON: 1,
  SET_TITLE: 2,
  SET_COLOR: 4,
  SET_CWD: 7,
  HYPERLINK: 8,
  ITERM2: 9,
  // iTerm2 proprietary sequences
  SET_FG_COLOR: 10,
  SET_BG_COLOR: 11,
  SET_CURSOR_COLOR: 12,
  CLIPBOARD: 52,
  KITTY: 99,
  // Kitty notification protocol
  RESET_COLOR: 104,
  RESET_FG_COLOR: 110,
  RESET_BG_COLOR: 111,
  RESET_CURSOR_COLOR: 112,
  SEMANTIC_PROMPT: 133,
  GHOSTTY: 777,
  // Ghostty notification protocol
  TAB_STATUS: 21337
  // Tab status extension
};
function parseOSC(content) {
  const semicolonIdx = content.indexOf(";");
  const command = semicolonIdx >= 0 ? content.slice(0, semicolonIdx) : content;
  const data = semicolonIdx >= 0 ? content.slice(semicolonIdx + 1) : "";
  const commandNum = parseInt(command, 10);
  if (commandNum === OSC.SET_TITLE_AND_ICON) {
    return { type: "title", action: { type: "both", title: data } };
  }
  if (commandNum === OSC.SET_ICON) {
    return { type: "title", action: { type: "iconName", name: data } };
  }
  if (commandNum === OSC.SET_TITLE) {
    return { type: "title", action: { type: "windowTitle", title: data } };
  }
  if (commandNum === OSC.HYPERLINK) {
    const parts = data.split(";");
    const paramsStr = parts[0] ?? "";
    const url = parts.slice(1).join(";");
    if (url === "") {
      return { type: "link", action: { type: "end" } };
    }
    const params = {};
    if (paramsStr) {
      for (const pair of paramsStr.split(":")) {
        const eqIdx = pair.indexOf("=");
        if (eqIdx >= 0) {
          params[pair.slice(0, eqIdx)] = pair.slice(eqIdx + 1);
        }
      }
    }
    return {
      type: "link",
      action: {
        type: "start",
        url,
        params: Object.keys(params).length > 0 ? params : void 0
      }
    };
  }
  if (commandNum === OSC.TAB_STATUS) {
    return { type: "tabStatus", action: parseTabStatus(data) };
  }
  return { type: "unknown", sequence: `\x1B]${content}` };
}
function parseOscColor(spec) {
  const hex = spec.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
  if (hex) {
    return {
      type: "rgb",
      r: parseInt(hex[1], 16),
      g: parseInt(hex[2], 16),
      b: parseInt(hex[3], 16)
    };
  }
  const rgb2 = spec.match(/^rgb:([0-9a-f]{1,4})\/([0-9a-f]{1,4})\/([0-9a-f]{1,4})$/i);
  if (rgb2) {
    const scale = (s) => Math.round(parseInt(s, 16) / (16 ** s.length - 1) * 255);
    return {
      type: "rgb",
      r: scale(rgb2[1]),
      g: scale(rgb2[2]),
      b: scale(rgb2[3])
    };
  }
  return null;
}
function parseTabStatus(data) {
  const action = {};
  for (const [key, value] of splitTabStatusPairs(data)) {
    switch (key) {
      case "indicator":
        action.indicator = value === "" ? null : parseOscColor(value);
        break;
      case "status":
        action.status = value === "" ? null : value;
        break;
      case "status-color":
        action.statusColor = value === "" ? null : parseOscColor(value);
        break;
    }
  }
  return action;
}
function* splitTabStatusPairs(data) {
  let key = "";
  let val = "";
  let inVal = false;
  let esc = false;
  for (const c of data) {
    if (esc) {
      if (inVal) {
        val += c;
      } else {
        key += c;
      }
      esc = false;
    } else if (c === "\\") {
      esc = true;
    } else if (c === ";") {
      yield [key, val];
      key = "";
      val = "";
      inVal = false;
    } else if (c === "=" && !inVal) {
      inVal = true;
    } else if (inVal) {
      val += c;
    } else {
      key += c;
    }
  }
  if (key || inVal) {
    yield [key, val];
  }
}
function link(url, params) {
  if (!url) {
    return LINK_END;
  }
  const p = { id: osc8Id(url), ...params };
  const paramStr = Object.entries(p).map(([k, v]) => `${k}=${v}`).join(":");
  return osc(OSC.HYPERLINK, paramStr, url);
}
function osc8Id(url) {
  let h = 0;
  for (let i = 0; i < url.length; i++) {
    h = (h << 5) - h + url.charCodeAt(i) | 0;
  }
  return (h >>> 0).toString(36);
}
var LINK_END = osc(OSC.HYPERLINK, "", "");
var ITERM2 = {
  NOTIFY: 0,
  BADGE: 2,
  PROGRESS: 4
};
var PROGRESS = {
  CLEAR: 0,
  SET: 1,
  ERROR: 2,
  INDETERMINATE: 3
};
var CLEAR_ITERM2_PROGRESS = `${OSC_PREFIX}${OSC.ITERM2};${ITERM2.PROGRESS};${PROGRESS.CLEAR};${BEL}`;
var CLEAR_TERMINAL_TITLE = `${OSC_PREFIX}${OSC.SET_TITLE_AND_ICON};${BEL}`;
var CLEAR_TAB_STATUS = osc(OSC.TAB_STATUS, "indicator=;status=;status-color=");
function supportsTabStatus() {
  return process.env.USER_TYPE === "ant";
}
function tabStatus(fields) {
  const parts = [];
  const rgb2 = (c) => c.type === "rgb" ? `#${[c.r, c.g, c.b].map((n) => n.toString(16).padStart(2, "0")).join("")}` : "";
  if ("indicator" in fields) {
    parts.push(`indicator=${fields.indicator ? rgb2(fields.indicator) : ""}`);
  }
  if ("status" in fields) {
    parts.push(`status=${fields.status?.replaceAll("\\", "\\\\").replaceAll(";", "\\;") ?? ""}`);
  }
  if ("statusColor" in fields) {
    parts.push(`status-color=${fields.statusColor ? rgb2(fields.statusColor) : ""}`);
  }
  return osc(OSC.TAB_STATUS, parts.join(";"));
}

// src/ink/termio/types.ts
function defaultStyle() {
  return {
    bold: false,
    dim: false,
    italic: false,
    underline: "none",
    blink: false,
    inverse: false,
    hidden: false,
    strikethrough: false,
    overline: false,
    fg: { type: "default" },
    bg: { type: "default" },
    underlineColor: { type: "default" }
  };
}

// src/ink/termio/sgr.ts
var NAMED_COLORS = [
  "black",
  "red",
  "green",
  "yellow",
  "blue",
  "magenta",
  "cyan",
  "white",
  "brightBlack",
  "brightRed",
  "brightGreen",
  "brightYellow",
  "brightBlue",
  "brightMagenta",
  "brightCyan",
  "brightWhite"
];
var UNDERLINE_STYLES = ["none", "single", "double", "curly", "dotted", "dashed"];
function parseParams(str) {
  if (str === "") {
    return [{ value: 0, subparams: [], colon: false }];
  }
  const result = [];
  let current = { value: null, subparams: [], colon: false };
  let num = "";
  let inSub = false;
  for (let i = 0; i <= str.length; i++) {
    const c = str[i];
    if (c === ";" || c === void 0) {
      const n = num === "" ? null : parseInt(num, 10);
      if (inSub) {
        if (n !== null) {
          current.subparams.push(n);
        }
      } else {
        current.value = n;
      }
      result.push(current);
      current = { value: null, subparams: [], colon: false };
      num = "";
      inSub = false;
    } else if (c === ":") {
      const n = num === "" ? null : parseInt(num, 10);
      if (!inSub) {
        current.value = n;
        current.colon = true;
        inSub = true;
      } else {
        if (n !== null) {
          current.subparams.push(n);
        }
      }
      num = "";
    } else if (c >= "0" && c <= "9") {
      num += c;
    }
  }
  return result;
}
function parseExtendedColor(params, idx) {
  const p = params[idx];
  if (!p) {
    return null;
  }
  if (p.colon && p.subparams.length >= 1) {
    if (p.subparams[0] === 5 && p.subparams.length >= 2) {
      return { index: p.subparams[1] };
    }
    if (p.subparams[0] === 2 && p.subparams.length >= 4) {
      const off = p.subparams.length >= 5 ? 1 : 0;
      return {
        r: p.subparams[1 + off],
        g: p.subparams[2 + off],
        b: p.subparams[3 + off]
      };
    }
  }
  const next = params[idx + 1];
  if (!next) {
    return null;
  }
  if (next.value === 5 && params[idx + 2]?.value !== null && params[idx + 2]?.value !== void 0) {
    return { index: params[idx + 2].value };
  }
  if (next.value === 2) {
    const r = params[idx + 2]?.value;
    const g = params[idx + 3]?.value;
    const b = params[idx + 4]?.value;
    if (r !== null && r !== void 0 && g !== null && g !== void 0 && b !== null && b !== void 0) {
      return { r, g, b };
    }
  }
  return null;
}
function applySGR(paramStr, style) {
  const params = parseParams(paramStr);
  let s = { ...style };
  let i = 0;
  while (i < params.length) {
    const p = params[i];
    const code = p.value ?? 0;
    if (code === 0) {
      s = defaultStyle();
      i++;
      continue;
    }
    if (code === 1) {
      s.bold = true;
      i++;
      continue;
    }
    if (code === 2) {
      s.dim = true;
      i++;
      continue;
    }
    if (code === 3) {
      s.italic = true;
      i++;
      continue;
    }
    if (code === 4) {
      s.underline = p.colon ? UNDERLINE_STYLES[p.subparams[0]] ?? "single" : "single";
      i++;
      continue;
    }
    if (code === 5 || code === 6) {
      s.blink = true;
      i++;
      continue;
    }
    if (code === 7) {
      s.inverse = true;
      i++;
      continue;
    }
    if (code === 8) {
      s.hidden = true;
      i++;
      continue;
    }
    if (code === 9) {
      s.strikethrough = true;
      i++;
      continue;
    }
    if (code === 21) {
      s.underline = "double";
      i++;
      continue;
    }
    if (code === 22) {
      s.bold = false;
      s.dim = false;
      i++;
      continue;
    }
    if (code === 23) {
      s.italic = false;
      i++;
      continue;
    }
    if (code === 24) {
      s.underline = "none";
      i++;
      continue;
    }
    if (code === 25) {
      s.blink = false;
      i++;
      continue;
    }
    if (code === 27) {
      s.inverse = false;
      i++;
      continue;
    }
    if (code === 28) {
      s.hidden = false;
      i++;
      continue;
    }
    if (code === 29) {
      s.strikethrough = false;
      i++;
      continue;
    }
    if (code === 53) {
      s.overline = true;
      i++;
      continue;
    }
    if (code === 55) {
      s.overline = false;
      i++;
      continue;
    }
    if (code >= 30 && code <= 37) {
      s.fg = { type: "named", name: NAMED_COLORS[code - 30] };
      i++;
      continue;
    }
    if (code === 39) {
      s.fg = { type: "default" };
      i++;
      continue;
    }
    if (code >= 40 && code <= 47) {
      s.bg = { type: "named", name: NAMED_COLORS[code - 40] };
      i++;
      continue;
    }
    if (code === 49) {
      s.bg = { type: "default" };
      i++;
      continue;
    }
    if (code >= 90 && code <= 97) {
      s.fg = { type: "named", name: NAMED_COLORS[code - 90 + 8] };
      i++;
      continue;
    }
    if (code >= 100 && code <= 107) {
      s.bg = { type: "named", name: NAMED_COLORS[code - 100 + 8] };
      i++;
      continue;
    }
    if (code === 38) {
      const c = parseExtendedColor(params, i);
      if (c) {
        s.fg = "index" in c ? { type: "indexed", index: c.index } : { type: "rgb", ...c };
        i += p.colon ? 1 : "index" in c ? 3 : 5;
        continue;
      }
    }
    if (code === 48) {
      const c = parseExtendedColor(params, i);
      if (c) {
        s.bg = "index" in c ? { type: "indexed", index: c.index } : { type: "rgb", ...c };
        i += p.colon ? 1 : "index" in c ? 3 : 5;
        continue;
      }
    }
    if (code === 58) {
      const c = parseExtendedColor(params, i);
      if (c) {
        s.underlineColor = "index" in c ? { type: "indexed", index: c.index } : { type: "rgb", ...c };
        i += p.colon ? 1 : "index" in c ? 3 : 5;
        continue;
      }
    }
    if (code === 59) {
      s.underlineColor = { type: "default" };
      i++;
      continue;
    }
    i++;
  }
  return s;
}

// src/ink/termio/tokenize.ts
function createTokenizer(options) {
  let currentState = "ground";
  let currentBuffer = "";
  const x10Mouse = options?.x10Mouse ?? false;
  return {
    feed(input) {
      const result = tokenize(input, currentState, currentBuffer, false, x10Mouse);
      currentState = result.state.state;
      currentBuffer = result.state.buffer;
      return result.tokens;
    },
    flush() {
      const result = tokenize("", currentState, currentBuffer, true, x10Mouse);
      currentState = result.state.state;
      currentBuffer = result.state.buffer;
      return result.tokens;
    },
    reset() {
      currentState = "ground";
      currentBuffer = "";
    },
    buffer() {
      return currentBuffer;
    }
  };
}
function tokenize(input, initialState, initialBuffer, flush, x10Mouse) {
  const tokens = [];
  const result = {
    state: initialState,
    buffer: ""
  };
  const data = initialBuffer + input;
  let i = 0;
  let textStart = 0;
  let seqStart = 0;
  const flushText = () => {
    if (i > textStart) {
      const text = data.slice(textStart, i);
      if (text) {
        tokens.push({ type: "text", value: text });
      }
    }
    textStart = i;
  };
  const emitSequence = (seq) => {
    if (seq) {
      tokens.push({ type: "sequence", value: seq });
    }
    result.state = "ground";
    textStart = i;
  };
  while (i < data.length) {
    const code = data.charCodeAt(i);
    switch (result.state) {
      case "ground":
        if (code === C0.ESC) {
          flushText();
          seqStart = i;
          result.state = "escape";
          i++;
        } else {
          i++;
        }
        break;
      case "escape":
        if (code === ESC_TYPE.CSI) {
          result.state = "csi";
          i++;
        } else if (code === ESC_TYPE.OSC) {
          result.state = "osc";
          i++;
        } else if (code === ESC_TYPE.DCS) {
          result.state = "dcs";
          i++;
        } else if (code === ESC_TYPE.APC) {
          result.state = "apc";
          i++;
        } else if (code === 79) {
          result.state = "ss3";
          i++;
        } else if (isCSIIntermediate(code)) {
          result.state = "escapeIntermediate";
          i++;
        } else if (isEscFinal(code)) {
          i++;
          emitSequence(data.slice(seqStart, i));
        } else if (code === C0.ESC) {
          emitSequence(data.slice(seqStart, i));
          seqStart = i;
          result.state = "escape";
          i++;
        } else {
          result.state = "ground";
          textStart = seqStart;
        }
        break;
      case "escapeIntermediate":
        if (isCSIIntermediate(code)) {
          i++;
        } else if (isEscFinal(code)) {
          i++;
          emitSequence(data.slice(seqStart, i));
        } else {
          result.state = "ground";
          textStart = seqStart;
        }
        break;
      case "csi":
        if (x10Mouse && code === 77 && i - seqStart === 2 && (i + 1 >= data.length || data.charCodeAt(i + 1) >= 32) && (i + 2 >= data.length || data.charCodeAt(i + 2) >= 32) && (i + 3 >= data.length || data.charCodeAt(i + 3) >= 32)) {
          if (i + 4 <= data.length) {
            i += 4;
            emitSequence(data.slice(seqStart, i));
          } else {
            i = data.length;
          }
          break;
        }
        if (isCSIFinal(code)) {
          i++;
          emitSequence(data.slice(seqStart, i));
        } else if (isCSIParam(code) || isCSIIntermediate(code)) {
          i++;
        } else {
          result.state = "ground";
          textStart = seqStart;
        }
        break;
      case "ss3":
        if (code >= 64 && code <= 126) {
          i++;
          emitSequence(data.slice(seqStart, i));
        } else {
          result.state = "ground";
          textStart = seqStart;
        }
        break;
      case "osc":
        if (code === C0.BEL) {
          i++;
          emitSequence(data.slice(seqStart, i));
        } else if (code === C0.ESC && i + 1 < data.length && data.charCodeAt(i + 1) === ESC_TYPE.ST) {
          i += 2;
          emitSequence(data.slice(seqStart, i));
        } else {
          i++;
        }
        break;
      case "dcs":
      case "apc":
        if (code === C0.BEL) {
          i++;
          emitSequence(data.slice(seqStart, i));
        } else if (code === C0.ESC && i + 1 < data.length && data.charCodeAt(i + 1) === ESC_TYPE.ST) {
          i += 2;
          emitSequence(data.slice(seqStart, i));
        } else {
          i++;
        }
        break;
    }
  }
  if (result.state === "ground") {
    flushText();
  } else if (flush) {
    const remaining = data.slice(seqStart);
    if (remaining) {
      tokens.push({ type: "sequence", value: remaining });
    }
    result.state = "ground";
  } else {
    result.buffer = data.slice(seqStart);
  }
  return { tokens, state: result };
}

// src/ink/termio/parser.ts
function isEmoji(codePoint) {
  return codePoint >= 9728 && codePoint <= 9983 || codePoint >= 9984 && codePoint <= 10175 || codePoint >= 127744 && codePoint <= 129535 || codePoint >= 129536 && codePoint <= 129791 || codePoint >= 127456 && codePoint <= 127487;
}
function isEastAsianWide(codePoint) {
  return codePoint >= 4352 && codePoint <= 4447 || codePoint >= 11904 && codePoint <= 40959 || codePoint >= 44032 && codePoint <= 55203 || codePoint >= 63744 && codePoint <= 64255 || codePoint >= 65040 && codePoint <= 65055 || codePoint >= 65072 && codePoint <= 65135 || codePoint >= 65280 && codePoint <= 65376 || codePoint >= 65504 && codePoint <= 65510 || codePoint >= 131072 && codePoint <= 196605 || codePoint >= 196608 && codePoint <= 262141;
}
function hasMultipleCodepoints(str) {
  let count = 0;
  for (const _ of str) {
    count++;
    if (count > 1) {
      return true;
    }
  }
  return false;
}
function graphemeWidth(grapheme) {
  if (hasMultipleCodepoints(grapheme)) {
    return 2;
  }
  const codePoint = grapheme.codePointAt(0);
  if (codePoint === void 0) {
    return 1;
  }
  if (isEmoji(codePoint) || isEastAsianWide(codePoint)) {
    return 2;
  }
  return 1;
}
function* segmentGraphemes(str) {
  for (const { segment } of getGraphemeSegmenter().segment(str)) {
    yield { value: segment, width: graphemeWidth(segment) };
  }
}
function parseCSIParams(paramStr) {
  if (paramStr === "") {
    return [];
  }
  return paramStr.split(/[;:]/).map((s) => s === "" ? 0 : parseInt(s, 10));
}
function parseCSI(rawSequence) {
  const inner = rawSequence.slice(2);
  if (inner.length === 0) {
    return null;
  }
  const finalByte = inner.charCodeAt(inner.length - 1);
  const beforeFinal = inner.slice(0, -1);
  let privateMode = "";
  let paramStr = beforeFinal;
  let intermediate = "";
  if (beforeFinal.length > 0 && "?>=".includes(beforeFinal[0])) {
    privateMode = beforeFinal[0];
    paramStr = beforeFinal.slice(1);
  }
  const intermediateMatch = paramStr.match(/([^0-9;:]+)$/);
  if (intermediateMatch) {
    intermediate = intermediateMatch[1];
    paramStr = paramStr.slice(0, -intermediate.length);
  }
  const params = parseCSIParams(paramStr);
  const p0 = params[0] ?? 1;
  const p1 = params[1] ?? 1;
  if (finalByte === CSI.SGR && privateMode === "") {
    return { type: "sgr", params: paramStr };
  }
  if (finalByte === CSI.CUU) {
    return {
      type: "cursor",
      action: { type: "move", direction: "up", count: p0 }
    };
  }
  if (finalByte === CSI.CUD) {
    return {
      type: "cursor",
      action: { type: "move", direction: "down", count: p0 }
    };
  }
  if (finalByte === CSI.CUF) {
    return {
      type: "cursor",
      action: { type: "move", direction: "forward", count: p0 }
    };
  }
  if (finalByte === CSI.CUB) {
    return {
      type: "cursor",
      action: { type: "move", direction: "back", count: p0 }
    };
  }
  if (finalByte === CSI.CNL) {
    return { type: "cursor", action: { type: "nextLine", count: p0 } };
  }
  if (finalByte === CSI.CPL) {
    return { type: "cursor", action: { type: "prevLine", count: p0 } };
  }
  if (finalByte === CSI.CHA) {
    return { type: "cursor", action: { type: "column", col: p0 } };
  }
  if (finalByte === CSI.CUP || finalByte === CSI.HVP) {
    return { type: "cursor", action: { type: "position", row: p0, col: p1 } };
  }
  if (finalByte === CSI.VPA) {
    return { type: "cursor", action: { type: "row", row: p0 } };
  }
  if (finalByte === CSI.ED) {
    const region = ERASE_DISPLAY[params[0] ?? 0] ?? "toEnd";
    return { type: "erase", action: { type: "display", region } };
  }
  if (finalByte === CSI.EL) {
    const region = ERASE_LINE_REGION[params[0] ?? 0] ?? "toEnd";
    return { type: "erase", action: { type: "line", region } };
  }
  if (finalByte === CSI.ECH) {
    return { type: "erase", action: { type: "chars", count: p0 } };
  }
  if (finalByte === CSI.SU) {
    return { type: "scroll", action: { type: "up", count: p0 } };
  }
  if (finalByte === CSI.SD) {
    return { type: "scroll", action: { type: "down", count: p0 } };
  }
  if (finalByte === CSI.DECSTBM) {
    return {
      type: "scroll",
      action: { type: "setRegion", top: p0, bottom: p1 }
    };
  }
  if (finalByte === CSI.SCOSC) {
    return { type: "cursor", action: { type: "save" } };
  }
  if (finalByte === CSI.SCORC) {
    return { type: "cursor", action: { type: "restore" } };
  }
  if (finalByte === CSI.DECSCUSR && intermediate === " ") {
    const styleInfo = CURSOR_STYLES[p0] ?? CURSOR_STYLES[0];
    return { type: "cursor", action: { type: "style", ...styleInfo } };
  }
  if (privateMode === "?" && (finalByte === CSI.SM || finalByte === CSI.RM)) {
    const enabled = finalByte === CSI.SM;
    if (p0 === DEC.CURSOR_VISIBLE) {
      return {
        type: "cursor",
        action: enabled ? { type: "show" } : { type: "hide" }
      };
    }
    if (p0 === DEC.ALT_SCREEN_CLEAR || p0 === DEC.ALT_SCREEN) {
      return { type: "mode", action: { type: "alternateScreen", enabled } };
    }
    if (p0 === DEC.BRACKETED_PASTE) {
      return { type: "mode", action: { type: "bracketedPaste", enabled } };
    }
    if (p0 === DEC.MOUSE_NORMAL) {
      return {
        type: "mode",
        action: { type: "mouseTracking", mode: enabled ? "normal" : "off" }
      };
    }
    if (p0 === DEC.MOUSE_BUTTON) {
      return {
        type: "mode",
        action: { type: "mouseTracking", mode: enabled ? "button" : "off" }
      };
    }
    if (p0 === DEC.MOUSE_ANY) {
      return {
        type: "mode",
        action: { type: "mouseTracking", mode: enabled ? "any" : "off" }
      };
    }
    if (p0 === DEC.FOCUS_EVENTS) {
      return { type: "mode", action: { type: "focusEvents", enabled } };
    }
  }
  return { type: "unknown", sequence: rawSequence };
}
function identifySequence(seq) {
  if (seq.length < 2) {
    return "unknown";
  }
  if (seq.charCodeAt(0) !== C0.ESC) {
    return "unknown";
  }
  const second = seq.charCodeAt(1);
  if (second === 91) {
    return "csi";
  }
  if (second === 93) {
    return "osc";
  }
  if (second === 79) {
    return "ss3";
  }
  return "esc";
}
var Parser = class {
  tokenizer = createTokenizer();
  style = defaultStyle();
  inLink = false;
  linkUrl;
  reset() {
    this.tokenizer.reset();
    this.style = defaultStyle();
    this.inLink = false;
    this.linkUrl = void 0;
  }
  /** Feed input and get resulting actions */
  feed(input) {
    const tokens = this.tokenizer.feed(input);
    const actions = [];
    for (const token of tokens) {
      const tokenActions = this.processToken(token);
      actions.push(...tokenActions);
    }
    return actions;
  }
  processToken(token) {
    switch (token.type) {
      case "text":
        return this.processText(token.value);
      case "sequence":
        return this.processSequence(token.value);
    }
  }
  processText(text) {
    const actions = [];
    let current = "";
    for (const char of text) {
      if (char.charCodeAt(0) === C0.BEL) {
        if (current) {
          const graphemes = [...segmentGraphemes(current)];
          if (graphemes.length > 0) {
            actions.push({ type: "text", graphemes, style: { ...this.style } });
          }
          current = "";
        }
        actions.push({ type: "bell" });
      } else {
        current += char;
      }
    }
    if (current) {
      const graphemes = [...segmentGraphemes(current)];
      if (graphemes.length > 0) {
        actions.push({ type: "text", graphemes, style: { ...this.style } });
      }
    }
    return actions;
  }
  processSequence(seq) {
    const seqType = identifySequence(seq);
    switch (seqType) {
      case "csi": {
        const action = parseCSI(seq);
        if (!action) {
          return [];
        }
        if (action.type === "sgr") {
          this.style = applySGR(action.params, this.style);
          return [];
        }
        return [action];
      }
      case "osc": {
        let content = seq.slice(2);
        if (content.endsWith("\x07")) {
          content = content.slice(0, -1);
        } else if (content.endsWith("\x1B\\")) {
          content = content.slice(0, -2);
        }
        const action = parseOSC(content);
        if (action) {
          if (action.type === "link") {
            if (action.action.type === "start") {
              this.inLink = true;
              this.linkUrl = action.action.url;
            } else {
              this.inLink = false;
              this.linkUrl = void 0;
            }
          }
          return [action];
        }
        return [];
      }
      case "esc": {
        const escContent = seq.slice(1);
        const action = parseEsc(escContent);
        return action ? [action] : [];
      }
      case "ss3":
        return [{ type: "unknown", sequence: seq }];
      default:
        return [{ type: "unknown", sequence: seq }];
    }
  }
};

// src/ink/Ansi.tsx
import { jsx as jsx3 } from "react/jsx-runtime";
var Ansi = React.memo(function Ansi2(t0) {
  const $ = _c3(12);
  const { children, dimColor } = t0;
  if (typeof children !== "string") {
    let t12;
    if ($[0] !== children || $[1] !== dimColor) {
      t12 = dimColor ? /* @__PURE__ */ jsx3(Text, { dim: true, children: String(children) }) : /* @__PURE__ */ jsx3(Text, { children: String(children) });
      $[0] = children;
      $[1] = dimColor;
      $[2] = t12;
    } else {
      t12 = $[2];
    }
    return t12;
  }
  if (children === "") {
    return null;
  }
  let t1;
  let t2;
  if ($[3] !== children || $[4] !== dimColor) {
    t2 = Symbol.for("react.early_return_sentinel");
    bb0: {
      const spans = parseToSpans(children);
      if (spans.length === 0) {
        t2 = null;
        break bb0;
      }
      if (spans.length === 1 && !hasAnyProps(spans[0].props)) {
        t2 = dimColor ? /* @__PURE__ */ jsx3(Text, { dim: true, children: spans[0].text }) : /* @__PURE__ */ jsx3(Text, { children: spans[0].text });
        break bb0;
      }
      let t32;
      if ($[7] !== dimColor) {
        t32 = (span, i) => {
          const hyperlink = span.props.hyperlink;
          if (dimColor) {
            span.props.dim = true;
          }
          const hasTextProps = hasAnyTextProps(span.props);
          if (hyperlink) {
            return hasTextProps ? /* @__PURE__ */ jsx3(Link, { url: hyperlink, children: /* @__PURE__ */ jsx3(
              StyledText,
              {
                backgroundColor: span.props.backgroundColor,
                bold: span.props.bold,
                color: span.props.color,
                dim: span.props.dim,
                inverse: span.props.inverse,
                italic: span.props.italic,
                strikethrough: span.props.strikethrough,
                underline: span.props.underline,
                children: span.text
              }
            ) }, i) : /* @__PURE__ */ jsx3(Link, { url: hyperlink, children: span.text }, i);
          }
          return hasTextProps ? /* @__PURE__ */ jsx3(
            StyledText,
            {
              backgroundColor: span.props.backgroundColor,
              bold: span.props.bold,
              color: span.props.color,
              dim: span.props.dim,
              inverse: span.props.inverse,
              italic: span.props.italic,
              strikethrough: span.props.strikethrough,
              underline: span.props.underline,
              children: span.text
            },
            i
          ) : span.text;
        };
        $[7] = dimColor;
        $[8] = t32;
      } else {
        t32 = $[8];
      }
      t1 = spans.map(t32);
    }
    $[3] = children;
    $[4] = dimColor;
    $[5] = t1;
    $[6] = t2;
  } else {
    t1 = $[5];
    t2 = $[6];
  }
  if (t2 !== Symbol.for("react.early_return_sentinel")) {
    return t2;
  }
  const content = t1;
  let t3;
  if ($[9] !== content || $[10] !== dimColor) {
    t3 = dimColor ? /* @__PURE__ */ jsx3(Text, { dim: true, children: content }) : /* @__PURE__ */ jsx3(Text, { children: content });
    $[9] = content;
    $[10] = dimColor;
    $[11] = t3;
  } else {
    t3 = $[11];
  }
  return t3;
});
function parseToSpans(input) {
  const parser = new Parser();
  const actions = parser.feed(input);
  const spans = [];
  let currentHyperlink;
  for (const action of actions) {
    if (action.type === "link") {
      if (action.action.type === "start") {
        currentHyperlink = action.action.url;
      } else {
        currentHyperlink = void 0;
      }
      continue;
    }
    if (action.type === "text") {
      const text = action.graphemes.map((g) => g.value).join("");
      if (!text) {
        continue;
      }
      const props = textStyleToSpanProps(action.style);
      if (currentHyperlink) {
        props.hyperlink = currentHyperlink;
      }
      const lastSpan = spans[spans.length - 1];
      if (lastSpan && propsEqual(lastSpan.props, props)) {
        lastSpan.text += text;
      } else {
        spans.push({
          text,
          props
        });
      }
    }
  }
  return spans;
}
function textStyleToSpanProps(style) {
  const props = {};
  if (style.bold) {
    props.bold = true;
  }
  if (style.dim) {
    props.dim = true;
  }
  if (style.italic) {
    props.italic = true;
  }
  if (style.underline !== "none") {
    props.underline = true;
  }
  if (style.strikethrough) {
    props.strikethrough = true;
  }
  if (style.inverse) {
    props.inverse = true;
  }
  const fgColor = colorToString(style.fg);
  if (fgColor) {
    props.color = fgColor;
  }
  const bgColor = colorToString(style.bg);
  if (bgColor) {
    props.backgroundColor = bgColor;
  }
  return props;
}
var NAMED_COLOR_MAP = {
  black: "ansi:black",
  red: "ansi:red",
  green: "ansi:green",
  yellow: "ansi:yellow",
  blue: "ansi:blue",
  magenta: "ansi:magenta",
  cyan: "ansi:cyan",
  white: "ansi:white",
  brightBlack: "ansi:blackBright",
  brightRed: "ansi:redBright",
  brightGreen: "ansi:greenBright",
  brightYellow: "ansi:yellowBright",
  brightBlue: "ansi:blueBright",
  brightMagenta: "ansi:magentaBright",
  brightCyan: "ansi:cyanBright",
  brightWhite: "ansi:whiteBright"
};
function colorToString(color) {
  switch (color.type) {
    case "named":
      return NAMED_COLOR_MAP[color.name];
    case "indexed":
      return `ansi256(${color.index})`;
    case "rgb":
      return `rgb(${color.r},${color.g},${color.b})`;
    case "default":
      return void 0;
  }
}
function propsEqual(a, b) {
  return a.color === b.color && a.backgroundColor === b.backgroundColor && a.bold === b.bold && a.dim === b.dim && a.italic === b.italic && a.underline === b.underline && a.strikethrough === b.strikethrough && a.inverse === b.inverse && a.hyperlink === b.hyperlink;
}
function hasAnyProps(props) {
  return props.color !== void 0 || props.backgroundColor !== void 0 || props.dim === true || props.bold === true || props.italic === true || props.underline === true || props.strikethrough === true || props.inverse === true || props.hyperlink !== void 0;
}
function hasAnyTextProps(props) {
  return props.color !== void 0 || props.backgroundColor !== void 0 || props.dim === true || props.bold === true || props.italic === true || props.underline === true || props.strikethrough === true || props.inverse === true;
}
function StyledText(t0) {
  const $ = _c3(14);
  let bold;
  let children;
  let dim;
  let rest;
  if ($[0] !== t0) {
    ;
    ({ bold, dim, children, ...rest } = t0);
    $[0] = t0;
    $[1] = bold;
    $[2] = children;
    $[3] = dim;
    $[4] = rest;
  } else {
    bold = $[1];
    children = $[2];
    dim = $[3];
    rest = $[4];
  }
  if (dim) {
    let t12;
    if ($[5] !== children || $[6] !== rest) {
      t12 = /* @__PURE__ */ jsx3(Text, { ...rest, dim: true, children });
      $[5] = children;
      $[6] = rest;
      $[7] = t12;
    } else {
      t12 = $[7];
    }
    return t12;
  }
  if (bold) {
    let t12;
    if ($[8] !== children || $[9] !== rest) {
      t12 = /* @__PURE__ */ jsx3(Text, { ...rest, bold: true, children });
      $[8] = children;
      $[9] = rest;
      $[10] = t12;
    } else {
      t12 = $[10];
    }
    return t12;
  }
  let t1;
  if ($[11] !== children || $[12] !== rest) {
    t1 = /* @__PURE__ */ jsx3(Text, { ...rest, children });
    $[11] = children;
    $[12] = rest;
    $[13] = t1;
  } else {
    t1 = $[13];
  }
  return t1;
}

// src/ink/components/AlternateScreen.tsx
import { useContext as useContext2, useInsertionEffect } from "react";
import { c as _c5 } from "react/compiler-runtime";

// src/ink/instances.ts
var instances = /* @__PURE__ */ new Map();
var instances_default = instances;

// src/ink/useTerminalNotification.ts
import { createContext, useCallback, useContext, useMemo as useMemo3 } from "react";

// src/ink/terminal.ts
import { coerce } from "semver";

// src/ink/clearTerminal.ts
var CURSOR_HOME_WINDOWS = csi(0, "f");
function isWindowsTerminal() {
  return process.platform === "win32" && !!process.env.WT_SESSION;
}
function isMintty() {
  if (process.env.TERM_PROGRAM === "mintty") {
    return true;
  }
  if (process.platform === "win32" && process.env.MSYSTEM) {
    return true;
  }
  return false;
}
function isModernWindowsTerminal() {
  if (isWindowsTerminal()) {
    return true;
  }
  if (process.platform === "win32" && process.env.TERM_PROGRAM === "vscode" && process.env.TERM_PROGRAM_VERSION) {
    return true;
  }
  if (isMintty()) {
    return true;
  }
  return false;
}
function getClearTerminalSequence() {
  if (process.platform === "win32") {
    if (isModernWindowsTerminal()) {
      return ERASE_SCREEN + ERASE_SCROLLBACK + CURSOR_HOME;
    } else {
      return ERASE_SCREEN + CURSOR_HOME_WINDOWS;
    }
  }
  return ERASE_SCREEN + ERASE_SCROLLBACK + CURSOR_HOME;
}
var clearTerminal = getClearTerminalSequence();

// src/ink/terminal.ts
function isSynchronizedOutputSupported() {
  if (process.env.TMUX) {
    return false;
  }
  const termProgram = process.env.TERM_PROGRAM;
  const term = process.env.TERM;
  if (termProgram === "iTerm.app" || termProgram === "WezTerm" || termProgram === "WarpTerminal" || termProgram === "ghostty" || termProgram === "contour" || termProgram === "vscode" || termProgram === "alacritty") {
    return true;
  }
  if (term?.includes("kitty") || process.env.KITTY_WINDOW_ID) {
    return true;
  }
  if (term === "xterm-ghostty") {
    return true;
  }
  if (term?.startsWith("foot")) {
    return true;
  }
  if (term?.includes("alacritty")) {
    return true;
  }
  if (process.env.ZED_TERM) {
    return true;
  }
  if (process.env.WT_SESSION) {
    return true;
  }
  const vteVersion = process.env.VTE_VERSION;
  if (vteVersion) {
    const version = parseInt(vteVersion, 10);
    if (version >= 6800) {
      return true;
    }
  }
  return false;
}
var xtversionName;
function setXtversionName(name) {
  if (xtversionName === void 0) {
    xtversionName = name;
  }
}
function isXtermJs() {
  if (process.env.TERM_PROGRAM === "vscode") {
    return true;
  }
  return xtversionName?.startsWith("xterm.js") ?? false;
}
var EXTENDED_KEYS_TERMINALS = ["iTerm.app", "kitty", "WezTerm", "ghostty", "tmux", "windows-terminal"];
function supportsExtendedKeys() {
  return EXTENDED_KEYS_TERMINALS.includes(env.terminal ?? "");
}
var SYNC_OUTPUT_SUPPORTED = isSynchronizedOutputSupported();
function writeDiffToTerminal(terminal, diff2, skipSyncMarkers = false) {
  if (diff2.length === 0) {
    return;
  }
  const useSync = !skipSyncMarkers;
  let buffer = useSync ? BSU : "";
  for (const patch of diff2) {
    switch (patch.type) {
      case "stdout":
        buffer += patch.content;
        break;
      case "clear":
        if (patch.count > 0) {
          buffer += eraseLines(patch.count);
        }
        break;
      case "clearTerminal":
        buffer += getClearTerminalSequence();
        break;
      case "cursorHide":
        buffer += HIDE_CURSOR;
        break;
      case "cursorShow":
        buffer += SHOW_CURSOR;
        break;
      case "cursorMove":
        buffer += cursorMove(patch.x, patch.y);
        break;
      case "cursorTo":
        buffer += cursorTo(patch.col);
        break;
      case "carriageReturn":
        buffer += "\r";
        break;
      case "hyperlink":
        buffer += link(patch.uri);
        break;
      case "styleStr":
        buffer += patch.str;
        break;
    }
  }
  if (useSync) {
    buffer += ESU;
  }
  terminal.stdout.write(buffer);
}

// src/ink/useTerminalNotification.ts
var TerminalWriteContext = createContext(null);
var TerminalWriteProvider = TerminalWriteContext.Provider;

// src/ink/components/Box.tsx
import { c as _c4 } from "react/compiler-runtime";

// src/utils/debug.ts
function logForDebugging(_message, _options = {}) {
}

// src/ink/warn.ts
function ifNotInteger(value, name) {
  if (value === void 0) {
    return;
  }
  if (Number.isInteger(value)) {
    return;
  }
  logForDebugging(`${name} should be an integer, got ${value}`, {
    level: "warn"
  });
}

// src/ink/components/Box.tsx
import { jsx as jsx4 } from "react/jsx-runtime";
function Box(t0) {
  const $ = _c4(48);
  let autoFocus;
  let children;
  let flexDirection;
  let flexGrow;
  let flexShrink;
  let flexWrap;
  let onBlur;
  let onBlurCapture;
  let onClick;
  let onFocus;
  let onFocusCapture;
  let onKeyDown;
  let onKeyDownCapture;
  let onMouseDown;
  let onMouseDrag;
  let onMouseEnter;
  let onMouseLeave;
  let onMouseUp;
  let ref;
  let style;
  let tabIndex;
  if ($[0] !== t0) {
    const {
      children: t12,
      flexWrap: t22,
      flexDirection: t32,
      flexGrow: t42,
      flexShrink: t5,
      ref: t6,
      tabIndex: t7,
      autoFocus: t8,
      onClick: t9,
      onFocus: t10,
      onFocusCapture: t11,
      onBlur: t122,
      onBlurCapture: t13,
      onMouseDown: t14,
      onMouseUp: t15,
      onMouseDrag: t16,
      onMouseEnter: t17,
      onMouseLeave: t18,
      onKeyDown: t19,
      onKeyDownCapture: t20,
      ...t21
    } = t0;
    children = t12;
    ref = t6;
    tabIndex = t7;
    autoFocus = t8;
    onClick = t9;
    onFocus = t10;
    onFocusCapture = t11;
    onBlur = t122;
    onBlurCapture = t13;
    onMouseDown = t14;
    onMouseUp = t15;
    onMouseDrag = t16;
    onMouseEnter = t17;
    onMouseLeave = t18;
    onKeyDown = t19;
    onKeyDownCapture = t20;
    style = t21;
    flexWrap = t22 === void 0 ? "nowrap" : t22;
    flexDirection = t32 === void 0 ? "row" : t32;
    flexGrow = t42 === void 0 ? 0 : t42;
    flexShrink = t5 === void 0 ? 1 : t5;
    ifNotInteger(style.margin, "margin");
    ifNotInteger(style.marginX, "marginX");
    ifNotInteger(style.marginY, "marginY");
    ifNotInteger(style.marginTop, "marginTop");
    ifNotInteger(style.marginBottom, "marginBottom");
    ifNotInteger(style.marginLeft, "marginLeft");
    ifNotInteger(style.marginRight, "marginRight");
    ifNotInteger(style.padding, "padding");
    ifNotInteger(style.paddingX, "paddingX");
    ifNotInteger(style.paddingY, "paddingY");
    ifNotInteger(style.paddingTop, "paddingTop");
    ifNotInteger(style.paddingBottom, "paddingBottom");
    ifNotInteger(style.paddingLeft, "paddingLeft");
    ifNotInteger(style.paddingRight, "paddingRight");
    ifNotInteger(style.gap, "gap");
    ifNotInteger(style.columnGap, "columnGap");
    ifNotInteger(style.rowGap, "rowGap");
    $[0] = t0;
    $[1] = autoFocus;
    $[2] = children;
    $[3] = flexDirection;
    $[4] = flexGrow;
    $[5] = flexShrink;
    $[6] = flexWrap;
    $[7] = onBlur;
    $[8] = onBlurCapture;
    $[9] = onClick;
    $[10] = onFocus;
    $[11] = onFocusCapture;
    $[12] = onKeyDown;
    $[13] = onKeyDownCapture;
    $[14] = onMouseDown;
    $[15] = onMouseUp;
    $[16] = onMouseDrag;
    $[17] = onMouseEnter;
    $[18] = onMouseLeave;
    $[19] = ref;
    $[20] = style;
    $[21] = tabIndex;
  } else {
    autoFocus = $[1];
    children = $[2];
    flexDirection = $[3];
    flexGrow = $[4];
    flexShrink = $[5];
    flexWrap = $[6];
    onBlur = $[7];
    onBlurCapture = $[8];
    onClick = $[9];
    onFocus = $[10];
    onFocusCapture = $[11];
    onKeyDown = $[12];
    onKeyDownCapture = $[13];
    onMouseDown = $[14];
    onMouseUp = $[15];
    onMouseDrag = $[16];
    onMouseEnter = $[17];
    onMouseLeave = $[18];
    ref = $[19];
    style = $[20];
    tabIndex = $[21];
  }
  const t1 = style.overflowX ?? style.overflow ?? "visible";
  const t2 = style.overflowY ?? style.overflow ?? "visible";
  let t3;
  if ($[22] !== flexDirection || $[23] !== flexGrow || $[24] !== flexShrink || $[25] !== flexWrap || $[26] !== style || $[27] !== t1 || $[28] !== t2) {
    t3 = {
      flexWrap,
      flexDirection,
      flexGrow,
      flexShrink,
      ...style,
      overflowX: t1,
      overflowY: t2
    };
    $[22] = flexDirection;
    $[23] = flexGrow;
    $[24] = flexShrink;
    $[25] = flexWrap;
    $[26] = style;
    $[27] = t1;
    $[28] = t2;
    $[29] = t3;
  } else {
    t3 = $[29];
  }
  let t4;
  if ($[30] !== autoFocus || $[31] !== children || $[32] !== onBlur || $[33] !== onBlurCapture || $[34] !== onClick || $[35] !== onFocus || $[36] !== onFocusCapture || $[37] !== onKeyDown || $[38] !== onKeyDownCapture || $[39] !== onMouseDown || $[40] !== onMouseUp || $[41] !== onMouseDrag || $[42] !== onMouseEnter || $[43] !== onMouseLeave || $[44] !== ref || $[45] !== t3 || $[46] !== tabIndex) {
    t4 = /* @__PURE__ */ jsx4(
      "ink-box",
      {
        autoFocus,
        onBlur,
        onBlurCapture,
        onClick,
        onFocus,
        onFocusCapture,
        onKeyDown,
        onKeyDownCapture,
        onMouseDown,
        onMouseDrag,
        onMouseEnter,
        onMouseLeave,
        onMouseUp,
        ref,
        style: t3,
        tabIndex,
        children
      }
    );
    $[30] = autoFocus;
    $[31] = children;
    $[32] = onBlur;
    $[33] = onBlurCapture;
    $[34] = onClick;
    $[35] = onFocus;
    $[36] = onFocusCapture;
    $[37] = onKeyDown;
    $[38] = onKeyDownCapture;
    $[39] = onMouseDown;
    $[40] = onMouseUp;
    $[41] = onMouseDrag;
    $[42] = onMouseEnter;
    $[43] = onMouseLeave;
    $[44] = ref;
    $[45] = t3;
    $[46] = tabIndex;
    $[47] = t4;
  } else {
    t4 = $[47];
  }
  return t4;
}
var Box_default = Box;

// src/ink/components/TerminalSizeContext.tsx
import { createContext as createContext2 } from "react";
var TerminalSizeContext = createContext2(null);

// src/ink/components/AlternateScreen.tsx
import { jsx as jsx5 } from "react/jsx-runtime";
function AlternateScreen(t0) {
  const $ = _c5(7);
  const { children, mouseTracking: t1 } = t0;
  const mouseTracking = t1 === void 0 ? true : t1;
  const size = useContext2(TerminalSizeContext);
  const writeRaw = useContext2(TerminalWriteContext);
  let t2;
  let t3;
  if ($[0] !== mouseTracking || $[1] !== writeRaw) {
    t2 = () => {
      const ink = instances_default.get(process.stdout);
      if (!writeRaw) {
        return;
      }
      writeRaw(ENTER_ALT_SCREEN + "\x1B[2J\x1B[H" + (mouseTracking ? ENABLE_MOUSE_TRACKING : ""));
      ink?.setAltScreenActive(true, mouseTracking);
      return () => {
        ink?.setAltScreenActive(false);
        ink?.clearTextSelection();
        writeRaw((mouseTracking ? DISABLE_MOUSE_TRACKING : "") + EXIT_ALT_SCREEN);
      };
    };
    t3 = [writeRaw, mouseTracking];
    $[0] = mouseTracking;
    $[1] = writeRaw;
    $[2] = t2;
    $[3] = t3;
  } else {
    t2 = $[2];
    t3 = $[3];
  }
  useInsertionEffect(t2, t3);
  const t4 = size?.rows ?? 24;
  let t5;
  if ($[4] !== children || $[5] !== t4) {
    t5 = /* @__PURE__ */ jsx5(Box_default, { flexDirection: "column", flexShrink: 0, height: t4, width: "100%", children });
    $[4] = children;
    $[5] = t4;
    $[6] = t5;
  } else {
    t5 = $[6];
  }
  return t5;
}

// src/ink/components/Newline.tsx
import { c as _c6 } from "react/compiler-runtime";
import { jsx as jsx6 } from "react/jsx-runtime";
function Newline(t0) {
  const $ = _c6(4);
  const { count: t1 } = t0;
  const count = t1 === void 0 ? 1 : t1;
  let t2;
  if ($[0] !== count) {
    t2 = "\n".repeat(count);
    $[0] = count;
    $[1] = t2;
  } else {
    t2 = $[1];
  }
  let t3;
  if ($[2] !== t2) {
    t3 = /* @__PURE__ */ jsx6("ink-text", { children: t2 });
    $[2] = t2;
    $[3] = t3;
  } else {
    t3 = $[3];
  }
  return t3;
}

// src/ink/components/NoSelect.tsx
import { c as _c7 } from "react/compiler-runtime";
import { jsx as jsx7 } from "react/jsx-runtime";
function NoSelect(t0) {
  const $ = _c7(8);
  let boxProps;
  let children;
  let fromLeftEdge;
  if ($[0] !== t0) {
    ;
    ({ children, fromLeftEdge, ...boxProps } = t0);
    $[0] = t0;
    $[1] = boxProps;
    $[2] = children;
    $[3] = fromLeftEdge;
  } else {
    boxProps = $[1];
    children = $[2];
    fromLeftEdge = $[3];
  }
  const t1 = fromLeftEdge ? "from-left-edge" : true;
  let t2;
  if ($[4] !== boxProps || $[5] !== children || $[6] !== t1) {
    t2 = /* @__PURE__ */ jsx7(Box_default, { ...boxProps, noSelect: t1, children });
    $[4] = boxProps;
    $[5] = children;
    $[6] = t1;
    $[7] = t2;
  } else {
    t2 = $[7];
  }
  return t2;
}

// src/ink/components/RawAnsi.tsx
import { c as _c8 } from "react/compiler-runtime";
import { jsx as jsx8 } from "react/jsx-runtime";
function RawAnsi(t0) {
  const $ = _c8(6);
  const { lines, width } = t0;
  if (lines.length === 0) {
    return null;
  }
  let t1;
  if ($[0] !== lines) {
    t1 = lines.join("\n");
    $[0] = lines;
    $[1] = t1;
  } else {
    t1 = $[1];
  }
  let t2;
  if ($[2] !== lines.length || $[3] !== t1 || $[4] !== width) {
    t2 = /* @__PURE__ */ jsx8("ink-raw-ansi", { rawHeight: lines.length, rawText: t1, rawWidth: width });
    $[2] = lines.length;
    $[3] = t1;
    $[4] = width;
    $[5] = t2;
  } else {
    t2 = $[5];
  }
  return t2;
}

// src/ink/components/ScrollBox.tsx
import { useImperativeHandle, useRef, useState } from "react";

// src/bootstrap/state.ts
function flushInteractionTime() {
}
function updateLastInteractionTime() {
}
function markScrollActivity() {
}

// src/native-ts/yoga-layout/enums.ts
var Align = {
  Auto: 0,
  FlexStart: 1,
  Center: 2,
  FlexEnd: 3,
  Stretch: 4,
  Baseline: 5,
  SpaceBetween: 6,
  SpaceAround: 7,
  SpaceEvenly: 8
};
var Direction = {
  Inherit: 0,
  LTR: 1,
  RTL: 2
};
var Display = {
  Flex: 0,
  None: 1,
  Contents: 2
};
var Edge = {
  Left: 0,
  Top: 1,
  Right: 2,
  Bottom: 3,
  Start: 4,
  End: 5,
  Horizontal: 6,
  Vertical: 7,
  All: 8
};
var Errata = {
  None: 0,
  StretchFlexBasis: 1,
  AbsolutePositionWithoutInsetsExcludesPadding: 2,
  AbsolutePercentAgainstInnerSize: 4,
  All: 2147483647,
  Classic: 2147483646
};
var FlexDirection = {
  Column: 0,
  ColumnReverse: 1,
  Row: 2,
  RowReverse: 3
};
var Gutter = {
  Column: 0,
  Row: 1,
  All: 2
};
var Justify = {
  FlexStart: 0,
  Center: 1,
  FlexEnd: 2,
  SpaceBetween: 3,
  SpaceAround: 4,
  SpaceEvenly: 5
};
var MeasureMode = {
  Undefined: 0,
  Exactly: 1,
  AtMost: 2
};
var Overflow = {
  Visible: 0,
  Hidden: 1,
  Scroll: 2
};
var PositionType = {
  Static: 0,
  Relative: 1,
  Absolute: 2
};
var Unit = {
  Undefined: 0,
  Point: 1,
  Percent: 2,
  Auto: 3
};
var Wrap = {
  NoWrap: 0,
  Wrap: 1,
  WrapReverse: 2
};

// src/native-ts/yoga-layout/index.ts
var UNDEFINED_VALUE = { unit: Unit.Undefined, value: NaN };
var AUTO_VALUE = { unit: Unit.Auto, value: NaN };
function pointValue(v) {
  return { unit: Unit.Point, value: v };
}
function percentValue(v) {
  return { unit: Unit.Percent, value: v };
}
function resolveValue(v, ownerSize) {
  switch (v.unit) {
    case Unit.Point:
      return v.value;
    case Unit.Percent:
      return isNaN(ownerSize) ? NaN : v.value * ownerSize / 100;
    default:
      return NaN;
  }
}
function isDefined(n) {
  return !isNaN(n);
}
function sameFloat(a, b) {
  return a === b || a !== a && b !== b;
}
function defaultStyle2() {
  return {
    direction: Direction.Inherit,
    flexDirection: FlexDirection.Column,
    justifyContent: Justify.FlexStart,
    alignItems: Align.Stretch,
    alignSelf: Align.Auto,
    alignContent: Align.FlexStart,
    flexWrap: Wrap.NoWrap,
    overflow: Overflow.Visible,
    display: Display.Flex,
    positionType: PositionType.Relative,
    flexGrow: 0,
    flexShrink: 0,
    flexBasis: AUTO_VALUE,
    margin: new Array(9).fill(UNDEFINED_VALUE),
    padding: new Array(9).fill(UNDEFINED_VALUE),
    border: new Array(9).fill(UNDEFINED_VALUE),
    position: new Array(9).fill(UNDEFINED_VALUE),
    gap: new Array(3).fill(UNDEFINED_VALUE),
    width: AUTO_VALUE,
    height: AUTO_VALUE,
    minWidth: UNDEFINED_VALUE,
    minHeight: UNDEFINED_VALUE,
    maxWidth: UNDEFINED_VALUE,
    maxHeight: UNDEFINED_VALUE
  };
}
var EDGE_LEFT = 0;
var EDGE_TOP = 1;
var EDGE_RIGHT = 2;
var EDGE_BOTTOM = 3;
function resolveEdge(edges, physicalEdge2, ownerSize, allowAuto = false) {
  let v = edges[physicalEdge2];
  if (v.unit === Unit.Undefined) {
    if (physicalEdge2 === EDGE_LEFT || physicalEdge2 === EDGE_RIGHT) {
      v = edges[Edge.Horizontal];
    } else {
      v = edges[Edge.Vertical];
    }
  }
  if (v.unit === Unit.Undefined) {
    v = edges[Edge.All];
  }
  if (v.unit === Unit.Undefined) {
    if (physicalEdge2 === EDGE_LEFT) {
      v = edges[Edge.Start];
    }
    if (physicalEdge2 === EDGE_RIGHT) {
      v = edges[Edge.End];
    }
  }
  if (v.unit === Unit.Undefined) {
    return 0;
  }
  if (v.unit === Unit.Auto) {
    return allowAuto ? NaN : 0;
  }
  return resolveValue(v, ownerSize);
}
function resolveEdgeRaw(edges, physicalEdge2) {
  let v = edges[physicalEdge2];
  if (v.unit === Unit.Undefined) {
    if (physicalEdge2 === EDGE_LEFT || physicalEdge2 === EDGE_RIGHT) {
      v = edges[Edge.Horizontal];
    } else {
      v = edges[Edge.Vertical];
    }
  }
  if (v.unit === Unit.Undefined) {
    v = edges[Edge.All];
  }
  if (v.unit === Unit.Undefined) {
    if (physicalEdge2 === EDGE_LEFT) {
      v = edges[Edge.Start];
    }
    if (physicalEdge2 === EDGE_RIGHT) {
      v = edges[Edge.End];
    }
  }
  return v;
}
function isMarginAuto(edges, physicalEdge2) {
  return resolveEdgeRaw(edges, physicalEdge2).unit === Unit.Auto;
}
function hasAnyAutoEdge(edges) {
  for (let i = 0; i < 9; i++) {
    if (edges[i].unit === 3) {
      return true;
    }
  }
  return false;
}
function hasAnyDefinedEdge(edges) {
  for (let i = 0; i < 9; i++) {
    if (edges[i].unit !== 0) {
      return true;
    }
  }
  return false;
}
function resolveEdges4Into(edges, ownerSize, out) {
  const eH = edges[6];
  const eV = edges[7];
  const eA = edges[8];
  const eS = edges[4];
  const eE = edges[5];
  const pctDenom = isNaN(ownerSize) ? NaN : ownerSize / 100;
  let v = edges[0];
  if (v.unit === 0) {
    v = eH;
  }
  if (v.unit === 0) {
    v = eA;
  }
  if (v.unit === 0) {
    v = eS;
  }
  out[0] = v.unit === 1 ? v.value : v.unit === 2 ? v.value * pctDenom : 0;
  v = edges[1];
  if (v.unit === 0) {
    v = eV;
  }
  if (v.unit === 0) {
    v = eA;
  }
  out[1] = v.unit === 1 ? v.value : v.unit === 2 ? v.value * pctDenom : 0;
  v = edges[2];
  if (v.unit === 0) {
    v = eH;
  }
  if (v.unit === 0) {
    v = eA;
  }
  if (v.unit === 0) {
    v = eE;
  }
  out[2] = v.unit === 1 ? v.value : v.unit === 2 ? v.value * pctDenom : 0;
  v = edges[3];
  if (v.unit === 0) {
    v = eV;
  }
  if (v.unit === 0) {
    v = eA;
  }
  out[3] = v.unit === 1 ? v.value : v.unit === 2 ? v.value * pctDenom : 0;
}
function isRow(dir) {
  return dir === FlexDirection.Row || dir === FlexDirection.RowReverse;
}
function isReverse(dir) {
  return dir === FlexDirection.RowReverse || dir === FlexDirection.ColumnReverse;
}
function crossAxis(dir) {
  return isRow(dir) ? FlexDirection.Column : FlexDirection.Row;
}
function leadingEdge(dir) {
  switch (dir) {
    case FlexDirection.Row:
      return EDGE_LEFT;
    case FlexDirection.RowReverse:
      return EDGE_RIGHT;
    case FlexDirection.Column:
      return EDGE_TOP;
    case FlexDirection.ColumnReverse:
      return EDGE_BOTTOM;
  }
}
function trailingEdge(dir) {
  switch (dir) {
    case FlexDirection.Row:
      return EDGE_RIGHT;
    case FlexDirection.RowReverse:
      return EDGE_LEFT;
    case FlexDirection.Column:
      return EDGE_BOTTOM;
    case FlexDirection.ColumnReverse:
      return EDGE_TOP;
  }
}
function createConfig() {
  const config = {
    pointScaleFactor: 1,
    errata: Errata.None,
    useWebDefaults: false,
    free() {
    },
    isExperimentalFeatureEnabled() {
      return false;
    },
    setExperimentalFeatureEnabled() {
    },
    setPointScaleFactor(f) {
      config.pointScaleFactor = f;
    },
    getErrata() {
      return config.errata;
    },
    setErrata(e) {
      config.errata = e;
    },
    setUseWebDefaults(v) {
      config.useWebDefaults = v;
    }
  };
  return config;
}
var Node = class {
  style;
  layout;
  parent;
  children;
  measureFunc;
  config;
  isDirty_;
  isReferenceBaseline_;
  _flexBasis = 0;
  _mainSize = 0;
  _crossSize = 0;
  _lineIndex = 0;
  _hasAutoMargin = false;
  _hasPosition = false;
  _hasPadding = false;
  _hasBorder = false;
  _hasMargin = false;
  _lW = NaN;
  _lH = NaN;
  _lWM = 0;
  _lHM = 0;
  _lOW = NaN;
  _lOH = NaN;
  _lFW = false;
  _lFH = false;
  _lOutW = NaN;
  _lOutH = NaN;
  _hasL = false;
  _mW = NaN;
  _mH = NaN;
  _mWM = 0;
  _mHM = 0;
  _mOW = NaN;
  _mOH = NaN;
  _mOutW = NaN;
  _mOutH = NaN;
  _hasM = false;
  _fbBasis = NaN;
  _fbOwnerW = NaN;
  _fbOwnerH = NaN;
  _fbAvailMain = NaN;
  _fbAvailCross = NaN;
  _fbCrossMode = 0;
  _fbGen = -1;
  _cIn = null;
  _cOut = null;
  _cGen = -1;
  _cN = 0;
  _cWr = 0;
  constructor(config) {
    this.style = defaultStyle2();
    this.layout = {
      left: 0,
      top: 0,
      width: 0,
      height: 0,
      border: [0, 0, 0, 0],
      padding: [0, 0, 0, 0],
      margin: [0, 0, 0, 0]
    };
    this.parent = null;
    this.children = [];
    this.measureFunc = null;
    this.config = config ?? DEFAULT_CONFIG;
    this.isDirty_ = true;
    this.isReferenceBaseline_ = false;
    _yogaLiveNodes++;
  }
  insertChild(child, index) {
    child.parent = this;
    this.children.splice(index, 0, child);
    this.markDirty();
  }
  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx >= 0) {
      this.children.splice(idx, 1);
      child.parent = null;
      this.markDirty();
    }
  }
  getChild(index) {
    return this.children[index];
  }
  getChildCount() {
    return this.children.length;
  }
  getParent() {
    return this.parent;
  }
  free() {
    this.parent = null;
    this.children = [];
    this.measureFunc = null;
    this._cIn = null;
    this._cOut = null;
    _yogaLiveNodes--;
  }
  freeRecursive() {
    for (const c of this.children) {
      c.freeRecursive();
    }
    this.free();
  }
  reset() {
    this.style = defaultStyle2();
    this.children = [];
    this.parent = null;
    this.measureFunc = null;
    this.isDirty_ = true;
    this._hasAutoMargin = false;
    this._hasPosition = false;
    this._hasPadding = false;
    this._hasBorder = false;
    this._hasMargin = false;
    this._hasL = false;
    this._hasM = false;
    this._cN = 0;
    this._cWr = 0;
    this._fbBasis = NaN;
  }
  markDirty() {
    this.isDirty_ = true;
    if (this.parent && !this.parent.isDirty_) {
      this.parent.markDirty();
    }
  }
  isDirty() {
    return this.isDirty_;
  }
  hasNewLayout() {
    return true;
  }
  markLayoutSeen() {
  }
  setMeasureFunc(fn) {
    this.measureFunc = fn;
    this.markDirty();
  }
  unsetMeasureFunc() {
    this.measureFunc = null;
    this.markDirty();
  }
  getComputedLeft() {
    return this.layout.left;
  }
  getComputedTop() {
    return this.layout.top;
  }
  getComputedWidth() {
    return this.layout.width;
  }
  getComputedHeight() {
    return this.layout.height;
  }
  getComputedRight() {
    const p = this.parent;
    return p ? p.layout.width - this.layout.left - this.layout.width : 0;
  }
  getComputedBottom() {
    const p = this.parent;
    return p ? p.layout.height - this.layout.top - this.layout.height : 0;
  }
  getComputedLayout() {
    return {
      left: this.layout.left,
      top: this.layout.top,
      right: this.getComputedRight(),
      bottom: this.getComputedBottom(),
      width: this.layout.width,
      height: this.layout.height
    };
  }
  getComputedBorder(edge) {
    return this.layout.border[physicalEdge(edge)];
  }
  getComputedPadding(edge) {
    return this.layout.padding[physicalEdge(edge)];
  }
  getComputedMargin(edge) {
    return this.layout.margin[physicalEdge(edge)];
  }
  setWidth(v) {
    this.style.width = parseDimension(v);
    this.markDirty();
  }
  setWidthPercent(v) {
    this.style.width = percentValue(v);
    this.markDirty();
  }
  setWidthAuto() {
    this.style.width = AUTO_VALUE;
    this.markDirty();
  }
  setHeight(v) {
    this.style.height = parseDimension(v);
    this.markDirty();
  }
  setHeightPercent(v) {
    this.style.height = percentValue(v);
    this.markDirty();
  }
  setHeightAuto() {
    this.style.height = AUTO_VALUE;
    this.markDirty();
  }
  setMinWidth(v) {
    this.style.minWidth = parseDimension(v);
    this.markDirty();
  }
  setMinWidthPercent(v) {
    this.style.minWidth = percentValue(v);
    this.markDirty();
  }
  setMinHeight(v) {
    this.style.minHeight = parseDimension(v);
    this.markDirty();
  }
  setMinHeightPercent(v) {
    this.style.minHeight = percentValue(v);
    this.markDirty();
  }
  setMaxWidth(v) {
    this.style.maxWidth = parseDimension(v);
    this.markDirty();
  }
  setMaxWidthPercent(v) {
    this.style.maxWidth = percentValue(v);
    this.markDirty();
  }
  setMaxHeight(v) {
    this.style.maxHeight = parseDimension(v);
    this.markDirty();
  }
  setMaxHeightPercent(v) {
    this.style.maxHeight = percentValue(v);
    this.markDirty();
  }
  setFlexDirection(dir) {
    this.style.flexDirection = dir;
    this.markDirty();
  }
  setFlexGrow(v) {
    this.style.flexGrow = v ?? 0;
    this.markDirty();
  }
  setFlexShrink(v) {
    this.style.flexShrink = v ?? 0;
    this.markDirty();
  }
  setFlex(v) {
    if (v === void 0 || isNaN(v)) {
      this.style.flexGrow = 0;
      this.style.flexShrink = 0;
    } else if (v > 0) {
      this.style.flexGrow = v;
      this.style.flexShrink = 1;
      this.style.flexBasis = pointValue(0);
    } else if (v < 0) {
      this.style.flexGrow = 0;
      this.style.flexShrink = -v;
    } else {
      this.style.flexGrow = 0;
      this.style.flexShrink = 0;
    }
    this.markDirty();
  }
  setFlexBasis(v) {
    this.style.flexBasis = parseDimension(v);
    this.markDirty();
  }
  setFlexBasisPercent(v) {
    this.style.flexBasis = percentValue(v);
    this.markDirty();
  }
  setFlexBasisAuto() {
    this.style.flexBasis = AUTO_VALUE;
    this.markDirty();
  }
  setFlexWrap(wrap) {
    this.style.flexWrap = wrap;
    this.markDirty();
  }
  setAlignItems(a) {
    this.style.alignItems = a;
    this.markDirty();
  }
  setAlignSelf(a) {
    this.style.alignSelf = a;
    this.markDirty();
  }
  setAlignContent(a) {
    this.style.alignContent = a;
    this.markDirty();
  }
  setJustifyContent(j) {
    this.style.justifyContent = j;
    this.markDirty();
  }
  setDisplay(d) {
    this.style.display = d;
    this.markDirty();
  }
  getDisplay() {
    return this.style.display;
  }
  setPositionType(t) {
    this.style.positionType = t;
    this.markDirty();
  }
  setPosition(edge, v) {
    this.style.position[edge] = parseDimension(v);
    this._hasPosition = hasAnyDefinedEdge(this.style.position);
    this.markDirty();
  }
  setPositionPercent(edge, v) {
    this.style.position[edge] = percentValue(v);
    this._hasPosition = true;
    this.markDirty();
  }
  setPositionAuto(edge) {
    this.style.position[edge] = AUTO_VALUE;
    this._hasPosition = true;
    this.markDirty();
  }
  setOverflow(o) {
    this.style.overflow = o;
    this.markDirty();
  }
  setDirection(d) {
    this.style.direction = d;
    this.markDirty();
  }
  setBoxSizing(_) {
  }
  setMargin(edge, v) {
    const val = parseDimension(v);
    this.style.margin[edge] = val;
    if (val.unit === Unit.Auto) {
      this._hasAutoMargin = true;
    } else {
      this._hasAutoMargin = hasAnyAutoEdge(this.style.margin);
    }
    this._hasMargin = this._hasAutoMargin || hasAnyDefinedEdge(this.style.margin);
    this.markDirty();
  }
  setMarginPercent(edge, v) {
    this.style.margin[edge] = percentValue(v);
    this._hasAutoMargin = hasAnyAutoEdge(this.style.margin);
    this._hasMargin = true;
    this.markDirty();
  }
  setMarginAuto(edge) {
    this.style.margin[edge] = AUTO_VALUE;
    this._hasAutoMargin = true;
    this._hasMargin = true;
    this.markDirty();
  }
  setPadding(edge, v) {
    this.style.padding[edge] = parseDimension(v);
    this._hasPadding = hasAnyDefinedEdge(this.style.padding);
    this.markDirty();
  }
  setPaddingPercent(edge, v) {
    this.style.padding[edge] = percentValue(v);
    this._hasPadding = true;
    this.markDirty();
  }
  setBorder(edge, v) {
    this.style.border[edge] = v === void 0 ? UNDEFINED_VALUE : pointValue(v);
    this._hasBorder = hasAnyDefinedEdge(this.style.border);
    this.markDirty();
  }
  setGap(gutter, v) {
    this.style.gap[gutter] = parseDimension(v);
    this.markDirty();
  }
  setGapPercent(gutter, v) {
    this.style.gap[gutter] = percentValue(v);
    this.markDirty();
  }
  getFlexDirection() {
    return this.style.flexDirection;
  }
  getJustifyContent() {
    return this.style.justifyContent;
  }
  getAlignItems() {
    return this.style.alignItems;
  }
  getAlignSelf() {
    return this.style.alignSelf;
  }
  getAlignContent() {
    return this.style.alignContent;
  }
  getFlexGrow() {
    return this.style.flexGrow;
  }
  getFlexShrink() {
    return this.style.flexShrink;
  }
  getFlexBasis() {
    return this.style.flexBasis;
  }
  getFlexWrap() {
    return this.style.flexWrap;
  }
  getWidth() {
    return this.style.width;
  }
  getHeight() {
    return this.style.height;
  }
  getOverflow() {
    return this.style.overflow;
  }
  getPositionType() {
    return this.style.positionType;
  }
  getDirection() {
    return this.style.direction;
  }
  copyStyle(_) {
  }
  setDirtiedFunc(_) {
  }
  unsetDirtiedFunc() {
  }
  setIsReferenceBaseline(v) {
    this.isReferenceBaseline_ = v;
    this.markDirty();
  }
  isReferenceBaseline() {
    return this.isReferenceBaseline_;
  }
  setAspectRatio(_) {
  }
  getAspectRatio() {
    return NaN;
  }
  setAlwaysFormsContainingBlock(_) {
  }
  calculateLayout(ownerWidth, ownerHeight, _direction) {
    _yogaNodesVisited = 0;
    _yogaMeasureCalls = 0;
    _yogaCacheHits = 0;
    _generation++;
    const w = ownerWidth === void 0 ? NaN : ownerWidth;
    const h = ownerHeight === void 0 ? NaN : ownerHeight;
    layoutNode(
      this,
      w,
      h,
      isDefined(w) ? MeasureMode.Exactly : MeasureMode.Undefined,
      isDefined(h) ? MeasureMode.Exactly : MeasureMode.Undefined,
      w,
      h,
      true
    );
    const mar = this.layout.margin;
    const posL = resolveValue(resolveEdgeRaw(this.style.position, EDGE_LEFT), isDefined(w) ? w : 0);
    const posT = resolveValue(resolveEdgeRaw(this.style.position, EDGE_TOP), isDefined(w) ? w : 0);
    this.layout.left = mar[EDGE_LEFT] + (isDefined(posL) ? posL : 0);
    this.layout.top = mar[EDGE_TOP] + (isDefined(posT) ? posT : 0);
    roundLayout(this, this.config.pointScaleFactor, 0, 0);
  }
};
var DEFAULT_CONFIG = createConfig();
var CACHE_SLOTS = 4;
function cacheWrite(node, aW, aH, wM, hM, oW, oH, fW, fH, wasDirty) {
  if (!node._cIn) {
    node._cIn = new Float64Array(CACHE_SLOTS * 8);
    node._cOut = new Float64Array(CACHE_SLOTS * 2);
  }
  if (wasDirty && node._cGen !== _generation) {
    node._cN = 0;
    node._cWr = 0;
  }
  const i = node._cWr++ % CACHE_SLOTS;
  if (node._cN < CACHE_SLOTS) {
    node._cN = node._cWr;
  }
  const o = i * 8;
  const cIn = node._cIn;
  cIn[o] = aW;
  cIn[o + 1] = aH;
  cIn[o + 2] = wM;
  cIn[o + 3] = hM;
  cIn[o + 4] = oW;
  cIn[o + 5] = oH;
  cIn[o + 6] = fW ? 1 : 0;
  cIn[o + 7] = fH ? 1 : 0;
  node._cOut[i * 2] = node.layout.width;
  node._cOut[i * 2 + 1] = node.layout.height;
  node._cGen = _generation;
}
function commitCacheOutputs(node, performLayout) {
  if (performLayout) {
    node._lOutW = node.layout.width;
    node._lOutH = node.layout.height;
  } else {
    node._mOutW = node.layout.width;
    node._mOutH = node.layout.height;
  }
}
var _generation = 0;
var _yogaNodesVisited = 0;
var _yogaMeasureCalls = 0;
var _yogaCacheHits = 0;
var _yogaLiveNodes = 0;
function getYogaCounters() {
  return {
    visited: _yogaNodesVisited,
    measured: _yogaMeasureCalls,
    cacheHits: _yogaCacheHits,
    live: _yogaLiveNodes
  };
}
function layoutNode(node, availableWidth, availableHeight, widthMode, heightMode, ownerWidth, ownerHeight, performLayout, forceWidth = false, forceHeight = false) {
  _yogaNodesVisited++;
  const style = node.style;
  const layout = node.layout;
  const sameGen = node._cGen === _generation && !performLayout;
  if (!node.isDirty_ || sameGen) {
    if (!node.isDirty_ && node._hasL && node._lWM === widthMode && node._lHM === heightMode && node._lFW === forceWidth && node._lFH === forceHeight && sameFloat(node._lW, availableWidth) && sameFloat(node._lH, availableHeight) && sameFloat(node._lOW, ownerWidth) && sameFloat(node._lOH, ownerHeight)) {
      _yogaCacheHits++;
      layout.width = node._lOutW;
      layout.height = node._lOutH;
      return;
    }
    if (node._cN > 0 && (sameGen || !node.isDirty_)) {
      const cIn = node._cIn;
      for (let i = 0; i < node._cN; i++) {
        const o = i * 8;
        if (cIn[o + 2] === widthMode && cIn[o + 3] === heightMode && cIn[o + 6] === (forceWidth ? 1 : 0) && cIn[o + 7] === (forceHeight ? 1 : 0) && sameFloat(cIn[o], availableWidth) && sameFloat(cIn[o + 1], availableHeight) && sameFloat(cIn[o + 4], ownerWidth) && sameFloat(cIn[o + 5], ownerHeight)) {
          layout.width = node._cOut[i * 2];
          layout.height = node._cOut[i * 2 + 1];
          _yogaCacheHits++;
          return;
        }
      }
    }
    if (!node.isDirty_ && !performLayout && node._hasM && node._mWM === widthMode && node._mHM === heightMode && sameFloat(node._mW, availableWidth) && sameFloat(node._mH, availableHeight) && sameFloat(node._mOW, ownerWidth) && sameFloat(node._mOH, ownerHeight)) {
      layout.width = node._mOutW;
      layout.height = node._mOutH;
      _yogaCacheHits++;
      return;
    }
  }
  const wasDirty = node.isDirty_;
  if (performLayout) {
    node._lW = availableWidth;
    node._lH = availableHeight;
    node._lWM = widthMode;
    node._lHM = heightMode;
    node._lOW = ownerWidth;
    node._lOH = ownerHeight;
    node._lFW = forceWidth;
    node._lFH = forceHeight;
    node._hasL = true;
    node.isDirty_ = false;
    if (wasDirty) {
      node._hasM = false;
    }
  } else {
    node._mW = availableWidth;
    node._mH = availableHeight;
    node._mWM = widthMode;
    node._mHM = heightMode;
    node._mOW = ownerWidth;
    node._mOH = ownerHeight;
    node._hasM = true;
    if (wasDirty) {
      node._hasL = false;
    }
  }
  const pad = layout.padding;
  const bor = layout.border;
  const mar = layout.margin;
  if (node._hasPadding) {
    resolveEdges4Into(style.padding, ownerWidth, pad);
  } else {
    pad[0] = pad[1] = pad[2] = pad[3] = 0;
  }
  if (node._hasBorder) {
    resolveEdges4Into(style.border, ownerWidth, bor);
  } else {
    bor[0] = bor[1] = bor[2] = bor[3] = 0;
  }
  if (node._hasMargin) {
    resolveEdges4Into(style.margin, ownerWidth, mar);
  } else {
    mar[0] = mar[1] = mar[2] = mar[3] = 0;
  }
  const paddingBorderWidth = pad[0] + pad[2] + bor[0] + bor[2];
  const paddingBorderHeight = pad[1] + pad[3] + bor[1] + bor[3];
  const styleWidth = forceWidth ? NaN : resolveValue(style.width, ownerWidth);
  const styleHeight = forceHeight ? NaN : resolveValue(style.height, ownerHeight);
  let width = availableWidth;
  let height = availableHeight;
  let wMode = widthMode;
  let hMode = heightMode;
  if (isDefined(styleWidth)) {
    width = styleWidth;
    wMode = MeasureMode.Exactly;
  }
  if (isDefined(styleHeight)) {
    height = styleHeight;
    hMode = MeasureMode.Exactly;
  }
  width = boundAxis(style, true, width, ownerWidth, ownerHeight);
  height = boundAxis(style, false, height, ownerWidth, ownerHeight);
  if (node.measureFunc && node.children.length === 0) {
    const innerW = wMode === MeasureMode.Undefined ? NaN : Math.max(0, width - paddingBorderWidth);
    const innerH = hMode === MeasureMode.Undefined ? NaN : Math.max(0, height - paddingBorderHeight);
    _yogaMeasureCalls++;
    const measured = node.measureFunc(innerW, wMode, innerH, hMode);
    node.layout.width = wMode === MeasureMode.Exactly ? width : boundAxis(style, true, (measured.width ?? 0) + paddingBorderWidth, ownerWidth, ownerHeight);
    node.layout.height = hMode === MeasureMode.Exactly ? height : boundAxis(style, false, (measured.height ?? 0) + paddingBorderHeight, ownerWidth, ownerHeight);
    commitCacheOutputs(node, performLayout);
    cacheWrite(
      node,
      availableWidth,
      availableHeight,
      widthMode,
      heightMode,
      ownerWidth,
      ownerHeight,
      forceWidth,
      forceHeight,
      wasDirty
    );
    return;
  }
  if (node.children.length === 0) {
    node.layout.width = wMode === MeasureMode.Exactly ? width : boundAxis(style, true, paddingBorderWidth, ownerWidth, ownerHeight);
    node.layout.height = hMode === MeasureMode.Exactly ? height : boundAxis(style, false, paddingBorderHeight, ownerWidth, ownerHeight);
    commitCacheOutputs(node, performLayout);
    cacheWrite(
      node,
      availableWidth,
      availableHeight,
      widthMode,
      heightMode,
      ownerWidth,
      ownerHeight,
      forceWidth,
      forceHeight,
      wasDirty
    );
    return;
  }
  const mainAxis = style.flexDirection;
  const crossAx = crossAxis(mainAxis);
  const isMainRow = isRow(mainAxis);
  const mainSize = isMainRow ? width : height;
  const crossSize = isMainRow ? height : width;
  const mainMode = isMainRow ? wMode : hMode;
  const crossMode = isMainRow ? hMode : wMode;
  const mainPadBorder = isMainRow ? paddingBorderWidth : paddingBorderHeight;
  const crossPadBorder = isMainRow ? paddingBorderHeight : paddingBorderWidth;
  const innerMainSize = isDefined(mainSize) ? Math.max(0, mainSize - mainPadBorder) : NaN;
  const innerCrossSize = isDefined(crossSize) ? Math.max(0, crossSize - crossPadBorder) : NaN;
  const gapMain = resolveGap(style, isMainRow ? Gutter.Column : Gutter.Row, innerMainSize);
  const flowChildren = [];
  const absChildren = [];
  collectLayoutChildren(node, flowChildren, absChildren);
  const ownerW = isDefined(width) ? width : NaN;
  const ownerH = isDefined(height) ? height : NaN;
  const isWrap = style.flexWrap !== Wrap.NoWrap;
  const gapCross = resolveGap(style, isMainRow ? Gutter.Row : Gutter.Column, innerCrossSize);
  for (const c of flowChildren) {
    c._flexBasis = computeFlexBasis(c, mainAxis, innerMainSize, innerCrossSize, crossMode, ownerW, ownerH);
  }
  const lines = [];
  if (!isWrap || !isDefined(innerMainSize) || flowChildren.length === 0) {
    for (const c of flowChildren) {
      c._lineIndex = 0;
    }
    lines.push(flowChildren);
  } else {
    let lineStart = 0;
    let lineLen = 0;
    for (let i = 0; i < flowChildren.length; i++) {
      const c = flowChildren[i];
      const hypo = boundAxis(c.style, isMainRow, c._flexBasis, ownerW, ownerH);
      const outer = Math.max(0, hypo) + childMarginForAxis(c, mainAxis, ownerW);
      const withGap = i > lineStart ? gapMain : 0;
      if (i > lineStart && lineLen + withGap + outer > innerMainSize) {
        lines.push(flowChildren.slice(lineStart, i));
        lineStart = i;
        lineLen = outer;
      } else {
        lineLen += withGap + outer;
      }
      c._lineIndex = lines.length;
    }
    lines.push(flowChildren.slice(lineStart));
  }
  const lineCount = lines.length;
  const isBaseline = isBaselineLayout(node, flowChildren);
  const lineConsumedMain = new Array(lineCount);
  const lineCrossSizes = new Array(lineCount);
  const lineMaxAscent = isBaseline ? new Array(lineCount).fill(0) : [];
  let maxLineMain = 0;
  let totalLinesCross = 0;
  for (let li = 0; li < lineCount; li++) {
    const line = lines[li];
    const lineGap = line.length > 1 ? gapMain * (line.length - 1) : 0;
    let lineBasis = lineGap;
    for (const c of line) {
      lineBasis += c._flexBasis + childMarginForAxis(c, mainAxis, ownerW);
    }
    let availMain = innerMainSize;
    if (!isDefined(availMain)) {
      const mainOwner = isMainRow ? ownerWidth : ownerHeight;
      const minM = resolveValue(isMainRow ? style.minWidth : style.minHeight, mainOwner);
      const maxM = resolveValue(isMainRow ? style.maxWidth : style.maxHeight, mainOwner);
      if (isDefined(maxM) && lineBasis > maxM - mainPadBorder) {
        availMain = Math.max(0, maxM - mainPadBorder);
      } else if (isDefined(minM) && lineBasis < minM - mainPadBorder) {
        availMain = Math.max(0, minM - mainPadBorder);
      }
    }
    resolveFlexibleLengths(line, availMain, lineBasis, isMainRow, ownerW, ownerH);
    let lineCross = 0;
    for (const c of line) {
      const cStyle = c.style;
      const childAlign = cStyle.alignSelf === Align.Auto ? style.alignItems : cStyle.alignSelf;
      const cMarginCross = childMarginForAxis(c, crossAx, ownerW);
      let childCrossSize = NaN;
      let childCrossMode = MeasureMode.Undefined;
      const resolvedCrossStyle = resolveValue(isMainRow ? cStyle.height : cStyle.width, isMainRow ? ownerH : ownerW);
      const crossLeadE = isMainRow ? EDGE_TOP : EDGE_LEFT;
      const crossTrailE = isMainRow ? EDGE_BOTTOM : EDGE_RIGHT;
      const hasCrossAutoMargin = c._hasAutoMargin && (isMarginAuto(cStyle.margin, crossLeadE) || isMarginAuto(cStyle.margin, crossTrailE));
      if (isDefined(resolvedCrossStyle)) {
        childCrossSize = resolvedCrossStyle;
        childCrossMode = MeasureMode.Exactly;
      } else if (childAlign === Align.Stretch && !hasCrossAutoMargin && !isWrap && isDefined(innerCrossSize) && crossMode === MeasureMode.Exactly) {
        childCrossSize = Math.max(0, innerCrossSize - cMarginCross);
        childCrossMode = MeasureMode.Exactly;
      } else if (!isWrap && isDefined(innerCrossSize)) {
        childCrossSize = Math.max(0, innerCrossSize - cMarginCross);
        childCrossMode = MeasureMode.AtMost;
      }
      const cw = isMainRow ? c._mainSize : childCrossSize;
      const ch = isMainRow ? childCrossSize : c._mainSize;
      layoutNode(
        c,
        cw,
        ch,
        isMainRow ? MeasureMode.Exactly : childCrossMode,
        isMainRow ? childCrossMode : MeasureMode.Exactly,
        ownerW,
        ownerH,
        performLayout,
        isMainRow,
        !isMainRow
      );
      c._crossSize = isMainRow ? c.layout.height : c.layout.width;
      lineCross = Math.max(lineCross, c._crossSize + cMarginCross);
    }
    if (isBaseline) {
      let maxAscent = 0;
      let maxDescent = 0;
      for (const c of line) {
        if (resolveChildAlign(node, c) !== Align.Baseline) {
          continue;
        }
        const mTop = resolveEdge(c.style.margin, EDGE_TOP, ownerW);
        const mBot = resolveEdge(c.style.margin, EDGE_BOTTOM, ownerW);
        const ascent = calculateBaseline(c) + mTop;
        const descent = c.layout.height + mTop + mBot - ascent;
        if (ascent > maxAscent) {
          maxAscent = ascent;
        }
        if (descent > maxDescent) {
          maxDescent = descent;
        }
      }
      lineMaxAscent[li] = maxAscent;
      if (maxAscent + maxDescent > lineCross) {
        lineCross = maxAscent + maxDescent;
      }
    }
    const mainLead = leadingEdge(mainAxis);
    const mainTrail = trailingEdge(mainAxis);
    let consumed = lineGap;
    for (const c of line) {
      const cm = c.layout.margin;
      consumed += c._mainSize + cm[mainLead] + cm[mainTrail];
    }
    lineConsumedMain[li] = consumed;
    lineCrossSizes[li] = lineCross;
    maxLineMain = Math.max(maxLineMain, consumed);
    totalLinesCross += lineCross;
  }
  const totalCrossGap = lineCount > 1 ? gapCross * (lineCount - 1) : 0;
  totalLinesCross += totalCrossGap;
  const isScroll = style.overflow === Overflow.Scroll;
  const contentMain = maxLineMain + mainPadBorder;
  const finalMainSize = mainMode === MeasureMode.Exactly ? mainSize : mainMode === MeasureMode.AtMost && isScroll ? Math.max(Math.min(mainSize, contentMain), mainPadBorder) : isWrap && lineCount > 1 && mainMode === MeasureMode.AtMost ? mainSize : contentMain;
  const contentCross = totalLinesCross + crossPadBorder;
  const finalCrossSize = crossMode === MeasureMode.Exactly ? crossSize : crossMode === MeasureMode.AtMost && isScroll ? Math.max(Math.min(crossSize, contentCross), crossPadBorder) : contentCross;
  node.layout.width = boundAxis(style, true, isMainRow ? finalMainSize : finalCrossSize, ownerWidth, ownerHeight);
  node.layout.height = boundAxis(style, false, isMainRow ? finalCrossSize : finalMainSize, ownerWidth, ownerHeight);
  commitCacheOutputs(node, performLayout);
  cacheWrite(
    node,
    availableWidth,
    availableHeight,
    widthMode,
    heightMode,
    ownerWidth,
    ownerHeight,
    forceWidth,
    forceHeight,
    wasDirty
  );
  if (!performLayout) {
    return;
  }
  const actualInnerMain = (isMainRow ? node.layout.width : node.layout.height) - mainPadBorder;
  const actualInnerCross = (isMainRow ? node.layout.height : node.layout.width) - crossPadBorder;
  const mainLeadEdgePhys = leadingEdge(mainAxis);
  const mainTrailEdgePhys = trailingEdge(mainAxis);
  const crossLeadEdgePhys = isMainRow ? EDGE_TOP : EDGE_LEFT;
  const crossTrailEdgePhys = isMainRow ? EDGE_BOTTOM : EDGE_RIGHT;
  const reversed = isReverse(mainAxis);
  const mainContainerSize = isMainRow ? node.layout.width : node.layout.height;
  const crossLead = pad[crossLeadEdgePhys] + bor[crossLeadEdgePhys];
  let lineCrossOffset = crossLead;
  let betweenLines = gapCross;
  const freeCross = actualInnerCross - totalLinesCross;
  if (lineCount === 1 && !isWrap && !isBaseline) {
    lineCrossSizes[0] = actualInnerCross;
  } else {
    const remCross = Math.max(0, freeCross);
    switch (style.alignContent) {
      case Align.FlexStart:
        break;
      case Align.Center:
        lineCrossOffset += freeCross / 2;
        break;
      case Align.FlexEnd:
        lineCrossOffset += freeCross;
        break;
      case Align.Stretch:
        if (lineCount > 0 && remCross > 0) {
          const add = remCross / lineCount;
          for (let i = 0; i < lineCount; i++) {
            lineCrossSizes[i] += add;
          }
        }
        break;
      case Align.SpaceBetween:
        if (lineCount > 1) {
          betweenLines += remCross / (lineCount - 1);
        }
        break;
      case Align.SpaceAround:
        if (lineCount > 0) {
          betweenLines += remCross / lineCount;
          lineCrossOffset += remCross / lineCount / 2;
        }
        break;
      case Align.SpaceEvenly:
        if (lineCount > 0) {
          betweenLines += remCross / (lineCount + 1);
          lineCrossOffset += remCross / (lineCount + 1);
        }
        break;
      default:
        break;
    }
  }
  const wrapReverse = style.flexWrap === Wrap.WrapReverse;
  const crossContainerSize = isMainRow ? node.layout.height : node.layout.width;
  let lineCrossPos = lineCrossOffset;
  for (let li = 0; li < lineCount; li++) {
    const line = lines[li];
    const lineCross = lineCrossSizes[li];
    const consumedMain = lineConsumedMain[li];
    const n = line.length;
    if (isWrap || crossMode !== MeasureMode.Exactly) {
      for (const c of line) {
        const cStyle = c.style;
        const childAlign = cStyle.alignSelf === Align.Auto ? style.alignItems : cStyle.alignSelf;
        const crossStyleDef = isDefined(
          resolveValue(isMainRow ? cStyle.height : cStyle.width, isMainRow ? ownerH : ownerW)
        );
        const hasCrossAutoMargin = c._hasAutoMargin && (isMarginAuto(cStyle.margin, crossLeadEdgePhys) || isMarginAuto(cStyle.margin, crossTrailEdgePhys));
        if (childAlign === Align.Stretch && !crossStyleDef && !hasCrossAutoMargin) {
          const cMarginCross = childMarginForAxis(c, crossAx, ownerW);
          const target = Math.max(0, lineCross - cMarginCross);
          if (c._crossSize !== target) {
            const cw = isMainRow ? c._mainSize : target;
            const ch = isMainRow ? target : c._mainSize;
            layoutNode(
              c,
              cw,
              ch,
              MeasureMode.Exactly,
              MeasureMode.Exactly,
              ownerW,
              ownerH,
              performLayout,
              isMainRow,
              !isMainRow
            );
            c._crossSize = target;
          }
        }
      }
    }
    let mainOffset = pad[mainLeadEdgePhys] + bor[mainLeadEdgePhys];
    let betweenMain = gapMain;
    let numAutoMarginsMain = 0;
    for (const c of line) {
      if (!c._hasAutoMargin) {
        continue;
      }
      if (isMarginAuto(c.style.margin, mainLeadEdgePhys)) {
        numAutoMarginsMain++;
      }
      if (isMarginAuto(c.style.margin, mainTrailEdgePhys)) {
        numAutoMarginsMain++;
      }
    }
    const freeMain = actualInnerMain - consumedMain;
    const remainingMain = Math.max(0, freeMain);
    const autoMarginMainSize = numAutoMarginsMain > 0 && remainingMain > 0 ? remainingMain / numAutoMarginsMain : 0;
    if (numAutoMarginsMain === 0) {
      switch (style.justifyContent) {
        case Justify.FlexStart:
          break;
        case Justify.Center:
          mainOffset += freeMain / 2;
          break;
        case Justify.FlexEnd:
          mainOffset += freeMain;
          break;
        case Justify.SpaceBetween:
          if (n > 1) {
            betweenMain += remainingMain / (n - 1);
          }
          break;
        case Justify.SpaceAround:
          if (n > 0) {
            betweenMain += remainingMain / n;
            mainOffset += remainingMain / n / 2;
          }
          break;
        case Justify.SpaceEvenly:
          if (n > 0) {
            betweenMain += remainingMain / (n + 1);
            mainOffset += remainingMain / (n + 1);
          }
          break;
      }
    }
    const effectiveLineCrossPos = wrapReverse ? crossContainerSize - lineCrossPos - lineCross : lineCrossPos;
    let pos = mainOffset;
    for (const c of line) {
      const cMargin = c.style.margin;
      const cLayoutMargin = c.layout.margin;
      let autoMainLead = false;
      let autoMainTrail = false;
      let autoCrossLead = false;
      let autoCrossTrail = false;
      let mMainLead;
      let mMainTrail;
      let mCrossLead;
      let mCrossTrail;
      if (c._hasAutoMargin) {
        autoMainLead = isMarginAuto(cMargin, mainLeadEdgePhys);
        autoMainTrail = isMarginAuto(cMargin, mainTrailEdgePhys);
        autoCrossLead = isMarginAuto(cMargin, crossLeadEdgePhys);
        autoCrossTrail = isMarginAuto(cMargin, crossTrailEdgePhys);
        mMainLead = autoMainLead ? autoMarginMainSize : cLayoutMargin[mainLeadEdgePhys];
        mMainTrail = autoMainTrail ? autoMarginMainSize : cLayoutMargin[mainTrailEdgePhys];
        mCrossLead = autoCrossLead ? 0 : cLayoutMargin[crossLeadEdgePhys];
        mCrossTrail = autoCrossTrail ? 0 : cLayoutMargin[crossTrailEdgePhys];
      } else {
        mMainLead = cLayoutMargin[mainLeadEdgePhys];
        mMainTrail = cLayoutMargin[mainTrailEdgePhys];
        mCrossLead = cLayoutMargin[crossLeadEdgePhys];
        mCrossTrail = cLayoutMargin[crossTrailEdgePhys];
      }
      const mainPos = reversed ? mainContainerSize - (pos + mMainLead) - c._mainSize : pos + mMainLead;
      const childAlign = c.style.alignSelf === Align.Auto ? style.alignItems : c.style.alignSelf;
      let crossPos = effectiveLineCrossPos + mCrossLead;
      const crossFree = lineCross - c._crossSize - mCrossLead - mCrossTrail;
      if (autoCrossLead && autoCrossTrail) {
        crossPos += Math.max(0, crossFree) / 2;
      } else if (autoCrossLead) {
        crossPos += Math.max(0, crossFree);
      } else if (autoCrossTrail) {
      } else {
        switch (childAlign) {
          case Align.FlexStart:
          case Align.Stretch:
            if (wrapReverse) {
              crossPos += crossFree;
            }
            break;
          case Align.Center:
            crossPos += crossFree / 2;
            break;
          case Align.FlexEnd:
            if (!wrapReverse) {
              crossPos += crossFree;
            }
            break;
          case Align.Baseline:
            if (isBaseline) {
              crossPos = effectiveLineCrossPos + lineMaxAscent[li] - calculateBaseline(c);
            }
            break;
          default:
            break;
        }
      }
      let relX = 0;
      let relY = 0;
      if (c._hasPosition) {
        const relLeft = resolveValue(resolveEdgeRaw(c.style.position, EDGE_LEFT), ownerW);
        const relRight = resolveValue(resolveEdgeRaw(c.style.position, EDGE_RIGHT), ownerW);
        const relTop = resolveValue(resolveEdgeRaw(c.style.position, EDGE_TOP), ownerW);
        const relBottom = resolveValue(resolveEdgeRaw(c.style.position, EDGE_BOTTOM), ownerW);
        relX = isDefined(relLeft) ? relLeft : isDefined(relRight) ? -relRight : 0;
        relY = isDefined(relTop) ? relTop : isDefined(relBottom) ? -relBottom : 0;
      }
      if (isMainRow) {
        c.layout.left = mainPos + relX;
        c.layout.top = crossPos + relY;
      } else {
        c.layout.left = crossPos + relX;
        c.layout.top = mainPos + relY;
      }
      pos += c._mainSize + mMainLead + mMainTrail + betweenMain;
    }
    lineCrossPos += lineCross + betweenLines;
  }
  for (const c of absChildren) {
    layoutAbsoluteChild(node, c, node.layout.width, node.layout.height, pad, bor);
  }
}
function layoutAbsoluteChild(parent, child, parentWidth, parentHeight, pad, bor) {
  const cs = child.style;
  const posLeft = resolveEdgeRaw(cs.position, EDGE_LEFT);
  const posRight = resolveEdgeRaw(cs.position, EDGE_RIGHT);
  const posTop = resolveEdgeRaw(cs.position, EDGE_TOP);
  const posBottom = resolveEdgeRaw(cs.position, EDGE_BOTTOM);
  const rLeft = resolveValue(posLeft, parentWidth);
  const rRight = resolveValue(posRight, parentWidth);
  const rTop = resolveValue(posTop, parentHeight);
  const rBottom = resolveValue(posBottom, parentHeight);
  const paddingBoxW = parentWidth - bor[0] - bor[2];
  const paddingBoxH = parentHeight - bor[1] - bor[3];
  let cw = resolveValue(cs.width, paddingBoxW);
  let ch = resolveValue(cs.height, paddingBoxH);
  if (!isDefined(cw) && isDefined(rLeft) && isDefined(rRight)) {
    cw = paddingBoxW - rLeft - rRight;
  }
  if (!isDefined(ch) && isDefined(rTop) && isDefined(rBottom)) {
    ch = paddingBoxH - rTop - rBottom;
  }
  layoutNode(
    child,
    cw,
    ch,
    isDefined(cw) ? MeasureMode.Exactly : MeasureMode.Undefined,
    isDefined(ch) ? MeasureMode.Exactly : MeasureMode.Undefined,
    paddingBoxW,
    paddingBoxH,
    true
  );
  const mL = resolveEdge(cs.margin, EDGE_LEFT, parentWidth);
  const mT = resolveEdge(cs.margin, EDGE_TOP, parentWidth);
  const mR = resolveEdge(cs.margin, EDGE_RIGHT, parentWidth);
  const mB = resolveEdge(cs.margin, EDGE_BOTTOM, parentWidth);
  const mainAxis = parent.style.flexDirection;
  const reversed = isReverse(mainAxis);
  const mainRow = isRow(mainAxis);
  const wrapReverse = parent.style.flexWrap === Wrap.WrapReverse;
  const alignment = cs.alignSelf === Align.Auto ? parent.style.alignItems : cs.alignSelf;
  let left;
  if (isDefined(rLeft)) {
    left = bor[0] + rLeft + mL;
  } else if (isDefined(rRight)) {
    left = parentWidth - bor[2] - rRight - child.layout.width - mR;
  } else if (mainRow) {
    const lead = pad[0] + bor[0];
    const trail = parentWidth - pad[2] - bor[2];
    left = reversed ? trail - child.layout.width - mR : justifyAbsolute(parent.style.justifyContent, lead, trail, child.layout.width) + mL;
  } else {
    left = alignAbsolute(alignment, pad[0] + bor[0], parentWidth - pad[2] - bor[2], child.layout.width, wrapReverse) + mL;
  }
  let top;
  if (isDefined(rTop)) {
    top = bor[1] + rTop + mT;
  } else if (isDefined(rBottom)) {
    top = parentHeight - bor[3] - rBottom - child.layout.height - mB;
  } else if (mainRow) {
    top = alignAbsolute(alignment, pad[1] + bor[1], parentHeight - pad[3] - bor[3], child.layout.height, wrapReverse) + mT;
  } else {
    const lead = pad[1] + bor[1];
    const trail = parentHeight - pad[3] - bor[3];
    top = reversed ? trail - child.layout.height - mB : justifyAbsolute(parent.style.justifyContent, lead, trail, child.layout.height) + mT;
  }
  child.layout.left = left;
  child.layout.top = top;
}
function justifyAbsolute(justify, leadEdge, trailEdge, childSize) {
  switch (justify) {
    case Justify.Center:
      return leadEdge + (trailEdge - leadEdge - childSize) / 2;
    case Justify.FlexEnd:
      return trailEdge - childSize;
    default:
      return leadEdge;
  }
}
function alignAbsolute(align, leadEdge, trailEdge, childSize, wrapReverse) {
  switch (align) {
    case Align.Center:
      return leadEdge + (trailEdge - leadEdge - childSize) / 2;
    case Align.FlexEnd:
      return wrapReverse ? leadEdge : trailEdge - childSize;
    default:
      return wrapReverse ? trailEdge - childSize : leadEdge;
  }
}
function computeFlexBasis(child, mainAxis, availableMain, availableCross, crossMode, ownerWidth, ownerHeight) {
  const sameGen = child._fbGen === _generation;
  if ((sameGen || !child.isDirty_) && child._fbCrossMode === crossMode && sameFloat(child._fbOwnerW, ownerWidth) && sameFloat(child._fbOwnerH, ownerHeight) && sameFloat(child._fbAvailMain, availableMain) && sameFloat(child._fbAvailCross, availableCross)) {
    return child._fbBasis;
  }
  const cs = child.style;
  const isMainRow = isRow(mainAxis);
  const basis = resolveValue(cs.flexBasis, availableMain);
  if (isDefined(basis)) {
    const b2 = Math.max(0, basis);
    child._fbBasis = b2;
    child._fbOwnerW = ownerWidth;
    child._fbOwnerH = ownerHeight;
    child._fbAvailMain = availableMain;
    child._fbAvailCross = availableCross;
    child._fbCrossMode = crossMode;
    child._fbGen = _generation;
    return b2;
  }
  const mainStyleDim = isMainRow ? cs.width : cs.height;
  const mainOwner = isMainRow ? ownerWidth : ownerHeight;
  const resolved = resolveValue(mainStyleDim, mainOwner);
  if (isDefined(resolved)) {
    const b2 = Math.max(0, resolved);
    child._fbBasis = b2;
    child._fbOwnerW = ownerWidth;
    child._fbOwnerH = ownerHeight;
    child._fbAvailMain = availableMain;
    child._fbAvailCross = availableCross;
    child._fbCrossMode = crossMode;
    child._fbGen = _generation;
    return b2;
  }
  const crossStyleDim = isMainRow ? cs.height : cs.width;
  const crossOwner = isMainRow ? ownerHeight : ownerWidth;
  let crossConstraint = resolveValue(crossStyleDim, crossOwner);
  let crossConstraintMode = isDefined(crossConstraint) ? MeasureMode.Exactly : MeasureMode.Undefined;
  if (!isDefined(crossConstraint) && isDefined(availableCross)) {
    crossConstraint = availableCross;
    crossConstraintMode = crossMode === MeasureMode.Exactly && isStretchAlign(child) ? MeasureMode.Exactly : MeasureMode.AtMost;
  }
  let mainConstraint = NaN;
  let mainConstraintMode = MeasureMode.Undefined;
  if (isMainRow && isDefined(availableMain) && hasMeasureFuncInSubtree(child)) {
    mainConstraint = availableMain;
    mainConstraintMode = MeasureMode.AtMost;
  }
  const mw = isMainRow ? mainConstraint : crossConstraint;
  const mh = isMainRow ? crossConstraint : mainConstraint;
  const mwMode = isMainRow ? mainConstraintMode : crossConstraintMode;
  const mhMode = isMainRow ? crossConstraintMode : mainConstraintMode;
  layoutNode(child, mw, mh, mwMode, mhMode, ownerWidth, ownerHeight, false);
  const b = isMainRow ? child.layout.width : child.layout.height;
  child._fbBasis = b;
  child._fbOwnerW = ownerWidth;
  child._fbOwnerH = ownerHeight;
  child._fbAvailMain = availableMain;
  child._fbAvailCross = availableCross;
  child._fbCrossMode = crossMode;
  child._fbGen = _generation;
  return b;
}
function hasMeasureFuncInSubtree(node) {
  if (node.measureFunc) {
    return true;
  }
  for (const c of node.children) {
    if (hasMeasureFuncInSubtree(c)) {
      return true;
    }
  }
  return false;
}
function resolveFlexibleLengths(children, availableInnerMain, totalFlexBasis, isMainRow, ownerW, ownerH) {
  const n = children.length;
  const frozen = new Array(n).fill(false);
  const initialFree = isDefined(availableInnerMain) ? availableInnerMain - totalFlexBasis : 0;
  for (let i = 0; i < n; i++) {
    const c = children[i];
    const clamped = boundAxis(c.style, isMainRow, c._flexBasis, ownerW, ownerH);
    const inflexible = !isDefined(availableInnerMain) || (initialFree >= 0 ? c.style.flexGrow === 0 : c.style.flexShrink === 0);
    if (inflexible) {
      c._mainSize = Math.max(0, clamped);
      frozen[i] = true;
    } else {
      c._mainSize = c._flexBasis;
    }
  }
  const unclamped = new Array(n);
  for (let iter = 0; iter <= n; iter++) {
    let frozenDelta = 0;
    let totalGrow = 0;
    let totalShrinkScaled = 0;
    let unfrozenCount = 0;
    for (let i = 0; i < n; i++) {
      const c = children[i];
      if (frozen[i]) {
        frozenDelta += c._mainSize - c._flexBasis;
      } else {
        totalGrow += c.style.flexGrow;
        totalShrinkScaled += c.style.flexShrink * c._flexBasis;
        unfrozenCount++;
      }
    }
    if (unfrozenCount === 0) {
      break;
    }
    let remaining = initialFree - frozenDelta;
    if (remaining > 0 && totalGrow > 0 && totalGrow < 1) {
      const scaled = initialFree * totalGrow;
      if (scaled < remaining) {
        remaining = scaled;
      }
    } else if (remaining < 0 && totalShrinkScaled > 0) {
      let totalShrink = 0;
      for (let i = 0; i < n; i++) {
        if (!frozen[i]) {
          totalShrink += children[i].style.flexShrink;
        }
      }
      if (totalShrink < 1) {
        const scaled = initialFree * totalShrink;
        if (scaled > remaining) {
          remaining = scaled;
        }
      }
    }
    let totalViolation = 0;
    for (let i = 0; i < n; i++) {
      if (frozen[i]) {
        continue;
      }
      const c = children[i];
      let t = c._flexBasis;
      if (remaining > 0 && totalGrow > 0) {
        t += remaining * c.style.flexGrow / totalGrow;
      } else if (remaining < 0 && totalShrinkScaled > 0) {
        t += remaining * (c.style.flexShrink * c._flexBasis) / totalShrinkScaled;
      }
      unclamped[i] = t;
      const clamped = Math.max(0, boundAxis(c.style, isMainRow, t, ownerW, ownerH));
      c._mainSize = clamped;
      totalViolation += clamped - t;
    }
    if (totalViolation === 0) {
      break;
    }
    let anyFrozen = false;
    for (let i = 0; i < n; i++) {
      if (frozen[i]) {
        continue;
      }
      const v = children[i]._mainSize - unclamped[i];
      if (totalViolation > 0 && v > 0 || totalViolation < 0 && v < 0) {
        frozen[i] = true;
        anyFrozen = true;
      }
    }
    if (!anyFrozen) {
      break;
    }
  }
}
function isStretchAlign(child) {
  const p = child.parent;
  if (!p) {
    return false;
  }
  const align = child.style.alignSelf === Align.Auto ? p.style.alignItems : child.style.alignSelf;
  return align === Align.Stretch;
}
function resolveChildAlign(parent, child) {
  return child.style.alignSelf === Align.Auto ? parent.style.alignItems : child.style.alignSelf;
}
function calculateBaseline(node) {
  let baselineChild = null;
  for (const c of node.children) {
    if (c._lineIndex > 0) {
      break;
    }
    if (c.style.positionType === PositionType.Absolute) {
      continue;
    }
    if (c.style.display === Display.None) {
      continue;
    }
    if (resolveChildAlign(node, c) === Align.Baseline || c.isReferenceBaseline_) {
      baselineChild = c;
      break;
    }
    if (baselineChild === null) {
      baselineChild = c;
    }
  }
  if (baselineChild === null) {
    return node.layout.height;
  }
  return calculateBaseline(baselineChild) + baselineChild.layout.top;
}
function isBaselineLayout(node, flowChildren) {
  if (!isRow(node.style.flexDirection)) {
    return false;
  }
  if (node.style.alignItems === Align.Baseline) {
    return true;
  }
  for (const c of flowChildren) {
    if (c.style.alignSelf === Align.Baseline) {
      return true;
    }
  }
  return false;
}
function childMarginForAxis(child, axis, ownerWidth) {
  if (!child._hasMargin) {
    return 0;
  }
  const lead = resolveEdge(child.style.margin, leadingEdge(axis), ownerWidth);
  const trail = resolveEdge(child.style.margin, trailingEdge(axis), ownerWidth);
  return lead + trail;
}
function resolveGap(style, gutter, ownerSize) {
  let v = style.gap[gutter];
  if (v.unit === Unit.Undefined) {
    v = style.gap[Gutter.All];
  }
  const r = resolveValue(v, ownerSize);
  return isDefined(r) ? Math.max(0, r) : 0;
}
function boundAxis(style, isWidth, value, ownerWidth, ownerHeight) {
  const minV = isWidth ? style.minWidth : style.minHeight;
  const maxV = isWidth ? style.maxWidth : style.maxHeight;
  const minU = minV.unit;
  const maxU = maxV.unit;
  if (minU === 0 && maxU === 0) {
    return value;
  }
  const owner = isWidth ? ownerWidth : ownerHeight;
  let v = value;
  if (maxU === 1) {
    if (v > maxV.value) {
      v = maxV.value;
    }
  } else if (maxU === 2) {
    const m = maxV.value * owner / 100;
    if (m === m && v > m) {
      v = m;
    }
  }
  if (minU === 1) {
    if (v < minV.value) {
      v = minV.value;
    }
  } else if (minU === 2) {
    const m = minV.value * owner / 100;
    if (m === m && v < m) {
      v = m;
    }
  }
  return v;
}
function zeroLayoutRecursive(node) {
  for (const c of node.children) {
    c.layout.left = 0;
    c.layout.top = 0;
    c.layout.width = 0;
    c.layout.height = 0;
    c.isDirty_ = true;
    c._hasL = false;
    c._hasM = false;
    zeroLayoutRecursive(c);
  }
}
function collectLayoutChildren(node, flow, abs) {
  for (const c of node.children) {
    const disp = c.style.display;
    if (disp === Display.None) {
      c.layout.left = 0;
      c.layout.top = 0;
      c.layout.width = 0;
      c.layout.height = 0;
      zeroLayoutRecursive(c);
    } else if (disp === Display.Contents) {
      c.layout.left = 0;
      c.layout.top = 0;
      c.layout.width = 0;
      c.layout.height = 0;
      collectLayoutChildren(c, flow, abs);
    } else if (c.style.positionType === PositionType.Absolute) {
      abs.push(c);
    } else {
      flow.push(c);
    }
  }
}
function roundLayout(node, scale, absLeft, absTop) {
  if (scale === 0) {
    return;
  }
  const l = node.layout;
  const nodeLeft = l.left;
  const nodeTop = l.top;
  const nodeWidth = l.width;
  const nodeHeight = l.height;
  const absNodeLeft = absLeft + nodeLeft;
  const absNodeTop = absTop + nodeTop;
  const isText = node.measureFunc !== null;
  l.left = roundValue(nodeLeft, scale, false, isText);
  l.top = roundValue(nodeTop, scale, false, isText);
  const absRight = absNodeLeft + nodeWidth;
  const absBottom = absNodeTop + nodeHeight;
  const hasFracW = !isWholeNumber(nodeWidth * scale);
  const hasFracH = !isWholeNumber(nodeHeight * scale);
  l.width = roundValue(absRight, scale, isText && hasFracW, isText && !hasFracW) - roundValue(absNodeLeft, scale, false, isText);
  l.height = roundValue(absBottom, scale, isText && hasFracH, isText && !hasFracH) - roundValue(absNodeTop, scale, false, isText);
  for (const c of node.children) {
    roundLayout(c, scale, absNodeLeft, absNodeTop);
  }
}
function isWholeNumber(v) {
  const frac = v - Math.floor(v);
  return frac < 1e-4 || frac > 0.9999;
}
function roundValue(v, scale, forceCeil, forceFloor) {
  let scaled = v * scale;
  let frac = scaled - Math.floor(scaled);
  if (frac < 0) {
    frac += 1;
  }
  if (frac < 1e-4) {
    scaled = Math.floor(scaled);
  } else if (frac > 0.9999) {
    scaled = Math.ceil(scaled);
  } else if (forceCeil) {
    scaled = Math.ceil(scaled);
  } else if (forceFloor) {
    scaled = Math.floor(scaled);
  } else {
    scaled = Math.floor(scaled) + (frac >= 0.4999 ? 1 : 0);
  }
  return scaled / scale;
}
function parseDimension(v) {
  if (v === void 0) {
    return UNDEFINED_VALUE;
  }
  if (v === "auto") {
    return AUTO_VALUE;
  }
  if (typeof v === "number") {
    return Number.isFinite(v) ? pointValue(v) : UNDEFINED_VALUE;
  }
  if (typeof v === "string" && v.endsWith("%")) {
    return percentValue(parseFloat(v));
  }
  const n = parseFloat(v);
  return isNaN(n) ? UNDEFINED_VALUE : pointValue(n);
}
function physicalEdge(edge) {
  switch (edge) {
    case Edge.Left:
    case Edge.Start:
      return EDGE_LEFT;
    case Edge.Top:
      return EDGE_TOP;
    case Edge.Right:
    case Edge.End:
      return EDGE_RIGHT;
    case Edge.Bottom:
      return EDGE_BOTTOM;
    default:
      return EDGE_LEFT;
  }
}
var YOGA_INSTANCE = {
  Config: {
    create: createConfig,
    destroy() {
    }
  },
  Node: {
    create: (config) => new Node(config),
    createDefault: () => new Node(),
    createWithConfig: (config) => new Node(config),
    destroy() {
    }
  }
};
var yoga_layout_default = YOGA_INSTANCE;

// src/ink/layout/node.ts
var LayoutEdge = {
  All: "all",
  Horizontal: "horizontal",
  Vertical: "vertical",
  Left: "left",
  Right: "right",
  Top: "top",
  Bottom: "bottom",
  Start: "start",
  End: "end"
};
var LayoutGutter = {
  All: "all",
  Column: "column",
  Row: "row"
};
var LayoutDisplay = {
  Flex: "flex",
  None: "none"
};
var LayoutFlexDirection = {
  Row: "row",
  RowReverse: "row-reverse",
  Column: "column",
  ColumnReverse: "column-reverse"
};
var LayoutAlign = {
  Auto: "auto",
  Stretch: "stretch",
  FlexStart: "flex-start",
  Center: "center",
  FlexEnd: "flex-end"
};
var LayoutJustify = {
  FlexStart: "flex-start",
  Center: "center",
  FlexEnd: "flex-end",
  SpaceBetween: "space-between",
  SpaceAround: "space-around",
  SpaceEvenly: "space-evenly"
};
var LayoutWrap = {
  NoWrap: "nowrap",
  Wrap: "wrap",
  WrapReverse: "wrap-reverse"
};
var LayoutPositionType = {
  Relative: "relative",
  Absolute: "absolute"
};
var LayoutOverflow = {
  Visible: "visible",
  Hidden: "hidden",
  Scroll: "scroll"
};
var LayoutMeasureMode = {
  Undefined: "undefined",
  Exactly: "exactly",
  AtMost: "at-most"
};

// src/ink/layout/yoga.ts
var EDGE_MAP = {
  all: Edge.All,
  horizontal: Edge.Horizontal,
  vertical: Edge.Vertical,
  left: Edge.Left,
  right: Edge.Right,
  top: Edge.Top,
  bottom: Edge.Bottom,
  start: Edge.Start,
  end: Edge.End
};
var GUTTER_MAP = {
  all: Gutter.All,
  column: Gutter.Column,
  row: Gutter.Row
};
var YogaLayoutNode = class _YogaLayoutNode {
  yoga;
  constructor(yoga) {
    this.yoga = yoga;
  }
  // Tree
  insertChild(child, index) {
    this.yoga.insertChild(child.yoga, index);
  }
  removeChild(child) {
    this.yoga.removeChild(child.yoga);
  }
  getChildCount() {
    return this.yoga.getChildCount();
  }
  getParent() {
    const p = this.yoga.getParent();
    return p ? new _YogaLayoutNode(p) : null;
  }
  // Layout
  calculateLayout(width, _height) {
    this.yoga.calculateLayout(width, void 0, Direction.LTR);
  }
  setMeasureFunc(fn) {
    this.yoga.setMeasureFunc((w, wMode) => {
      const mode = wMode === MeasureMode.Exactly ? LayoutMeasureMode.Exactly : wMode === MeasureMode.AtMost ? LayoutMeasureMode.AtMost : LayoutMeasureMode.Undefined;
      return fn(w, mode);
    });
  }
  unsetMeasureFunc() {
    this.yoga.unsetMeasureFunc();
  }
  markDirty() {
    this.yoga.markDirty();
  }
  // Computed layout
  getComputedLeft() {
    return this.yoga.getComputedLeft();
  }
  getComputedTop() {
    return this.yoga.getComputedTop();
  }
  getComputedWidth() {
    return this.yoga.getComputedWidth();
  }
  getComputedHeight() {
    return this.yoga.getComputedHeight();
  }
  getComputedBorder(edge) {
    return this.yoga.getComputedBorder(EDGE_MAP[edge]);
  }
  getComputedPadding(edge) {
    return this.yoga.getComputedPadding(EDGE_MAP[edge]);
  }
  // Style setters
  setWidth(value) {
    this.yoga.setWidth(value);
  }
  setWidthPercent(value) {
    this.yoga.setWidthPercent(value);
  }
  setWidthAuto() {
    this.yoga.setWidthAuto();
  }
  setHeight(value) {
    this.yoga.setHeight(value);
  }
  setHeightPercent(value) {
    this.yoga.setHeightPercent(value);
  }
  setHeightAuto() {
    this.yoga.setHeightAuto();
  }
  setMinWidth(value) {
    this.yoga.setMinWidth(value);
  }
  setMinWidthPercent(value) {
    this.yoga.setMinWidthPercent(value);
  }
  setMinHeight(value) {
    this.yoga.setMinHeight(value);
  }
  setMinHeightPercent(value) {
    this.yoga.setMinHeightPercent(value);
  }
  setMaxWidth(value) {
    this.yoga.setMaxWidth(value);
  }
  setMaxWidthPercent(value) {
    this.yoga.setMaxWidthPercent(value);
  }
  setMaxHeight(value) {
    this.yoga.setMaxHeight(value);
  }
  setMaxHeightPercent(value) {
    this.yoga.setMaxHeightPercent(value);
  }
  setFlexDirection(dir) {
    const map = {
      row: FlexDirection.Row,
      "row-reverse": FlexDirection.RowReverse,
      column: FlexDirection.Column,
      "column-reverse": FlexDirection.ColumnReverse
    };
    this.yoga.setFlexDirection(map[dir]);
  }
  setFlexGrow(value) {
    this.yoga.setFlexGrow(value);
  }
  setFlexShrink(value) {
    this.yoga.setFlexShrink(value);
  }
  setFlexBasis(value) {
    this.yoga.setFlexBasis(value);
  }
  setFlexBasisPercent(value) {
    this.yoga.setFlexBasisPercent(value);
  }
  setFlexWrap(wrap) {
    const map = {
      nowrap: Wrap.NoWrap,
      wrap: Wrap.Wrap,
      "wrap-reverse": Wrap.WrapReverse
    };
    this.yoga.setFlexWrap(map[wrap]);
  }
  setAlignItems(align) {
    const map = {
      auto: Align.Auto,
      stretch: Align.Stretch,
      "flex-start": Align.FlexStart,
      center: Align.Center,
      "flex-end": Align.FlexEnd
    };
    this.yoga.setAlignItems(map[align]);
  }
  setAlignSelf(align) {
    const map = {
      auto: Align.Auto,
      stretch: Align.Stretch,
      "flex-start": Align.FlexStart,
      center: Align.Center,
      "flex-end": Align.FlexEnd
    };
    this.yoga.setAlignSelf(map[align]);
  }
  setJustifyContent(justify) {
    const map = {
      "flex-start": Justify.FlexStart,
      center: Justify.Center,
      "flex-end": Justify.FlexEnd,
      "space-between": Justify.SpaceBetween,
      "space-around": Justify.SpaceAround,
      "space-evenly": Justify.SpaceEvenly
    };
    this.yoga.setJustifyContent(map[justify]);
  }
  setDisplay(display) {
    this.yoga.setDisplay(display === "flex" ? Display.Flex : Display.None);
  }
  getDisplay() {
    return this.yoga.getDisplay() === Display.None ? LayoutDisplay.None : LayoutDisplay.Flex;
  }
  setPositionType(type) {
    this.yoga.setPositionType(type === "absolute" ? PositionType.Absolute : PositionType.Relative);
  }
  setPosition(edge, value) {
    this.yoga.setPosition(EDGE_MAP[edge], value);
  }
  setPositionPercent(edge, value) {
    this.yoga.setPositionPercent(EDGE_MAP[edge], value);
  }
  setOverflow(overflow) {
    const map = {
      visible: Overflow.Visible,
      hidden: Overflow.Hidden,
      scroll: Overflow.Scroll
    };
    this.yoga.setOverflow(map[overflow]);
  }
  setMargin(edge, value) {
    this.yoga.setMargin(EDGE_MAP[edge], value);
  }
  setPadding(edge, value) {
    this.yoga.setPadding(EDGE_MAP[edge], value);
  }
  setBorder(edge, value) {
    this.yoga.setBorder(EDGE_MAP[edge], value);
  }
  setGap(gutter, value) {
    this.yoga.setGap(GUTTER_MAP[gutter], value);
  }
  // Lifecycle
  free() {
    this.yoga.free();
  }
  freeRecursive() {
    this.yoga.freeRecursive();
  }
};
function createYogaLayoutNode() {
  return new YogaLayoutNode(yoga_layout_default.Node.create());
}

// src/ink/layout/engine.ts
function createLayoutNode() {
  return createYogaLayoutNode();
}

// src/ink/stringWidth.ts
import emojiRegex from "emoji-regex";
import { eastAsianWidth } from "get-east-asian-width";
import stripAnsi from "strip-ansi";
var EMOJI_REGEX = emojiRegex();
function stringWidthJavaScript(str) {
  if (typeof str !== "string" || str.length === 0) {
    return 0;
  }
  let isPureAscii = true;
  for (let i = 0; i < str.length; i++) {
    const code = str.charCodeAt(i);
    if (code >= 127 || code === 27) {
      isPureAscii = false;
      break;
    }
  }
  if (isPureAscii) {
    let width2 = 0;
    for (let i = 0; i < str.length; i++) {
      const code = str.charCodeAt(i);
      if (code > 31) {
        width2++;
      }
    }
    return width2;
  }
  if (str.includes("\x1B")) {
    str = stripAnsi(str);
    if (str.length === 0) {
      return 0;
    }
  }
  if (!needsSegmentation(str)) {
    let width2 = 0;
    for (const char of str) {
      const codePoint = char.codePointAt(0);
      if (!isZeroWidth(codePoint)) {
        width2 += eastAsianWidth(codePoint, { ambiguousAsWide: false });
      }
    }
    return width2;
  }
  let width = 0;
  for (const { segment: grapheme } of getGraphemeSegmenter().segment(str)) {
    EMOJI_REGEX.lastIndex = 0;
    if (EMOJI_REGEX.test(grapheme)) {
      width += getEmojiWidth(grapheme);
      continue;
    }
    for (const char of grapheme) {
      const codePoint = char.codePointAt(0);
      if (!isZeroWidth(codePoint)) {
        width += eastAsianWidth(codePoint, { ambiguousAsWide: false });
        break;
      }
    }
  }
  return width;
}
function needsSegmentation(str) {
  for (const char of str) {
    const cp = char.codePointAt(0);
    if (cp >= 127744 && cp <= 129791) {
      return true;
    }
    if (cp >= 9728 && cp <= 10175) {
      return true;
    }
    if (cp >= 127462 && cp <= 127487) {
      return true;
    }
    if (cp >= 65024 && cp <= 65039) {
      return true;
    }
    if (cp === 8205) {
      return true;
    }
  }
  return false;
}
function getEmojiWidth(grapheme) {
  const first = grapheme.codePointAt(0);
  if (first >= 127462 && first <= 127487) {
    let count = 0;
    for (const _ of grapheme) {
      count++;
    }
    return count === 1 ? 1 : 2;
  }
  if (grapheme.length === 2) {
    const second = grapheme.codePointAt(1);
    if (second === 65039 && (first >= 48 && first <= 57 || first === 35 || first === 42)) {
      return 1;
    }
  }
  return 2;
}
function isZeroWidth(codePoint) {
  if (codePoint >= 32 && codePoint < 127) {
    return false;
  }
  if (codePoint >= 160 && codePoint < 768) {
    return codePoint === 173;
  }
  if (codePoint <= 31 || codePoint >= 127 && codePoint <= 159) {
    return true;
  }
  if (codePoint >= 8203 && codePoint <= 8205 || // ZW space/joiner
  codePoint === 65279 || // BOM
  codePoint >= 8288 && codePoint <= 8292) {
    return true;
  }
  if (codePoint >= 65024 && codePoint <= 65039 || codePoint >= 917760 && codePoint <= 917999) {
    return true;
  }
  if (codePoint >= 768 && codePoint <= 879 || codePoint >= 6832 && codePoint <= 6911 || codePoint >= 7616 && codePoint <= 7679 || codePoint >= 8400 && codePoint <= 8447 || codePoint >= 65056 && codePoint <= 65071) {
    return true;
  }
  if (codePoint >= 2304 && codePoint <= 3407) {
    const offset = codePoint & 127;
    if (offset <= 3) {
      return true;
    }
    if (offset >= 58 && offset <= 79) {
      return true;
    }
    if (offset >= 81 && offset <= 87) {
      return true;
    }
    if (offset >= 98 && offset <= 99) {
      return true;
    }
  }
  if (codePoint === 3633 || // Thai MAI HAN-AKAT
  codePoint >= 3636 && codePoint <= 3642 || // Thai vowel signs (skip U+0E32, U+0E33)
  codePoint >= 3655 && codePoint <= 3662 || // Thai vowel signs and marks
  codePoint === 3761 || // Lao MAI KAN
  codePoint >= 3764 && codePoint <= 3772 || // Lao vowel signs (skip U+0EB2, U+0EB3)
  codePoint >= 3784 && codePoint <= 3789) {
    return true;
  }
  if (codePoint >= 1536 && codePoint <= 1541 || codePoint === 1757 || codePoint === 1807 || codePoint === 2274) {
    return true;
  }
  if (codePoint >= 55296 && codePoint <= 57343) {
    return true;
  }
  if (codePoint >= 917504 && codePoint <= 917631) {
    return true;
  }
  return false;
}
var bunStringWidth = typeof Bun !== "undefined" && typeof Bun.stringWidth === "function" ? Bun.stringWidth : null;
var BUN_STRING_WIDTH_OPTS = { ambiguousIsNarrow: true };
var stringWidth = bunStringWidth ? (str) => bunStringWidth(str, BUN_STRING_WIDTH_OPTS) : stringWidthJavaScript;

// src/ink/line-width-cache.ts
var cache = /* @__PURE__ */ new Map();
var MAX_CACHE_SIZE = 4096;
function lineWidth(line) {
  const cached = cache.get(line);
  if (cached !== void 0) {
    return cached;
  }
  const width = stringWidth(line);
  if (cache.size >= MAX_CACHE_SIZE) {
    cache.clear();
  }
  cache.set(line, width);
  return width;
}

// src/ink/measure-text.ts
function measureText(text, maxWidth) {
  if (text.length === 0) {
    return {
      width: 0,
      height: 0
    };
  }
  const noWrap = maxWidth <= 0 || !Number.isFinite(maxWidth);
  let height = 0;
  let width = 0;
  let start = 0;
  while (start <= text.length) {
    const end = text.indexOf("\n", start);
    const line = end === -1 ? text.substring(start) : text.substring(start, end);
    const w = lineWidth(line);
    width = Math.max(width, w);
    if (noWrap) {
      height++;
    } else {
      height += w === 0 ? 1 : Math.ceil(w / maxWidth);
    }
    if (end === -1) {
      break;
    }
    start = end + 1;
  }
  return { width, height };
}
var measure_text_default = measureText;

// src/ink/node-cache.ts
var nodeCache = /* @__PURE__ */ new WeakMap();
var pendingClears = /* @__PURE__ */ new WeakMap();
var absoluteNodeRemoved = false;
function addPendingClear(parent, rect, isAbsolute) {
  const existing = pendingClears.get(parent);
  if (existing) {
    existing.push(rect);
  } else {
    pendingClears.set(parent, [rect]);
  }
  if (isAbsolute) {
    absoluteNodeRemoved = true;
  }
}
function consumeAbsoluteRemovedFlag() {
  const had = absoluteNodeRemoved;
  absoluteNodeRemoved = false;
  return had;
}

// src/ink/squash-text-nodes.ts
function squashTextNodesToSegments(node, inheritedStyles = {}, inheritedHyperlink, out = []) {
  const mergedStyles = node.textStyles ? { ...inheritedStyles, ...node.textStyles } : inheritedStyles;
  for (const childNode of node.childNodes) {
    if (childNode === void 0) {
      continue;
    }
    if (childNode.nodeName === "#text") {
      if (childNode.nodeValue.length > 0) {
        out.push({
          text: childNode.nodeValue,
          styles: mergedStyles,
          hyperlink: inheritedHyperlink
        });
      }
    } else if (childNode.nodeName === "ink-text" || childNode.nodeName === "ink-virtual-text") {
      squashTextNodesToSegments(childNode, mergedStyles, inheritedHyperlink, out);
    } else if (childNode.nodeName === "ink-link") {
      const href = childNode.attributes["href"];
      squashTextNodesToSegments(childNode, mergedStyles, href || inheritedHyperlink, out);
    }
  }
  return out;
}
function squashTextNodes(node) {
  let text = "";
  for (const childNode of node.childNodes) {
    if (childNode === void 0) {
      continue;
    }
    if (childNode.nodeName === "#text") {
      text += childNode.nodeValue;
    } else if (childNode.nodeName === "ink-text" || childNode.nodeName === "ink-virtual-text") {
      text += squashTextNodes(childNode);
    } else if (childNode.nodeName === "ink-link") {
      text += squashTextNodes(childNode);
    }
  }
  return text;
}
var squash_text_nodes_default = squashTextNodes;

// src/ink/tabstops.ts
var DEFAULT_TAB_INTERVAL = 8;
function expandTabs(text, interval = DEFAULT_TAB_INTERVAL) {
  if (!text.includes("	")) {
    return text;
  }
  const tokenizer = createTokenizer();
  const tokens = tokenizer.feed(text);
  tokens.push(...tokenizer.flush());
  let result = "";
  let column = 0;
  for (const token of tokens) {
    if (token.type === "sequence") {
      result += token.value;
    } else {
      const parts = token.value.split(/(\t|\n)/);
      for (const part of parts) {
        if (part === "	") {
          const spaces = interval - column % interval;
          result += " ".repeat(spaces);
          column += spaces;
        } else if (part === "\n") {
          result += part;
          column = 0;
        } else {
          result += part;
          column += stringWidth(part);
        }
      }
    }
  }
  return result;
}

// src/utils/sliceAnsi.ts
import { ansiCodesToString, reduceAnsiCodes, tokenize as tokenize2, undoAnsiCodes } from "@alcalzone/ansi-tokenize";
function isEndCode(code) {
  return code.code === code.endCode;
}
function filterStartCodes(codes) {
  return codes.filter((c) => !isEndCode(c));
}
function sliceAnsi(str, start, end) {
  const tokens = tokenize2(str);
  let activeCodes = [];
  let position = 0;
  let result = "";
  let include = false;
  for (const token of tokens) {
    const width = token.type === "ansi" ? 0 : token.fullWidth ? 2 : stringWidth(token.value);
    if (end !== void 0 && position >= end) {
      if (token.type === "ansi" || width > 0 || !include) {
        break;
      }
    }
    if (token.type === "ansi") {
      activeCodes.push(token);
      if (include) {
        result += token.code;
      }
    } else {
      if (!include && position >= start) {
        if (start > 0 && width === 0) {
          continue;
        }
        include = true;
        activeCodes = filterStartCodes(reduceAnsiCodes(activeCodes));
        result = ansiCodesToString(activeCodes);
      }
      if (include) {
        result += token.value;
      }
      position += width;
    }
  }
  const activeStartCodes = filterStartCodes(reduceAnsiCodes(activeCodes));
  result += ansiCodesToString(undoAnsiCodes(activeStartCodes));
  return result;
}

// src/ink/wrapAnsi.ts
import wrapAnsiNpm from "wrap-ansi";
var wrapAnsiBun = typeof Bun !== "undefined" && typeof Bun.wrapAnsi === "function" ? Bun.wrapAnsi : null;
var wrapAnsi = wrapAnsiBun ?? wrapAnsiNpm;

// src/ink/wrap-text.ts
var ELLIPSIS = "\u2026";
function sliceFit(text, start, end) {
  const s = sliceAnsi(text, start, end);
  return stringWidth(s) > end - start ? sliceAnsi(text, start, end - 1) : s;
}
function truncate(text, columns, position) {
  if (columns < 1) {
    return "";
  }
  if (columns === 1) {
    return ELLIPSIS;
  }
  const length = stringWidth(text);
  if (length <= columns) {
    return text;
  }
  if (position === "start") {
    return ELLIPSIS + sliceFit(text, length - columns + 1, length);
  }
  if (position === "middle") {
    const half = Math.floor(columns / 2);
    return sliceFit(text, 0, half) + ELLIPSIS + sliceFit(text, length - (columns - half) + 1, length);
  }
  return sliceFit(text, 0, columns - 1) + ELLIPSIS;
}
function wrapText(text, maxWidth, wrapType) {
  if (wrapType === "wrap") {
    return wrapAnsi(text, maxWidth, {
      trim: false,
      hard: true
    });
  }
  if (wrapType === "wrap-trim") {
    return wrapAnsi(text, maxWidth, {
      trim: true,
      hard: true
    });
  }
  if (wrapType.startsWith("truncate")) {
    let position = "end";
    if (wrapType === "truncate-middle") {
      position = "middle";
    }
    if (wrapType === "truncate-start") {
      position = "start";
    }
    return truncate(text, maxWidth, position);
  }
  return text;
}

// src/ink/dom.ts
var createNode = (nodeName) => {
  const needsYogaNode = nodeName !== "ink-virtual-text" && nodeName !== "ink-link" && nodeName !== "ink-progress";
  const node = {
    nodeName,
    style: {},
    attributes: {},
    childNodes: [],
    parentNode: void 0,
    yogaNode: needsYogaNode ? createLayoutNode() : void 0,
    dirty: false
  };
  if (nodeName === "ink-text") {
    node.yogaNode?.setMeasureFunc(measureTextNode.bind(null, node));
  } else if (nodeName === "ink-raw-ansi") {
    node.yogaNode?.setMeasureFunc(measureRawAnsiNode.bind(null, node));
  }
  return node;
};
var appendChildNode = (node, childNode) => {
  if (childNode.parentNode) {
    removeChildNode(childNode.parentNode, childNode);
  }
  childNode.parentNode = node;
  node.childNodes.push(childNode);
  if (childNode.yogaNode) {
    node.yogaNode?.insertChild(childNode.yogaNode, node.yogaNode.getChildCount());
  }
  markDirty(node);
};
var insertBeforeNode = (node, newChildNode, beforeChildNode) => {
  if (newChildNode.parentNode) {
    removeChildNode(newChildNode.parentNode, newChildNode);
  }
  newChildNode.parentNode = node;
  const index = node.childNodes.indexOf(beforeChildNode);
  if (index >= 0) {
    let yogaIndex = 0;
    if (newChildNode.yogaNode && node.yogaNode) {
      for (let i = 0; i < index; i++) {
        if (node.childNodes[i]?.yogaNode) {
          yogaIndex++;
        }
      }
    }
    node.childNodes.splice(index, 0, newChildNode);
    if (newChildNode.yogaNode && node.yogaNode) {
      node.yogaNode.insertChild(newChildNode.yogaNode, yogaIndex);
    }
    markDirty(node);
    return;
  }
  node.childNodes.push(newChildNode);
  if (newChildNode.yogaNode) {
    node.yogaNode?.insertChild(newChildNode.yogaNode, node.yogaNode.getChildCount());
  }
  markDirty(node);
};
var removeChildNode = (node, removeNode) => {
  if (removeNode.yogaNode) {
    removeNode.parentNode?.yogaNode?.removeChild(removeNode.yogaNode);
  }
  collectRemovedRects(node, removeNode);
  removeNode.parentNode = void 0;
  const index = node.childNodes.indexOf(removeNode);
  if (index >= 0) {
    node.childNodes.splice(index, 1);
  }
  markDirty(node);
};
function collectRemovedRects(parent, removed, underAbsolute = false) {
  if (removed.nodeName === "#text") {
    return;
  }
  const elem = removed;
  const isAbsolute = underAbsolute || elem.style.position === "absolute";
  const cached = nodeCache.get(elem);
  if (cached) {
    addPendingClear(parent, cached, isAbsolute);
    nodeCache.delete(elem);
  }
  for (const child of elem.childNodes) {
    collectRemovedRects(parent, child, isAbsolute);
  }
}
var setAttribute = (node, key, value) => {
  if (key === "children") {
    return;
  }
  if (node.attributes[key] === value) {
    return;
  }
  node.attributes[key] = value;
  markDirty(node);
};
var setStyle = (node, style) => {
  if (stylesEqual2(node.style, style)) {
    return;
  }
  node.style = style;
  markDirty(node);
};
var setTextStyles = (node, textStyles) => {
  if (shallowEqual(node.textStyles, textStyles)) {
    return;
  }
  node.textStyles = textStyles;
  markDirty(node);
};
function stylesEqual2(a, b) {
  return shallowEqual(a, b);
}
function shallowEqual(a, b) {
  if (a === b) {
    return true;
  }
  if (a === void 0 || b === void 0) {
    return false;
  }
  const aKeys = Object.keys(a);
  const bKeys = Object.keys(b);
  if (aKeys.length !== bKeys.length) {
    return false;
  }
  for (const key of aKeys) {
    if (a[key] !== b[key]) {
      return false;
    }
  }
  return true;
}
var createTextNode = (text) => {
  const node = {
    nodeName: "#text",
    nodeValue: text,
    yogaNode: void 0,
    parentNode: void 0,
    style: {}
  };
  setTextNodeValue(node, text);
  return node;
};
var measureTextNode = function(node, width, widthMode) {
  const rawText = node.nodeName === "#text" ? node.nodeValue : squash_text_nodes_default(node);
  const text = expandTabs(rawText);
  const dimensions = measure_text_default(text, width);
  if (dimensions.width <= width) {
    return dimensions;
  }
  if (dimensions.width >= 1 && width > 0 && width < 1) {
    return dimensions;
  }
  if (text.includes("\n") && widthMode === LayoutMeasureMode.Undefined) {
    const effectiveWidth = Math.max(width, dimensions.width);
    return measure_text_default(text, effectiveWidth);
  }
  const textWrap = node.style?.textWrap ?? "wrap";
  const wrappedText = wrapText(text, width, textWrap);
  return measure_text_default(wrappedText, width);
};
var measureRawAnsiNode = function(node) {
  return {
    width: node.attributes["rawWidth"],
    height: node.attributes["rawHeight"]
  };
};
var markDirty = (node) => {
  let current = node;
  let markedYoga = false;
  while (current) {
    if (current.nodeName !== "#text") {
      ;
      current.dirty = true;
      if (!markedYoga && (current.nodeName === "ink-text" || current.nodeName === "ink-raw-ansi") && current.yogaNode) {
        current.yogaNode.markDirty();
        markedYoga = true;
      }
    }
    current = current.parentNode;
  }
};
var scheduleRenderFrom = (node) => {
  let cur = node;
  while (cur?.parentNode) {
    cur = cur.parentNode;
  }
  if (cur && cur.nodeName !== "#text") {
    ;
    cur.onRender?.();
  }
};
var setTextNodeValue = (node, text) => {
  if (typeof text !== "string") {
    text = String(text);
  }
  if (node.nodeValue === text) {
    return;
  }
  node.nodeValue = text;
  markDirty(node);
};
var clearYogaNodeReferences = (node) => {
  if ("childNodes" in node) {
    for (const child of node.childNodes) {
      clearYogaNodeReferences(child);
    }
  }
  node.yogaNode = void 0;
};

// src/ink/reconciler.ts
import createReconciler from "react-reconciler";

// src/ink/events/dispatcher.ts
import {
  ContinuousEventPriority,
  DefaultEventPriority,
  DiscreteEventPriority,
  NoEventPriority
} from "react-reconciler/constants.js";

// src/utils/log.ts
function logError(error) {
  if (!process.env.HERMES_INK_DEBUG_ERRORS) {
    return;
  }
  console.error(error);
}

// src/ink/events/event-handlers.ts
var HANDLER_FOR_EVENT = {
  keydown: { bubble: "onKeyDown", capture: "onKeyDownCapture" },
  focus: { bubble: "onFocus", capture: "onFocusCapture" },
  blur: { bubble: "onBlur", capture: "onBlurCapture" },
  paste: { bubble: "onPaste", capture: "onPasteCapture" },
  resize: { bubble: "onResize" },
  click: { bubble: "onClick" },
  mousedown: { bubble: "onMouseDown" },
  mouseup: { bubble: "onMouseUp" },
  mousedrag: { bubble: "onMouseDrag" }
};
var EVENT_HANDLER_PROPS = /* @__PURE__ */ new Set([
  "onKeyDown",
  "onKeyDownCapture",
  "onFocus",
  "onFocusCapture",
  "onBlur",
  "onBlurCapture",
  "onPaste",
  "onPasteCapture",
  "onResize",
  "onClick",
  "onMouseDown",
  "onMouseUp",
  "onMouseDrag",
  "onMouseEnter",
  "onMouseLeave"
]);

// src/ink/events/dispatcher.ts
function getHandler(node, eventType, capture) {
  const handlers = node._eventHandlers;
  if (!handlers) {
    return void 0;
  }
  const mapping = HANDLER_FOR_EVENT[eventType];
  if (!mapping) {
    return void 0;
  }
  const propName = capture ? mapping.capture : mapping.bubble;
  if (!propName) {
    return void 0;
  }
  return handlers[propName];
}
function collectListeners(target, event) {
  const listeners = [];
  let node = target;
  while (node) {
    const isTarget = node === target;
    const captureHandler = getHandler(node, event.type, true);
    const bubbleHandler = getHandler(node, event.type, false);
    if (captureHandler) {
      listeners.unshift({
        node,
        handler: captureHandler,
        phase: isTarget ? "at_target" : "capturing"
      });
    }
    if (bubbleHandler && (event.bubbles || isTarget)) {
      listeners.push({
        node,
        handler: bubbleHandler,
        phase: isTarget ? "at_target" : "bubbling"
      });
    }
    node = node.parentNode;
  }
  return listeners;
}
function processDispatchQueue(listeners, event) {
  let previousNode;
  for (const { node, handler, phase } of listeners) {
    if (event._isImmediatePropagationStopped()) {
      break;
    }
    if (event._isPropagationStopped() && node !== previousNode) {
      break;
    }
    event._setEventPhase(phase);
    event._setCurrentTarget(node);
    event._prepareForTarget(node);
    try {
      handler(event);
    } catch (error) {
      logError(error);
    }
    previousNode = node;
  }
}
function getEventPriority(eventType) {
  switch (eventType) {
    case "keydown":
    case "keyup":
    case "click":
    case "focus":
    case "blur":
    case "paste":
      return DiscreteEventPriority;
    case "resize":
    case "scroll":
    case "mousemove":
      return ContinuousEventPriority;
    default:
      return DefaultEventPriority;
  }
}
var Dispatcher = class {
  currentEvent = null;
  currentUpdatePriority = DefaultEventPriority;
  discreteUpdates = null;
  /**
   * Infer event priority from the currently-dispatching event.
   * Called by the reconciler host config's resolveUpdatePriority
   * when no explicit priority has been set.
   */
  resolveEventPriority() {
    if (this.currentUpdatePriority !== NoEventPriority) {
      return this.currentUpdatePriority;
    }
    if (this.currentEvent) {
      return getEventPriority(this.currentEvent.type);
    }
    return DefaultEventPriority;
  }
  /**
   * Dispatch an event through capture and bubble phases.
   * Returns true if preventDefault() was NOT called.
   */
  dispatch(target, event) {
    const previousEvent = this.currentEvent;
    this.currentEvent = event;
    try {
      event._setTarget(target);
      const listeners = collectListeners(target, event);
      processDispatchQueue(listeners, event);
      event._setEventPhase("none");
      event._setCurrentTarget(null);
      return !event.defaultPrevented;
    } finally {
      this.currentEvent = previousEvent;
    }
  }
  /**
   * Dispatch with discrete (sync) priority.
   * For user-initiated events: keyboard, click, focus, paste.
   */
  dispatchDiscrete(target, event) {
    if (!this.discreteUpdates) {
      return this.dispatch(target, event);
    }
    return this.discreteUpdates((t, e) => this.dispatch(t, e), target, event, void 0, void 0);
  }
  /**
   * Dispatch with continuous priority.
   * For high-frequency events: resize, scroll, mouse move.
   */
  dispatchContinuous(target, event) {
    const previousPriority = this.currentUpdatePriority;
    try {
      this.currentUpdatePriority = ContinuousEventPriority;
      return this.dispatch(target, event);
    } finally {
      this.currentUpdatePriority = previousPriority;
    }
  }
};

// src/ink/events/event.ts
var Event = class {
  _didStopImmediatePropagation = false;
  didStopImmediatePropagation() {
    return this._didStopImmediatePropagation;
  }
  stopImmediatePropagation() {
    this._didStopImmediatePropagation = true;
  }
};

// src/ink/events/terminal-event.ts
var TerminalEvent = class extends Event {
  type;
  timeStamp;
  bubbles;
  cancelable;
  _target = null;
  _currentTarget = null;
  _eventPhase = "none";
  _propagationStopped = false;
  _defaultPrevented = false;
  constructor(type, init) {
    super();
    this.type = type;
    this.timeStamp = performance.now();
    this.bubbles = init?.bubbles ?? true;
    this.cancelable = init?.cancelable ?? true;
  }
  get target() {
    return this._target;
  }
  get currentTarget() {
    return this._currentTarget;
  }
  get eventPhase() {
    return this._eventPhase;
  }
  get defaultPrevented() {
    return this._defaultPrevented;
  }
  stopPropagation() {
    this._propagationStopped = true;
  }
  stopImmediatePropagation() {
    super.stopImmediatePropagation();
    this._propagationStopped = true;
  }
  preventDefault() {
    if (this.cancelable) {
      this._defaultPrevented = true;
    }
  }
  // -- Internal setters used by the Dispatcher
  /** @internal */
  _setTarget(target) {
    this._target = target;
  }
  /** @internal */
  _setCurrentTarget(target) {
    this._currentTarget = target;
  }
  /** @internal */
  _setEventPhase(phase) {
    this._eventPhase = phase;
  }
  /** @internal */
  _isPropagationStopped() {
    return this._propagationStopped;
  }
  /** @internal */
  _isImmediatePropagationStopped() {
    return this.didStopImmediatePropagation();
  }
  /**
   * Hook for subclasses to do per-node setup before each handler fires.
   * Default is a no-op.
   */
  _prepareForTarget(_target) {
  }
};

// src/ink/events/focus-event.ts
var FocusEvent = class extends TerminalEvent {
  relatedTarget;
  constructor(type, relatedTarget = null) {
    super(type, { bubbles: true, cancelable: false });
    this.relatedTarget = relatedTarget;
  }
};

// src/ink/focus.ts
var MAX_FOCUS_STACK = 32;
var FocusManager = class {
  activeElement = null;
  dispatchFocusEvent;
  enabled = true;
  focusStack = [];
  constructor(dispatchFocusEvent) {
    this.dispatchFocusEvent = dispatchFocusEvent;
  }
  focus(node) {
    if (node === this.activeElement) {
      return;
    }
    if (!this.enabled) {
      return;
    }
    const previous = this.activeElement;
    if (previous) {
      const idx = this.focusStack.indexOf(previous);
      if (idx !== -1) {
        this.focusStack.splice(idx, 1);
      }
      this.focusStack.push(previous);
      if (this.focusStack.length > MAX_FOCUS_STACK) {
        this.focusStack.shift();
      }
      this.dispatchFocusEvent(previous, new FocusEvent("blur", node));
    }
    this.activeElement = node;
    this.dispatchFocusEvent(node, new FocusEvent("focus", previous));
  }
  blur() {
    if (!this.activeElement) {
      return;
    }
    const previous = this.activeElement;
    this.activeElement = null;
    this.dispatchFocusEvent(previous, new FocusEvent("blur", null));
  }
  /**
   * Called by the reconciler when a node is removed from the tree.
   * Handles both the exact node and any focused descendant within
   * the removed subtree. Dispatches blur and restores focus from stack.
   */
  handleNodeRemoved(node, root) {
    this.focusStack = this.focusStack.filter((n) => n !== node && isInTree(n, root));
    if (!this.activeElement) {
      return;
    }
    if (this.activeElement !== node && isInTree(this.activeElement, root)) {
      return;
    }
    const removed = this.activeElement;
    this.activeElement = null;
    this.dispatchFocusEvent(removed, new FocusEvent("blur", null));
    while (this.focusStack.length > 0) {
      const candidate = this.focusStack.pop();
      if (isInTree(candidate, root)) {
        this.activeElement = candidate;
        this.dispatchFocusEvent(candidate, new FocusEvent("focus", removed));
        return;
      }
    }
  }
  handleAutoFocus(node) {
    this.focus(node);
  }
  handleClickFocus(node) {
    const tabIndex = node.attributes["tabIndex"];
    if (typeof tabIndex !== "number") {
      return;
    }
    this.focus(node);
  }
  enable() {
    this.enabled = true;
  }
  disable() {
    this.enabled = false;
  }
  focusNext(root) {
    this.moveFocus(1, root);
  }
  focusPrevious(root) {
    this.moveFocus(-1, root);
  }
  moveFocus(direction, root) {
    if (!this.enabled) {
      return;
    }
    const tabbable = collectTabbable(root);
    if (tabbable.length === 0) {
      return;
    }
    const currentIndex = this.activeElement ? tabbable.indexOf(this.activeElement) : -1;
    const nextIndex = currentIndex === -1 ? direction === 1 ? 0 : tabbable.length - 1 : (currentIndex + direction + tabbable.length) % tabbable.length;
    const next = tabbable[nextIndex];
    if (next) {
      this.focus(next);
    }
  }
};
function collectTabbable(root) {
  const result = [];
  walkTree(root, result);
  return result;
}
function walkTree(node, result) {
  const tabIndex = node.attributes["tabIndex"];
  if (typeof tabIndex === "number" && tabIndex >= 0) {
    result.push(node);
  }
  for (const child of node.childNodes) {
    if (child.nodeName !== "#text") {
      walkTree(child, result);
    }
  }
}
function isInTree(node, root) {
  let current = node;
  while (current) {
    if (current === root) {
      return true;
    }
    current = current.parentNode;
  }
  return false;
}
function getRootNode(node) {
  let current = node;
  while (current) {
    if (current.focusManager) {
      return current;
    }
    current = current.parentNode;
  }
  throw new Error("Node is not in a tree with a FocusManager");
}
function getFocusManager(node) {
  return getRootNode(node).focusManager;
}

// src/ink/styles.ts
var applyPositionStyles = (node, style) => {
  if ("position" in style) {
    node.setPositionType(style.position === "absolute" ? LayoutPositionType.Absolute : LayoutPositionType.Relative);
  }
  if ("top" in style) {
    applyPositionEdge(node, "top", style.top);
  }
  if ("bottom" in style) {
    applyPositionEdge(node, "bottom", style.bottom);
  }
  if ("left" in style) {
    applyPositionEdge(node, "left", style.left);
  }
  if ("right" in style) {
    applyPositionEdge(node, "right", style.right);
  }
};
function applyPositionEdge(node, edge, v) {
  if (typeof v === "string") {
    node.setPositionPercent(edge, Number.parseInt(v, 10));
  } else if (typeof v === "number") {
    node.setPosition(edge, v);
  } else {
    node.setPosition(edge, Number.NaN);
  }
}
var applyOverflowStyles = (node, style) => {
  const y = style.overflowY ?? style.overflow;
  const x = style.overflowX ?? style.overflow;
  if (y === "scroll" || x === "scroll") {
    node.setOverflow(LayoutOverflow.Scroll);
  } else if (y === "hidden" || x === "hidden") {
    node.setOverflow(LayoutOverflow.Hidden);
  } else if ("overflow" in style || "overflowX" in style || "overflowY" in style) {
    node.setOverflow(LayoutOverflow.Visible);
  }
};
var applyMarginStyles = (node, style) => {
  if ("margin" in style) {
    node.setMargin(LayoutEdge.All, style.margin ?? 0);
  }
  if ("marginX" in style) {
    node.setMargin(LayoutEdge.Horizontal, style.marginX ?? 0);
  }
  if ("marginY" in style) {
    node.setMargin(LayoutEdge.Vertical, style.marginY ?? 0);
  }
  if ("marginLeft" in style) {
    node.setMargin(LayoutEdge.Start, style.marginLeft || 0);
  }
  if ("marginRight" in style) {
    node.setMargin(LayoutEdge.End, style.marginRight || 0);
  }
  if ("marginTop" in style) {
    node.setMargin(LayoutEdge.Top, style.marginTop || 0);
  }
  if ("marginBottom" in style) {
    node.setMargin(LayoutEdge.Bottom, style.marginBottom || 0);
  }
};
var applyPaddingStyles = (node, style) => {
  if ("padding" in style) {
    node.setPadding(LayoutEdge.All, style.padding ?? 0);
  }
  if ("paddingX" in style) {
    node.setPadding(LayoutEdge.Horizontal, style.paddingX ?? 0);
  }
  if ("paddingY" in style) {
    node.setPadding(LayoutEdge.Vertical, style.paddingY ?? 0);
  }
  if ("paddingLeft" in style) {
    node.setPadding(LayoutEdge.Left, style.paddingLeft || 0);
  }
  if ("paddingRight" in style) {
    node.setPadding(LayoutEdge.Right, style.paddingRight || 0);
  }
  if ("paddingTop" in style) {
    node.setPadding(LayoutEdge.Top, style.paddingTop || 0);
  }
  if ("paddingBottom" in style) {
    node.setPadding(LayoutEdge.Bottom, style.paddingBottom || 0);
  }
};
var applyFlexStyles = (node, style) => {
  if ("flexGrow" in style) {
    node.setFlexGrow(style.flexGrow ?? 0);
  }
  if ("flexShrink" in style) {
    node.setFlexShrink(typeof style.flexShrink === "number" ? style.flexShrink : 1);
  }
  if ("flexWrap" in style) {
    if (style.flexWrap === "nowrap") {
      node.setFlexWrap(LayoutWrap.NoWrap);
    }
    if (style.flexWrap === "wrap") {
      node.setFlexWrap(LayoutWrap.Wrap);
    }
    if (style.flexWrap === "wrap-reverse") {
      node.setFlexWrap(LayoutWrap.WrapReverse);
    }
  }
  if ("flexDirection" in style) {
    if (style.flexDirection === "row") {
      node.setFlexDirection(LayoutFlexDirection.Row);
    }
    if (style.flexDirection === "row-reverse") {
      node.setFlexDirection(LayoutFlexDirection.RowReverse);
    }
    if (style.flexDirection === "column") {
      node.setFlexDirection(LayoutFlexDirection.Column);
    }
    if (style.flexDirection === "column-reverse") {
      node.setFlexDirection(LayoutFlexDirection.ColumnReverse);
    }
  }
  if ("flexBasis" in style) {
    if (typeof style.flexBasis === "number") {
      node.setFlexBasis(style.flexBasis);
    } else if (typeof style.flexBasis === "string") {
      node.setFlexBasisPercent(Number.parseInt(style.flexBasis, 10));
    } else {
      node.setFlexBasis(Number.NaN);
    }
  }
  if ("alignItems" in style) {
    if (style.alignItems === "stretch" || !style.alignItems) {
      node.setAlignItems(LayoutAlign.Stretch);
    }
    if (style.alignItems === "flex-start") {
      node.setAlignItems(LayoutAlign.FlexStart);
    }
    if (style.alignItems === "center") {
      node.setAlignItems(LayoutAlign.Center);
    }
    if (style.alignItems === "flex-end") {
      node.setAlignItems(LayoutAlign.FlexEnd);
    }
  }
  if ("alignSelf" in style) {
    if (style.alignSelf === "auto" || !style.alignSelf) {
      node.setAlignSelf(LayoutAlign.Auto);
    }
    if (style.alignSelf === "flex-start") {
      node.setAlignSelf(LayoutAlign.FlexStart);
    }
    if (style.alignSelf === "center") {
      node.setAlignSelf(LayoutAlign.Center);
    }
    if (style.alignSelf === "flex-end") {
      node.setAlignSelf(LayoutAlign.FlexEnd);
    }
  }
  if ("justifyContent" in style) {
    if (style.justifyContent === "flex-start" || !style.justifyContent) {
      node.setJustifyContent(LayoutJustify.FlexStart);
    }
    if (style.justifyContent === "center") {
      node.setJustifyContent(LayoutJustify.Center);
    }
    if (style.justifyContent === "flex-end") {
      node.setJustifyContent(LayoutJustify.FlexEnd);
    }
    if (style.justifyContent === "space-between") {
      node.setJustifyContent(LayoutJustify.SpaceBetween);
    }
    if (style.justifyContent === "space-around") {
      node.setJustifyContent(LayoutJustify.SpaceAround);
    }
    if (style.justifyContent === "space-evenly") {
      node.setJustifyContent(LayoutJustify.SpaceEvenly);
    }
  }
};
var applyDimensionStyles = (node, style) => {
  if ("width" in style) {
    if (typeof style.width === "number") {
      node.setWidth(style.width);
    } else if (typeof style.width === "string") {
      node.setWidthPercent(Number.parseInt(style.width, 10));
    } else {
      node.setWidthAuto();
    }
  }
  if ("height" in style) {
    if (typeof style.height === "number") {
      node.setHeight(style.height);
    } else if (typeof style.height === "string") {
      node.setHeightPercent(Number.parseInt(style.height, 10));
    } else {
      node.setHeightAuto();
    }
  }
  if ("minWidth" in style) {
    if (typeof style.minWidth === "string") {
      node.setMinWidthPercent(Number.parseInt(style.minWidth, 10));
    } else {
      node.setMinWidth(style.minWidth ?? 0);
    }
  }
  if ("minHeight" in style) {
    if (typeof style.minHeight === "string") {
      node.setMinHeightPercent(Number.parseInt(style.minHeight, 10));
    } else {
      node.setMinHeight(style.minHeight ?? 0);
    }
  }
  if ("maxWidth" in style) {
    if (typeof style.maxWidth === "string") {
      node.setMaxWidthPercent(Number.parseInt(style.maxWidth, 10));
    } else {
      node.setMaxWidth(style.maxWidth ?? 0);
    }
  }
  if ("maxHeight" in style) {
    if (typeof style.maxHeight === "string") {
      node.setMaxHeightPercent(Number.parseInt(style.maxHeight, 10));
    } else {
      node.setMaxHeight(style.maxHeight ?? 0);
    }
  }
};
var applyDisplayStyles = (node, style) => {
  if ("display" in style) {
    node.setDisplay(style.display === "flex" ? LayoutDisplay.Flex : LayoutDisplay.None);
  }
};
var applyBorderStyles = (node, style, resolvedStyle) => {
  const resolved = resolvedStyle ?? style;
  if ("borderStyle" in style) {
    const borderWidth = style.borderStyle ? 1 : 0;
    node.setBorder(LayoutEdge.Top, resolved.borderTop !== false ? borderWidth : 0);
    node.setBorder(LayoutEdge.Bottom, resolved.borderBottom !== false ? borderWidth : 0);
    node.setBorder(LayoutEdge.Left, resolved.borderLeft !== false ? borderWidth : 0);
    node.setBorder(LayoutEdge.Right, resolved.borderRight !== false ? borderWidth : 0);
  } else {
    if ("borderTop" in style && style.borderTop !== void 0) {
      node.setBorder(LayoutEdge.Top, style.borderTop === false ? 0 : 1);
    }
    if ("borderBottom" in style && style.borderBottom !== void 0) {
      node.setBorder(LayoutEdge.Bottom, style.borderBottom === false ? 0 : 1);
    }
    if ("borderLeft" in style && style.borderLeft !== void 0) {
      node.setBorder(LayoutEdge.Left, style.borderLeft === false ? 0 : 1);
    }
    if ("borderRight" in style && style.borderRight !== void 0) {
      node.setBorder(LayoutEdge.Right, style.borderRight === false ? 0 : 1);
    }
  }
};
var applyGapStyles = (node, style) => {
  if ("gap" in style) {
    node.setGap(LayoutGutter.All, style.gap ?? 0);
  }
  if ("columnGap" in style) {
    node.setGap(LayoutGutter.Column, style.columnGap ?? 0);
  }
  if ("rowGap" in style) {
    node.setGap(LayoutGutter.Row, style.rowGap ?? 0);
  }
};
var styles = (node, style = {}, resolvedStyle) => {
  applyPositionStyles(node, style);
  applyOverflowStyles(node, style);
  applyMarginStyles(node, style);
  applyPaddingStyles(node, style);
  applyFlexStyles(node, style);
  applyDimensionStyles(node, style);
  applyDisplayStyles(node, style);
  applyBorderStyles(node, style, resolvedStyle);
  applyGapStyles(node, style);
};
var styles_default = styles;

// src/ink/reconciler.ts
if (process.env.NODE_ENV === "development") {
  try {
    void Promise.resolve().then(() => (init_devtools(), devtools_exports));
  } catch (error) {
    if (error.code === "ERR_MODULE_NOT_FOUND") {
      console.warn(
        `
The environment variable DEV is set to true, so Ink tried to import \`react-devtools-core\`,
but this failed as it was not installed. Debugging with React Devtools requires it.

To install use this command:

$ npm install --save-dev react-devtools-core
				`.trim() + "\n"
      );
    } else {
      throw error;
    }
  }
}
var diff = (before, after) => {
  if (before === after) {
    return;
  }
  if (!before) {
    return after;
  }
  const changed = {};
  let isChanged = false;
  for (const key of Object.keys(before)) {
    const isDeleted = after ? !Object.hasOwn(after, key) : true;
    if (isDeleted) {
      changed[key] = void 0;
      isChanged = true;
    }
  }
  if (after) {
    for (const key of Object.keys(after)) {
      if (after[key] !== before[key]) {
        changed[key] = after[key];
        isChanged = true;
      }
    }
  }
  return isChanged ? changed : void 0;
};
var cleanupYogaNode = (node) => {
  const yogaNode = node.yogaNode;
  if (yogaNode) {
    yogaNode.unsetMeasureFunc();
    clearYogaNodeReferences(node);
    yogaNode.freeRecursive();
  }
};
function setEventHandler(node, key, value) {
  if (!node._eventHandlers) {
    node._eventHandlers = {};
  }
  node._eventHandlers[key] = value;
}
function applyProp(node, key, value) {
  if (key === "children") {
    return;
  }
  if (key === "style") {
    setStyle(node, value);
    if (node.yogaNode) {
      styles_default(node.yogaNode, value);
    }
    return;
  }
  if (key === "textStyles") {
    node.textStyles = value;
    return;
  }
  if (EVENT_HANDLER_PROPS.has(key)) {
    setEventHandler(node, key, value);
    return;
  }
  setAttribute(node, key, value);
}
var dispatcher = new Dispatcher();
var _lastYogaMs = 0;
var _lastCommitMs = 0;
var _commitStart = 0;
function recordYogaMs(ms) {
  _lastYogaMs = ms;
}
function getLastYogaMs() {
  return _lastYogaMs;
}
function markCommitStart() {
  _commitStart = performance.now();
}
function getLastCommitMs() {
  return _lastCommitMs;
}
function resetProfileCounters() {
  _lastYogaMs = 0;
  _lastCommitMs = 0;
  _commitStart = 0;
}
var reconciler = createReconciler({
  getRootHostContext: () => ({ isInsideText: false }),
  prepareForCommit: () => null,
  preparePortalMount: () => null,
  clearContainer: () => false,
  resetAfterCommit(rootNode) {
    _lastCommitMs = _commitStart > 0 ? performance.now() - _commitStart : 0;
    _commitStart = 0;
    if (typeof rootNode.onComputeLayout === "function") {
      rootNode.onComputeLayout();
    }
    if (process.env.NODE_ENV === "test") {
      if (rootNode.childNodes.length === 0 && rootNode.hasRenderedContent) {
        return;
      }
      if (rootNode.childNodes.length > 0) {
        rootNode.hasRenderedContent = true;
      }
      rootNode.onImmediateRender?.();
      return;
    }
    rootNode.onRender?.();
  },
  getChildHostContext(parentHostContext, type) {
    const previousIsInsideText = parentHostContext.isInsideText;
    const isInsideText = type === "ink-text" || type === "ink-virtual-text" || type === "ink-link";
    if (previousIsInsideText === isInsideText) {
      return parentHostContext;
    }
    return { isInsideText };
  },
  shouldSetTextContent: () => false,
  createInstance(originalType, newProps, _root, hostContext, _internalHandle) {
    if (hostContext.isInsideText && originalType === "ink-box") {
      throw new Error(`<Box> can't be nested inside <Text> component`);
    }
    const type = originalType === "ink-text" && hostContext.isInsideText ? "ink-virtual-text" : originalType;
    const node = createNode(type);
    for (const [key, value] of Object.entries(newProps)) {
      applyProp(node, key, value);
    }
    return node;
  },
  createTextInstance(text, _root, hostContext) {
    if (!hostContext.isInsideText) {
      throw new Error(`Text string "${text}" must be rendered inside <Text> component`);
    }
    return createTextNode(text);
  },
  resetTextContent() {
  },
  hideTextInstance(node) {
    setTextNodeValue(node, "");
  },
  unhideTextInstance(node, text) {
    setTextNodeValue(node, text);
  },
  getPublicInstance: (instance) => instance,
  hideInstance(node) {
    node.isHidden = true;
    node.yogaNode?.setDisplay(LayoutDisplay.None);
    markDirty(node);
  },
  unhideInstance(node) {
    node.isHidden = false;
    node.yogaNode?.setDisplay(LayoutDisplay.Flex);
    markDirty(node);
  },
  appendInitialChild: appendChildNode,
  appendChild: appendChildNode,
  insertBefore: insertBeforeNode,
  finalizeInitialChildren(_node, _type, props) {
    return props["autoFocus"] === true;
  },
  commitMount(node) {
    getFocusManager(node).handleAutoFocus(node);
  },
  isPrimaryRenderer: true,
  supportsMutation: true,
  supportsPersistence: false,
  supportsHydration: false,
  scheduleTimeout: setTimeout,
  cancelTimeout: clearTimeout,
  noTimeout: -1,
  getCurrentUpdatePriority: () => dispatcher.currentUpdatePriority,
  beforeActiveInstanceBlur() {
  },
  afterActiveInstanceBlur() {
  },
  detachDeletedInstance() {
  },
  getInstanceFromNode: () => null,
  prepareScopeUpdate() {
  },
  getInstanceFromScope: () => null,
  appendChildToContainer: appendChildNode,
  insertInContainerBefore: insertBeforeNode,
  removeChildFromContainer(node, removeNode) {
    removeChildNode(node, removeNode);
    cleanupYogaNode(removeNode);
    getFocusManager(node).handleNodeRemoved(removeNode, node);
  },
  // React 19 commitUpdate receives old and new props directly instead of an updatePayload
  commitUpdate(node, _type, oldProps, newProps) {
    const props = diff(oldProps, newProps);
    const style = diff(oldProps["style"], newProps["style"]);
    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (key === "style") {
          setStyle(node, value);
          continue;
        }
        if (key === "textStyles") {
          setTextStyles(node, value);
          continue;
        }
        if (EVENT_HANDLER_PROPS.has(key)) {
          setEventHandler(node, key, value);
          continue;
        }
        setAttribute(node, key, value);
      }
    }
    if (style && node.yogaNode) {
      styles_default(node.yogaNode, style, newProps["style"]);
    }
  },
  commitTextUpdate(node, _oldText, newText) {
    setTextNodeValue(node, newText);
  },
  removeChild(node, removeNode) {
    removeChildNode(node, removeNode);
    cleanupYogaNode(removeNode);
    if (removeNode.nodeName !== "#text") {
      const root = getRootNode(node);
      root.focusManager.handleNodeRemoved(removeNode, root);
    }
  },
  // React 19 required methods
  maySuspendCommit() {
    return false;
  },
  preloadInstance() {
    return true;
  },
  startSuspendingCommit() {
  },
  suspendInstance() {
  },
  waitForCommitToBeReady() {
    return null;
  },
  NotPendingTransition: null,
  HostTransitionContext: {
    $$typeof: Symbol.for("react.context"),
    _currentValue: null
  },
  setCurrentUpdatePriority(newPriority) {
    dispatcher.currentUpdatePriority = newPriority;
  },
  resolveUpdatePriority() {
    return dispatcher.resolveEventPriority();
  },
  resetFormInstance() {
  },
  requestPostPaintCallback() {
  },
  shouldAttemptEagerTransition() {
    return false;
  },
  trackSchedulerEvent() {
  },
  resolveEventType() {
    return dispatcher.currentEvent?.type ?? null;
  },
  resolveEventTimeStamp() {
    return dispatcher.currentEvent?.timeStamp ?? -1.1;
  }
});
dispatcher.discreteUpdates = reconciler.discreteUpdates.bind(reconciler);
var reconciler_default = reconciler;

// src/ink/components/ScrollBox.tsx
import { jsx as jsx9 } from "react/jsx-runtime";
function ScrollBox({ children, ref, stickyScroll, ...style }) {
  const domRef = useRef(null);
  const [, forceRender] = useState(0);
  const listenersRef = useRef(/* @__PURE__ */ new Set());
  const renderQueuedRef = useRef(false);
  const notify = () => {
    for (const l of listenersRef.current) {
      l();
    }
  };
  function scrollMutated(el) {
    markScrollActivity();
    markDirty(el);
    markCommitStart();
    notify();
    if (renderQueuedRef.current) {
      return;
    }
    renderQueuedRef.current = true;
    queueMicrotask(() => {
      renderQueuedRef.current = false;
      scheduleRenderFrom(el);
    });
  }
  useImperativeHandle(
    ref,
    () => ({
      scrollTo(y) {
        const el = domRef.current;
        if (!el) {
          return;
        }
        el.stickyScroll = false;
        el.pendingScrollDelta = void 0;
        el.scrollAnchor = void 0;
        el.scrollTop = Math.max(0, Math.floor(y));
        scrollMutated(el);
      },
      scrollToElement(el, offset = 0) {
        const box = domRef.current;
        if (!box) {
          return;
        }
        box.stickyScroll = false;
        box.pendingScrollDelta = void 0;
        box.scrollAnchor = {
          el,
          offset
        };
        scrollMutated(box);
      },
      scrollBy(dy) {
        const el = domRef.current;
        if (!el) {
          return;
        }
        el.stickyScroll = false;
        el.scrollAnchor = void 0;
        el.pendingScrollDelta = (el.pendingScrollDelta ?? 0) + Math.floor(dy);
        scrollMutated(el);
      },
      scrollToBottom() {
        const el = domRef.current;
        if (!el) {
          return;
        }
        el.pendingScrollDelta = void 0;
        el.stickyScroll = true;
        markDirty(el);
        notify();
        forceRender((n) => n + 1);
      },
      getScrollTop() {
        return domRef.current?.scrollTop ?? 0;
      },
      getPendingDelta() {
        return domRef.current?.pendingScrollDelta ?? 0;
      },
      getScrollHeight() {
        return domRef.current?.scrollHeight ?? 0;
      },
      getFreshScrollHeight() {
        const content = domRef.current?.childNodes[0];
        return content?.yogaNode?.getComputedHeight() ?? domRef.current?.scrollHeight ?? 0;
      },
      getViewportHeight() {
        return domRef.current?.scrollViewportHeight ?? 0;
      },
      getViewportTop() {
        return domRef.current?.scrollViewportTop ?? 0;
      },
      isSticky() {
        const el = domRef.current;
        if (!el) {
          return false;
        }
        return el.stickyScroll ?? Boolean(el.attributes["stickyScroll"]);
      },
      subscribe(listener) {
        listenersRef.current.add(listener);
        return () => listenersRef.current.delete(listener);
      },
      setClampBounds(min, max) {
        const el = domRef.current;
        if (!el) {
          return;
        }
        el.scrollClampMin = min;
        el.scrollClampMax = max;
      }
    }),
    // notify/scrollMutated are inline (no useCallback) but only close over
    // refs + imports — stable. Empty deps avoids rebuilding the handle on
    // every render (which re-registers the ref = churn).
    []
  );
  return /* @__PURE__ */ jsx9(
    "ink-box",
    {
      ref: (el) => {
        domRef.current = el;
        if (el) {
          el.scrollTop ??= 0;
        }
      },
      style: {
        flexWrap: "nowrap",
        flexDirection: style.flexDirection ?? "row",
        flexGrow: style.flexGrow ?? 0,
        flexShrink: style.flexShrink ?? 1,
        ...style,
        overflowX: "scroll",
        overflowY: "scroll"
      },
      ...stickyScroll ? {
        stickyScroll: true
      } : {},
      children: /* @__PURE__ */ jsx9(Box_default, { flexDirection: "column", flexGrow: 1, flexShrink: 0, width: "100%", children })
    }
  );
}
var ScrollBox_default = ScrollBox;

// src/ink/components/Spacer.tsx
import { c as _c9 } from "react/compiler-runtime";
import { jsx as jsx10 } from "react/jsx-runtime";
function Spacer() {
  const $ = _c9(1);
  let t0;
  if ($[0] === Symbol.for("react.memo_cache_sentinel")) {
    t0 = /* @__PURE__ */ jsx10(Box_default, { flexGrow: 1 });
    $[0] = t0;
  } else {
    t0 = $[0];
  }
  return t0;
}

// src/ink/hooks/use-app.ts
import { useContext as useContext3 } from "react";

// src/ink/components/AppContext.ts
import { createContext as createContext3 } from "react";
var AppContext = createContext3({
  exit() {
  }
});
AppContext.displayName = "InternalAppContext";
var AppContext_default = AppContext;

// src/ink/hooks/use-app.ts
var useApp = () => useContext3(AppContext_default);
var use_app_default = useApp;

// src/ink/hooks/use-declared-cursor.ts
import { useCallback as useCallback2, useContext as useContext4, useLayoutEffect, useRef as useRef2 } from "react";

// src/ink/components/CursorDeclarationContext.ts
import { createContext as createContext4 } from "react";
var CursorDeclarationContext = createContext4(() => {
});
var CursorDeclarationContext_default = CursorDeclarationContext;

// src/ink/hooks/use-declared-cursor.ts
function useDeclaredCursor({
  line,
  column,
  active
}) {
  const setCursorDeclaration = useContext4(CursorDeclarationContext_default);
  const nodeRef = useRef2(null);
  const setNode = useCallback2((node) => {
    nodeRef.current = node;
  }, []);
  useLayoutEffect(() => {
    const node = nodeRef.current;
    if (active && node) {
      setCursorDeclaration({ relativeX: column, relativeY: line, node });
    } else {
      setCursorDeclaration(null, node);
    }
  });
  useLayoutEffect(() => {
    return () => {
      setCursorDeclaration(null, nodeRef.current);
    };
  }, [setCursorDeclaration]);
  return setNode;
}

// src/ink/hooks/use-external-process.ts
import { useCallback as useCallback3 } from "react";
async function withInkSuspended(run) {
  const ink = instances_default.get(process.stdout);
  if (!ink) {
    await run();
    return;
  }
  ink.enterAlternateScreen();
  try {
    await run();
  } finally {
    ink.exitAlternateScreen();
  }
}
function useExternalProcess() {
  return useCallback3((run) => withInkSuspended(run), []);
}

// src/ink/hooks/use-input.ts
import { useEffect, useLayoutEffect as useLayoutEffect2 } from "react";
import { useEventCallback } from "usehooks-ts";

// src/ink/hooks/use-stdin.ts
import { useContext as useContext5 } from "react";

// src/ink/components/StdinContext.ts
import { createContext as createContext5 } from "react";

// src/ink/events/emitter.ts
import { EventEmitter as NodeEventEmitter } from "events";
var EventEmitter = class extends NodeEventEmitter {
  constructor() {
    super();
    this.setMaxListeners(0);
  }
  emit(type, ...args) {
    if (type === "error") {
      return super.emit(type, ...args);
    }
    const listeners = this.rawListeners(type);
    if (listeners.length === 0) {
      return false;
    }
    const ccEvent = args[0] instanceof Event ? args[0] : null;
    for (const listener of listeners) {
      listener.apply(this, args);
      if (ccEvent?.didStopImmediatePropagation()) {
        break;
      }
    }
    return true;
  }
};

// src/ink/components/StdinContext.ts
var StdinContext = createContext5({
  stdin: process.stdin,
  inputEmitter: new EventEmitter(),
  setRawMode() {
  },
  isRawModeSupported: false,
  exitOnCtrlC: true,
  querier: null
});
StdinContext.displayName = "StdinContext";
var StdinContext_default = StdinContext;

// src/ink/hooks/use-stdin.ts
var useStdin = () => useContext5(StdinContext_default);
var use_stdin_default = useStdin;

// src/ink/hooks/use-input.ts
var useInput = (inputHandler, options = {}) => {
  const { setRawMode, exitOnCtrlC, inputEmitter } = use_stdin_default();
  useLayoutEffect2(() => {
    if (options.isActive === false) {
      return;
    }
    setRawMode(true);
    return () => {
      setRawMode(false);
    };
  }, [options.isActive, setRawMode]);
  const handleData = useEventCallback((event) => {
    if (options.isActive === false) {
      return;
    }
    const { input, key } = event;
    if (!(input === "c" && key.ctrl) || !exitOnCtrlC) {
      inputHandler(input, key, event);
    }
  });
  useEffect(() => {
    inputEmitter?.on("input", handleData);
    return () => {
      inputEmitter?.removeListener("input", handleData);
    };
  }, [inputEmitter, handleData]);
};
var use_input_default = useInput;

// src/ink/hooks/use-selection.ts
import { useContext as useContext6, useMemo as useMemo4, useSyncExternalStore } from "react";

// src/ink/layout/geometry.ts
function unionRect(a, b) {
  const minX = Math.min(a.x, b.x);
  const minY = Math.min(a.y, b.y);
  const maxX = Math.max(a.x + a.width, b.x + b.width);
  const maxY = Math.max(a.y + a.height, b.y + b.height);
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}
function clamp(value, min, max) {
  if (min !== void 0 && value < min) {
    return min;
  }
  if (max !== void 0 && value > max) {
    return max;
  }
  return value;
}

// src/ink/screen.ts
import { ansiCodesToString as ansiCodesToString2, diffAnsiCodes } from "@alcalzone/ansi-tokenize";
var CharPool = class {
  strings = [" ", ""];
  // Index 0 = space, 1 = empty (spacer)
  stringMap = /* @__PURE__ */ new Map([
    [" ", 0],
    ["", 1]
  ]);
  ascii = initCharAscii();
  // charCode → index, -1 = not interned
  intern(char) {
    if (char.length === 1) {
      const code = char.charCodeAt(0);
      if (code < 128) {
        const cached = this.ascii[code];
        if (cached !== -1) {
          return cached;
        }
        const index2 = this.strings.length;
        this.strings.push(char);
        this.ascii[code] = index2;
        return index2;
      }
    }
    const existing = this.stringMap.get(char);
    if (existing !== void 0) {
      return existing;
    }
    const index = this.strings.length;
    this.strings.push(char);
    this.stringMap.set(char, index);
    return index;
  }
  get(index) {
    return this.strings[index] ?? " ";
  }
};
var HyperlinkPool = class {
  strings = [""];
  // Index 0 = no hyperlink
  stringMap = /* @__PURE__ */ new Map();
  intern(hyperlink) {
    if (!hyperlink) {
      return 0;
    }
    let id = this.stringMap.get(hyperlink);
    if (id === void 0) {
      id = this.strings.length;
      this.strings.push(hyperlink);
      this.stringMap.set(hyperlink, id);
    }
    return id;
  }
  get(id) {
    return id === 0 ? void 0 : this.strings[id];
  }
};
var INVERSE_CODE = {
  type: "ansi",
  code: "\x1B[7m",
  endCode: "\x1B[27m"
};
var BOLD_CODE = {
  type: "ansi",
  code: "\x1B[1m",
  endCode: "\x1B[22m"
};
var UNDERLINE_CODE = {
  type: "ansi",
  code: "\x1B[4m",
  endCode: "\x1B[24m"
};
var YELLOW_FG_CODE = {
  type: "ansi",
  code: "\x1B[33m",
  endCode: "\x1B[39m"
};
var StylePool = class {
  ids = /* @__PURE__ */ new Map();
  styles = [];
  transitionCache = /* @__PURE__ */ new Map();
  none;
  constructor() {
    this.none = this.intern([]);
  }
  /**
   * Intern a style and return its ID. Bit 0 of the ID encodes whether the
   * style has a visible effect on space characters (background, inverse,
   * underline, etc.). Foreground-only styles get even IDs; styles visible
   * on spaces get odd IDs. This lets the renderer skip invisible spaces
   * with a single bitmask check on the packed word.
   */
  intern(styles2) {
    const key = styles2.length === 0 ? "" : styles2.map((s) => s.code).join("\0");
    let id = this.ids.get(key);
    if (id === void 0) {
      const rawId = this.styles.length;
      this.styles.push(styles2.length === 0 ? [] : styles2);
      id = rawId << 1 | (styles2.length > 0 && hasVisibleSpaceEffect(styles2) ? 1 : 0);
      this.ids.set(key, id);
    }
    return id;
  }
  /** Recover styles from an encoded ID. Strips the bit-0 flag via >>> 1. */
  get(id) {
    return this.styles[id >>> 1] ?? [];
  }
  /**
   * Returns the pre-serialized ANSI string to transition from one style to
   * another. Cached by (fromId, toId) — zero allocations after first call
   * for a given pair.
   */
  transition(fromId, toId) {
    if (fromId === toId) {
      return "";
    }
    const key = fromId * 1048576 + toId;
    let str = this.transitionCache.get(key);
    if (str === void 0) {
      str = ansiCodesToString2(diffAnsiCodes(this.get(fromId), this.get(toId)));
      this.transitionCache.set(key, str);
    }
    return str;
  }
  /**
   * Intern a style that is `base + inverse`. Cached by base ID so
   * repeated calls for the same underlying style don't re-scan the
   * AnsiCode[] array. Used by the selection overlay.
   */
  inverseCache = /* @__PURE__ */ new Map();
  withInverse(baseId) {
    let id = this.inverseCache.get(baseId);
    if (id === void 0) {
      const baseCodes = this.get(baseId);
      const hasInverse = baseCodes.some((c) => c.endCode === "\x1B[27m");
      id = hasInverse ? baseId : this.intern([...baseCodes, INVERSE_CODE]);
      this.inverseCache.set(baseId, id);
    }
    return id;
  }
  /** Inverse + bold + yellow-bg-via-fg-swap for the CURRENT search match.
   *  OTHER matches are plain inverse — bg inherits from the theme. Current
   *  gets a distinct yellow bg (via fg-then-inverse swap) plus bold weight
   *  so it stands out in a sea of inverse. Underline was too subtle. Zero
   *  reflow risk: all pure SGR overlays, per-cell, post-layout. The yellow
   *  overrides any existing fg (syntax highlighting) on those cells — fine,
   *  the "you are here" signal IS the point, syntax color can yield. */
  currentMatchCache = /* @__PURE__ */ new Map();
  withCurrentMatch(baseId) {
    let id = this.currentMatchCache.get(baseId);
    if (id === void 0) {
      const baseCodes = this.get(baseId);
      const codes = baseCodes.filter((c) => c.endCode !== "\x1B[39m" && c.endCode !== "\x1B[49m");
      codes.push(YELLOW_FG_CODE);
      if (!baseCodes.some((c) => c.endCode === "\x1B[27m")) {
        codes.push(INVERSE_CODE);
      }
      if (!baseCodes.some((c) => c.endCode === "\x1B[22m")) {
        codes.push(BOLD_CODE);
      }
      if (!baseCodes.some((c) => c.endCode === "\x1B[24m")) {
        codes.push(UNDERLINE_CODE);
      }
      id = this.intern(codes);
      this.currentMatchCache.set(baseId, id);
    }
    return id;
  }
  /**
   * Selection overlay: REPLACE the cell's background with a solid color
   * while preserving its foreground (color, bold, italic, dim, underline).
   * Matches native terminal selection — a dedicated bg color, not SGR-7
   * inverse. Inverse swaps fg/bg per-cell, which fragments visually over
   * syntax-highlighted text (every fg color becomes a different bg stripe).
   *
   * Strips any existing bg (endCode 49m — REPLACES, so diff-added green
   * etc. don't bleed through) and any existing inverse (endCode 27m —
   * inverse on top of a solid bg would re-swap and look wrong).
   *
   * bg is set via setSelectionBg(); null → fallback to withInverse() so the
   * overlay still works before theme wiring sets a color (tests, first frame).
   * Cache is keyed by baseId only — setSelectionBg() clears it on change.
   */
  selectionBgCode = null;
  selectionBgCache = /* @__PURE__ */ new Map();
  setSelectionBg(bg) {
    if (this.selectionBgCode?.code === bg?.code) {
      return;
    }
    this.selectionBgCode = bg;
    this.selectionBgCache.clear();
  }
  withSelectionBg(baseId) {
    const bg = this.selectionBgCode;
    if (bg === null) {
      return this.withInverse(baseId);
    }
    let id = this.selectionBgCache.get(baseId);
    if (id === void 0) {
      const kept = this.get(baseId).filter((c) => c.endCode !== "\x1B[49m" && c.endCode !== "\x1B[27m");
      kept.push(bg);
      id = this.intern(kept);
      this.selectionBgCache.set(baseId, id);
    }
    return id;
  }
};
var VISIBLE_ON_SPACE = /* @__PURE__ */ new Set([
  "\x1B[49m",
  // background color
  "\x1B[27m",
  // inverse
  "\x1B[24m",
  // underline
  "\x1B[29m",
  // strikethrough
  "\x1B[55m"
  // overline
]);
function hasVisibleSpaceEffect(styles2) {
  for (const style of styles2) {
    if (VISIBLE_ON_SPACE.has(style.endCode)) {
      return true;
    }
  }
  return false;
}
var EMPTY_CHAR_INDEX = 0;
var SPACER_CHAR_INDEX = 1;
function initCharAscii() {
  const table = new Int32Array(128);
  table.fill(-1);
  table[32] = EMPTY_CHAR_INDEX;
  return table;
}
var STYLE_SHIFT = 17;
var HYPERLINK_SHIFT = 2;
var HYPERLINK_MASK = 32767;
var WIDTH_MASK = 3;
function packWord1(styleId, hyperlinkId, width) {
  return styleId << STYLE_SHIFT | hyperlinkId << HYPERLINK_SHIFT | width;
}
var EMPTY_CELL_VALUE = 0n;
function isEmptyCellByIndex(screen, index) {
  const ci = index << 1;
  return screen.cells[ci] === 0 && screen.cells[ci | 1] === 0;
}
function isEmptyCellAt(screen, x, y) {
  if (x < 0 || y < 0 || x >= screen.width || y >= screen.height) {
    return true;
  }
  return isEmptyCellByIndex(screen, y * screen.width + x);
}
function internHyperlink(screen, hyperlink) {
  return screen.hyperlinkPool.intern(hyperlink);
}
function createScreen(width, height, styles2, charPool, hyperlinkPool) {
  ifNotInteger(width, "createScreen width");
  ifNotInteger(height, "createScreen height");
  if (!Number.isInteger(width) || width < 0) {
    width = Math.max(0, Math.floor(width) || 0);
  }
  if (!Number.isInteger(height) || height < 0) {
    height = Math.max(0, Math.floor(height) || 0);
  }
  const size = width * height;
  const buf = new ArrayBuffer(size << 3);
  const cells = new Int32Array(buf);
  const cells64 = new BigInt64Array(buf);
  return {
    width,
    height,
    cells,
    cells64,
    charPool,
    hyperlinkPool,
    emptyStyleId: styles2.none,
    damage: void 0,
    noSelect: new Uint8Array(size),
    softWrap: new Int32Array(height)
  };
}
function resetScreen(screen, width, height) {
  ifNotInteger(width, "resetScreen width");
  ifNotInteger(height, "resetScreen height");
  if (!Number.isInteger(width) || width < 0) {
    width = Math.max(0, Math.floor(width) || 0);
  }
  if (!Number.isInteger(height) || height < 0) {
    height = Math.max(0, Math.floor(height) || 0);
  }
  const size = width * height;
  if (screen.cells64.length < size) {
    const buf = new ArrayBuffer(size << 3);
    screen.cells = new Int32Array(buf);
    screen.cells64 = new BigInt64Array(buf);
    screen.noSelect = new Uint8Array(size);
  }
  if (screen.softWrap.length < height) {
    screen.softWrap = new Int32Array(height);
  }
  screen.cells64.fill(EMPTY_CELL_VALUE, 0, size);
  screen.noSelect.fill(0, 0, size);
  screen.softWrap.fill(0, 0, height);
  screen.width = width;
  screen.height = height;
  screen.damage = void 0;
}
function migrateScreenPools(screen, charPool, hyperlinkPool) {
  const oldCharPool = screen.charPool;
  const oldHyperlinkPool = screen.hyperlinkPool;
  if (oldCharPool === charPool && oldHyperlinkPool === hyperlinkPool) {
    return;
  }
  const size = screen.width * screen.height;
  const cells = screen.cells;
  for (let ci = 0; ci < size << 1; ci += 2) {
    const oldCharId = cells[ci];
    cells[ci] = charPool.intern(oldCharPool.get(oldCharId));
    const word1 = cells[ci + 1];
    const oldHyperlinkId = word1 >>> HYPERLINK_SHIFT & HYPERLINK_MASK;
    if (oldHyperlinkId !== 0) {
      const oldStr = oldHyperlinkPool.get(oldHyperlinkId);
      const newHyperlinkId = hyperlinkPool.intern(oldStr);
      const styleId = word1 >>> STYLE_SHIFT;
      const width = word1 & WIDTH_MASK;
      cells[ci + 1] = packWord1(styleId, newHyperlinkId, width);
    }
  }
  screen.charPool = charPool;
  screen.hyperlinkPool = hyperlinkPool;
}
function cellAt(screen, x, y) {
  if (x < 0 || y < 0 || x >= screen.width || y >= screen.height) {
    return void 0;
  }
  return cellAtIndex(screen, y * screen.width + x);
}
function cellAtIndex(screen, index) {
  const ci = index << 1;
  const word1 = screen.cells[ci + 1];
  const hid = word1 >>> HYPERLINK_SHIFT & HYPERLINK_MASK;
  return {
    // Unwritten cells have charIndex=0 (EMPTY_CHAR_INDEX); charPool.get(0) returns ' '
    char: screen.charPool.get(screen.cells[ci]),
    styleId: word1 >>> STYLE_SHIFT,
    width: word1 & WIDTH_MASK,
    hyperlink: hid === 0 ? void 0 : screen.hyperlinkPool.get(hid)
  };
}
function visibleCellAtIndex(cells, charPool, hyperlinkPool, index, lastRenderedStyleId) {
  const ci = index << 1;
  const charId = cells[ci];
  if (charId === 1) {
    return void 0;
  }
  const word1 = cells[ci + 1];
  if (charId === 0 && (word1 & 262140) === 0) {
    const fgStyle = word1 >>> STYLE_SHIFT;
    if (fgStyle === 0 || fgStyle === lastRenderedStyleId) {
      return void 0;
    }
  }
  const hid = word1 >>> HYPERLINK_SHIFT & HYPERLINK_MASK;
  return {
    char: charPool.get(charId),
    styleId: word1 >>> STYLE_SHIFT,
    width: word1 & WIDTH_MASK,
    hyperlink: hid === 0 ? void 0 : hyperlinkPool.get(hid)
  };
}
function cellAtCI(screen, ci, out) {
  const w1 = ci | 1;
  const word1 = screen.cells[w1];
  out.char = screen.charPool.get(screen.cells[ci]);
  out.styleId = word1 >>> STYLE_SHIFT;
  out.width = word1 & WIDTH_MASK;
  const hid = word1 >>> HYPERLINK_SHIFT & HYPERLINK_MASK;
  out.hyperlink = hid === 0 ? void 0 : screen.hyperlinkPool.get(hid);
}
function charInCellAt(screen, x, y) {
  if (x < 0 || y < 0 || x >= screen.width || y >= screen.height) {
    return void 0;
  }
  const ci = y * screen.width + x << 1;
  return screen.charPool.get(screen.cells[ci]);
}
function setCellAt(screen, x, y, cell) {
  if (x < 0 || y < 0 || x >= screen.width || y >= screen.height) {
    return;
  }
  const ci = y * screen.width + x << 1;
  const cells = screen.cells;
  const prevWidth = cells[ci + 1] & WIDTH_MASK;
  if (prevWidth === 1 /* Wide */ && cell.width !== 1 /* Wide */) {
    const spacerX = x + 1;
    if (spacerX < screen.width) {
      const spacerCI = ci + 2;
      if ((cells[spacerCI + 1] & WIDTH_MASK) === 2 /* SpacerTail */) {
        cells[spacerCI] = EMPTY_CHAR_INDEX;
        cells[spacerCI + 1] = packWord1(screen.emptyStyleId, 0, 0 /* Narrow */);
      }
    }
  }
  let clearedWideX = -1;
  if (prevWidth === 2 /* SpacerTail */ && cell.width !== 2 /* SpacerTail */) {
    if (x > 0) {
      const wideCI = ci - 2;
      if ((cells[wideCI + 1] & WIDTH_MASK) === 1 /* Wide */) {
        cells[wideCI] = EMPTY_CHAR_INDEX;
        cells[wideCI + 1] = packWord1(screen.emptyStyleId, 0, 0 /* Narrow */);
        clearedWideX = x - 1;
      }
    }
  }
  cells[ci] = internCharString(screen, cell.char);
  cells[ci + 1] = packWord1(cell.styleId, internHyperlink(screen, cell.hyperlink), cell.width);
  const minX = clearedWideX >= 0 ? Math.min(x, clearedWideX) : x;
  const damage = screen.damage;
  if (damage) {
    const right = damage.x + damage.width;
    const bottom = damage.y + damage.height;
    if (minX < damage.x) {
      damage.width += damage.x - minX;
      damage.x = minX;
    } else if (x >= right) {
      damage.width = x - damage.x + 1;
    }
    if (y < damage.y) {
      damage.height += damage.y - y;
      damage.y = y;
    } else if (y >= bottom) {
      damage.height = y - damage.y + 1;
    }
  } else {
    screen.damage = { x: minX, y, width: x - minX + 1, height: 1 };
  }
  if (cell.width === 1 /* Wide */) {
    const spacerX = x + 1;
    if (spacerX < screen.width) {
      const spacerCI = ci + 2;
      if ((cells[spacerCI + 1] & WIDTH_MASK) === 1 /* Wide */) {
        const orphanCI = spacerCI + 2;
        if (spacerX + 1 < screen.width && (cells[orphanCI + 1] & WIDTH_MASK) === 2 /* SpacerTail */) {
          cells[orphanCI] = EMPTY_CHAR_INDEX;
          cells[orphanCI + 1] = packWord1(screen.emptyStyleId, 0, 0 /* Narrow */);
        }
      }
      cells[spacerCI] = SPACER_CHAR_INDEX;
      cells[spacerCI + 1] = packWord1(screen.emptyStyleId, 0, 2 /* SpacerTail */);
      const d = screen.damage;
      if (d && spacerX >= d.x + d.width) {
        d.width = spacerX - d.x + 1;
      }
    }
  }
}
function setCellStyleId(screen, x, y, styleId) {
  if (x < 0 || y < 0 || x >= screen.width || y >= screen.height) {
    return;
  }
  const ci = y * screen.width + x << 1;
  const cells = screen.cells;
  const word1 = cells[ci + 1];
  const width = word1 & WIDTH_MASK;
  if (width === 2 /* SpacerTail */ || width === 3 /* SpacerHead */) {
    return;
  }
  const hid = word1 >>> HYPERLINK_SHIFT & HYPERLINK_MASK;
  cells[ci + 1] = packWord1(styleId, hid, width);
  const d = screen.damage;
  if (d) {
    screen.damage = unionRect(d, { x, y, width: 1, height: 1 });
  } else {
    screen.damage = { x, y, width: 1, height: 1 };
  }
}
function internCharString(screen, char) {
  return screen.charPool.intern(char);
}
function blitRegion(dst, src, regionX, regionY, maxX, maxY) {
  regionX = Math.max(0, regionX);
  regionY = Math.max(0, regionY);
  if (regionX >= maxX || regionY >= maxY) {
    return;
  }
  const rowLen = maxX - regionX;
  const srcStride = src.width << 1;
  const dstStride = dst.width << 1;
  const rowBytes = rowLen << 1;
  const srcCells = src.cells;
  const dstCells = dst.cells;
  const srcNoSel = src.noSelect;
  const dstNoSel = dst.noSelect;
  dst.softWrap.set(src.softWrap.subarray(regionY, maxY), regionY);
  if (regionX === 0 && maxX === src.width && src.width === dst.width) {
    const srcStart = regionY * srcStride;
    const totalBytes = (maxY - regionY) * srcStride;
    dstCells.set(
      srcCells.subarray(srcStart, srcStart + totalBytes),
      srcStart
      // srcStart === dstStart when strides match and regionX === 0
    );
    const nsStart = regionY * src.width;
    const nsLen = (maxY - regionY) * src.width;
    dstNoSel.set(srcNoSel.subarray(nsStart, nsStart + nsLen), nsStart);
  } else {
    let srcRowCI = regionY * srcStride + (regionX << 1);
    let dstRowCI = regionY * dstStride + (regionX << 1);
    let srcRowNS = regionY * src.width + regionX;
    let dstRowNS = regionY * dst.width + regionX;
    for (let y = regionY; y < maxY; y++) {
      dstCells.set(srcCells.subarray(srcRowCI, srcRowCI + rowBytes), dstRowCI);
      dstNoSel.set(srcNoSel.subarray(srcRowNS, srcRowNS + rowLen), dstRowNS);
      srcRowCI += srcStride;
      dstRowCI += dstStride;
      srcRowNS += src.width;
      dstRowNS += dst.width;
    }
  }
  const regionRect = {
    x: regionX,
    y: regionY,
    width: rowLen,
    height: maxY - regionY
  };
  if (dst.damage) {
    dst.damage = unionRect(dst.damage, regionRect);
  } else {
    dst.damage = regionRect;
  }
  if (maxX < dst.width) {
    let srcLastCI = regionY * src.width + (maxX - 1) << 1;
    let dstSpacerCI = regionY * dst.width + maxX << 1;
    let wroteSpacerOutsideRegion = false;
    for (let y = regionY; y < maxY; y++) {
      if ((srcCells[srcLastCI + 1] & WIDTH_MASK) === 1 /* Wide */) {
        dstCells[dstSpacerCI] = SPACER_CHAR_INDEX;
        dstCells[dstSpacerCI + 1] = packWord1(dst.emptyStyleId, 0, 2 /* SpacerTail */);
        wroteSpacerOutsideRegion = true;
      }
      srcLastCI += srcStride;
      dstSpacerCI += dstStride;
    }
    if (wroteSpacerOutsideRegion && dst.damage) {
      const rightEdge = dst.damage.x + dst.damage.width;
      if (rightEdge === maxX) {
        dst.damage = { ...dst.damage, width: dst.damage.width + 1 };
      }
    }
  }
}
function shiftRows(screen, top, bottom, n) {
  if (n === 0 || top < 0 || bottom >= screen.height || top > bottom) {
    return;
  }
  const w = screen.width;
  const cells64 = screen.cells64;
  const noSel = screen.noSelect;
  const sw = screen.softWrap;
  const absN = Math.abs(n);
  if (absN > bottom - top) {
    cells64.fill(EMPTY_CELL_VALUE, top * w, (bottom + 1) * w);
    noSel.fill(0, top * w, (bottom + 1) * w);
    sw.fill(0, top, bottom + 1);
    return;
  }
  if (n > 0) {
    cells64.copyWithin(top * w, (top + n) * w, (bottom + 1) * w);
    noSel.copyWithin(top * w, (top + n) * w, (bottom + 1) * w);
    sw.copyWithin(top, top + n, bottom + 1);
    cells64.fill(EMPTY_CELL_VALUE, (bottom - n + 1) * w, (bottom + 1) * w);
    noSel.fill(0, (bottom - n + 1) * w, (bottom + 1) * w);
    sw.fill(0, bottom - n + 1, bottom + 1);
  } else {
    cells64.copyWithin((top - n) * w, top * w, (bottom + n + 1) * w);
    noSel.copyWithin((top - n) * w, top * w, (bottom + n + 1) * w);
    sw.copyWithin(top - n, top, bottom + n + 1);
    cells64.fill(EMPTY_CELL_VALUE, top * w, (top - n) * w);
    noSel.fill(0, top * w, (top - n) * w);
    sw.fill(0, top, top - n);
  }
}
var OSC8_REGEX = new RegExp(`^${ESC}\\]8${SEP}${SEP}([^${BEL}]*)${BEL}$`);
var OSC8_PREFIX = `${ESC}]8${SEP}`;
function extractHyperlinkFromStyles(styles2) {
  for (const style of styles2) {
    const code = style.code;
    if (code.length < 5 || !code.startsWith(OSC8_PREFIX)) {
      continue;
    }
    const match = code.match(OSC8_REGEX);
    if (match) {
      return match[1] || null;
    }
  }
  return null;
}
function filterOutHyperlinkStyles(styles2) {
  return styles2.filter((style) => !style.code.startsWith(OSC8_PREFIX) || !OSC8_REGEX.test(style.code));
}
function diffEach(prev, next, cb) {
  const prevWidth = prev.width;
  const nextWidth = next.width;
  const prevHeight = prev.height;
  const nextHeight = next.height;
  let region;
  if (prevWidth === 0 && prevHeight === 0) {
    region = { x: 0, y: 0, width: nextWidth, height: nextHeight };
  } else if (next.damage) {
    region = next.damage;
    if (prev.damage) {
      region = unionRect(region, prev.damage);
    }
  } else if (prev.damage) {
    region = prev.damage;
  } else {
    region = { x: 0, y: 0, width: 0, height: 0 };
  }
  if (prevHeight > nextHeight) {
    region = unionRect(region, {
      x: 0,
      y: nextHeight,
      width: prevWidth,
      height: prevHeight - nextHeight
    });
  }
  if (prevWidth > nextWidth) {
    region = unionRect(region, {
      x: nextWidth,
      y: 0,
      width: prevWidth - nextWidth,
      height: prevHeight
    });
  }
  const maxHeight = Math.max(prevHeight, nextHeight);
  const maxWidth = Math.max(prevWidth, nextWidth);
  const endY = Math.min(region.y + region.height, maxHeight);
  const endX = Math.min(region.x + region.width, maxWidth);
  if (prevWidth === nextWidth) {
    return diffSameWidth(prev, next, region.x, endX, region.y, endY, cb);
  }
  return diffDifferentWidth(prev, next, region.x, endX, region.y, endY, cb);
}
function findNextDiff(a, b, w0, count) {
  for (let i = 0; i < count; i++, w0 += 2) {
    const w1 = w0 | 1;
    if (a[w0] !== b[w0] || a[w1] !== b[w1]) {
      return i;
    }
  }
  return count;
}
function diffRowBoth(prevCells, nextCells, prev, next, ci, y, startX, endX, prevCell, nextCell, cb) {
  let x = startX;
  while (x < endX) {
    const skip = findNextDiff(prevCells, nextCells, ci, endX - x);
    x += skip;
    ci += skip << 1;
    if (x >= endX) {
      break;
    }
    cellAtCI(prev, ci, prevCell);
    cellAtCI(next, ci, nextCell);
    if (cb(x, y, prevCell, nextCell)) {
      return true;
    }
    x++;
    ci += 2;
  }
  return false;
}
function diffRowRemoved(prev, ci, y, startX, endX, prevCell, cb) {
  for (let x = startX; x < endX; x++, ci += 2) {
    cellAtCI(prev, ci, prevCell);
    if (cb(x, y, prevCell, void 0)) {
      return true;
    }
  }
  return false;
}
function diffRowAdded(nextCells, next, ci, y, startX, endX, nextCell, cb) {
  for (let x = startX; x < endX; x++, ci += 2) {
    if (nextCells[ci] === 0 && nextCells[ci | 1] === 0) {
      continue;
    }
    cellAtCI(next, ci, nextCell);
    if (cb(x, y, void 0, nextCell)) {
      return true;
    }
  }
  return false;
}
function diffSameWidth(prev, next, startX, endX, startY, endY, cb) {
  const prevCells = prev.cells;
  const nextCells = next.cells;
  const width = prev.width;
  const prevHeight = prev.height;
  const nextHeight = next.height;
  const stride = width << 1;
  const prevCell = {
    char: " ",
    styleId: 0,
    width: 0 /* Narrow */,
    hyperlink: void 0
  };
  const nextCell = {
    char: " ",
    styleId: 0,
    width: 0 /* Narrow */,
    hyperlink: void 0
  };
  const rowEndX = Math.min(endX, width);
  let rowCI = startY * width + startX << 1;
  for (let y = startY; y < endY; y++) {
    const prevIn = y < prevHeight;
    const nextIn = y < nextHeight;
    if (prevIn && nextIn) {
      if (diffRowBoth(prevCells, nextCells, prev, next, rowCI, y, startX, rowEndX, prevCell, nextCell, cb)) {
        return true;
      }
    } else if (prevIn) {
      if (diffRowRemoved(prev, rowCI, y, startX, rowEndX, prevCell, cb)) {
        return true;
      }
    } else if (nextIn) {
      if (diffRowAdded(nextCells, next, rowCI, y, startX, rowEndX, nextCell, cb)) {
        return true;
      }
    }
    rowCI += stride;
  }
  return false;
}
function diffDifferentWidth(prev, next, startX, endX, startY, endY, cb) {
  const prevWidth = prev.width;
  const nextWidth = next.width;
  const prevCells = prev.cells;
  const nextCells = next.cells;
  const prevCell = {
    char: " ",
    styleId: 0,
    width: 0 /* Narrow */,
    hyperlink: void 0
  };
  const nextCell = {
    char: " ",
    styleId: 0,
    width: 0 /* Narrow */,
    hyperlink: void 0
  };
  const prevStride = prevWidth << 1;
  const nextStride = nextWidth << 1;
  let prevRowCI = startY * prevWidth + startX << 1;
  let nextRowCI = startY * nextWidth + startX << 1;
  for (let y = startY; y < endY; y++) {
    const prevIn = y < prev.height;
    const nextIn = y < next.height;
    const prevEndX = prevIn ? Math.min(endX, prevWidth) : startX;
    const nextEndX = nextIn ? Math.min(endX, nextWidth) : startX;
    const bothEndX = Math.min(prevEndX, nextEndX);
    let prevCI = prevRowCI;
    let nextCI = nextRowCI;
    for (let x = startX; x < bothEndX; x++) {
      if (prevCells[prevCI] === nextCells[nextCI] && prevCells[prevCI + 1] === nextCells[nextCI + 1]) {
        prevCI += 2;
        nextCI += 2;
        continue;
      }
      cellAtCI(prev, prevCI, prevCell);
      cellAtCI(next, nextCI, nextCell);
      prevCI += 2;
      nextCI += 2;
      if (cb(x, y, prevCell, nextCell)) {
        return true;
      }
    }
    if (prevEndX > bothEndX) {
      prevCI = prevRowCI + (bothEndX - startX << 1);
      for (let x = bothEndX; x < prevEndX; x++) {
        cellAtCI(prev, prevCI, prevCell);
        prevCI += 2;
        if (cb(x, y, prevCell, void 0)) {
          return true;
        }
      }
    }
    if (nextEndX > bothEndX) {
      nextCI = nextRowCI + (bothEndX - startX << 1);
      for (let x = bothEndX; x < nextEndX; x++) {
        if (nextCells[nextCI] === 0 && nextCells[nextCI | 1] === 0) {
          nextCI += 2;
          continue;
        }
        cellAtCI(next, nextCI, nextCell);
        nextCI += 2;
        if (cb(x, y, void 0, nextCell)) {
          return true;
        }
      }
    }
    prevRowCI += prevStride;
    nextRowCI += nextStride;
  }
  return false;
}
function markNoSelectRegion(screen, x, y, width, height) {
  const maxX = Math.min(x + width, screen.width);
  const maxY = Math.min(y + height, screen.height);
  const noSel = screen.noSelect;
  const stride = screen.width;
  for (let row = Math.max(0, y); row < maxY; row++) {
    const rowStart = row * stride;
    noSel.fill(1, rowStart + Math.max(0, x), rowStart + maxX);
  }
}

// src/ink/selection.ts
function createSelectionState() {
  return {
    anchor: null,
    focus: null,
    isDragging: false,
    anchorSpan: null,
    scrolledOffAbove: [],
    scrolledOffBelow: [],
    scrolledOffAboveSW: [],
    scrolledOffBelowSW: [],
    lastPressHadAlt: false
  };
}
function startSelection(s, col, row) {
  s.anchor = { col, row };
  s.focus = null;
  s.isDragging = true;
  s.anchorSpan = null;
  s.scrolledOffAbove = [];
  s.scrolledOffBelow = [];
  s.scrolledOffAboveSW = [];
  s.scrolledOffBelowSW = [];
  s.virtualAnchorRow = void 0;
  s.virtualFocusRow = void 0;
  s.lastPressHadAlt = false;
}
function updateSelection(s, col, row) {
  if (!s.isDragging) {
    return;
  }
  if (!s.focus && s.anchor && s.anchor.col === col && s.anchor.row === row) {
    return;
  }
  s.focus = { col, row };
}
function finishSelection(s) {
  s.isDragging = false;
}
function clearSelection(s) {
  s.anchor = null;
  s.focus = null;
  s.isDragging = false;
  s.anchorSpan = null;
  s.scrolledOffAbove = [];
  s.scrolledOffBelow = [];
  s.scrolledOffAboveSW = [];
  s.scrolledOffBelowSW = [];
  s.virtualAnchorRow = void 0;
  s.virtualFocusRow = void 0;
  s.lastPressHadAlt = false;
}
var WORD_CHAR = /[\p{L}\p{N}_/.\-+~\\]/u;
function charClass(c) {
  if (c === " " || c === "") {
    return 0;
  }
  if (WORD_CHAR.test(c)) {
    return 1;
  }
  return 2;
}
function wordBoundsAt(screen, col, row) {
  if (row < 0 || row >= screen.height) {
    return null;
  }
  const width = screen.width;
  const noSelect = screen.noSelect;
  const rowOff = row * width;
  let c = col;
  if (c > 0) {
    const cell = cellAt(screen, c, row);
    if (cell && cell.width === 2 /* SpacerTail */) {
      c -= 1;
    }
  }
  if (c < 0 || c >= width || noSelect[rowOff + c] === 1) {
    return null;
  }
  const startCell = cellAt(screen, c, row);
  if (!startCell) {
    return null;
  }
  const cls = charClass(startCell.char);
  let lo = c;
  while (lo > 0) {
    const prev = lo - 1;
    if (noSelect[rowOff + prev] === 1) {
      break;
    }
    const pc = cellAt(screen, prev, row);
    if (!pc) {
      break;
    }
    if (pc.width === 2 /* SpacerTail */) {
      if (prev === 0 || noSelect[rowOff + prev - 1] === 1) {
        break;
      }
      const head = cellAt(screen, prev - 1, row);
      if (!head || charClass(head.char) !== cls) {
        break;
      }
      lo = prev - 1;
      continue;
    }
    if (charClass(pc.char) !== cls) {
      break;
    }
    lo = prev;
  }
  let hi = c;
  while (hi < width - 1) {
    const next = hi + 1;
    if (noSelect[rowOff + next] === 1) {
      break;
    }
    const nc = cellAt(screen, next, row);
    if (!nc) {
      break;
    }
    if (nc.width === 2 /* SpacerTail */) {
      hi = next;
      continue;
    }
    if (charClass(nc.char) !== cls) {
      break;
    }
    hi = next;
  }
  return { lo, hi };
}
function comparePoints(a, b) {
  if (a.row !== b.row) {
    return a.row < b.row ? -1 : 1;
  }
  if (a.col !== b.col) {
    return a.col < b.col ? -1 : 1;
  }
  return 0;
}
function selectWordAt(s, screen, col, row) {
  const b = wordBoundsAt(screen, col, row);
  if (!b) {
    return;
  }
  const lo = { col: b.lo, row };
  const hi = { col: b.hi, row };
  s.anchor = lo;
  s.focus = hi;
  s.isDragging = true;
  s.anchorSpan = { lo, hi, kind: "word" };
}
var URL_BOUNDARY = /* @__PURE__ */ new Set([..."<>\"'` "]);
function isUrlChar(c) {
  if (c.length !== 1) {
    return false;
  }
  const code = c.charCodeAt(0);
  return code >= 33 && code <= 126 && !URL_BOUNDARY.has(c);
}
function findPlainTextUrlAt(screen, col, row) {
  if (row < 0 || row >= screen.height) {
    return void 0;
  }
  const width = screen.width;
  const noSelect = screen.noSelect;
  const rowOff = row * width;
  let c = col;
  if (c > 0) {
    const cell = cellAt(screen, c, row);
    if (cell && cell.width === 2 /* SpacerTail */) {
      c -= 1;
    }
  }
  if (c < 0 || c >= width || noSelect[rowOff + c] === 1) {
    return void 0;
  }
  const startCell = cellAt(screen, c, row);
  if (!startCell || !isUrlChar(startCell.char)) {
    return void 0;
  }
  let lo = c;
  while (lo > 0) {
    const prev = lo - 1;
    if (noSelect[rowOff + prev] === 1) {
      break;
    }
    const pc = cellAt(screen, prev, row);
    if (!pc || pc.width !== 0 /* Narrow */ || !isUrlChar(pc.char)) {
      break;
    }
    lo = prev;
  }
  let hi = c;
  while (hi < width - 1) {
    const next = hi + 1;
    if (noSelect[rowOff + next] === 1) {
      break;
    }
    const nc = cellAt(screen, next, row);
    if (!nc || nc.width !== 0 /* Narrow */ || !isUrlChar(nc.char)) {
      break;
    }
    hi = next;
  }
  let token = "";
  for (let i = lo; i <= hi; i++) {
    token += cellAt(screen, i, row).char;
  }
  const clickIdx = c - lo;
  const schemeRe = /(?:https?|file):\/\//g;
  let urlStart = -1;
  let urlEnd = token.length;
  for (let m; m = schemeRe.exec(token); ) {
    if (m.index > clickIdx) {
      urlEnd = m.index;
      break;
    }
    urlStart = m.index;
  }
  if (urlStart < 0) {
    return void 0;
  }
  let url = token.slice(urlStart, urlEnd);
  const OPENER = { ")": "(", "]": "[", "}": "{" };
  while (url.length > 0) {
    const last = url.at(-1);
    if (".,;:!?".includes(last)) {
      url = url.slice(0, -1);
      continue;
    }
    const opener = OPENER[last];
    if (!opener) {
      break;
    }
    let opens = 0;
    let closes = 0;
    for (let i = 0; i < url.length; i++) {
      const ch = url.charAt(i);
      if (ch === opener) {
        opens++;
      } else if (ch === last) {
        closes++;
      }
    }
    if (closes > opens) {
      url = url.slice(0, -1);
    } else {
      break;
    }
  }
  if (clickIdx >= urlStart + url.length) {
    return void 0;
  }
  return url;
}
function selectLineAt(s, screen, row) {
  if (row < 0 || row >= screen.height) {
    return;
  }
  const lo = { col: 0, row };
  const hi = { col: screen.width - 1, row };
  s.anchor = lo;
  s.focus = hi;
  s.isDragging = true;
  s.anchorSpan = { lo, hi, kind: "line" };
}
function extendSelection(s, screen, col, row) {
  if (!s.isDragging || !s.anchorSpan) {
    return;
  }
  const span = s.anchorSpan;
  let mLo;
  let mHi;
  if (span.kind === "word") {
    const b = wordBoundsAt(screen, col, row);
    mLo = { col: b ? b.lo : col, row };
    mHi = { col: b ? b.hi : col, row };
  } else {
    const r = clamp(row, 0, screen.height - 1);
    mLo = { col: 0, row: r };
    mHi = { col: screen.width - 1, row: r };
  }
  if (comparePoints(mHi, span.lo) < 0) {
    s.anchor = span.hi;
    s.focus = mLo;
  } else if (comparePoints(mLo, span.hi) > 0) {
    s.anchor = span.lo;
    s.focus = mHi;
  } else {
    s.anchor = span.lo;
    s.focus = span.hi;
  }
}
function moveFocus(s, col, row) {
  if (!s.focus) {
    return;
  }
  s.anchorSpan = null;
  s.focus = { col, row };
  s.virtualFocusRow = void 0;
}
function shiftSelection(s, dRow, minRow, maxRow, width) {
  if (!s.anchor || !s.focus) {
    return;
  }
  const vAnchor = (s.virtualAnchorRow ?? s.anchor.row) + dRow;
  const vFocus = (s.virtualFocusRow ?? s.focus.row) + dRow;
  if (vAnchor < minRow && vFocus < minRow || vAnchor > maxRow && vFocus > maxRow) {
    clearSelection(s);
    return;
  }
  const oldMin = Math.min(s.virtualAnchorRow ?? s.anchor.row, s.virtualFocusRow ?? s.focus.row);
  const oldMax = Math.max(s.virtualAnchorRow ?? s.anchor.row, s.virtualFocusRow ?? s.focus.row);
  const oldAboveDebt = Math.max(0, minRow - oldMin);
  const oldBelowDebt = Math.max(0, oldMax - maxRow);
  const newAboveDebt = Math.max(0, minRow - Math.min(vAnchor, vFocus));
  const newBelowDebt = Math.max(0, Math.max(vAnchor, vFocus) - maxRow);
  if (newAboveDebt < oldAboveDebt) {
    const drop = oldAboveDebt - newAboveDebt;
    s.scrolledOffAbove.length -= drop;
    s.scrolledOffAboveSW.length = s.scrolledOffAbove.length;
  }
  if (newBelowDebt < oldBelowDebt) {
    const drop = oldBelowDebt - newBelowDebt;
    s.scrolledOffBelow.splice(0, drop);
    s.scrolledOffBelowSW.splice(0, drop);
  }
  if (s.scrolledOffAbove.length > newAboveDebt) {
    s.scrolledOffAbove = newAboveDebt > 0 ? s.scrolledOffAbove.slice(-newAboveDebt) : [];
    s.scrolledOffAboveSW = newAboveDebt > 0 ? s.scrolledOffAboveSW.slice(-newAboveDebt) : [];
  }
  if (s.scrolledOffBelow.length > newBelowDebt) {
    s.scrolledOffBelow = s.scrolledOffBelow.slice(0, newBelowDebt);
    s.scrolledOffBelowSW = s.scrolledOffBelowSW.slice(0, newBelowDebt);
  }
  const shift = (p, vRow) => {
    if (vRow < minRow) {
      return { col: 0, row: minRow };
    }
    if (vRow > maxRow) {
      return { col: width - 1, row: maxRow };
    }
    return { col: p.col, row: vRow };
  };
  s.anchor = shift(s.anchor, vAnchor);
  s.focus = shift(s.focus, vFocus);
  s.virtualAnchorRow = vAnchor < minRow || vAnchor > maxRow ? vAnchor : void 0;
  s.virtualFocusRow = vFocus < minRow || vFocus > maxRow ? vFocus : void 0;
  if (s.anchorSpan) {
    const sp = (p) => {
      const r = p.row + dRow;
      if (r < minRow) {
        return { col: 0, row: minRow };
      }
      if (r > maxRow) {
        return { col: width - 1, row: maxRow };
      }
      return { col: p.col, row: r };
    };
    s.anchorSpan = {
      lo: sp(s.anchorSpan.lo),
      hi: sp(s.anchorSpan.hi),
      kind: s.anchorSpan.kind
    };
  }
}
function shiftAnchor(s, dRow, minRow, maxRow) {
  if (!s.anchor) {
    return;
  }
  const raw = (s.virtualAnchorRow ?? s.anchor.row) + dRow;
  s.anchor = { col: s.anchor.col, row: clamp(raw, minRow, maxRow) };
  s.virtualAnchorRow = raw < minRow || raw > maxRow ? raw : void 0;
  if (s.anchorSpan) {
    const shift = (p) => ({
      col: p.col,
      row: clamp(p.row + dRow, minRow, maxRow)
    });
    s.anchorSpan = {
      lo: shift(s.anchorSpan.lo),
      hi: shift(s.anchorSpan.hi),
      kind: s.anchorSpan.kind
    };
  }
}
function shiftSelectionForFollow(s, dRow, minRow, maxRow) {
  if (!s.anchor) {
    return false;
  }
  const rawAnchor = (s.virtualAnchorRow ?? s.anchor.row) + dRow;
  const rawFocus = s.focus ? (s.virtualFocusRow ?? s.focus.row) + dRow : void 0;
  if (rawAnchor < minRow && rawFocus !== void 0 && rawFocus < minRow) {
    clearSelection(s);
    return true;
  }
  s.anchor = { col: s.anchor.col, row: clamp(rawAnchor, minRow, maxRow) };
  if (s.focus && rawFocus !== void 0) {
    s.focus = { col: s.focus.col, row: clamp(rawFocus, minRow, maxRow) };
  }
  s.virtualAnchorRow = rawAnchor < minRow || rawAnchor > maxRow ? rawAnchor : void 0;
  s.virtualFocusRow = rawFocus !== void 0 && (rawFocus < minRow || rawFocus > maxRow) ? rawFocus : void 0;
  if (s.anchorSpan) {
    const shift = (p) => ({
      col: p.col,
      row: clamp(p.row + dRow, minRow, maxRow)
    });
    s.anchorSpan = {
      lo: shift(s.anchorSpan.lo),
      hi: shift(s.anchorSpan.hi),
      kind: s.anchorSpan.kind
    };
  }
  return false;
}
function hasSelection(s) {
  return s.anchor !== null && s.focus !== null;
}
function selectionBounds(s) {
  if (!s.anchor || !s.focus) {
    return null;
  }
  return comparePoints(s.anchor, s.focus) <= 0 ? { start: s.anchor, end: s.focus } : { start: s.focus, end: s.anchor };
}
function extractRowText(screen, row, colStart, colEnd) {
  const noSelect = screen.noSelect;
  const rowOff = row * screen.width;
  const contentEnd = row + 1 < screen.height ? screen.softWrap[row + 1] : 0;
  const lastCol = contentEnd > 0 ? Math.min(colEnd, contentEnd - 1) : colEnd;
  let line = "";
  for (let col = colStart; col <= lastCol; col++) {
    if (noSelect[rowOff + col] === 1) {
      continue;
    }
    const cell = cellAt(screen, col, row);
    if (!cell) {
      continue;
    }
    if (cell.width === 2 /* SpacerTail */ || cell.width === 3 /* SpacerHead */) {
      continue;
    }
    line += cell.char;
  }
  return contentEnd > 0 ? line : line.replace(/\s+$/, "");
}
function joinRows(lines, text, sw) {
  if (sw && lines.length > 0) {
    lines[lines.length - 1] += text;
  } else {
    lines.push(text);
  }
}
function getSelectedText(s, screen) {
  const b = selectionBounds(s);
  if (!b) {
    return "";
  }
  const { start, end } = b;
  const sw = screen.softWrap;
  const lines = [];
  for (let i = 0; i < s.scrolledOffAbove.length; i++) {
    joinRows(lines, s.scrolledOffAbove[i], s.scrolledOffAboveSW[i]);
  }
  for (let row = start.row; row <= end.row; row++) {
    const rowStart = row === start.row ? start.col : 0;
    const rowEnd = row === end.row ? end.col : screen.width - 1;
    joinRows(lines, extractRowText(screen, row, rowStart, rowEnd), sw[row] > 0);
  }
  for (let i = 0; i < s.scrolledOffBelow.length; i++) {
    joinRows(lines, s.scrolledOffBelow[i], s.scrolledOffBelowSW[i]);
  }
  return lines.join("\n");
}
function captureScrolledRows(s, screen, firstRow, lastRow, side) {
  const b = selectionBounds(s);
  if (!b || firstRow > lastRow) {
    return;
  }
  const { start, end } = b;
  const lo = Math.max(firstRow, start.row);
  const hi = Math.min(lastRow, end.row);
  if (lo > hi) {
    return;
  }
  const width = screen.width;
  const sw = screen.softWrap;
  const captured = [];
  const capturedSW = [];
  for (let row = lo; row <= hi; row++) {
    const colStart = row === start.row ? start.col : 0;
    const colEnd = row === end.row ? end.col : width - 1;
    captured.push(extractRowText(screen, row, colStart, colEnd));
    capturedSW.push(sw[row] > 0);
  }
  if (side === "above") {
    s.scrolledOffAbove.push(...captured);
    s.scrolledOffAboveSW.push(...capturedSW);
    if (s.anchor && s.anchor.row === start.row && lo === start.row) {
      s.anchor = { col: 0, row: s.anchor.row };
      if (s.anchorSpan) {
        s.anchorSpan = {
          kind: s.anchorSpan.kind,
          lo: { col: 0, row: s.anchorSpan.lo.row },
          hi: { col: width - 1, row: s.anchorSpan.hi.row }
        };
      }
    }
  } else {
    s.scrolledOffBelow.unshift(...captured);
    s.scrolledOffBelowSW.unshift(...capturedSW);
    if (s.anchor && s.anchor.row === end.row && hi === end.row) {
      s.anchor = { col: width - 1, row: s.anchor.row };
      if (s.anchorSpan) {
        s.anchorSpan = {
          kind: s.anchorSpan.kind,
          lo: { col: 0, row: s.anchorSpan.lo.row },
          hi: { col: width - 1, row: s.anchorSpan.hi.row }
        };
      }
    }
  }
}
function applySelectionOverlay(screen, selection, stylePool) {
  const b = selectionBounds(selection);
  if (!b) {
    return;
  }
  const { start, end } = b;
  const width = screen.width;
  const noSelect = screen.noSelect;
  for (let row = start.row; row <= end.row && row < screen.height; row++) {
    const colStart = row === start.row ? start.col : 0;
    const colEnd = row === end.row ? Math.min(end.col, width - 1) : width - 1;
    const rowOff = row * width;
    for (let col = colStart; col <= colEnd; col++) {
      const idx = rowOff + col;
      if (noSelect[idx] === 1) {
        continue;
      }
      const cell = cellAtIndex(screen, idx);
      setCellStyleId(screen, col, row, stylePool.withSelectionBg(cell.styleId));
    }
  }
}

// src/ink/hooks/use-selection.ts
function useSelection() {
  useContext6(StdinContext_default);
  const ink = instances_default.get(process.stdout);
  return useMemo4(() => {
    if (!ink) {
      return {
        copySelection: () => "",
        copySelectionNoClear: () => "",
        clearSelection: () => {
        },
        hasSelection: () => false,
        getState: () => null,
        subscribe: () => () => {
        },
        shiftAnchor: () => {
        },
        shiftSelection: () => {
        },
        moveFocus: () => {
        },
        captureScrolledRows: () => {
        },
        setSelectionBgColor: () => {
        }
      };
    }
    return {
      copySelection: () => ink.copySelection(),
      copySelectionNoClear: () => ink.copySelectionNoClear(),
      clearSelection: () => ink.clearTextSelection(),
      hasSelection: () => ink.hasTextSelection(),
      getState: () => ink.selection,
      subscribe: (cb) => ink.subscribeToSelectionChange(cb),
      shiftAnchor: (dRow, minRow, maxRow) => shiftAnchor(ink.selection, dRow, minRow, maxRow),
      shiftSelection: (dRow, minRow, maxRow) => ink.shiftSelectionForScroll(dRow, minRow, maxRow),
      moveFocus: (move) => ink.moveSelectionFocus(move),
      captureScrolledRows: (firstRow, lastRow, side) => ink.captureScrolledRows(firstRow, lastRow, side),
      setSelectionBgColor: (color) => ink.setSelectionBgColor(color)
    };
  }, [ink]);
}
var NO_SUBSCRIBE = () => () => {
};
var ALWAYS_FALSE = () => false;
function useHasSelection() {
  useContext6(StdinContext_default);
  const ink = instances_default.get(process.stdout);
  return useSyncExternalStore(
    ink ? ink.subscribeToSelectionChange : NO_SUBSCRIBE,
    ink ? ink.hasTextSelection : ALWAYS_FALSE
  );
}

// src/ink/hooks/use-tab-status.ts
import { useContext as useContext7, useEffect as useEffect2, useRef as useRef3 } from "react";
var rgb = (r, g, b) => ({
  type: "rgb",
  r,
  g,
  b
});
var TAB_STATUS_PRESETS = {
  idle: {
    indicator: rgb(0, 215, 95),
    status: "Idle",
    statusColor: rgb(136, 136, 136)
  },
  busy: {
    indicator: rgb(255, 149, 0),
    status: "Working\u2026",
    statusColor: rgb(255, 149, 0)
  },
  waiting: {
    indicator: rgb(95, 135, 255),
    status: "Waiting",
    statusColor: rgb(95, 135, 255)
  }
};
function useTabStatus(kind) {
  const writeRaw = useContext7(TerminalWriteContext);
  const prevKindRef = useRef3(null);
  useEffect2(() => {
    if (kind === null) {
      if (prevKindRef.current !== null && writeRaw && supportsTabStatus()) {
        writeRaw(wrapForMultiplexer(CLEAR_TAB_STATUS));
      }
      prevKindRef.current = null;
      return;
    }
    prevKindRef.current = kind;
    if (!writeRaw || !supportsTabStatus()) {
      return;
    }
    writeRaw(wrapForMultiplexer(tabStatus(TAB_STATUS_PRESETS[kind])));
  }, [kind, writeRaw]);
}

// src/ink/hooks/use-terminal-focus.ts
import { useContext as useContext8 } from "react";

// src/ink/components/TerminalFocusContext.tsx
import { createContext as createContext6, useSyncExternalStore as useSyncExternalStore2 } from "react";
import { c as _c10 } from "react/compiler-runtime";

// src/ink/terminal-focus-state.ts
var focusState = "unknown";
var resolvers = /* @__PURE__ */ new Set();
var subscribers = /* @__PURE__ */ new Set();
function setTerminalFocused(v) {
  focusState = v ? "focused" : "blurred";
  for (const cb of subscribers) {
    cb();
  }
  if (!v) {
    for (const resolve of resolvers) {
      resolve();
    }
    resolvers.clear();
  }
}
function getTerminalFocused() {
  return focusState !== "blurred";
}
function getTerminalFocusState() {
  return focusState;
}
function subscribeTerminalFocus(cb) {
  subscribers.add(cb);
  return () => {
    subscribers.delete(cb);
  };
}

// src/ink/components/TerminalFocusContext.tsx
import { jsx as jsx11 } from "react/jsx-runtime";
var TerminalFocusContext = createContext6({
  isTerminalFocused: true,
  terminalFocusState: "unknown"
});
TerminalFocusContext.displayName = "TerminalFocusContext";
function TerminalFocusProvider(t0) {
  const $ = _c10(6);
  const { children } = t0;
  const isTerminalFocused = useSyncExternalStore2(subscribeTerminalFocus, getTerminalFocused);
  const terminalFocusState = useSyncExternalStore2(subscribeTerminalFocus, getTerminalFocusState);
  let t1;
  if ($[0] !== isTerminalFocused || $[1] !== terminalFocusState) {
    t1 = {
      isTerminalFocused,
      terminalFocusState
    };
    $[0] = isTerminalFocused;
    $[1] = terminalFocusState;
    $[2] = t1;
  } else {
    t1 = $[2];
  }
  const value = t1;
  let t2;
  if ($[3] !== children || $[4] !== value) {
    t2 = /* @__PURE__ */ jsx11(TerminalFocusContext.Provider, { value, children });
    $[3] = children;
    $[4] = value;
    $[5] = t2;
  } else {
    t2 = $[5];
  }
  return t2;
}
var TerminalFocusContext_default = TerminalFocusContext;

// src/ink/hooks/use-terminal-focus.ts
function useTerminalFocus() {
  const { isTerminalFocused } = useContext8(TerminalFocusContext_default);
  return isTerminalFocused;
}

// src/ink/hooks/use-terminal-title.ts
import { useContext as useContext9, useEffect as useEffect3 } from "react";
import stripAnsi2 from "strip-ansi";
function useTerminalTitle(title) {
  const writeRaw = useContext9(TerminalWriteContext);
  useEffect3(() => {
    if (title === null || !writeRaw) {
      return;
    }
    const clean = stripAnsi2(title);
    if (process.platform === "win32") {
      process.title = clean;
    } else {
      writeRaw(osc(OSC.SET_TITLE_AND_ICON, clean));
    }
  }, [title, writeRaw]);
}

// src/ink/hooks/use-terminal-viewport.ts
import { useCallback as useCallback4, useContext as useContext10, useLayoutEffect as useLayoutEffect3, useRef as useRef4 } from "react";
function useTerminalViewport() {
  const terminalSize = useContext10(TerminalSizeContext);
  const elementRef = useRef4(null);
  const entryRef = useRef4({ isVisible: true });
  const setElement = useCallback4((el) => {
    elementRef.current = el;
  }, []);
  useLayoutEffect3(() => {
    const element = elementRef.current;
    if (!element?.yogaNode || !terminalSize) {
      return;
    }
    const height = element.yogaNode.getComputedHeight();
    const rows = terminalSize.rows;
    let absoluteTop = element.yogaNode.getComputedTop();
    let parent = element.parentNode;
    let root = element.yogaNode;
    while (parent) {
      if (parent.yogaNode) {
        absoluteTop += parent.yogaNode.getComputedTop();
        root = parent.yogaNode;
      }
      if (parent.scrollTop) {
        absoluteTop -= parent.scrollTop;
      }
      parent = parent.parentNode;
    }
    const screenHeight = root.getComputedHeight();
    const bottom = absoluteTop + height;
    const cursorRestoreScroll = screenHeight > rows ? 1 : 0;
    const viewportY = Math.max(0, screenHeight - rows) + cursorRestoreScroll;
    const viewportBottom = viewportY + rows;
    const visible = bottom > viewportY && absoluteTop < viewportBottom;
    if (visible !== entryRef.current.isVisible) {
      entryRef.current = { isVisible: visible };
    }
  });
  return [setElement, entryRef.current];
}

// src/ink/measure-element.ts
var measureElement = (node) => ({
  width: node.yogaNode?.getComputedWidth() ?? 0,
  height: node.yogaNode?.getComputedHeight() ?? 0
});
var measure_element_default = measureElement;

// src/ink/root.ts
import { Stream } from "stream";

// src/ink/ink.tsx
import { closeSync, constants as fsConstants, openSync, readSync, writeSync } from "fs";
import { format } from "util";
import autoBind from "auto-bind";
import noop2 from "lodash-es/noop.js";
import throttle from "lodash-es/throttle.js";
import { ConcurrentRoot } from "react-reconciler/constants.js";
import { onExit } from "signal-exit";

// src/ink/colorize.ts
import chalk from "chalk";
function boostChalkLevelForXtermJs() {
  if (process.env.TERM_PROGRAM === "vscode" && chalk.level === 2) {
    chalk.level = 3;
    return true;
  }
  return false;
}
function clampChalkLevelForTmux() {
  if (process.env.TMUX && chalk.level > 2) {
    chalk.level = 2;
    return true;
  }
  return false;
}
var CHALK_BOOSTED_FOR_XTERMJS = boostChalkLevelForXtermJs();
var CHALK_CLAMPED_FOR_TMUX = clampChalkLevelForTmux();
var RGB_REGEX = /^rgb\(\s?(\d+),\s?(\d+),\s?(\d+)\s?\)$/;
var ANSI_REGEX = /^ansi256\(\s?(\d+)\s?\)$/;
var colorize = (str, color, type) => {
  if (!color) {
    return str;
  }
  if (color.startsWith("ansi:")) {
    const value = color.substring("ansi:".length);
    switch (value) {
      case "black":
        return type === "foreground" ? chalk.black(str) : chalk.bgBlack(str);
      case "red":
        return type === "foreground" ? chalk.red(str) : chalk.bgRed(str);
      case "green":
        return type === "foreground" ? chalk.green(str) : chalk.bgGreen(str);
      case "yellow":
        return type === "foreground" ? chalk.yellow(str) : chalk.bgYellow(str);
      case "blue":
        return type === "foreground" ? chalk.blue(str) : chalk.bgBlue(str);
      case "magenta":
        return type === "foreground" ? chalk.magenta(str) : chalk.bgMagenta(str);
      case "cyan":
        return type === "foreground" ? chalk.cyan(str) : chalk.bgCyan(str);
      case "white":
        return type === "foreground" ? chalk.white(str) : chalk.bgWhite(str);
      case "blackBright":
        return type === "foreground" ? chalk.blackBright(str) : chalk.bgBlackBright(str);
      case "redBright":
        return type === "foreground" ? chalk.redBright(str) : chalk.bgRedBright(str);
      case "greenBright":
        return type === "foreground" ? chalk.greenBright(str) : chalk.bgGreenBright(str);
      case "yellowBright":
        return type === "foreground" ? chalk.yellowBright(str) : chalk.bgYellowBright(str);
      case "blueBright":
        return type === "foreground" ? chalk.blueBright(str) : chalk.bgBlueBright(str);
      case "magentaBright":
        return type === "foreground" ? chalk.magentaBright(str) : chalk.bgMagentaBright(str);
      case "cyanBright":
        return type === "foreground" ? chalk.cyanBright(str) : chalk.bgCyanBright(str);
      case "whiteBright":
        return type === "foreground" ? chalk.whiteBright(str) : chalk.bgWhiteBright(str);
    }
  }
  if (color.startsWith("#")) {
    return type === "foreground" ? chalk.hex(color)(str) : chalk.bgHex(color)(str);
  }
  if (color.startsWith("ansi256")) {
    const matches = ANSI_REGEX.exec(color);
    if (!matches) {
      return str;
    }
    const value = Number(matches[1]);
    return type === "foreground" ? chalk.ansi256(value)(str) : chalk.bgAnsi256(value)(str);
  }
  if (color.startsWith("rgb")) {
    const matches = RGB_REGEX.exec(color);
    if (!matches) {
      return str;
    }
    const firstValue = Number(matches[1]);
    const secondValue = Number(matches[2]);
    const thirdValue = Number(matches[3]);
    return type === "foreground" ? chalk.rgb(firstValue, secondValue, thirdValue)(str) : chalk.bgRgb(firstValue, secondValue, thirdValue)(str);
  }
  return str;
};
function applyTextStyles(text, styles2) {
  let result = text;
  if (styles2.inverse) {
    result = chalk.inverse(result);
  }
  if (styles2.strikethrough) {
    result = chalk.strikethrough(result);
  }
  if (styles2.underline) {
    result = chalk.underline(result);
  }
  if (styles2.italic) {
    result = chalk.italic(result);
  }
  if (styles2.bold) {
    result = chalk.bold(result);
  }
  if (styles2.dim) {
    result = chalk.dim(result);
  }
  if (styles2.color) {
    result = colorize(result, styles2.color, "foreground");
  }
  if (styles2.backgroundColor) {
    result = colorize(result, styles2.backgroundColor, "background");
  }
  return result;
}
function applyColor(text, color) {
  if (!color) {
    return text;
  }
  return colorize(text, color, "foreground");
}

// src/ink/components/App.tsx
import { PureComponent } from "react";

// src/utils/earlyInput.ts
var isCapturing = false;
var readableHandler = null;
function stopCapturingEarlyInput() {
  if (!isCapturing) {
    return;
  }
  isCapturing = false;
  if (readableHandler) {
    process.stdin.removeListener("readable", readableHandler);
    readableHandler = null;
  }
}

// src/utils/fullscreen.ts
function isMouseClicksDisabled() {
  return /^(1|true|yes|on)$/.test((process.env.HERMES_TUI_DISABLE_MOUSE_CLICKS ?? "").trim().toLowerCase());
}

// src/ink/parse-keypress.ts
import { Buffer as Buffer3 } from "buffer";
var META_KEY_CODE_RE = /^(?:\x1b)([a-zA-Z0-9])$/;
var FN_KEY_RE = (
  // eslint-disable-next-line no-control-regex
  /^(?:\x1b+)(O|N|\[|\[\[)(?:(\d+)(?:;(\d+))?([~^$])|(?:1;)?(\d+)?([a-zA-Z]))/
);
var CSI_U_RE = /^\x1b\[(\d+)(?:;(\d+))?u/;
var MODIFY_OTHER_KEYS_RE = /^\x1b\[27;(\d+);(\d+)~/;
var DECRPM_RE = /^\x1b\[\?(\d+);(\d+)\$y$/;
var DA1_RE = /^\x1b\[\?([\d;]*)c$/;
var DA2_RE = /^\x1b\[>([\d;]*)c$/;
var KITTY_FLAGS_RE = /^\x1b\[\?(\d+)u$/;
var CURSOR_POSITION_RE = /^\x1b\[\?(\d+);(\d+)R$/;
var OSC_RESPONSE_RE = /^\x1b\](\d+);(.*?)(?:\x07|\x1b\\)$/s;
var XTVERSION_RE = /^\x1bP>\|(.*?)(?:\x07|\x1b\\)$/s;
var SGR_MOUSE_RE = /^\x1b\[<(\d+);(\d+);(\d+)([Mm])$/;
function createPasteKey(content) {
  return {
    kind: "key",
    name: "",
    fn: false,
    ctrl: false,
    meta: false,
    shift: false,
    option: false,
    super: false,
    sequence: content,
    raw: content,
    isPasted: true
  };
}
function parseTerminalResponse(s) {
  if (s.startsWith("\x1B[")) {
    let m;
    if (m = DECRPM_RE.exec(s)) {
      return {
        type: "decrpm",
        mode: parseInt(m[1], 10),
        status: parseInt(m[2], 10)
      };
    }
    if (m = DA1_RE.exec(s)) {
      return { type: "da1", params: splitNumericParams(m[1]) };
    }
    if (m = DA2_RE.exec(s)) {
      return { type: "da2", params: splitNumericParams(m[1]) };
    }
    if (m = KITTY_FLAGS_RE.exec(s)) {
      return { type: "kittyKeyboard", flags: parseInt(m[1], 10) };
    }
    if (m = CURSOR_POSITION_RE.exec(s)) {
      return {
        type: "cursorPosition",
        row: parseInt(m[1], 10),
        col: parseInt(m[2], 10)
      };
    }
    return null;
  }
  if (s.startsWith("\x1B]")) {
    const m = OSC_RESPONSE_RE.exec(s);
    if (m) {
      return { type: "osc", code: parseInt(m[1], 10), data: m[2] };
    }
  }
  if (s.startsWith("\x1BP")) {
    const m = XTVERSION_RE.exec(s);
    if (m) {
      return { type: "xtversion", name: m[1] };
    }
  }
  return null;
}
function splitNumericParams(params) {
  if (!params) {
    return [];
  }
  return params.split(";").map((p) => parseInt(p, 10));
}
var INITIAL_STATE = {
  mode: "NORMAL",
  incomplete: "",
  pasteBuffer: ""
};
function inputToString(input) {
  if (Buffer3.isBuffer(input)) {
    if (input[0] > 127 && input[1] === void 0) {
      ;
      input[0] -= 128;
      return "\x1B" + String(input);
    } else {
      return String(input);
    }
  } else if (input !== void 0 && typeof input !== "string") {
    return String(input);
  } else if (!input) {
    return "";
  } else {
    return input;
  }
}
function parseMultipleKeypresses(prevState, input = "") {
  const isFlush = input === null;
  const inputString = isFlush ? "" : inputToString(input);
  const tokenizer = prevState._tokenizer ?? createTokenizer({ x10Mouse: true });
  const tokens = isFlush ? tokenizer.flush() : tokenizer.feed(inputString);
  const keys = [];
  let inPaste = prevState.mode === "IN_PASTE";
  let pasteBuffer = prevState.pasteBuffer;
  for (const token of tokens) {
    if (token.type === "sequence") {
      if (token.value === PASTE_START) {
        inPaste = true;
        pasteBuffer = "";
      } else if (token.value === PASTE_END) {
        keys.push(createPasteKey(pasteBuffer));
        inPaste = false;
        pasteBuffer = "";
      } else if (inPaste) {
        pasteBuffer += token.value;
      } else {
        const response = parseTerminalResponse(token.value);
        if (response) {
          keys.push({ kind: "response", sequence: token.value, response });
        } else {
          const mouse = parseMouseEvent(token.value);
          if (mouse) {
            keys.push(mouse);
          } else {
            keys.push(parseKeypress(token.value));
          }
        }
      }
    } else if (token.type === "text") {
      if (inPaste) {
        pasteBuffer += token.value;
      } else if (/^\[<\d+;\d+;\d+[Mm]$/.test(token.value) || /^\[M[\x60-\x7f][\x20-\uffff]{2}$/.test(token.value)) {
        const resynthesized = "\x1B" + token.value;
        const mouse = parseMouseEvent(resynthesized);
        keys.push(mouse ?? parseKeypress(resynthesized));
      } else {
        keys.push(parseKeypress(token.value));
      }
    }
  }
  if (isFlush && inPaste && pasteBuffer) {
    keys.push(createPasteKey(pasteBuffer));
    inPaste = false;
    pasteBuffer = "";
  }
  const newState = {
    mode: inPaste ? "IN_PASTE" : "NORMAL",
    incomplete: tokenizer.buffer(),
    pasteBuffer,
    _tokenizer: tokenizer
  };
  return [keys, newState];
}
var keyName = {
  /* xterm/gnome ESC O letter */
  OP: "f1",
  OQ: "f2",
  OR: "f3",
  OS: "f4",
  /* Application keypad mode (numpad digits 0-9) */
  Op: "0",
  Oq: "1",
  Or: "2",
  Os: "3",
  Ot: "4",
  Ou: "5",
  Ov: "6",
  Ow: "7",
  Ox: "8",
  Oy: "9",
  /* Application keypad mode (numpad operators) */
  Oj: "*",
  Ok: "+",
  Ol: ",",
  Om: "-",
  On: ".",
  Oo: "/",
  OM: "return",
  /* xterm/rxvt ESC [ number ~ */
  "[11~": "f1",
  "[12~": "f2",
  "[13~": "f3",
  "[14~": "f4",
  /* from Cygwin and used in libuv */
  "[[A": "f1",
  "[[B": "f2",
  "[[C": "f3",
  "[[D": "f4",
  "[[E": "f5",
  /* common */
  "[15~": "f5",
  "[17~": "f6",
  "[18~": "f7",
  "[19~": "f8",
  "[20~": "f9",
  "[21~": "f10",
  "[23~": "f11",
  "[24~": "f12",
  /* xterm ESC [ letter */
  "[A": "up",
  "[B": "down",
  "[C": "right",
  "[D": "left",
  "[E": "clear",
  "[F": "end",
  "[H": "home",
  /* xterm/gnome ESC O letter */
  OA: "up",
  OB: "down",
  OC: "right",
  OD: "left",
  OE: "clear",
  OF: "end",
  OH: "home",
  /* xterm/rxvt ESC [ number ~ */
  "[1~": "home",
  "[2~": "insert",
  "[3~": "delete",
  "[4~": "end",
  "[5~": "pageup",
  "[6~": "pagedown",
  /* putty */
  "[[5~": "pageup",
  "[[6~": "pagedown",
  /* rxvt */
  "[7~": "home",
  "[8~": "end",
  /* rxvt keys with modifiers */
  "[a": "up",
  "[b": "down",
  "[c": "right",
  "[d": "left",
  "[e": "clear",
  "[2$": "insert",
  "[3$": "delete",
  "[5$": "pageup",
  "[6$": "pagedown",
  "[7$": "home",
  "[8$": "end",
  Oa: "up",
  Ob: "down",
  Oc: "right",
  Od: "left",
  Oe: "clear",
  "[2^": "insert",
  "[3^": "delete",
  "[5^": "pageup",
  "[6^": "pagedown",
  "[7^": "home",
  "[8^": "end",
  /* misc. */
  "[Z": "tab"
};
var nonAlphanumericKeys = [
  // Filter out single-character values (digits, operators from numpad) since
  // those are printable characters that should produce input
  ...Object.values(keyName).filter((v) => v.length > 1),
  // escape and backspace are assigned directly in parseKeypress (not via the
  // keyName map), so the spread above misses them. Without these, ctrl+escape
  // via Kitty/modifyOtherKeys leaks the literal word "escape" as input text
  // (input-event.ts:58 assigns keypress.name when ctrl is set).
  "escape",
  "backspace",
  "wheelup",
  "wheeldown",
  "mouse"
];
var isShiftKey = (code) => {
  return ["[a", "[b", "[c", "[d", "[e", "[2$", "[3$", "[5$", "[6$", "[7$", "[8$", "[Z"].includes(code);
};
var isCtrlKey = (code) => {
  return ["Oa", "Ob", "Oc", "Od", "Oe", "[2^", "[3^", "[5^", "[6^", "[7^", "[8^"].includes(code);
};
function decodeModifier(modifier) {
  const m = modifier - 1;
  return {
    shift: !!(m & 1),
    meta: !!(m & 2),
    ctrl: !!(m & 4),
    super: !!(m & 8)
  };
}
function keycodeToName(keycode) {
  switch (keycode) {
    case 9:
      return "tab";
    case 13:
      return "return";
    case 27:
      return "escape";
    case 32:
      return "space";
    case 127:
      return "backspace";
    // Kitty keyboard protocol numpad keys (KP_0 through KP_9)
    case 57399:
      return "0";
    case 57400:
      return "1";
    case 57401:
      return "2";
    case 57402:
      return "3";
    case 57403:
      return "4";
    case 57404:
      return "5";
    case 57405:
      return "6";
    case 57406:
      return "7";
    case 57407:
      return "8";
    case 57408:
      return "9";
    case 57409:
      return ".";
    case 57410:
      return "/";
    case 57411:
      return "*";
    case 57412:
      return "-";
    case 57413:
      return "+";
    case 57414:
      return "return";
    case 57415:
      return "=";
    default:
      if (keycode >= 32 && keycode <= 126) {
        return String.fromCharCode(keycode).toLowerCase();
      }
      return void 0;
  }
}
function parseMouseEvent(s) {
  const match = SGR_MOUSE_RE.exec(s);
  if (!match) {
    return null;
  }
  const button = parseInt(match[1], 10);
  if ((button & 64) !== 0) {
    return null;
  }
  return {
    kind: "mouse",
    button,
    action: match[4] === "M" ? "press" : "release",
    col: parseInt(match[2], 10),
    row: parseInt(match[3], 10),
    sequence: s
  };
}
function parseKeypress(s = "") {
  let parts;
  const key = {
    kind: "key",
    name: "",
    fn: false,
    ctrl: false,
    meta: false,
    shift: false,
    option: false,
    super: false,
    sequence: s,
    raw: s,
    isPasted: false
  };
  key.sequence = key.sequence || s || key.name;
  let match;
  if (match = CSI_U_RE.exec(s)) {
    const codepoint = parseInt(match[1], 10);
    const modifier = match[2] ? parseInt(match[2], 10) : 1;
    const mods = decodeModifier(modifier);
    const name = keycodeToName(codepoint);
    return {
      kind: "key",
      name,
      fn: false,
      ctrl: mods.ctrl,
      meta: mods.meta,
      shift: mods.shift,
      option: false,
      super: mods.super,
      sequence: s,
      raw: s,
      isPasted: false
    };
  }
  if (match = MODIFY_OTHER_KEYS_RE.exec(s)) {
    const mods = decodeModifier(parseInt(match[1], 10));
    const name = keycodeToName(parseInt(match[2], 10));
    return {
      kind: "key",
      name,
      fn: false,
      ctrl: mods.ctrl,
      meta: mods.meta,
      shift: mods.shift,
      option: false,
      super: mods.super,
      sequence: s,
      raw: s,
      isPasted: false
    };
  }
  if (match = SGR_MOUSE_RE.exec(s)) {
    const button = parseInt(match[1], 10);
    if ((button & 67) === 64) {
      return createNavKey(s, "wheelup", false);
    }
    if ((button & 67) === 65) {
      return createNavKey(s, "wheeldown", false);
    }
    return createNavKey(s, "mouse", false);
  }
  if (s.length === 6 && s.startsWith("\x1B[M")) {
    const button = s.charCodeAt(3) - 32;
    if ((button & 67) === 64) {
      return createNavKey(s, "wheelup", false);
    }
    if ((button & 67) === 65) {
      return createNavKey(s, "wheeldown", false);
    }
    return createNavKey(s, "mouse", false);
  }
  if (s === "\r" || s === "\n") {
    key.raw = void 0;
    key.name = "return";
  } else if (s === "	") {
    key.name = "tab";
  } else if (s === "\b" || s === "\x1B\b") {
    key.name = "backspace";
    key.meta = s.charAt(0) === "\x1B";
  } else if (s === "\x7F" || s === "\x1B\x7F") {
    key.name = "backspace";
    key.meta = s.charAt(0) === "\x1B";
  } else if (s === "\x1B" || s === "\x1B\x1B") {
    key.name = "escape";
    key.meta = s.length === 2;
  } else if (s === " " || s === "\x1B ") {
    key.name = "space";
    key.meta = s.length === 2;
  } else if (s === "") {
    key.name = "_";
    key.ctrl = true;
  } else if (s <= "" && s.length === 1) {
    key.name = String.fromCharCode(s.charCodeAt(0) + "a".charCodeAt(0) - 1);
    key.ctrl = true;
  } else if (s.length === 1 && s >= "0" && s <= "9") {
    key.name = "number";
  } else if (s.length === 1 && s >= "a" && s <= "z") {
    key.name = s;
  } else if (s.length === 1 && s >= "A" && s <= "Z") {
    key.name = s.toLowerCase();
    key.shift = true;
  } else if (parts = META_KEY_CODE_RE.exec(s)) {
    key.meta = true;
    key.shift = /^[A-Z]$/.test(parts[1]);
  } else if (parts = FN_KEY_RE.exec(s)) {
    const segs = [...s];
    if (segs[0] === "\x1B" && segs[1] === "\x1B") {
      key.option = true;
    }
    const code = [parts[1], parts[2], parts[4], parts[6]].filter(Boolean).join("");
    const modifier = (parts[3] || parts[5] || 1) - 1;
    key.ctrl = !!(modifier & 4);
    key.meta = !!(modifier & 2);
    key.super = !!(modifier & 8);
    key.shift = !!(modifier & 1);
    key.code = code;
    key.name = keyName[code];
    key.shift = isShiftKey(code) || key.shift;
    key.ctrl = isCtrlKey(code) || key.ctrl;
  }
  if (key.raw === "\x1Bb") {
    key.meta = true;
    key.name = "left";
  } else if (key.raw === "\x1Bf") {
    key.meta = true;
    key.name = "right";
  }
  switch (s) {
    case "\x1B[1~":
      return createNavKey(s, "home", false);
    case "\x1B[4~":
      return createNavKey(s, "end", false);
    case "\x1B[5~":
      return createNavKey(s, "pageup", false);
    case "\x1B[6~":
      return createNavKey(s, "pagedown", false);
    case "\x1B[1;5D":
      return createNavKey(s, "left", true);
    case "\x1B[1;5C":
      return createNavKey(s, "right", true);
  }
  return key;
}
function createNavKey(s, name, ctrl) {
  return {
    kind: "key",
    name,
    ctrl,
    meta: false,
    shift: false,
    option: false,
    super: false,
    fn: false,
    sequence: s,
    raw: s,
    isPasted: false
  };
}

// src/ink/events/input-event.ts
function parseKey(keypress) {
  const key = {
    upArrow: keypress.name === "up",
    downArrow: keypress.name === "down",
    leftArrow: keypress.name === "left",
    rightArrow: keypress.name === "right",
    pageDown: keypress.name === "pagedown",
    pageUp: keypress.name === "pageup",
    wheelUp: keypress.name === "wheelup",
    wheelDown: keypress.name === "wheeldown",
    home: keypress.name === "home",
    end: keypress.name === "end",
    return: keypress.name === "return",
    escape: keypress.name === "escape",
    fn: keypress.fn,
    ctrl: keypress.ctrl,
    shift: keypress.shift,
    tab: keypress.name === "tab",
    backspace: keypress.name === "backspace",
    delete: keypress.name === "delete",
    // `parseKeypress` parses \u001B\u001B[A (meta + up arrow) as meta = false
    // but with option = true, so we need to take this into account here
    // to avoid breaking changes in Ink.
    // TODO(vadimdemedes): consider removing this in the next major version.
    meta: keypress.meta || keypress.name === "escape" || keypress.option,
    // Super (Cmd on macOS / Win key) — only arrives via kitty keyboard
    // protocol CSI u sequences. Distinct from meta (Alt/Option) so
    // bindings like cmd+c can be expressed separately from opt+c.
    super: keypress.super
  };
  let input = keypress.ctrl ? keypress.name : keypress.sequence;
  if (input === void 0) {
    input = "";
  }
  if (keypress.ctrl && input === "space") {
    input = " ";
  }
  if (keypress.code && !keypress.name) {
    input = "";
  }
  if (!keypress.name && /^\[<\d+;\d+;\d+[Mm]/.test(input)) {
    input = "";
  }
  if (input.startsWith("\x1B")) {
    input = input.slice(1);
  }
  let processedAsSpecialSequence = false;
  if (/^\[\d/.test(input) && input.endsWith("u")) {
    if (!keypress.name) {
      input = "";
    } else {
      input = keypress.name === "space" ? " " : keypress.name === "escape" ? "" : keypress.name;
    }
    processedAsSpecialSequence = true;
  }
  if (input.startsWith("[27;") && input.endsWith("~")) {
    if (!keypress.name) {
      input = "";
    } else {
      input = keypress.name === "space" ? " " : keypress.name === "escape" ? "" : keypress.name;
    }
    processedAsSpecialSequence = true;
  }
  if (input.startsWith("O") && input.length === 2 && keypress.name && keypress.name.length === 1) {
    input = keypress.name;
    processedAsSpecialSequence = true;
  }
  if (!processedAsSpecialSequence && keypress.name && nonAlphanumericKeys.includes(keypress.name)) {
    input = "";
  }
  if (input.length === 1 && typeof input[0] === "string" && input[0] >= "A" && input[0] <= "Z") {
    key.shift = true;
  }
  return [key, input];
}
var InputEvent = class extends Event {
  keypress;
  key;
  input;
  constructor(keypress) {
    super();
    const [key, input] = parseKey(keypress);
    this.keypress = keypress;
    this.key = key;
    this.input = input;
  }
};

// src/ink/events/terminal-focus-event.ts
var TerminalFocusEvent = class extends Event {
  type;
  constructor(type) {
    super();
    this.type = type;
  }
};

// src/ink/terminal-querier.ts
function xtversion() {
  return {
    request: csi(">0q"),
    match: (r) => r.type === "xtversion"
  };
}
var SENTINEL = csi("c");
var TerminalQuerier = class {
  constructor(stdout) {
    this.stdout = stdout;
  }
  /**
   * Interleaved queue of queries and sentinels in send order. Terminals
   * respond in order, so each flush() barrier only drains queries queued
   * before it — concurrent batches from independent callers stay isolated.
   */
  queue = [];
  /**
   * Send a query and wait for its response.
   *
   * Resolves with the response when `query.match` matches an incoming
   * TerminalResponse, or with `undefined` when a flush() sentinel arrives
   * before any matching response (meaning the terminal ignored the query).
   *
   * Never rejects; never times out on its own. If you never call flush()
   * and the terminal doesn't respond, the promise remains pending.
   */
  send(query) {
    return new Promise((resolve) => {
      this.queue.push({
        kind: "query",
        match: query.match,
        resolve: (r) => resolve(r)
      });
      this.stdout.write(query.request);
    });
  }
  /**
   * Send the DA1 sentinel. Resolves when DA1's response arrives.
   *
   * As a side effect, all queries still pending when DA1 arrives are
   * resolved with `undefined` (terminal didn't respond → doesn't support
   * the query). This is the barrier that makes send() timeout-free.
   *
   * Safe to call with no pending queries — still waits for a round-trip.
   */
  flush() {
    return new Promise((resolve) => {
      this.queue.push({ kind: "sentinel", resolve });
      this.stdout.write(SENTINEL);
    });
  }
  /**
   * Dispatch a response parsed from stdin. Called by App.tsx's
   * processKeysInBatch for every `kind: 'response'` item.
   *
   * Matching strategy:
   * - First, try to match a pending query (FIFO, first match wins).
   *   This lets callers send(da1()) explicitly if they want the DA1
   *   params — a separate DA1 write means the terminal sends TWO DA1
   *   responses. The first matches the explicit query; the second
   *   (unmatched) fires the sentinel.
   * - Otherwise, if this is a DA1, fire the FIRST pending sentinel:
   *   resolve any queries queued before that sentinel with undefined
   *   (the terminal answered DA1 without answering them → unsupported)
   *   and signal its flush() completion. Only draining up to the first
   *   sentinel keeps later batches intact when multiple callers have
   *   concurrent queries in flight.
   * - Unsolicited responses (no match, no sentinel) are silently dropped.
   */
  onResponse(r) {
    const idx = this.queue.findIndex((p) => p.kind === "query" && p.match(r));
    if (idx !== -1) {
      const [q] = this.queue.splice(idx, 1);
      if (q?.kind === "query") {
        q.resolve(r);
      }
      return;
    }
    if (r.type === "da1") {
      const s = this.queue.findIndex((p) => p.kind === "sentinel");
      if (s === -1) {
        return;
      }
      for (const p of this.queue.splice(0, s + 1)) {
        if (p.kind === "query") {
          p.resolve(void 0);
        } else {
          p.resolve();
        }
      }
    }
  }
};

// src/ink/components/ClockContext.tsx
import { createContext as createContext7, useEffect as useEffect4, useState as useState2 } from "react";
import { c as _c11 } from "react/compiler-runtime";

// src/ink/constants.ts
var FRAME_INTERVAL_MS = 16;
var BLURRED_FRAME_INTERVAL_MS = FRAME_INTERVAL_MS;

// src/ink/components/ClockContext.tsx
import { jsx as jsx12 } from "react/jsx-runtime";
function createClock(tickIntervalMs) {
  const subscribers2 = /* @__PURE__ */ new Map();
  let interval = null;
  let currentTickIntervalMs = tickIntervalMs;
  let startTime = 0;
  let tickTime = 0;
  function tick() {
    tickTime = Date.now() - startTime;
    for (const onChange of subscribers2.keys()) {
      onChange();
    }
  }
  function updateInterval() {
    const anyKeepAlive = [...subscribers2.values()].some(Boolean);
    if (anyKeepAlive) {
      if (interval) {
        clearInterval(interval);
        interval = null;
      }
      if (startTime === 0) {
        startTime = Date.now();
      }
      interval = setInterval(tick, currentTickIntervalMs);
    } else if (interval) {
      clearInterval(interval);
      interval = null;
    }
  }
  return {
    subscribe(onChange, keepAlive) {
      subscribers2.set(onChange, keepAlive);
      updateInterval();
      return () => {
        subscribers2.delete(onChange);
        updateInterval();
      };
    },
    now() {
      if (startTime === 0) {
        startTime = Date.now();
      }
      if (interval && tickTime) {
        return tickTime;
      }
      return Date.now() - startTime;
    },
    setTickInterval(ms) {
      if (ms === currentTickIntervalMs) {
        return;
      }
      currentTickIntervalMs = ms;
      updateInterval();
    }
  };
}
var ClockContext = createContext7(null);
function ClockProvider(t0) {
  const $ = _c11(7);
  const { children } = t0;
  const [clock] = useState2(_temp);
  const focused = useTerminalFocus();
  let t1;
  let t2;
  if ($[0] !== clock || $[1] !== focused) {
    t1 = () => {
      clock.setTickInterval(focused ? FRAME_INTERVAL_MS : BLURRED_FRAME_INTERVAL_MS);
    };
    t2 = [clock, focused];
    $[0] = clock;
    $[1] = focused;
    $[2] = t1;
    $[3] = t2;
  } else {
    t1 = $[2];
    t2 = $[3];
  }
  useEffect4(t1, t2);
  let t3;
  if ($[4] !== children || $[5] !== clock) {
    t3 = /* @__PURE__ */ jsx12(ClockContext.Provider, { value: clock, children });
    $[4] = children;
    $[5] = clock;
    $[6] = t3;
  } else {
    t3 = $[6];
  }
  return t3;
}
function _temp() {
  return createClock(FRAME_INTERVAL_MS);
}

// src/ink/components/ErrorOverview.tsx
import { readFileSync } from "fs";
import codeExcerpt from "code-excerpt";
import StackUtils from "stack-utils";
import { jsx as jsx13, jsxs } from "react/jsx-runtime";
var cleanupPath = (path) => {
  return path?.replace(`file://${process.cwd()}/`, "");
};
var stackUtils;
function getStackUtils() {
  return stackUtils ??= new StackUtils({
    cwd: process.cwd(),
    internals: StackUtils.nodeInternals()
  });
}
function ErrorOverview({ error }) {
  const stack = error.stack ? error.stack.split("\n").slice(1) : void 0;
  const origin = stack ? getStackUtils().parseLine(stack[0]) : void 0;
  const filePath = cleanupPath(origin?.file);
  let excerpt;
  let lineWidth2 = 0;
  if (filePath && origin?.line) {
    try {
      const sourceCode = readFileSync(filePath, "utf8");
      excerpt = codeExcerpt(sourceCode, origin.line);
      if (excerpt) {
        for (const { line } of excerpt) {
          lineWidth2 = Math.max(lineWidth2, String(line).length);
        }
      }
    } catch {
    }
  }
  return /* @__PURE__ */ jsxs(Box_default, { flexDirection: "column", padding: 1, children: [
    /* @__PURE__ */ jsxs(Box_default, { children: [
      /* @__PURE__ */ jsxs(Text, { backgroundColor: "ansi:red", color: "ansi:white", children: [
        " ",
        "ERROR",
        " "
      ] }),
      /* @__PURE__ */ jsxs(Text, { children: [
        " ",
        error.message
      ] })
    ] }),
    origin && filePath && /* @__PURE__ */ jsx13(Box_default, { marginTop: 1, children: /* @__PURE__ */ jsxs(Text, { dim: true, children: [
      filePath,
      ":",
      origin.line,
      ":",
      origin.column
    ] }) }),
    origin && excerpt && /* @__PURE__ */ jsx13(Box_default, { flexDirection: "column", marginTop: 1, children: excerpt.map(({ line: line_0, value }) => /* @__PURE__ */ jsxs(Box_default, { children: [
      /* @__PURE__ */ jsx13(Box_default, { width: lineWidth2 + 1, children: /* @__PURE__ */ jsxs(
        Text,
        {
          backgroundColor: line_0 === origin.line ? "ansi:red" : void 0,
          color: line_0 === origin.line ? "ansi:white" : void 0,
          dim: line_0 !== origin.line,
          children: [
            String(line_0).padStart(lineWidth2, " "),
            ":"
          ]
        }
      ) }),
      /* @__PURE__ */ jsx13(
        Text,
        {
          backgroundColor: line_0 === origin.line ? "ansi:red" : void 0,
          color: line_0 === origin.line ? "ansi:white" : void 0,
          children: " " + value
        },
        line_0
      )
    ] }, line_0)) }),
    error.stack && /* @__PURE__ */ jsx13(Box_default, { flexDirection: "column", marginTop: 1, children: error.stack.split("\n").slice(1).map((line_1) => {
      const parsedLine = getStackUtils().parseLine(line_1);
      if (!parsedLine) {
        return /* @__PURE__ */ jsxs(Box_default, { children: [
          /* @__PURE__ */ jsx13(Text, { dim: true, children: "- " }),
          /* @__PURE__ */ jsx13(Text, { bold: true, children: line_1 })
        ] }, line_1);
      }
      return /* @__PURE__ */ jsxs(Box_default, { children: [
        /* @__PURE__ */ jsx13(Text, { dim: true, children: "- " }),
        /* @__PURE__ */ jsx13(Text, { bold: true, children: parsedLine.function }),
        /* @__PURE__ */ jsxs(Text, { dim: true, children: [
          " ",
          "(",
          cleanupPath(parsedLine.file) ?? "",
          ":",
          parsedLine.line,
          ":",
          parsedLine.column,
          ")"
        ] })
      ] }, line_1);
    }) })
  ] });
}

// src/ink/components/App.tsx
import { jsx as jsx14 } from "react/jsx-runtime";
var SUPPORTS_SUSPEND = false;
var STDIN_RESUME_GAP_MS = 5e3;
var MULTI_CLICK_TIMEOUT_MS = 500;
var MULTI_CLICK_DISTANCE = 1;
var App = class extends PureComponent {
  static displayName = "InternalApp";
  static getDerivedStateFromError(error) {
    return {
      error
    };
  }
  state = {
    error: void 0
  };
  // Count how many components enabled raw mode to avoid disabling
  // raw mode until all components don't need it anymore
  rawModeEnabledCount = 0;
  inputEmitter = new EventEmitter();
  keyParseState = INITIAL_STATE;
  // Timer for flushing incomplete escape sequences
  incompleteEscapeTimer = null;
  // Timeout durations for incomplete sequences (ms)
  NORMAL_TIMEOUT = 50;
  // Short timeout for regular esc sequences
  PASTE_TIMEOUT = 500;
  // Longer timeout for paste operations
  // Terminal query/response dispatch. Responses arrive on stdin (parsed
  // out by parse-keypress) and are routed to pending promise resolvers.
  querier = new TerminalQuerier(this.props.stdout);
  // Multi-click tracking for double/triple-click text selection. A click
  // within MULTI_CLICK_TIMEOUT_MS and MULTI_CLICK_DISTANCE of the previous
  // click increments clickCount; otherwise it resets to 1.
  lastClickTime = 0;
  lastClickCol = -1;
  lastClickRow = -1;
  clickCount = 0;
  // Deferred hyperlink-open timer — cancelled if a second click arrives
  // within MULTI_CLICK_TIMEOUT_MS (so double-clicking a hyperlink selects
  // the word without also opening the browser). DOM onClick dispatch is
  // NOT deferred — it returns true from onClickAt and skips this timer.
  pendingHyperlinkTimer = null;
  // Last mode-1003 motion position. Terminals already dedupe to cell
  // granularity but this also lets us skip dispatchHover entirely on
  // repeat events (drag-then-release at same cell, etc.).
  lastHoverCol = -1;
  lastHoverRow = -1;
  mouseCaptureTarget;
  // Timestamp of last stdin chunk. Used to detect long gaps (tmux attach,
  // ssh reconnect, laptop wake) and trigger terminal mode re-assert.
  // Initialized to now so startup doesn't false-trigger.
  lastStdinTime = Date.now();
  // Determines if TTY is supported on the provided stdin
  isRawModeSupported() {
    return this.props.stdin.isTTY;
  }
  render() {
    return /* @__PURE__ */ jsx14(
      TerminalSizeContext.Provider,
      {
        value: {
          columns: this.props.terminalColumns,
          rows: this.props.terminalRows
        },
        children: /* @__PURE__ */ jsx14(
          AppContext_default.Provider,
          {
            value: {
              exit: this.handleExit
            },
            children: /* @__PURE__ */ jsx14(
              StdinContext_default.Provider,
              {
                value: {
                  stdin: this.props.stdin,
                  setRawMode: this.handleSetRawMode,
                  isRawModeSupported: this.isRawModeSupported(),
                  exitOnCtrlC: this.props.exitOnCtrlC,
                  inputEmitter: this.inputEmitter,
                  querier: this.querier
                },
                children: /* @__PURE__ */ jsx14(TerminalFocusProvider, { children: /* @__PURE__ */ jsx14(ClockProvider, { children: /* @__PURE__ */ jsx14(CursorDeclarationContext_default.Provider, { value: this.props.onCursorDeclaration ?? (() => {
                }), children: this.state.error ? /* @__PURE__ */ jsx14(ErrorOverview, { error: this.state.error }) : this.props.children }) }) })
              }
            )
          }
        )
      }
    );
  }
  componentDidMount() {
    if (this.props.stdout.isTTY) {
      this.props.stdout.write(HIDE_CURSOR);
    }
  }
  componentWillUnmount() {
    if (this.props.stdout.isTTY) {
      this.props.stdout.write(SHOW_CURSOR);
    }
    if (this.incompleteEscapeTimer) {
      clearTimeout(this.incompleteEscapeTimer);
      this.incompleteEscapeTimer = null;
    }
    if (this.pendingHyperlinkTimer) {
      clearTimeout(this.pendingHyperlinkTimer);
      this.pendingHyperlinkTimer = null;
    }
    if (this.isRawModeSupported()) {
      this.handleSetRawMode(false);
    }
  }
  componentDidCatch(error) {
    this.handleExit(error);
  }
  handleSetRawMode = (isEnabled) => {
    const { stdin } = this.props;
    if (!this.isRawModeSupported()) {
      if (stdin === process.stdin) {
        throw new Error(
          "Raw mode is not supported on the current process.stdin, which Ink uses as input stream by default.\nRead about how to prevent this error on https://github.com/vadimdemedes/ink/#israwmodesupported"
        );
      } else {
        throw new Error(
          "Raw mode is not supported on the stdin provided to Ink.\nRead about how to prevent this error on https://github.com/vadimdemedes/ink/#israwmodesupported"
        );
      }
    }
    stdin.setEncoding("utf8");
    if (isEnabled) {
      if (this.rawModeEnabledCount === 0) {
        stopCapturingEarlyInput();
        stdin.ref();
        stdin.setRawMode(true);
        stdin.addListener("readable", this.handleReadable);
        this.props.stdout.write(EBP);
        this.props.stdout.write(EFE);
        if (supportsExtendedKeys()) {
          this.props.stdout.write(ENABLE_KITTY_KEYBOARD);
          this.props.stdout.write(ENABLE_MODIFY_OTHER_KEYS);
        }
        setImmediate(() => {
          void Promise.all([this.querier.send(xtversion()), this.querier.flush()]).then(([r]) => {
            if (r) {
              setXtversionName(r.name);
              logForDebugging(`XTVERSION: terminal identified as "${r.name}"`);
            } else {
              logForDebugging("XTVERSION: no reply (terminal ignored query)");
            }
          });
        });
      }
      this.rawModeEnabledCount++;
      return;
    }
    if (--this.rawModeEnabledCount === 0) {
      this.props.stdout.write(DISABLE_MODIFY_OTHER_KEYS);
      this.props.stdout.write(DISABLE_KITTY_KEYBOARD);
      this.props.stdout.write(DFE);
      this.props.stdout.write(DBP);
      stdin.setRawMode(false);
      stdin.removeListener("readable", this.handleReadable);
      stdin.unref();
    }
  };
  // Helper to flush incomplete escape sequences
  flushIncomplete = () => {
    this.incompleteEscapeTimer = null;
    if (!this.keyParseState.incomplete) {
      return;
    }
    if (this.props.stdin.readableLength > 0) {
      this.incompleteEscapeTimer = setTimeout(this.flushIncomplete, this.NORMAL_TIMEOUT);
      return;
    }
    this.processInput(null);
  };
  // Process input through the parser and handle the results
  processInput = (input) => {
    const [keys, newState] = parseMultipleKeypresses(this.keyParseState, input);
    this.keyParseState = newState;
    if (keys.length > 0) {
      reconciler_default.discreteUpdates(processKeysInBatch, this, keys, void 0, void 0);
    }
    if (this.keyParseState.incomplete) {
      if (this.incompleteEscapeTimer) {
        clearTimeout(this.incompleteEscapeTimer);
      }
      this.incompleteEscapeTimer = setTimeout(
        this.flushIncomplete,
        this.keyParseState.mode === "IN_PASTE" ? this.PASTE_TIMEOUT : this.NORMAL_TIMEOUT
      );
    }
  };
  handleReadable = () => {
    const now = Date.now();
    if (now - this.lastStdinTime > STDIN_RESUME_GAP_MS) {
      this.props.onStdinResume?.();
    }
    this.lastStdinTime = now;
    try {
      let chunk;
      while ((chunk = this.props.stdin.read()) !== null) {
        this.processInput(chunk);
      }
    } catch (error) {
      logError(error);
      const { stdin } = this.props;
      if (this.rawModeEnabledCount > 0 && !stdin.listeners("readable").includes(this.handleReadable)) {
        logForDebugging("handleReadable: re-attaching stdin readable listener after error recovery", {
          level: "warn"
        });
        stdin.addListener("readable", this.handleReadable);
      }
    }
  };
  handleInput = (input) => {
    if (input === "" && this.props.exitOnCtrlC) {
      this.handleExit();
    }
  };
  handleExit = (error) => {
    if (this.isRawModeSupported()) {
      this.handleSetRawMode(false);
    }
    this.props.onExit(error);
  };
  handleTerminalFocus = (isFocused) => {
    setTerminalFocused(isFocused);
  };
  handleSuspend = () => {
    if (!this.isRawModeSupported()) {
      return;
    }
    const rawModeCountBeforeSuspend = this.rawModeEnabledCount;
    while (this.rawModeEnabledCount > 0) {
      this.handleSetRawMode(false);
    }
    if (this.props.stdout.isTTY) {
      this.props.stdout.write(SHOW_CURSOR + DFE + DISABLE_MOUSE_TRACKING);
    }
    this.inputEmitter.emit("suspend");
    const resumeHandler = () => {
      for (let i = 0; i < rawModeCountBeforeSuspend; i++) {
        if (this.isRawModeSupported()) {
          this.handleSetRawMode(true);
        }
      }
      if (this.props.stdout.isTTY) {
        this.props.stdout.write(HIDE_CURSOR + EFE);
      }
      this.inputEmitter.emit("resume");
      process.removeListener("SIGCONT", resumeHandler);
    };
    process.on("SIGCONT", resumeHandler);
    process.kill(process.pid, "SIGSTOP");
  };
};
function processKeysInBatch(app, items, _unused1, _unused2) {
  if (items.some((i) => i.kind === "key" || i.kind === "mouse" && !((i.button & 32) !== 0 && (i.button & 3) === 3))) {
    updateLastInteractionTime();
  }
  for (const item of items) {
    if (item.kind === "response") {
      app.querier.onResponse(item.response);
      continue;
    }
    if (item.kind === "mouse") {
      handleMouseEvent(app, item);
      continue;
    }
    const sequence = item.sequence;
    if (sequence === FOCUS_IN) {
      app.handleTerminalFocus(true);
      const event2 = new TerminalFocusEvent("terminalfocus");
      app.inputEmitter.emit("terminalfocus", event2);
      continue;
    }
    if (sequence === FOCUS_OUT) {
      app.handleTerminalFocus(false);
      if (app.props.selection.isDragging) {
        finishSelection(app.props.selection);
        app.props.onSelectionChange();
      }
      const event2 = new TerminalFocusEvent("terminalblur");
      app.inputEmitter.emit("terminalblur", event2);
      continue;
    }
    if (!getTerminalFocused()) {
      setTerminalFocused(true);
    }
    if (item.name === "z" && item.ctrl && SUPPORTS_SUSPEND) {
      app.handleSuspend();
      continue;
    }
    app.handleInput(sequence);
    const event = new InputEvent(item);
    app.inputEmitter.emit("input", event);
    app.props.dispatchKeyboardEvent(item);
  }
}
function handleMouseEvent(app, m) {
  if (isMouseClicksDisabled()) {
    return;
  }
  const sel = app.props.selection;
  const col = m.col - 1;
  const row = m.row - 1;
  const baseButton = m.button & 3;
  if (m.action === "press") {
    if ((m.button & 32) !== 0 && baseButton === 3) {
      if (app.mouseCaptureTarget) {
        app.props.onMouseUpAt(app.mouseCaptureTarget, col, row, baseButton);
        app.mouseCaptureTarget = void 0;
      }
      if (sel.isDragging) {
        finishSelection(sel);
        app.props.onSelectionChange();
      }
      if (col === app.lastHoverCol && row === app.lastHoverRow) {
        return;
      }
      app.lastHoverCol = col;
      app.lastHoverRow = row;
      app.props.onHoverAt(col, row);
      return;
    }
    if (baseButton !== 0) {
      app.clickCount = 0;
      return;
    }
    if ((m.button & 32) !== 0) {
      if (app.mouseCaptureTarget) {
        app.props.onMouseDragAt(app.mouseCaptureTarget, col, row, baseButton);
        return;
      }
      app.props.onSelectionDrag(col, row);
      return;
    }
    if (sel.isDragging) {
      finishSelection(sel);
      app.props.onSelectionChange();
    }
    const capture = app.props.onMouseDownAt(col, row, baseButton);
    if (capture) {
      app.mouseCaptureTarget = capture;
      app.clickCount = 0;
      return;
    }
    const now = Date.now();
    const nearLast = now - app.lastClickTime < MULTI_CLICK_TIMEOUT_MS && Math.abs(col - app.lastClickCol) <= MULTI_CLICK_DISTANCE && Math.abs(row - app.lastClickRow) <= MULTI_CLICK_DISTANCE;
    app.clickCount = nearLast ? app.clickCount + 1 : 1;
    app.lastClickTime = now;
    app.lastClickCol = col;
    app.lastClickRow = row;
    if (app.clickCount >= 2) {
      if (app.pendingHyperlinkTimer) {
        clearTimeout(app.pendingHyperlinkTimer);
        app.pendingHyperlinkTimer = null;
      }
      const count = app.clickCount === 2 ? 2 : 3;
      app.props.onMultiClick(col, row, count);
      return;
    }
    startSelection(sel, col, row);
    sel.lastPressHadAlt = (m.button & 8) !== 0;
    app.props.onSelectionChange();
    return;
  }
  if (app.mouseCaptureTarget) {
    app.props.onMouseUpAt(app.mouseCaptureTarget, col, row, baseButton);
    app.mouseCaptureTarget = void 0;
    return;
  }
  if (baseButton !== 0) {
    if (!sel.isDragging) {
      return;
    }
    finishSelection(sel);
    app.props.onSelectionChange();
    return;
  }
  finishSelection(sel);
  if (!hasSelection(sel) && sel.anchor) {
    if (!app.props.onClickAt(col, row)) {
      const url = app.props.getHyperlinkAt(col, row);
      if (url && process.env.TERM_PROGRAM !== "vscode" && !isXtermJs()) {
        if (app.pendingHyperlinkTimer) {
          clearTimeout(app.pendingHyperlinkTimer);
        }
        app.pendingHyperlinkTimer = setTimeout(
          (app2, url2) => {
            app2.pendingHyperlinkTimer = null;
            app2.props.onOpenHyperlink(url2);
          },
          MULTI_CLICK_TIMEOUT_MS,
          app,
          url
        );
      }
    }
  }
  app.props.onSelectionChange();
}

// src/ink/events/keyboard-event.ts
var KeyboardEvent = class extends TerminalEvent {
  key;
  ctrl;
  shift;
  meta;
  superKey;
  fn;
  constructor(parsedKey) {
    super("keydown", { bubbles: true, cancelable: true });
    this.key = keyFromParsed(parsedKey);
    this.ctrl = parsedKey.ctrl;
    this.shift = parsedKey.shift;
    this.meta = parsedKey.meta || parsedKey.option;
    this.superKey = parsedKey.super;
    this.fn = parsedKey.fn;
  }
};
function keyFromParsed(parsed) {
  const seq = parsed.sequence ?? "";
  const name = parsed.name ?? "";
  if (parsed.ctrl) {
    return name;
  }
  if (seq.length === 1) {
    const code = seq.charCodeAt(0);
    if (code >= 32 && code !== 127) {
      return seq;
    }
  }
  return name || seq;
}

// src/ink/frame.ts
function emptyFrame(rows, columns, stylePool, charPool, hyperlinkPool) {
  return {
    screen: createScreen(0, 0, stylePool, charPool, hyperlinkPool),
    viewport: { width: columns, height: rows },
    cursor: { x: 0, y: 0, visible: true }
  };
}

// src/ink/events/click-event.ts
var ClickEvent = class extends Event {
  /** 0-indexed screen column of the click */
  col;
  /** 0-indexed screen row of the click */
  row;
  /**
   * Click column relative to the current handler's Box (col - box.x).
   * Recomputed by dispatchClick before each handler fires, so an onClick
   * on a container sees coords relative to that container, not to any
   * child the click landed on.
   */
  localCol = 0;
  /** Click row relative to the current handler's Box (row - box.y). */
  localRow = 0;
  /**
   * True if the clicked cell has no visible content (unwritten in the
   * screen buffer — both packed words are 0). Handlers can check this to
   * ignore clicks on blank space to the right of text, so accidental
   * clicks on empty terminal space don't toggle state.
   */
  cellIsBlank;
  constructor(col, row, cellIsBlank) {
    super();
    this.col = col;
    this.row = row;
    this.cellIsBlank = cellIsBlank;
  }
};

// src/ink/events/mouse-event.ts
var MouseEvent = class extends Event {
  col;
  row;
  localCol = 0;
  localRow = 0;
  cellIsBlank;
  button;
  constructor(col, row, cellIsBlank, button) {
    super();
    this.col = col;
    this.row = row;
    this.cellIsBlank = cellIsBlank;
    this.button = button;
  }
};

// src/ink/hit-test.ts
function hitTest(node, col, row) {
  const rect = nodeCache.get(node);
  if (!rect) {
    return null;
  }
  if (col < rect.x || col >= rect.x + rect.width || row < rect.y || row >= rect.y + rect.height) {
    return null;
  }
  for (let i = node.childNodes.length - 1; i >= 0; i--) {
    const child = node.childNodes[i];
    if (child.nodeName === "#text") {
      continue;
    }
    const hit = hitTest(child, col, row);
    if (hit) {
      return hit;
    }
  }
  return node;
}
function dispatchClick(root, col, row, cellIsBlank = false) {
  let target = hitTest(root, col, row) ?? void 0;
  if (!target) {
    return false;
  }
  if (root.focusManager) {
    let focusTarget = target;
    while (focusTarget) {
      if (typeof focusTarget.attributes["tabIndex"] === "number") {
        root.focusManager.handleClickFocus(focusTarget);
        break;
      }
      focusTarget = focusTarget.parentNode;
    }
  }
  const event = new ClickEvent(col, row, cellIsBlank);
  let handled = false;
  while (target) {
    const handler = target._eventHandlers?.onClick;
    if (handler) {
      handled = true;
      const rect = nodeCache.get(target);
      if (rect) {
        event.localCol = col - rect.x;
        event.localRow = row - rect.y;
      }
      handler(event);
      if (event.didStopImmediatePropagation()) {
        return true;
      }
    }
    target = target.parentNode;
  }
  return handled;
}
function dispatchMouse(root, col, row, handlerName, button, cellIsBlank = false, target) {
  let node = target ?? hitTest(root, col, row) ?? void 0;
  if (!node) {
    return void 0;
  }
  const event = new MouseEvent(col, row, cellIsBlank, button);
  let handled;
  while (node) {
    const handler = node._eventHandlers?.[handlerName];
    if (handler) {
      handled ??= node;
      const rect = nodeCache.get(node);
      if (rect) {
        event.localCol = col - rect.x;
        event.localRow = row - rect.y;
      }
      handler(event);
      if (event.didStopImmediatePropagation()) {
        return handled;
      }
    }
    node = node.parentNode;
  }
  return handled;
}
function dispatchHover(root, col, row, hovered) {
  const next = /* @__PURE__ */ new Set();
  let node = hitTest(root, col, row) ?? void 0;
  while (node) {
    const h = node._eventHandlers;
    if (h?.onMouseEnter || h?.onMouseLeave) {
      next.add(node);
    }
    node = node.parentNode;
  }
  for (const old of hovered) {
    if (!next.has(old)) {
      hovered.delete(old);
      if (old.parentNode) {
        ;
        old._eventHandlers?.onMouseLeave?.();
      }
    }
  }
  for (const n of next) {
    if (!hovered.has(n)) {
      hovered.add(n);
      n._eventHandlers?.onMouseEnter?.();
    }
  }
}

// src/ink/log-update.ts
import { ansiCodesToString as ansiCodesToString3, diffAnsiCodes as diffAnsiCodes2 } from "@alcalzone/ansi-tokenize";
var CARRIAGE_RETURN = { type: "carriageReturn" };
var NEWLINE = { type: "stdout", content: "\n" };
var LogUpdate = class {
  constructor(options) {
    this.options = options;
    this.state = {
      previousOutput: ""
    };
  }
  state;
  renderPreviousOutput_DEPRECATED(prevFrame) {
    if (!this.options.isTTY) {
      return [NEWLINE];
    }
    return this.getRenderOpsForDone(prevFrame);
  }
  // Called when process resumes from suspension (SIGCONT) to prevent clobbering terminal content
  reset() {
    this.state.previousOutput = "";
  }
  renderFullFrame(frame) {
    const { screen } = frame;
    const lines = [];
    let currentStyles = [];
    let currentHyperlink = void 0;
    for (let y = 0; y < screen.height; y++) {
      let line = "";
      for (let x = 0; x < screen.width; x++) {
        const cell = cellAt(screen, x, y);
        if (cell && cell.width !== 2 /* SpacerTail */) {
          if (cell.hyperlink !== currentHyperlink) {
            if (currentHyperlink !== void 0) {
              line += LINK_END;
            }
            if (cell.hyperlink !== void 0) {
              line += link(cell.hyperlink);
            }
            currentHyperlink = cell.hyperlink;
          }
          const cellStyles = this.options.stylePool.get(cell.styleId);
          const styleDiff = diffAnsiCodes2(currentStyles, cellStyles);
          if (styleDiff.length > 0) {
            line += ansiCodesToString3(styleDiff);
            currentStyles = cellStyles;
          }
          line += cell.char;
        }
      }
      if (currentHyperlink !== void 0) {
        line += LINK_END;
        currentHyperlink = void 0;
      }
      const resetCodes = diffAnsiCodes2(currentStyles, []);
      if (resetCodes.length > 0) {
        line += ansiCodesToString3(resetCodes);
        currentStyles = [];
      }
      lines.push(line.trimEnd());
    }
    if (lines.length === 0) {
      return [];
    }
    return [{ type: "stdout", content: lines.join("\n") }];
  }
  getRenderOpsForDone(prev) {
    this.state.previousOutput = "";
    if (!prev.cursor.visible) {
      return [{ type: "cursorShow" }];
    }
    return [];
  }
  render(prev, next, altScreen = false, decstbmSafe = true) {
    if (!this.options.isTTY) {
      return this.renderFullFrame(next);
    }
    const startTime = performance.now();
    const stylePool = this.options.stylePool;
    if (next.viewport.height < prev.viewport.height || prev.viewport.width !== 0 && next.viewport.width !== prev.viewport.width) {
      return fullResetSequence_CAUSES_FLICKER(next, "resize", stylePool);
    }
    let scrollPatch = [];
    if (altScreen && next.scrollHint && decstbmSafe) {
      const { top, bottom, delta } = next.scrollHint;
      if (top >= 0 && bottom < prev.screen.height && bottom < next.screen.height) {
        shiftRows(prev.screen, top, bottom, delta);
        scrollPatch = [
          {
            type: "stdout",
            content: setScrollRegion(top + 1, bottom + 1) + (delta > 0 ? scrollUp(delta) : scrollDown(-delta)) + RESET_SCROLL_REGION + CURSOR_HOME
          }
        ];
      }
    }
    const cursorAtBottom = prev.cursor.y >= prev.screen.height;
    const isGrowing = next.screen.height > prev.screen.height;
    const prevHadScrollback = cursorAtBottom && prev.screen.height >= prev.viewport.height;
    const isShrinking = next.screen.height < prev.screen.height;
    const nextFitsViewport = next.screen.height <= prev.viewport.height;
    if (prevHadScrollback && nextFitsViewport && isShrinking) {
      logForDebugging(
        `Full reset (shrink->below): prevHeight=${prev.screen.height}, nextHeight=${next.screen.height}, viewport=${prev.viewport.height}`
      );
      return fullResetSequence_CAUSES_FLICKER(next, "offscreen", stylePool);
    }
    if (prev.screen.height >= prev.viewport.height && prev.screen.height > 0 && cursorAtBottom && !isGrowing) {
      const viewportY2 = prev.screen.height - prev.viewport.height;
      const scrollbackRows = viewportY2 + 1;
      let scrollbackChangeY = -1;
      diffEach(prev.screen, next.screen, (_x, y) => {
        if (y < scrollbackRows) {
          scrollbackChangeY = y;
          return true;
        }
      });
      if (scrollbackChangeY >= 0) {
        const prevLine = readLine(prev.screen, scrollbackChangeY);
        const nextLine = readLine(next.screen, scrollbackChangeY);
        return fullResetSequence_CAUSES_FLICKER(next, "offscreen", stylePool, {
          triggerY: scrollbackChangeY,
          prevLine,
          nextLine
        });
      }
    }
    const screen = new VirtualScreen(prev.cursor, next.viewport.width);
    const heightDelta = Math.max(next.screen.height, 1) - Math.max(prev.screen.height, 1);
    const shrinking = heightDelta < 0;
    const growing = heightDelta > 0;
    if (shrinking) {
      const linesToClear = prev.screen.height - next.screen.height;
      if (linesToClear > prev.viewport.height) {
        return fullResetSequence_CAUSES_FLICKER(next, "offscreen", this.options.stylePool);
      }
      screen.txn((prev2) => [
        [
          { type: "clear", count: linesToClear },
          { type: "cursorMove", x: 0, y: -1 }
        ],
        { dx: -prev2.x, dy: -linesToClear }
      ]);
    }
    const cursorRestoreScroll = prevHadScrollback ? 1 : 0;
    const viewportY = growing ? Math.max(0, prev.screen.height - prev.viewport.height + cursorRestoreScroll) : Math.max(prev.screen.height, next.screen.height) - next.viewport.height + cursorRestoreScroll;
    let currentStyleId = stylePool.none;
    let currentHyperlink = void 0;
    let needsFullReset = false;
    let resetTriggerY = -1;
    diffEach(prev.screen, next.screen, (x, y, removed, added) => {
      if (growing && y >= prev.screen.height) {
        return;
      }
      if (added && (added.width === 2 /* SpacerTail */ || added.width === 3 /* SpacerHead */)) {
        return;
      }
      if (removed && (removed.width === 2 /* SpacerTail */ || removed.width === 3 /* SpacerHead */) && !added) {
        return;
      }
      if (added && isEmptyCellAt(next.screen, x, y) && !removed) {
        return;
      }
      if (y < viewportY) {
        needsFullReset = true;
        resetTriggerY = y;
        return true;
      }
      moveCursorTo(screen, x, y);
      if (added) {
        const targetHyperlink = added.hyperlink;
        currentHyperlink = transitionHyperlink(screen.diff, currentHyperlink, targetHyperlink);
        const styleStr = stylePool.transition(currentStyleId, added.styleId);
        if (writeCellWithStyleStr(screen, added, styleStr)) {
          currentStyleId = added.styleId;
        }
      } else if (removed) {
        const styleIdToReset = currentStyleId;
        const hyperlinkToReset = currentHyperlink;
        currentStyleId = stylePool.none;
        currentHyperlink = void 0;
        screen.txn(() => {
          const patches = [];
          transitionStyle(patches, stylePool, styleIdToReset, stylePool.none);
          transitionHyperlink(patches, hyperlinkToReset, void 0);
          patches.push({ type: "stdout", content: " " });
          return [patches, { dx: 1, dy: 0 }];
        });
      }
    });
    if (needsFullReset) {
      return fullResetSequence_CAUSES_FLICKER(next, "offscreen", stylePool, {
        triggerY: resetTriggerY,
        prevLine: readLine(prev.screen, resetTriggerY),
        nextLine: readLine(next.screen, resetTriggerY)
      });
    }
    currentStyleId = transitionStyle(screen.diff, stylePool, currentStyleId, stylePool.none);
    currentHyperlink = transitionHyperlink(screen.diff, currentHyperlink, void 0);
    if (growing) {
      renderFrameSlice(screen, next, prev.screen.height, next.screen.height, stylePool);
    }
    if (altScreen) {
    } else if (next.cursor.y >= next.screen.height) {
      screen.txn((prev2) => {
        const rowsToCreate = next.cursor.y - prev2.y;
        if (rowsToCreate > 0) {
          const patches = new Array(1 + rowsToCreate);
          patches[0] = CARRIAGE_RETURN;
          for (let i = 0; i < rowsToCreate; i++) {
            patches[1 + i] = NEWLINE;
          }
          return [patches, { dx: -prev2.x, dy: rowsToCreate }];
        }
        const dy = next.cursor.y - prev2.y;
        if (dy !== 0 || prev2.x !== next.cursor.x) {
          const patches = [CARRIAGE_RETURN];
          patches.push({ type: "cursorMove", x: next.cursor.x, y: dy });
          return [patches, { dx: next.cursor.x - prev2.x, dy }];
        }
        return [[], { dx: 0, dy: 0 }];
      });
    } else {
      moveCursorTo(screen, next.cursor.x, next.cursor.y);
    }
    const elapsed = performance.now() - startTime;
    if (elapsed > 50) {
      const damage = next.screen.damage;
      const damageInfo = damage ? `${damage.width}x${damage.height} at (${damage.x},${damage.y})` : "none";
      logForDebugging(
        `Slow render: ${elapsed.toFixed(1)}ms, screen: ${next.screen.height}x${next.screen.width}, damage: ${damageInfo}, changes: ${screen.diff.length}`
      );
    }
    return scrollPatch.length > 0 ? [...scrollPatch, ...screen.diff] : screen.diff;
  }
};
function transitionHyperlink(diff2, current, target) {
  if (current !== target) {
    diff2.push({ type: "hyperlink", uri: target ?? "" });
    return target;
  }
  return current;
}
function transitionStyle(diff2, stylePool, currentId, targetId) {
  const str = stylePool.transition(currentId, targetId);
  if (str.length > 0) {
    diff2.push({ type: "styleStr", str });
  }
  return targetId;
}
function readLine(screen, y) {
  let line = "";
  for (let x = 0; x < screen.width; x++) {
    line += charInCellAt(screen, x, y) ?? " ";
  }
  return line.trimEnd();
}
function fullResetSequence_CAUSES_FLICKER(frame, reason, stylePool, debug) {
  const screen = new VirtualScreen({ x: 0, y: 0 }, frame.viewport.width);
  renderFrame(screen, frame, stylePool);
  return [{ type: "clearTerminal", reason, debug }, ...screen.diff];
}
function renderFrame(screen, frame, stylePool) {
  renderFrameSlice(screen, frame, 0, frame.screen.height, stylePool);
}
function renderFrameSlice(screen, frame, startY, endY, stylePool) {
  let currentStyleId = stylePool.none;
  let currentHyperlink = void 0;
  let lastRenderedStyleId = -1;
  const { width: screenWidth, cells, charPool, hyperlinkPool } = frame.screen;
  let index = startY * screenWidth;
  for (let y = startY; y < endY; y += 1) {
    if (screen.cursor.y < y) {
      const rowsToAdvance = y - screen.cursor.y;
      screen.txn((prev) => {
        const patches = new Array(1 + rowsToAdvance);
        patches[0] = CARRIAGE_RETURN;
        for (let i = 0; i < rowsToAdvance; i++) {
          patches[1 + i] = NEWLINE;
        }
        return [patches, { dx: -prev.x, dy: rowsToAdvance }];
      });
    }
    lastRenderedStyleId = -1;
    for (let x = 0; x < screenWidth; x += 1, index += 1) {
      const cell = visibleCellAtIndex(cells, charPool, hyperlinkPool, index, lastRenderedStyleId);
      if (!cell) {
        continue;
      }
      moveCursorTo(screen, x, y);
      const targetHyperlink = cell.hyperlink;
      currentHyperlink = transitionHyperlink(screen.diff, currentHyperlink, targetHyperlink);
      const styleStr = stylePool.transition(currentStyleId, cell.styleId);
      if (writeCellWithStyleStr(screen, cell, styleStr)) {
        currentStyleId = cell.styleId;
        lastRenderedStyleId = cell.styleId;
      }
    }
    currentStyleId = transitionStyle(screen.diff, stylePool, currentStyleId, stylePool.none);
    currentHyperlink = transitionHyperlink(screen.diff, currentHyperlink, void 0);
    screen.txn((prev) => [[CARRIAGE_RETURN, NEWLINE], { dx: -prev.x, dy: 1 }]);
  }
  transitionStyle(screen.diff, stylePool, currentStyleId, stylePool.none);
  transitionHyperlink(screen.diff, currentHyperlink, void 0);
  return screen;
}
function writeCellWithStyleStr(screen, cell, styleStr) {
  const cellWidth = cell.width === 1 /* Wide */ ? 2 : 1;
  const px = screen.cursor.x;
  const vw = screen.viewportWidth;
  if (cellWidth === 2 && px < vw) {
    const threshold = cell.char.length > 2 ? vw : vw + 1;
    if (px + 2 >= threshold) {
      return false;
    }
  }
  const diff2 = screen.diff;
  if (styleStr.length > 0) {
    diff2.push({ type: "styleStr", str: styleStr });
  }
  const needsCompensation = cellWidth === 2 && needsWidthCompensation(cell.char);
  if (needsCompensation && px + 1 < vw) {
    diff2.push({ type: "cursorTo", col: px + 2 });
    diff2.push({ type: "stdout", content: " " });
    diff2.push({ type: "cursorTo", col: px + 1 });
  }
  diff2.push({ type: "stdout", content: cell.char });
  if (needsCompensation) {
    diff2.push({ type: "cursorTo", col: px + cellWidth + 1 });
  }
  if (px >= vw) {
    screen.cursor.x = cellWidth;
    screen.cursor.y++;
  } else {
    screen.cursor.x = px + cellWidth;
  }
  return true;
}
function moveCursorTo(screen, targetX, targetY) {
  screen.txn((prev) => {
    const dx = targetX - prev.x;
    const dy = targetY - prev.y;
    const inPendingWrap = prev.x >= screen.viewportWidth;
    if (inPendingWrap) {
      return [[CARRIAGE_RETURN, { type: "cursorMove", x: targetX, y: dy }], { dx, dy }];
    }
    if (dy !== 0) {
      return [[CARRIAGE_RETURN, { type: "cursorMove", x: targetX, y: dy }], { dx, dy }];
    }
    return [[{ type: "cursorMove", x: dx, y: dy }], { dx, dy }];
  });
}
function needsWidthCompensation(char) {
  const cp = char.codePointAt(0);
  if (cp === void 0) {
    return false;
  }
  if (cp >= 129648 && cp <= 129791 || cp >= 129792 && cp <= 130047) {
    return true;
  }
  if (char.length >= 2) {
    for (let i = 0; i < char.length; i++) {
      if (char.charCodeAt(i) === 65039) {
        return true;
      }
    }
  }
  return false;
}
var VirtualScreen = class {
  constructor(origin, viewportWidth) {
    this.viewportWidth = viewportWidth;
    this.cursor = { ...origin };
  }
  // Public for direct mutation by writeCellWithStyleStr (avoids txn overhead).
  // File-private class — not exposed outside log-update.ts.
  cursor;
  diff = [];
  txn(fn) {
    const [patches, next] = fn(this.cursor);
    for (const patch of patches) {
      this.diff.push(patch);
    }
    this.cursor.x += next.dx;
    this.cursor.y += next.dy;
  }
};

// src/ink/optimizer.ts
function optimize(diff2) {
  if (diff2.length <= 1) {
    return diff2;
  }
  const result = [];
  let len = 0;
  for (const patch of diff2) {
    const type = patch.type;
    if (type === "stdout") {
      if (patch.content === "") {
        continue;
      }
    } else if (type === "cursorMove") {
      if (patch.x === 0 && patch.y === 0) {
        continue;
      }
    } else if (type === "clear") {
      if (patch.count === 0) {
        continue;
      }
    }
    if (len > 0) {
      const lastIdx = len - 1;
      const last = result[lastIdx];
      const lastType = last.type;
      if (type === "cursorMove" && lastType === "cursorMove") {
        result[lastIdx] = {
          type: "cursorMove",
          x: last.x + patch.x,
          y: last.y + patch.y
        };
        continue;
      }
      if (type === "cursorTo" && lastType === "cursorTo") {
        result[lastIdx] = patch;
        continue;
      }
      if (type === "styleStr" && lastType === "styleStr") {
        result[lastIdx] = { type: "styleStr", str: last.str + patch.str };
        continue;
      }
      if (type === "hyperlink" && lastType === "hyperlink" && patch.uri === last.uri) {
        continue;
      }
      if (type === "cursorShow" && lastType === "cursorHide" || type === "cursorHide" && lastType === "cursorShow") {
        result.pop();
        len--;
        continue;
      }
    }
    result.push(patch);
    len++;
  }
  return result;
}

// src/ink/output.ts
import { styledCharsFromTokens, tokenize as tokenize3 } from "@alcalzone/ansi-tokenize";

// src/ink/bidi.ts
import bidiFactory from "bidi-js";
var bidiInstance;
var needsSoftwareBidi;
function needsBidi() {
  if (needsSoftwareBidi === void 0) {
    needsSoftwareBidi = process.platform === "win32" || typeof process.env["WT_SESSION"] === "string" || // WSL in Windows Terminal
    process.env["TERM_PROGRAM"] === "vscode";
  }
  return needsSoftwareBidi;
}
function getBidi() {
  if (!bidiInstance) {
    bidiInstance = bidiFactory();
  }
  return bidiInstance;
}
function reorderBidi(characters) {
  if (!needsBidi() || characters.length === 0) {
    return characters;
  }
  const plainText = characters.map((c) => c.value).join("");
  if (!hasRTLCharacters(plainText)) {
    return characters;
  }
  const bidi = getBidi();
  const { levels } = bidi.getEmbeddingLevels(plainText, "auto");
  const charLevels = [];
  let offset = 0;
  for (let i = 0; i < characters.length; i++) {
    charLevels.push(levels[offset]);
    offset += characters[i].value.length;
  }
  const reordered = [...characters];
  const maxLevel = Math.max(...charLevels);
  for (let level = maxLevel; level >= 1; level--) {
    let i = 0;
    while (i < reordered.length) {
      if (charLevels[i] >= level) {
        let j = i + 1;
        while (j < reordered.length && charLevels[j] >= level) {
          j++;
        }
        reverseRange(reordered, i, j - 1);
        reverseRangeNumbers(charLevels, i, j - 1);
        i = j;
      } else {
        i++;
      }
    }
  }
  return reordered;
}
function reverseRange(arr, start, end) {
  while (start < end) {
    const temp = arr[start];
    arr[start] = arr[end];
    arr[end] = temp;
    start++;
    end--;
  }
}
function reverseRangeNumbers(arr, start, end) {
  while (start < end) {
    const temp = arr[start];
    arr[start] = arr[end];
    arr[end] = temp;
    start++;
    end--;
  }
}
function hasRTLCharacters(text) {
  return /[\u0590-\u05FF\uFB1D-\uFB4F\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u0780-\u07BF\u0700-\u074F]/u.test(
    text
  );
}

// src/ink/widest-line.ts
function widestLine(string) {
  let maxWidth = 0;
  let start = 0;
  while (start <= string.length) {
    const end = string.indexOf("\n", start);
    const line = end === -1 ? string.substring(start) : string.substring(start, end);
    maxWidth = Math.max(maxWidth, lineWidth(line));
    if (end === -1) {
      break;
    }
    start = end + 1;
  }
  return maxWidth;
}

// src/ink/output.ts
function intersectClip(parent, child) {
  if (!parent) {
    return child;
  }
  return {
    x1: maxDefined(parent.x1, child.x1),
    x2: minDefined(parent.x2, child.x2),
    y1: maxDefined(parent.y1, child.y1),
    y2: minDefined(parent.y2, child.y2)
  };
}
function maxDefined(a, b) {
  if (a === void 0) {
    return b;
  }
  if (b === void 0) {
    return a;
  }
  return Math.max(a, b);
}
function minDefined(a, b) {
  if (a === void 0) {
    return b;
  }
  if (b === void 0) {
    return a;
  }
  return Math.min(a, b);
}
var Output = class {
  width;
  height;
  stylePool;
  screen;
  operations = [];
  charCache = /* @__PURE__ */ new Map();
  constructor(options) {
    const { width, height, stylePool, screen } = options;
    this.width = width;
    this.height = height;
    this.stylePool = stylePool;
    this.screen = screen;
    resetScreen(screen, width, height);
  }
  /**
   * Reuse this Output for a new frame. Zeroes the screen buffer, clears
   * the operation list (backing storage is retained), and caps charCache
   * growth. Preserving charCache across frames is the main win — most
   * lines don't change between renders, so tokenize + grapheme clustering
   * becomes a cache hit.
   */
  reset(width, height, screen) {
    this.width = width;
    this.height = height;
    this.screen = screen;
    this.operations.length = 0;
    resetScreen(screen, width, height);
    if (this.charCache.size > 16384) {
      this.charCache.clear();
    }
  }
  /**
   * Copy cells from a source screen region (blit = block image transfer).
   */
  blit(src, x, y, width, height) {
    this.operations.push({ type: "blit", src, x, y, width, height });
  }
  /**
   * Shift full-width rows within [top, bottom] by n. n > 0 = up. Mirrors
   * what DECSTBM + SU/SD does to the terminal. Paired with blit() to reuse
   * prevScreen content during pure scroll, avoiding full child re-render.
   */
  shift(top, bottom, n) {
    this.operations.push({ type: "shift", top, bottom, n });
  }
  /**
   * Clear a region by writing empty cells. Used when a node shrinks to
   * ensure stale content from the previous frame is removed.
   */
  clear(region, fromAbsolute) {
    this.operations.push({ type: "clear", region, fromAbsolute });
  }
  /**
   * Mark a region as non-selectable (excluded from fullscreen text
   * selection copy + highlight). Used by <NoSelect> to fence off
   * gutters (line numbers, diff sigils). Applied AFTER blit/write so
   * the mark wins regardless of what's blitted into the region.
   */
  noSelect(region) {
    this.operations.push({ type: "noSelect", region });
  }
  write(x, y, text, softWrap) {
    if (!text) {
      return;
    }
    this.operations.push({
      type: "write",
      x,
      y,
      text,
      softWrap
    });
  }
  clip(clip) {
    this.operations.push({
      type: "clip",
      clip
    });
  }
  unclip() {
    this.operations.push({
      type: "unclip"
    });
  }
  get() {
    const screen = this.screen;
    const screenWidth = this.width;
    const screenHeight = this.height;
    let blitCells = 0;
    let writeCells = 0;
    const absoluteClears = [];
    for (const operation of this.operations) {
      if (operation.type !== "clear") {
        continue;
      }
      const { x, y, width, height } = operation.region;
      const startX = Math.max(0, x);
      const startY = Math.max(0, y);
      const maxX = Math.min(x + width, screenWidth);
      const maxY = Math.min(y + height, screenHeight);
      if (startX >= maxX || startY >= maxY) {
        continue;
      }
      const rect = {
        x: startX,
        y: startY,
        width: maxX - startX,
        height: maxY - startY
      };
      screen.damage = screen.damage ? unionRect(screen.damage, rect) : rect;
      if (operation.fromAbsolute) {
        absoluteClears.push(rect);
      }
    }
    const clips = [];
    for (const operation of this.operations) {
      switch (operation.type) {
        case "clear":
          continue;
        case "clip":
          clips.push(intersectClip(clips.at(-1), operation.clip));
          continue;
        case "unclip":
          clips.pop();
          continue;
        case "blit": {
          const { src, x: regionX, y: regionY, width: regionWidth, height: regionHeight } = operation;
          const clip = clips.at(-1);
          const startX = Math.max(regionX, clip?.x1 ?? 0);
          const startY = Math.max(regionY, clip?.y1 ?? 0);
          const maxY = Math.min(regionY + regionHeight, screenHeight, src.height, clip?.y2 ?? Infinity);
          const maxX = Math.min(regionX + regionWidth, screenWidth, src.width, clip?.x2 ?? Infinity);
          if (startX >= maxX || startY >= maxY) {
            continue;
          }
          if (absoluteClears.length === 0) {
            blitRegion(screen, src, startX, startY, maxX, maxY);
            blitCells += (maxY - startY) * (maxX - startX);
            continue;
          }
          for (let row = startY; row < maxY; row++) {
            let spans = [[startX, maxX]];
            for (const r of absoluteClears) {
              if (row < r.y || row >= r.y + r.height || !spans.length) {
                break;
              }
              const cs = Math.max(startX, r.x);
              const ce = Math.min(maxX, r.x + r.width);
              if (cs >= ce) {
                continue;
              }
              const next = [];
              for (const [sx, ex] of spans) {
                if (ce <= sx || cs >= ex) {
                  next.push([sx, ex]);
                  continue;
                }
                if (sx < cs) {
                  next.push([sx, cs]);
                }
                if (ce < ex) {
                  next.push([ce, ex]);
                }
              }
              spans = next;
            }
            for (const [sx, ex] of spans) {
              blitRegion(screen, src, sx, row, ex, row + 1);
              blitCells += ex - sx;
            }
          }
          continue;
        }
        case "shift": {
          shiftRows(screen, operation.top, operation.bottom, operation.n);
          continue;
        }
        case "write": {
          const { text, softWrap } = operation;
          let { x, y } = operation;
          let lines = text.split("\n");
          let swFrom = 0;
          let prevContentEnd = 0;
          const clip = clips.at(-1);
          if (clip) {
            const clipHorizontally = typeof clip?.x1 === "number" && typeof clip?.x2 === "number";
            const clipVertically = typeof clip?.y1 === "number" && typeof clip?.y2 === "number";
            if (clipHorizontally) {
              const width = widestLine(text);
              if (x + width <= clip.x1 || x >= clip.x2) {
                continue;
              }
            }
            if (clipVertically) {
              const height = lines.length;
              if (y + height <= clip.y1 || y >= clip.y2) {
                continue;
              }
            }
            if (clipHorizontally) {
              lines = lines.map((line) => {
                const from = x < clip.x1 ? clip.x1 - x : 0;
                const width = stringWidth(line);
                const to = x + width > clip.x2 ? clip.x2 - x : width;
                let sliced = sliceAnsi(line, from, to);
                if (stringWidth(sliced) > to - from) {
                  sliced = sliceAnsi(line, from, to - 1);
                }
                return sliced;
              });
              if (x < clip.x1) {
                x = clip.x1;
              }
            }
            if (clipVertically) {
              const from = y < clip.y1 ? clip.y1 - y : 0;
              const height = lines.length;
              const to = y + height > clip.y2 ? clip.y2 - y : height;
              if (softWrap && from > 0 && softWrap[from] === true) {
                prevContentEnd = x + stringWidth(lines[from - 1]);
              }
              lines = lines.slice(from, to);
              swFrom = from;
              if (y < clip.y1) {
                y = clip.y1;
              }
            }
          }
          const swBits = screen.softWrap;
          let offsetY = 0;
          for (const line of lines) {
            const lineY = y + offsetY;
            if (lineY >= screenHeight) {
              break;
            }
            const contentEnd = writeLineToScreen(screen, line, x, lineY, screenWidth, this.stylePool, this.charCache);
            writeCells += contentEnd - x;
            if (softWrap) {
              const isSW = softWrap[swFrom + offsetY] === true;
              swBits[lineY] = isSW ? prevContentEnd : 0;
              prevContentEnd = contentEnd;
            }
            offsetY++;
          }
          continue;
        }
      }
    }
    for (const operation of this.operations) {
      if (operation.type === "noSelect") {
        const { x, y, width, height } = operation.region;
        markNoSelectRegion(screen, x, y, width, height);
      }
    }
    const totalCells = blitCells + writeCells;
    if (totalCells > 1e3 && writeCells > blitCells) {
      logForDebugging(
        `High write ratio: blit=${blitCells}, write=${writeCells} (${(writeCells / totalCells * 100).toFixed(1)}% writes), screen=${screenHeight}x${screenWidth}`
      );
    }
    return screen;
  }
};
function stylesEqual3(a, b) {
  if (a === b) {
    return true;
  }
  const len = a.length;
  if (len !== b.length) {
    return false;
  }
  if (len === 0) {
    return true;
  }
  for (let i = 0; i < len; i++) {
    if (a[i].code !== b[i].code) {
      return false;
    }
  }
  return true;
}
function styledCharsWithGraphemeClustering(chars, stylePool) {
  const charCount = chars.length;
  if (charCount === 0) {
    return [];
  }
  const result = [];
  const bufferChars = [];
  let bufferStyles = chars[0].styles;
  for (let i = 0; i < charCount; i++) {
    const char = chars[i];
    const styles2 = char.styles;
    if (bufferChars.length > 0 && !stylesEqual3(styles2, bufferStyles)) {
      flushBuffer(bufferChars.join(""), bufferStyles, stylePool, result);
      bufferChars.length = 0;
    }
    bufferChars.push(char.value);
    bufferStyles = styles2;
  }
  if (bufferChars.length > 0) {
    flushBuffer(bufferChars.join(""), bufferStyles, stylePool, result);
  }
  return result;
}
function flushBuffer(buffer, styles2, stylePool, out) {
  const hyperlink = extractHyperlinkFromStyles(styles2) ?? void 0;
  const hasOsc8Styles = hyperlink !== void 0 || styles2.some((s) => s.code.length >= OSC8_PREFIX.length && s.code.startsWith(OSC8_PREFIX));
  const filteredStyles = hasOsc8Styles ? filterOutHyperlinkStyles(styles2) : styles2;
  const styleId = stylePool.intern(filteredStyles);
  for (const { segment: grapheme } of getGraphemeSegmenter().segment(buffer)) {
    out.push({
      value: grapheme,
      width: stringWidth(grapheme),
      styleId,
      hyperlink
    });
  }
}
function writeLineToScreen(screen, line, x, y, screenWidth, stylePool, charCache) {
  let characters = charCache.get(line);
  if (!characters) {
    characters = reorderBidi(styledCharsWithGraphemeClustering(styledCharsFromTokens(tokenize3(line)), stylePool));
    charCache.set(line, characters);
  }
  let offsetX = x;
  for (let charIdx = 0; charIdx < characters.length; charIdx++) {
    const character = characters[charIdx];
    const codePoint = character.value.codePointAt(0);
    if (codePoint !== void 0 && codePoint <= 31) {
      if (codePoint === 9) {
        const tabWidth = 8;
        const spacesToNextStop = tabWidth - offsetX % tabWidth;
        for (let i = 0; i < spacesToNextStop && offsetX < screenWidth; i++) {
          setCellAt(screen, offsetX, y, {
            char: " ",
            styleId: stylePool.none,
            width: 0 /* Narrow */,
            hyperlink: void 0
          });
          offsetX++;
        }
      } else if (codePoint === 27) {
        const nextChar = characters[charIdx + 1]?.value;
        const nextCode = nextChar?.codePointAt(0);
        if (nextChar === "(" || nextChar === ")" || nextChar === "*" || nextChar === "+") {
          charIdx += 2;
        } else if (nextChar === "[") {
          charIdx++;
          while (charIdx < characters.length - 1) {
            charIdx++;
            const c = characters[charIdx]?.value.codePointAt(0);
            if (c !== void 0 && c >= 64 && c <= 126) {
              break;
            }
          }
        } else if (nextChar === "]" || nextChar === "P" || nextChar === "_" || nextChar === "^" || nextChar === "X") {
          charIdx++;
          while (charIdx < characters.length - 1) {
            charIdx++;
            const c = characters[charIdx]?.value;
            if (c === "\x07") {
              break;
            }
            if (c === "\x1B") {
              const nextC = characters[charIdx + 1]?.value;
              if (nextC === "\\") {
                charIdx++;
                break;
              }
            }
          }
        } else if (nextCode !== void 0 && nextCode >= 48 && nextCode <= 126) {
          charIdx++;
        }
      }
      continue;
    }
    const charWidth = character.width;
    if (charWidth === 0) {
      continue;
    }
    const isWideCharacter = charWidth >= 2;
    if (isWideCharacter && offsetX + 2 > screenWidth) {
      setCellAt(screen, offsetX, y, {
        char: " ",
        styleId: stylePool.none,
        width: 3 /* SpacerHead */,
        hyperlink: void 0
      });
      offsetX++;
      continue;
    }
    setCellAt(screen, offsetX, y, {
      char: character.value,
      styleId: character.styleId,
      width: isWideCharacter ? 1 /* Wide */ : 0 /* Narrow */,
      hyperlink: character.hyperlink
    });
    offsetX += isWideCharacter ? 2 : 1;
  }
  return offsetX;
}

// src/ink/render-node-to-output.ts
import indentString from "indent-string";

// src/ink/get-max-width.ts
var getMaxWidth = (yogaNode) => {
  return yogaNode.getComputedWidth() - yogaNode.getComputedPadding(LayoutEdge.Left) - yogaNode.getComputedPadding(LayoutEdge.Right) - yogaNode.getComputedBorder(LayoutEdge.Left) - yogaNode.getComputedBorder(LayoutEdge.Right);
};
var get_max_width_default = getMaxWidth;

// src/ink/render-border.ts
import chalk2 from "chalk";
import cliBoxes from "cli-boxes";
var CUSTOM_BORDER_STYLES = {
  dashed: {
    top: "\u254C",
    left: "\u254E",
    right: "\u254E",
    bottom: "\u254C",
    // there aren't any line-drawing characters for dashes unfortunately
    topLeft: " ",
    topRight: " ",
    bottomLeft: " ",
    bottomRight: " "
  }
};
function embedTextInBorder(borderLine, text, align, offset = 0, borderChar) {
  const textLength = stringWidth(text);
  const borderLength = borderLine.length;
  if (textLength >= borderLength - 2) {
    return ["", text.substring(0, borderLength), ""];
  }
  let position;
  if (align === "center") {
    position = Math.floor((borderLength - textLength) / 2);
  } else if (align === "start") {
    position = offset + 1;
  } else {
    position = borderLength - textLength - offset - 1;
  }
  position = Math.max(1, Math.min(position, borderLength - textLength - 1));
  const before = borderLine.substring(0, 1) + borderChar.repeat(position - 1);
  const after = borderChar.repeat(borderLength - position - textLength - 1) + borderLine.substring(borderLength - 1);
  return [before, text, after];
}
function styleBorderLine(line, color, dim) {
  let styled = applyColor(line, color);
  if (dim) {
    styled = chalk2.dim(styled);
  }
  return styled;
}
var renderBorder = (x, y, node, output) => {
  if (node.style.borderStyle) {
    const width = Math.floor(node.yogaNode.getComputedWidth());
    const height = Math.floor(node.yogaNode.getComputedHeight());
    const box = typeof node.style.borderStyle === "string" ? CUSTOM_BORDER_STYLES[node.style.borderStyle] ?? cliBoxes[node.style.borderStyle] : node.style.borderStyle;
    const topBorderColor = node.style.borderTopColor ?? node.style.borderColor;
    const bottomBorderColor = node.style.borderBottomColor ?? node.style.borderColor;
    const leftBorderColor = node.style.borderLeftColor ?? node.style.borderColor;
    const rightBorderColor = node.style.borderRightColor ?? node.style.borderColor;
    const dimTopBorderColor = node.style.borderTopDimColor ?? node.style.borderDimColor;
    const dimBottomBorderColor = node.style.borderBottomDimColor ?? node.style.borderDimColor;
    const dimLeftBorderColor = node.style.borderLeftDimColor ?? node.style.borderDimColor;
    const dimRightBorderColor = node.style.borderRightDimColor ?? node.style.borderDimColor;
    const showTopBorder = node.style.borderTop !== false;
    const showBottomBorder = node.style.borderBottom !== false;
    const showLeftBorder = node.style.borderLeft !== false;
    const showRightBorder = node.style.borderRight !== false;
    const contentWidth = Math.max(0, width - (showLeftBorder ? 1 : 0) - (showRightBorder ? 1 : 0));
    const topBorderLine = showTopBorder ? (showLeftBorder ? box.topLeft : "") + box.top.repeat(contentWidth) + (showRightBorder ? box.topRight : "") : "";
    let topBorder;
    if (showTopBorder && node.style.borderText?.position === "top") {
      const [before, text, after] = embedTextInBorder(
        topBorderLine,
        node.style.borderText.content,
        node.style.borderText.align,
        node.style.borderText.offset,
        box.top
      );
      topBorder = styleBorderLine(before, topBorderColor, dimTopBorderColor) + text + styleBorderLine(after, topBorderColor, dimTopBorderColor);
    } else if (showTopBorder) {
      topBorder = styleBorderLine(topBorderLine, topBorderColor, dimTopBorderColor);
    }
    let verticalBorderHeight = height;
    if (showTopBorder) {
      verticalBorderHeight -= 1;
    }
    if (showBottomBorder) {
      verticalBorderHeight -= 1;
    }
    verticalBorderHeight = Math.max(0, verticalBorderHeight);
    let leftBorder = (applyColor(box.left, leftBorderColor) + "\n").repeat(verticalBorderHeight);
    if (dimLeftBorderColor) {
      leftBorder = chalk2.dim(leftBorder);
    }
    let rightBorder = (applyColor(box.right, rightBorderColor) + "\n").repeat(verticalBorderHeight);
    if (dimRightBorderColor) {
      rightBorder = chalk2.dim(rightBorder);
    }
    const bottomBorderLine = showBottomBorder ? (showLeftBorder ? box.bottomLeft : "") + box.bottom.repeat(contentWidth) + (showRightBorder ? box.bottomRight : "") : "";
    let bottomBorder;
    if (showBottomBorder && node.style.borderText?.position === "bottom") {
      const [before, text, after] = embedTextInBorder(
        bottomBorderLine,
        node.style.borderText.content,
        node.style.borderText.align,
        node.style.borderText.offset,
        box.bottom
      );
      bottomBorder = styleBorderLine(before, bottomBorderColor, dimBottomBorderColor) + text + styleBorderLine(after, bottomBorderColor, dimBottomBorderColor);
    } else if (showBottomBorder) {
      bottomBorder = styleBorderLine(bottomBorderLine, bottomBorderColor, dimBottomBorderColor);
    }
    const offsetY = showTopBorder ? 1 : 0;
    if (topBorder) {
      output.write(x, y, topBorder);
    }
    if (showLeftBorder) {
      output.write(x, y + offsetY, leftBorder);
    }
    if (showRightBorder) {
      output.write(x + width - 1, y + offsetY, rightBorder);
    }
    if (bottomBorder) {
      output.write(x, y + height - 1, bottomBorder);
    }
  }
};
var render_border_default = renderBorder;

// src/ink/render-node-to-output.ts
function isXtermJsHost() {
  return process.env.TERM_PROGRAM === "vscode" || isXtermJs();
}
var layoutShifted = false;
var absoluteOverlayMoved = false;
function resetLayoutShifted() {
  layoutShifted = false;
  absoluteOverlayMoved = false;
}
function didLayoutShift() {
  return layoutShifted;
}
function didAbsoluteOverlayMove() {
  return absoluteOverlayMoved;
}
var scrollHint = null;
var absoluteRectsPrev = [];
var absoluteRectsCur = [];
function resetScrollHint() {
  scrollHint = null;
  absoluteRectsPrev = absoluteRectsCur;
  absoluteRectsCur = [];
}
function getScrollHint() {
  return scrollHint;
}
var scrollDrainNode = null;
function resetScrollDrainNode() {
  scrollDrainNode = null;
}
function getScrollDrainNode() {
  return scrollDrainNode;
}
var followScroll = null;
function consumeFollowScroll() {
  const f = followScroll;
  followScroll = null;
  return f;
}
var SCROLL_MIN_PER_FRAME = 4;
var SCROLL_INSTANT_THRESHOLD = 5;
var SCROLL_HIGH_PENDING = 12;
var SCROLL_STEP_MED = 2;
var SCROLL_STEP_HIGH = 3;
var SCROLL_MAX_PENDING = 30;
function drainAdaptive(node, pending, innerHeight) {
  const sign = pending > 0 ? 1 : -1;
  let abs = Math.abs(pending);
  let applied = 0;
  if (abs > SCROLL_MAX_PENDING) {
    applied += sign * (abs - SCROLL_MAX_PENDING);
    abs = SCROLL_MAX_PENDING;
  }
  const step = abs <= SCROLL_INSTANT_THRESHOLD ? abs : abs < SCROLL_HIGH_PENDING ? SCROLL_STEP_MED : SCROLL_STEP_HIGH;
  applied += sign * step;
  const rem = abs - step;
  const cap = Math.max(1, innerHeight - 1);
  const totalAbs = Math.abs(applied);
  if (totalAbs > cap) {
    const excess = totalAbs - cap;
    node.pendingScrollDelta = sign * (rem + excess);
    return sign * cap;
  }
  node.pendingScrollDelta = rem > 0 ? sign * rem : void 0;
  return applied;
}
function drainProportional(node, pending, innerHeight) {
  const abs = Math.abs(pending);
  const cap = Math.max(1, innerHeight - 1);
  const step = Math.min(cap, Math.max(SCROLL_MIN_PER_FRAME, abs * 3 >> 2));
  if (abs <= step) {
    node.pendingScrollDelta = void 0;
    return pending;
  }
  const applied = pending > 0 ? step : -step;
  node.pendingScrollDelta = pending - applied;
  return applied;
}
var OSC2 = "\x1B]";
var BEL2 = "\x07";
function wrapWithOsc8Link(text, url) {
  return `${OSC2}8;;${url}${BEL2}${text}${OSC2}8;;${BEL2}`;
}
function buildCharToSegmentMap(segments) {
  const map = [];
  for (let i = 0; i < segments.length; i++) {
    const len = segments[i].text.length;
    for (let j = 0; j < len; j++) {
      map.push(i);
    }
  }
  return map;
}
function applyStylesToWrappedText(wrappedPlain, segments, charToSegment, originalPlain, trimEnabled = false) {
  const lines = wrappedPlain.split("\n");
  const resultLines = [];
  let charIndex = 0;
  for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
    const line = lines[lineIdx];
    if (trimEnabled && line.length > 0) {
      const lineStartsWithWhitespace = /\s/.test(line[0]);
      const originalHasWhitespace = charIndex < originalPlain.length && /\s/.test(originalPlain[charIndex]);
      if (originalHasWhitespace && !lineStartsWithWhitespace) {
        while (charIndex < originalPlain.length && /\s/.test(originalPlain[charIndex])) {
          charIndex++;
        }
      }
    }
    let styledLine = "";
    let runStart = 0;
    let runSegmentIndex = charToSegment[charIndex] ?? 0;
    for (let i = 0; i < line.length; i++) {
      const currentSegmentIndex = charToSegment[charIndex] ?? runSegmentIndex;
      if (currentSegmentIndex !== runSegmentIndex) {
        const runText2 = line.slice(runStart, i);
        const segment2 = segments[runSegmentIndex];
        if (segment2) {
          let styled = applyTextStyles(runText2, segment2.styles);
          if (segment2.hyperlink) {
            styled = wrapWithOsc8Link(styled, segment2.hyperlink);
          }
          styledLine += styled;
        } else {
          styledLine += runText2;
        }
        runStart = i;
        runSegmentIndex = currentSegmentIndex;
      }
      charIndex++;
    }
    const runText = line.slice(runStart);
    const segment = segments[runSegmentIndex];
    if (segment) {
      let styled = applyTextStyles(runText, segment.styles);
      if (segment.hyperlink) {
        styled = wrapWithOsc8Link(styled, segment.hyperlink);
      }
      styledLine += styled;
    } else {
      styledLine += runText;
    }
    resultLines.push(styledLine);
    if (charIndex < originalPlain.length && originalPlain[charIndex] === "\n") {
      charIndex++;
    }
    if (trimEnabled && lineIdx < lines.length - 1) {
      const nextLine = lines[lineIdx + 1];
      const nextLineFirstChar = nextLine.length > 0 ? nextLine[0] : null;
      while (charIndex < originalPlain.length && /\s/.test(originalPlain[charIndex])) {
        if (nextLineFirstChar !== null && originalPlain[charIndex] === nextLineFirstChar) {
          break;
        }
        charIndex++;
      }
    }
  }
  return resultLines.join("\n");
}
function wrapWithSoftWrap(plainText, maxWidth, textWrap) {
  if (textWrap !== "wrap" && textWrap !== "wrap-trim") {
    return {
      wrapped: wrapText(plainText, maxWidth, textWrap),
      softWrap: void 0
    };
  }
  const origLines = plainText.split("\n");
  const outLines = [];
  const softWrap = [];
  for (const orig of origLines) {
    const pieces = wrapText(orig, maxWidth, textWrap).split("\n");
    for (let i = 0; i < pieces.length; i++) {
      outLines.push(pieces[i]);
      softWrap.push(i > 0);
    }
  }
  return { wrapped: outLines.join("\n"), softWrap };
}
function applyPaddingToText(node, text, softWrap) {
  const yogaNode = node.childNodes[0]?.yogaNode;
  if (yogaNode) {
    const offsetX = yogaNode.getComputedLeft();
    const offsetY = yogaNode.getComputedTop();
    text = "\n".repeat(offsetY) + indentString(text, offsetX);
    if (softWrap && offsetY > 0) {
      softWrap.unshift(...Array(offsetY).fill(false));
    }
  }
  return text;
}
function renderNodeToOutput(node, output, {
  offsetX = 0,
  offsetY = 0,
  prevScreen,
  skipSelfBlit = false,
  inheritedBackgroundColor
}) {
  const { yogaNode } = node;
  if (yogaNode) {
    if (yogaNode.getDisplay() === LayoutDisplay.None) {
      if (node.dirty) {
        const cached2 = nodeCache.get(node);
        if (cached2) {
          output.clear({
            x: Math.floor(cached2.x),
            y: Math.floor(cached2.y),
            width: Math.floor(cached2.width),
            height: Math.floor(cached2.height)
          });
          dropSubtreeCache(node);
          layoutShifted = true;
        }
      }
      return;
    }
    const x = offsetX + yogaNode.getComputedLeft();
    const yogaTop = yogaNode.getComputedTop();
    let y = offsetY + yogaTop;
    const width = yogaNode.getComputedWidth();
    const height = yogaNode.getComputedHeight();
    if (y < 0 && node.style.position === "absolute") {
      y = 0;
    }
    const cached = nodeCache.get(node);
    if (!node.dirty && !skipSelfBlit && node.pendingScrollDelta === void 0 && cached && cached.x === x && cached.y === y && cached.width === width && cached.height === height && prevScreen) {
      const fx = Math.floor(x);
      const fy = Math.floor(y);
      const fw = Math.floor(width);
      const fh = Math.floor(height);
      output.blit(prevScreen, fx, fy, fw, fh);
      if (node.style.position === "absolute") {
        absoluteRectsCur.push(cached);
      }
      blitEscapingAbsoluteDescendants(node, output, prevScreen, fx, fy, fw, fh);
      return;
    }
    const positionChanged = cached !== void 0 && (cached.x !== x || cached.y !== y || cached.width !== width || cached.height !== height);
    if (positionChanged) {
      layoutShifted = true;
      absoluteOverlayMoved ||= node.style.position === "absolute";
    }
    if (cached && (node.dirty || positionChanged)) {
      output.clear(
        {
          x: Math.floor(cached.x),
          y: Math.floor(cached.y),
          width: Math.floor(cached.width),
          height: Math.floor(cached.height)
        },
        node.style.position === "absolute"
      );
    }
    const clears = pendingClears.get(node);
    const hasRemovedChild = clears !== void 0;
    if (hasRemovedChild) {
      layoutShifted = true;
      for (const rect2 of clears) {
        output.clear({
          x: Math.floor(rect2.x),
          y: Math.floor(rect2.y),
          width: Math.floor(rect2.width),
          height: Math.floor(rect2.height)
        });
      }
      pendingClears.delete(node);
    }
    if (height === 0 && siblingSharesY(node, yogaNode)) {
      nodeCache.set(node, { x, y, width, height, top: yogaTop });
      node.dirty = false;
      return;
    }
    if (node.nodeName === "ink-raw-ansi") {
      const text = node.attributes["rawText"];
      if (text) {
        output.write(x, y, text);
      }
    } else if (node.nodeName === "ink-text") {
      const segments = squashTextNodesToSegments(
        node,
        inheritedBackgroundColor ? { backgroundColor: inheritedBackgroundColor } : void 0
      );
      const plainText = segments.map((s) => s.text).join("");
      if (plainText.length > 0) {
        const maxWidth = Math.min(get_max_width_default(yogaNode), output.width - x);
        const textWrap = node.style.textWrap ?? "wrap";
        const needsWrapping = widestLine(plainText) > maxWidth;
        let text;
        let softWrap;
        if (needsWrapping && segments.length === 1) {
          const segment = segments[0];
          const w = wrapWithSoftWrap(plainText, maxWidth, textWrap);
          softWrap = w.softWrap;
          text = w.wrapped.split("\n").map((line) => {
            let styled = applyTextStyles(line, segment.styles);
            if (segment.hyperlink) {
              styled = wrapWithOsc8Link(styled, segment.hyperlink);
            }
            return styled;
          }).join("\n");
        } else if (needsWrapping) {
          const w = wrapWithSoftWrap(plainText, maxWidth, textWrap);
          softWrap = w.softWrap;
          const charToSegment = buildCharToSegmentMap(segments);
          text = applyStylesToWrappedText(w.wrapped, segments, charToSegment, plainText, textWrap === "wrap-trim");
        } else {
          text = segments.map((segment) => {
            let styledText = applyTextStyles(segment.text, segment.styles);
            if (segment.hyperlink) {
              styledText = wrapWithOsc8Link(styledText, segment.hyperlink);
            }
            return styledText;
          }).join("");
        }
        text = applyPaddingToText(node, text, softWrap);
        output.write(x, y, text, softWrap);
      }
    } else if (node.nodeName === "ink-box") {
      const boxBackgroundColor = node.style.backgroundColor ?? inheritedBackgroundColor;
      if (node.style.noSelect) {
        const boxX = Math.floor(x);
        const fromEdge = node.style.noSelect === "from-left-edge";
        output.noSelect({
          x: fromEdge ? 0 : boxX,
          y: Math.floor(y),
          width: fromEdge ? boxX + Math.floor(width) : Math.floor(width),
          height: Math.floor(height)
        });
      }
      const overflowX = node.style.overflowX ?? node.style.overflow;
      const overflowY = node.style.overflowY ?? node.style.overflow;
      const clipHorizontally = overflowX === "hidden" || overflowX === "scroll";
      const clipVertically = overflowY === "hidden" || overflowY === "scroll";
      const isScrollY = overflowY === "scroll";
      const needsClip = clipHorizontally || clipVertically;
      let y1;
      let y2;
      if (needsClip) {
        const x1 = clipHorizontally ? x + yogaNode.getComputedBorder(LayoutEdge.Left) : void 0;
        const x2 = clipHorizontally ? x + yogaNode.getComputedWidth() - yogaNode.getComputedBorder(LayoutEdge.Right) : void 0;
        y1 = clipVertically ? y + yogaNode.getComputedBorder(LayoutEdge.Top) : void 0;
        y2 = clipVertically ? y + yogaNode.getComputedHeight() - yogaNode.getComputedBorder(LayoutEdge.Bottom) : void 0;
        output.clip({ x1, x2, y1, y2 });
      }
      if (isScrollY) {
        const padTop = yogaNode.getComputedPadding(LayoutEdge.Top);
        const innerHeight = Math.max(
          0,
          (y2 ?? y + height) - (y1 ?? y) - padTop - yogaNode.getComputedPadding(LayoutEdge.Bottom)
        );
        const content = node.childNodes.find((c) => c.yogaNode);
        const contentYoga = content?.yogaNode;
        const scrollHeight = contentYoga?.getComputedHeight() ?? 0;
        const prevScrollHeight = node.scrollHeight ?? scrollHeight;
        const prevInnerHeight = node.scrollViewportHeight ?? innerHeight;
        node.scrollHeight = scrollHeight;
        node.scrollViewportHeight = innerHeight;
        node.scrollViewportTop = (y1 ?? y) + padTop;
        const maxScroll = Math.max(0, scrollHeight - innerHeight);
        if (node.scrollAnchor) {
          const anchorTop = node.scrollAnchor.el.yogaNode?.getComputedTop();
          if (anchorTop != null) {
            node.scrollTop = anchorTop + node.scrollAnchor.offset;
            node.pendingScrollDelta = void 0;
          }
          node.scrollAnchor = void 0;
        }
        const scrollTopBeforeFollow = node.scrollTop ?? 0;
        const sticky = node.stickyScroll ?? Boolean(node.attributes["stickyScroll"]);
        const prevMaxScroll = Math.max(0, prevScrollHeight - prevInnerHeight);
        const grew = scrollHeight >= prevScrollHeight;
        const atBottom = sticky || grew && scrollTopBeforeFollow >= prevMaxScroll;
        if (atBottom && (node.pendingScrollDelta ?? 0) >= 0) {
          node.scrollTop = maxScroll;
          node.pendingScrollDelta = void 0;
          if (node.stickyScroll === false && scrollTopBeforeFollow >= prevMaxScroll) {
            node.stickyScroll = true;
          }
        }
        const followDelta = (node.scrollTop ?? 0) - scrollTopBeforeFollow;
        if (followDelta > 0) {
          const vpTop = node.scrollViewportTop ?? 0;
          followScroll = {
            delta: followDelta,
            viewportTop: vpTop,
            viewportBottom: vpTop + innerHeight - 1
          };
        }
        let cur = node.scrollTop ?? 0;
        const pending = node.pendingScrollDelta;
        const cMin = node.scrollClampMin;
        const cMax = node.scrollClampMax;
        const haveClamp = cMin !== void 0 && cMax !== void 0;
        if (pending !== void 0 && pending !== 0) {
          const pastClamp = haveClamp && (pending < 0 && cur < cMin || pending > 0 && cur > cMax);
          const eff = pastClamp ? Math.min(4, innerHeight >> 3) : innerHeight;
          cur += isXtermJsHost() ? drainAdaptive(node, pending, eff) : drainProportional(node, pending, eff);
        } else if (pending === 0) {
          node.pendingScrollDelta = void 0;
        }
        let scrollTop = Math.max(0, Math.min(cur, maxScroll));
        const clamped = haveClamp ? Math.max(cMin, Math.min(scrollTop, cMax)) : scrollTop;
        node.scrollTop = scrollTop;
        if (scrollTop !== cur) {
          node.pendingScrollDelta = void 0;
        }
        if (node.pendingScrollDelta !== void 0) {
          scrollDrainNode = node;
        }
        scrollTop = clamped;
        if (content && contentYoga) {
          const contentX = x + contentYoga.getComputedLeft();
          const contentY = y + contentYoga.getComputedTop() - scrollTop;
          const contentCached = nodeCache.get(content);
          let hint = null;
          if (contentCached && contentCached.y !== contentY) {
            const delta = contentCached.y - contentY;
            const regionTop = Math.floor(y + contentYoga.getComputedTop());
            const regionBottom = regionTop + innerHeight - 1;
            if (cached?.y === y && cached.height === height && innerHeight > 0 && Math.abs(delta) < innerHeight) {
              hint = { top: regionTop, bottom: regionBottom, delta };
              scrollHint = hint;
            } else {
              layoutShifted = true;
            }
          }
          const scrollHeight2 = contentYoga.getComputedHeight();
          const prevHeight = contentCached?.height ?? scrollHeight2;
          const heightDelta = scrollHeight2 - prevHeight;
          const safeForFastPath = !hint || heightDelta === 0 || hint.delta > 0 && heightDelta === hint.delta;
          if (!safeForFastPath) {
            scrollHint = null;
          }
          if (hint && prevScreen && safeForFastPath) {
            const { top, bottom, delta } = hint;
            const w = Math.floor(width);
            output.blit(prevScreen, Math.floor(x), top, w, bottom - top + 1);
            output.shift(top, bottom, delta);
            const edgeTop = delta > 0 ? bottom - delta + 1 : top;
            const edgeBottom = delta > 0 ? bottom : top - delta - 1;
            output.clear({
              x: Math.floor(x),
              y: edgeTop,
              width: w,
              height: edgeBottom - edgeTop + 1
            });
            output.clip({
              x1: void 0,
              x2: void 0,
              y1: edgeTop,
              y2: edgeBottom + 1
            });
            const dirtyChildren = content.dirty ? new Set(content.childNodes.filter((c) => c.dirty)) : null;
            renderScrolledChildren(
              content,
              output,
              contentX,
              contentY,
              hasRemovedChild,
              void 0,
              // Cull to edge in child-local coords (inverse of contentY offset).
              edgeTop - contentY,
              edgeBottom + 1 - contentY,
              boxBackgroundColor,
              true
            );
            output.unclip();
            if (dirtyChildren) {
              const edgeTopLocal = edgeTop - contentY;
              const edgeBottomLocal = edgeBottom + 1 - contentY;
              const spaces2 = " ".repeat(w);
              let cumHeightShift = 0;
              for (const childNode of content.childNodes) {
                const childElem = childNode;
                const isDirty = dirtyChildren.has(childNode);
                if (!isDirty && cumHeightShift === 0) {
                  if (nodeCache.has(childElem)) {
                    continue;
                  }
                }
                const cy = childElem.yogaNode;
                if (!cy) {
                  continue;
                }
                const childTop = cy.getComputedTop();
                const childH = cy.getComputedHeight();
                const childBottom = childTop + childH;
                if (isDirty) {
                  const prev = nodeCache.get(childElem);
                  cumHeightShift += childH - (prev ? prev.height : 0);
                }
                if (childBottom <= scrollTop || childTop >= scrollTop + innerHeight) {
                  continue;
                }
                if (childTop >= edgeTopLocal && childBottom <= edgeBottomLocal) {
                  continue;
                }
                const screenY = Math.floor(contentY + childTop);
                if (!isDirty) {
                  const childCached = nodeCache.get(childElem);
                  if (childCached && Math.floor(childCached.y) - delta === screenY) {
                    continue;
                  }
                }
                const screenBottom = Math.min(
                  Math.floor(contentY + childBottom),
                  Math.floor((y1 ?? y) + padTop + innerHeight)
                );
                if (screenY < screenBottom) {
                  const fill = Array(screenBottom - screenY).fill(spaces2).join("\n");
                  output.write(Math.floor(x), screenY, fill);
                  output.clip({
                    x1: void 0,
                    x2: void 0,
                    y1: screenY,
                    y2: screenBottom
                  });
                  renderNodeToOutput(childElem, output, {
                    offsetX: contentX,
                    offsetY: contentY,
                    prevScreen: void 0,
                    inheritedBackgroundColor: boxBackgroundColor
                  });
                  output.unclip();
                }
              }
            }
            const spaces = absoluteRectsPrev.length ? " ".repeat(w) : "";
            for (const r of absoluteRectsPrev) {
              if (r.y >= bottom + 1 || r.y + r.height <= top) {
                continue;
              }
              const shiftedTop = Math.max(top, Math.floor(r.y) - delta);
              const shiftedBottom = Math.min(bottom + 1, Math.floor(r.y + r.height) - delta);
              if (shiftedTop >= edgeTop && shiftedBottom <= edgeBottom + 1) {
                continue;
              }
              if (shiftedTop >= shiftedBottom) {
                continue;
              }
              const fill = Array(shiftedBottom - shiftedTop).fill(spaces).join("\n");
              output.write(Math.floor(x), shiftedTop, fill);
              output.clip({
                x1: void 0,
                x2: void 0,
                y1: shiftedTop,
                y2: shiftedBottom
              });
              renderScrolledChildren(
                content,
                output,
                contentX,
                contentY,
                hasRemovedChild,
                void 0,
                shiftedTop - contentY,
                shiftedBottom - contentY,
                boxBackgroundColor,
                true
              );
              output.unclip();
            }
          } else {
            const scrolled = contentCached && contentCached.y !== contentY;
            if (scrolled && y1 !== void 0 && y2 !== void 0) {
              output.clear({
                x: Math.floor(x),
                y: Math.floor(y1),
                width: Math.floor(width),
                height: Math.floor(y2 - y1)
              });
            }
            renderScrolledChildren(
              content,
              output,
              contentX,
              contentY,
              hasRemovedChild,
              scrolled || positionChanged ? void 0 : prevScreen,
              scrollTop,
              scrollTop + innerHeight,
              boxBackgroundColor
            );
          }
          nodeCache.set(content, {
            x: contentX,
            y: contentY,
            width: contentYoga.getComputedWidth(),
            height: contentYoga.getComputedHeight()
          });
          content.dirty = false;
        }
      } else {
        const ownBackgroundColor = node.style.backgroundColor;
        if (ownBackgroundColor || node.style.opaque) {
          const borderLeft = yogaNode.getComputedBorder(LayoutEdge.Left);
          const borderRight = yogaNode.getComputedBorder(LayoutEdge.Right);
          const borderTop = yogaNode.getComputedBorder(LayoutEdge.Top);
          const borderBottom = yogaNode.getComputedBorder(LayoutEdge.Bottom);
          const innerWidth = Math.floor(width) - borderLeft - borderRight;
          const innerHeight = Math.floor(height) - borderTop - borderBottom;
          if (innerWidth > 0 && innerHeight > 0) {
            const spaces = " ".repeat(innerWidth);
            const fillLine = ownBackgroundColor ? applyTextStyles(spaces, { backgroundColor: ownBackgroundColor }) : spaces;
            const fill = Array(innerHeight).fill(fillLine).join("\n");
            output.write(x + borderLeft, y + borderTop, fill);
          }
        }
        renderChildren(
          node,
          output,
          x,
          y,
          hasRemovedChild,
          // backgroundColor and opaque both disable child blit: the fill
          // overwrites the entire interior each render, so any child whose
          // layout position shifted would blit stale cells from prevScreen
          // on top of the fresh fill. Previously opaque kept blit enabled
          // on the assumption that plain-space fill + unchanged children =
          // valid composite, but children CAN reposition (ScrollBox remeasure
          // on re-render → /permissions body blanked on Down arrow, #25436).
          ownBackgroundColor || node.style.opaque ? void 0 : prevScreen,
          boxBackgroundColor
        );
      }
      if (needsClip) {
        output.unclip();
      }
      render_border_default(x, y, node, output);
    } else if (node.nodeName === "ink-root") {
      renderChildren(node, output, x, y, hasRemovedChild, prevScreen, inheritedBackgroundColor);
    }
    const rect = { x, y, width, height, top: yogaTop };
    nodeCache.set(node, rect);
    if (node.style.position === "absolute") {
      absoluteRectsCur.push(rect);
    }
    node.dirty = false;
  }
}
function renderChildren(node, output, offsetX, offsetY, hasRemovedChild, prevScreen, inheritedBackgroundColor) {
  let seenDirtyChild = false;
  let seenDirtyClipped = false;
  for (const childNode of node.childNodes) {
    const childElem = childNode;
    const wasDirty = childElem.dirty;
    const isAbsolute = childElem.style.position === "absolute";
    renderNodeToOutput(childElem, output, {
      offsetX,
      offsetY,
      prevScreen: hasRemovedChild || seenDirtyChild ? void 0 : prevScreen,
      // Short-circuits on seenDirtyClipped (false in the common case) so
      // the opaque/bg reads don't happen per-child per-frame.
      skipSelfBlit: seenDirtyClipped && isAbsolute && !childElem.style.opaque && childElem.style.backgroundColor === void 0,
      inheritedBackgroundColor
    });
    if (wasDirty && !seenDirtyChild) {
      if (!clipsBothAxes(childElem) || isAbsolute) {
        seenDirtyChild = true;
      } else {
        seenDirtyClipped = true;
      }
    }
  }
}
function clipsBothAxes(node) {
  const ox = node.style.overflowX ?? node.style.overflow;
  const oy = node.style.overflowY ?? node.style.overflow;
  return (ox === "hidden" || ox === "scroll") && (oy === "hidden" || oy === "scroll");
}
function siblingSharesY(node, yogaNode) {
  const parent = node.parentNode;
  if (!parent) {
    return false;
  }
  const myTop = yogaNode.getComputedTop();
  const siblings = parent.childNodes;
  const idx = siblings.indexOf(node);
  for (let i = idx + 1; i < siblings.length; i++) {
    const sib = siblings[i].yogaNode;
    if (!sib) {
      continue;
    }
    return sib.getComputedTop() === myTop;
  }
  for (let i = idx - 1; i >= 0; i--) {
    const sib = siblings[i].yogaNode;
    if (!sib) {
      continue;
    }
    return sib.getComputedTop() === myTop;
  }
  return false;
}
function blitEscapingAbsoluteDescendants(node, output, prevScreen, px, py, pw, ph) {
  const pr = px + pw;
  const pb = py + ph;
  for (const child of node.childNodes) {
    if (child.nodeName === "#text") {
      continue;
    }
    const elem = child;
    if (elem.style.position === "absolute") {
      const cached = nodeCache.get(elem);
      if (cached) {
        absoluteRectsCur.push(cached);
        const cx = Math.floor(cached.x);
        const cy = Math.floor(cached.y);
        const cw = Math.floor(cached.width);
        const ch = Math.floor(cached.height);
        if (cx < px || cy < py || cx + cw > pr || cy + ch > pb) {
          output.blit(prevScreen, cx, cy, cw, ch);
        }
      }
    }
    blitEscapingAbsoluteDescendants(elem, output, prevScreen, px, py, pw, ph);
  }
}
function renderScrolledChildren(node, output, offsetX, offsetY, hasRemovedChild, prevScreen, scrollTopY, scrollBottomY, inheritedBackgroundColor, preserveCulledCache = false) {
  let seenDirtyChild = false;
  let cumHeightShift = 0;
  for (const childNode of node.childNodes) {
    const childElem = childNode;
    const cy = childElem.yogaNode;
    if (cy) {
      const cached = nodeCache.get(childElem);
      let top;
      let height;
      if (cached?.top !== void 0 && !childElem.dirty && cumHeightShift === 0) {
        top = cached.top;
        height = cached.height;
      } else {
        top = cy.getComputedTop();
        height = cy.getComputedHeight();
        if (childElem.dirty) {
          cumHeightShift += height - (cached ? cached.height : 0);
        }
        if (cached) {
          cached.top = top;
        }
      }
      const bottom = top + height;
      if (bottom <= scrollTopY || top >= scrollBottomY) {
        if (!preserveCulledCache) {
          dropSubtreeCache(childElem);
        }
        continue;
      }
    }
    const wasDirty = childElem.dirty;
    renderNodeToOutput(childElem, output, {
      offsetX,
      offsetY,
      prevScreen: hasRemovedChild || seenDirtyChild ? void 0 : prevScreen,
      inheritedBackgroundColor
    });
    if (wasDirty) {
      seenDirtyChild = true;
    }
  }
}
function dropSubtreeCache(node) {
  nodeCache.delete(node);
  for (const child of node.childNodes) {
    if (child.nodeName !== "#text") {
      dropSubtreeCache(child);
    }
  }
}
var render_node_to_output_default = renderNodeToOutput;

// src/ink/render-to-screen.ts
import noop from "lodash-es/noop.js";
import { LegacyRoot } from "react-reconciler/constants.js";
var timing = { reconcile: 0, yoga: 0, paint: 0, scan: 0, calls: 0 };
function scanPositions(screen, query) {
  const lq = query.toLowerCase();
  if (!lq) {
    return [];
  }
  const qlen = lq.length;
  const w = screen.width;
  const h = screen.height;
  const noSelect = screen.noSelect;
  const positions = [];
  const t0 = performance.now();
  for (let row = 0; row < h; row++) {
    const rowOff = row * w;
    let text = "";
    const colOf = [];
    const codeUnitToCell = [];
    for (let col = 0; col < w; col++) {
      const idx = rowOff + col;
      const cell = cellAtIndex(screen, idx);
      if (cell.width === 2 /* SpacerTail */ || cell.width === 3 /* SpacerHead */ || noSelect[idx] === 1) {
        continue;
      }
      const lc = cell.char.toLowerCase();
      const cellIdx = colOf.length;
      for (let i = 0; i < lc.length; i++) {
        codeUnitToCell.push(cellIdx);
      }
      text += lc;
      colOf.push(col);
    }
    let pos = text.indexOf(lq);
    while (pos >= 0) {
      const startCi = codeUnitToCell[pos];
      const endCi = codeUnitToCell[pos + qlen - 1];
      const col = colOf[startCi];
      const endCol = colOf[endCi] + 1;
      positions.push({ row, col, len: endCol - col });
      pos = text.indexOf(lq, pos + qlen);
    }
  }
  timing.scan += performance.now() - t0;
  return positions;
}
function applyPositionedHighlight(screen, stylePool, positions, rowOffset, currentIdx) {
  if (currentIdx < 0 || currentIdx >= positions.length) {
    return false;
  }
  const p = positions[currentIdx];
  const row = p.row + rowOffset;
  if (row < 0 || row >= screen.height) {
    return false;
  }
  const transform = (id) => stylePool.withCurrentMatch(id);
  const rowOff = row * screen.width;
  for (let col = p.col; col < p.col + p.len; col++) {
    if (col < 0 || col >= screen.width) {
      continue;
    }
    const cell = cellAtIndex(screen, rowOff + col);
    setCellStyleId(screen, col, row, transform(cell.styleId));
  }
  return true;
}

// src/ink/renderer.ts
function createRenderer(node, stylePool) {
  let output;
  return (options) => {
    const { frontFrame, backFrame, isTTY, terminalWidth, terminalRows } = options;
    const prevScreen = frontFrame.screen;
    const backScreen = backFrame.screen;
    const charPool = backScreen.charPool;
    const hyperlinkPool = backScreen.hyperlinkPool;
    const computedHeight = node.yogaNode?.getComputedHeight();
    const computedWidth = node.yogaNode?.getComputedWidth();
    const hasInvalidHeight = computedHeight === void 0 || !Number.isFinite(computedHeight) || computedHeight < 0;
    const hasInvalidWidth = computedWidth === void 0 || !Number.isFinite(computedWidth) || computedWidth < 0;
    if (!node.yogaNode || hasInvalidHeight || hasInvalidWidth) {
      if (node.yogaNode && (hasInvalidHeight || hasInvalidWidth)) {
        logForDebugging(
          `Invalid yoga dimensions: width=${computedWidth}, height=${computedHeight}, childNodes=${node.childNodes.length}, terminalWidth=${terminalWidth}, terminalRows=${terminalRows}`
        );
      }
      return {
        screen: createScreen(terminalWidth, 0, stylePool, charPool, hyperlinkPool),
        viewport: { width: terminalWidth, height: terminalRows },
        cursor: { x: 0, y: 0, visible: true }
      };
    }
    const width = Math.floor(node.yogaNode.getComputedWidth());
    const yogaHeight = Math.floor(node.yogaNode.getComputedHeight());
    const height = options.altScreen ? terminalRows : yogaHeight;
    if (options.altScreen && yogaHeight > terminalRows) {
      logForDebugging(
        `alt-screen: yoga height ${yogaHeight} > terminalRows ${terminalRows} \u2014 something is rendering outside <AlternateScreen>. Overflow clipped.`,
        { level: "warn" }
      );
    }
    const screen = backScreen ?? createScreen(width, height, stylePool, charPool, hyperlinkPool);
    if (output) {
      output.reset(width, height, screen);
    } else {
      output = new Output({ width, height, stylePool, screen });
    }
    resetLayoutShifted();
    resetScrollHint();
    resetScrollDrainNode();
    const absoluteRemoved = consumeAbsoluteRemovedFlag();
    render_node_to_output_default(node, output, {
      prevScreen: absoluteRemoved || options.prevFrameContaminated ? void 0 : prevScreen
    });
    const renderedScreen = output.get();
    const drainNode = getScrollDrainNode();
    if (drainNode) {
      markDirty(drainNode);
    }
    return {
      absoluteOverlayMoved: didAbsoluteOverlayMove(),
      scrollHint: options.altScreen ? getScrollHint() : null,
      scrollDrainPending: drainNode !== null,
      screen: renderedScreen,
      viewport: {
        width: terminalWidth,
        // Alt screen: fake viewport.height = rows + 1 so that
        // shouldClearScreen()'s `screen.height >= viewport.height` check
        // (which treats exactly-filling content as "overflows" for
        // scrollback purposes) never fires. Alt-screen content is always
        // exactly `rows` tall (via <Box height={rows}>) but never
        // scrolls — the cursor.y clamp below keeps the cursor-restore
        // from emitting an LF. With the standard diff path, every frame
        // is incremental; no fullResetSequence_CAUSES_FLICKER.
        height: options.altScreen ? terminalRows + 1 : terminalRows
      },
      cursor: {
        x: 0,
        // In the alt screen, keep the cursor inside the viewport. When
        // screen.height === terminalRows exactly (content fills the alt
        // screen), cursor.y = screen.height would trigger log-update's
        // cursor-restore LF at the last row, scrolling one row off the top
        // of the alt buffer and desyncing the diff's cursor model. The
        // cursor is hidden so its position only matters for diff coords.
        y: options.altScreen ? Math.max(0, Math.min(screen.height, terminalRows) - 1) : screen.height,
        // Hide cursor when there's dynamic output to render (only in TTY mode)
        visible: !isTTY || screen.height === 0
      }
    };
  };
}

// src/ink/searchHighlight.ts
function applySearchHighlight(screen, query, stylePool) {
  if (!query) {
    return false;
  }
  const lq = query.toLowerCase();
  const qlen = lq.length;
  const w = screen.width;
  const noSelect = screen.noSelect;
  const height = screen.height;
  let applied = false;
  for (let row = 0; row < height; row++) {
    const rowOff = row * w;
    let text = "";
    const colOf = [];
    const codeUnitToCell = [];
    for (let col = 0; col < w; col++) {
      const idx = rowOff + col;
      const cell = cellAtIndex(screen, idx);
      if (cell.width === 2 /* SpacerTail */ || cell.width === 3 /* SpacerHead */ || noSelect[idx] === 1) {
        continue;
      }
      const lc = cell.char.toLowerCase();
      const cellIdx = colOf.length;
      for (let i = 0; i < lc.length; i++) {
        codeUnitToCell.push(cellIdx);
      }
      text += lc;
      colOf.push(col);
    }
    let pos = text.indexOf(lq);
    while (pos >= 0) {
      applied = true;
      const startCi = codeUnitToCell[pos];
      const endCi = codeUnitToCell[pos + qlen - 1];
      for (let ci = startCi; ci <= endCi; ci++) {
        const col = colOf[ci];
        const cell = cellAtIndex(screen, rowOff + col);
        setCellStyleId(screen, col, row, stylePool.withInverse(cell.styleId));
      }
      pos = text.indexOf(lq, pos + qlen);
    }
  }
  return applied;
}

// src/ink/ink.tsx
import { jsx as jsx15 } from "react/jsx-runtime";
var ALT_SCREEN_ANCHOR_CURSOR = Object.freeze({
  x: 0,
  y: 0,
  visible: false
});
var CURSOR_HOME_PATCH = Object.freeze({
  type: "stdout",
  content: CURSOR_HOME
});
var ERASE_THEN_HOME_PATCH = Object.freeze({
  type: "stdout",
  content: ERASE_SCREEN + CURSOR_HOME
});
function makeAltScreenParkPatch(terminalRows) {
  return Object.freeze({
    type: "stdout",
    content: cursorPosition(terminalRows, 1)
  });
}
var Ink = class {
  constructor(options) {
    this.options = options;
    autoBind(this);
    if (this.options.patchConsole) {
      this.restoreConsole = this.patchConsole();
      this.restoreStderr = this.patchStderr();
    }
    this.terminal = {
      stdout: options.stdout,
      stderr: options.stderr
    };
    this.terminalColumns = options.stdout.columns || 80;
    this.terminalRows = options.stdout.rows || 24;
    this.altScreenParkPatch = makeAltScreenParkPatch(this.terminalRows);
    this.stylePool = new StylePool();
    this.charPool = new CharPool();
    this.hyperlinkPool = new HyperlinkPool();
    this.frontFrame = emptyFrame(
      this.terminalRows,
      this.terminalColumns,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    );
    this.backFrame = emptyFrame(
      this.terminalRows,
      this.terminalColumns,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    );
    this.log = new LogUpdate({
      isTTY: options.stdout.isTTY || false,
      stylePool: this.stylePool
    });
    const deferredRender = () => queueMicrotask(this.onRender);
    this.scheduleRender = throttle(deferredRender, FRAME_INTERVAL_MS, {
      leading: true,
      trailing: true
    });
    this.isUnmounted = false;
    this.unsubscribeExit = onExit(this.unmount, {
      alwaysLast: false
    });
    if (options.stdout.isTTY) {
      options.stdout.on("resize", this.handleResize);
      process.on("SIGCONT", this.handleResume);
      this.unsubscribeTTYHandlers = () => {
        options.stdout.off("resize", this.handleResize);
        process.off("SIGCONT", this.handleResume);
      };
    }
    this.rootNode = createNode("ink-root");
    this.focusManager = new FocusManager((target, event) => dispatcher.dispatchDiscrete(target, event));
    this.rootNode.focusManager = this.focusManager;
    this.renderer = createRenderer(this.rootNode, this.stylePool);
    this.rootNode.onRender = this.scheduleRender;
    this.rootNode.onImmediateRender = this.onRender;
    this.rootNode.onComputeLayout = () => {
      if (this.isUnmounted) {
        return;
      }
      if (this.rootNode.yogaNode) {
        const t0 = performance.now();
        this.rootNode.yogaNode.setWidth(this.terminalColumns);
        this.rootNode.yogaNode.calculateLayout(this.terminalColumns);
        const ms = performance.now() - t0;
        recordYogaMs(ms);
        const c = getYogaCounters();
        this.lastYogaCounters = {
          ms,
          ...c
        };
      }
    };
    this.container = reconciler_default.createContainer(
      this.rootNode,
      ConcurrentRoot,
      null,
      false,
      null,
      "id",
      noop2,
      // onUncaughtError
      noop2,
      // onCaughtError
      noop2,
      // onRecoverableError
      noop2
      // onDefaultTransitionIndicator
    );
    if (process.env.NODE_ENV === "development") {
      reconciler_default.injectIntoDevTools({
        bundleType: 0,
        // Reporting React DOM's version, not Ink's
        // See https://github.com/facebook/react/issues/16666#issuecomment-532639905
        version: "16.13.1",
        rendererPackageName: "ink"
      });
    }
  }
  log;
  terminal;
  scheduleRender;
  // Ignore last render after unmounting a tree to prevent empty output before exit
  isUnmounted = false;
  isPaused = false;
  container;
  rootNode;
  focusManager;
  renderer;
  stylePool;
  charPool;
  hyperlinkPool;
  exitPromise;
  restoreConsole;
  restoreStderr;
  unsubscribeTTYHandlers;
  terminalColumns;
  terminalRows;
  currentNode = null;
  frontFrame;
  backFrame;
  lastPoolResetTime = performance.now();
  drainTimer = null;
  lastYogaCounters = {
    ms: 0,
    visited: 0,
    measured: 0,
    cacheHits: 0,
    live: 0
  };
  altScreenParkPatch;
  // Text selection state (alt-screen only). Owned here so the overlay
  // pass in onRender can read it and App.tsx can update it from mouse
  // events. Public so instances.get() callers can access.
  selection = createSelectionState();
  // Search highlight query (alt-screen only). Setter below triggers
  // scheduleRender; applySearchHighlight in onRender inverts matching cells.
  searchHighlightQuery = "";
  // Position-based highlight. VML scans positions ONCE (via
  // scanElementSubtree, when the target message is mounted), stores them
  // message-relative, sets this for every-frame apply. rowOffset =
  // message's current screen-top. currentIdx = which position is
  // "current" (yellow). null clears. Positions are known upfront —
  // navigation is index arithmetic, no scan-feedback loop.
  searchPositions = null;
  // React-land subscribers for selection state changes (useHasSelection).
  // Fired alongside the terminal repaint whenever the selection mutates
  // so UI (e.g. footer hints) can react to selection appearing/clearing.
  selectionListeners = /* @__PURE__ */ new Set();
  selectionWasActive = false;
  // DOM nodes currently under the pointer (mode-1003 motion). Held here
  // so App.tsx's handleMouseEvent is stateless — dispatchHover diffs
  // against this set and mutates it in place.
  hoveredNodes = /* @__PURE__ */ new Set();
  // Set by <AlternateScreen> via setAltScreenActive(). Controls the
  // renderer's cursor.y clamping (keeps cursor in-viewport to avoid
  // LF-induced scroll when screen.height === terminalRows) and gates
  // alt-screen-aware SIGCONT/resize/unmount handling.
  altScreenActive = false;
  // Set alongside altScreenActive so SIGCONT resume knows whether to
  // re-enable mouse tracking (not all <AlternateScreen> uses want it).
  altScreenMouseTracking = false;
  // True when the previous frame's screen buffer cannot be trusted for
  // blit — selection overlay mutated it, resetFramesForAltScreen()
  // replaced it with blanks, or forceRedraw() reset it to 0×0. Forces
  // one full-render frame; steady-state frames after clear it and regain
  // the blit + narrow-damage fast path.
  prevFrameContaminated = false;
  // Set by handleResize: prepend ERASE_SCREEN to the next onRender's patches
  // INSIDE the BSU/ESU block so clear+paint is atomic. Writing ERASE_SCREEN
  // synchronously in handleResize would leave the screen blank for the ~80ms
  // render() takes; deferring into the atomic block means old content stays
  // visible until the new frame is fully ready.
  needsEraseBeforePaint = false;
  // Native cursor positioning: a component (via useDeclaredCursor) declares
  // where the terminal cursor should be parked after each frame. Terminal
  // emulators render IME preedit text at the physical cursor position, and
  // screen readers / screen magnifiers track it — so parking at the text
  // input's caret makes CJK input appear inline and lets a11y tools follow.
  cursorDeclaration = null;
  // Main-screen: physical cursor position after the declared-cursor move,
  // tracked separately from frame.cursor (which must stay at content-bottom
  // for log-update's relative-move invariants). Alt-screen doesn't need
  // this — every frame begins with CSI H. null = no move emitted last frame.
  displayCursor = null;
  // Burst of SIGWINCH (vscode panel drag) → one React commit per
  // microtask. Dims are captured sync in handleResize; only the
  // expensive tree rebuild defers.
  pendingResizeRender = false;
  // Fold synchronous re-entry (selection fanout, onFrame callback)
  // into one follow-up microtask instead of stacking renders.
  isRendering = false;
  immediateRerenderRequested = false;
  handleResume = () => {
    if (!this.options.stdout.isTTY) {
      return;
    }
    if (this.altScreenActive) {
      this.reenterAltScreen();
      return;
    }
    this.frontFrame = emptyFrame(
      this.frontFrame.viewport.height,
      this.frontFrame.viewport.width,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    );
    this.backFrame = emptyFrame(
      this.backFrame.viewport.height,
      this.backFrame.viewport.width,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    );
    this.log.reset();
    this.displayCursor = null;
  };
  // Dims captured sync — closes the stale-dim window the original
  // debounce rejection warned about. Expensive React commit defers to
  // one microtask per burst: vscode fires many SIGWINCHes per panel
  // drag, each ~80ms uncoalesced = event loop visibly locks up.
  handleResize = () => {
    const cols = this.options.stdout.columns || 80;
    const rows = this.options.stdout.rows || 24;
    if (cols === this.terminalColumns && rows === this.terminalRows) {
      return;
    }
    this.terminalColumns = cols;
    this.terminalRows = rows;
    this.altScreenParkPatch = makeAltScreenParkPatch(this.terminalRows);
    this.scheduleRender.cancel?.();
    if (this.drainTimer !== null) {
      clearTimeout(this.drainTimer);
      this.drainTimer = null;
    }
    if (this.altScreenActive && !this.isPaused && this.options.stdout.isTTY) {
      if (this.altScreenMouseTracking) {
        this.options.stdout.write(ENABLE_MOUSE_TRACKING);
      }
      this.resetFramesForAltScreen();
      this.needsEraseBeforePaint = true;
    }
    if (this.pendingResizeRender) {
      return;
    }
    this.pendingResizeRender = true;
    queueMicrotask(() => {
      this.pendingResizeRender = false;
      if (this.isUnmounted || this.currentNode === null) {
        return;
      }
      this.render(this.currentNode);
    });
  };
  resolveExitPromise = () => {
  };
  rejectExitPromise = () => {
  };
  unsubscribeExit = () => {
  };
  /**
   * Pause Ink and hand the terminal over to an external TUI (e.g. git
   * commit editor). In non-fullscreen mode this enters the alt screen;
   * in fullscreen mode we're already in alt so we just clear it.
   * Call `exitAlternateScreen()` when done to restore Ink.
   */
  enterAlternateScreen() {
    this.pause();
    this.suspendStdin();
    this.options.stdout.write(
      // Disable extended key reporting first — editors that don't speak
      // CSI-u (e.g. nano) show "Unknown sequence" for every Ctrl-<key> if
      // kitty/modifyOtherKeys stays active. exitAlternateScreen re-enables.
      DISABLE_KITTY_KEYBOARD + DISABLE_MODIFY_OTHER_KEYS + (this.altScreenMouseTracking ? DISABLE_MOUSE_TRACKING : "") + // disable mouse (no-op if off)
      (this.altScreenActive ? "" : "\x1B[?1049h") + // enter alt (already in alt if fullscreen)
      "\x1B[?1004l\x1B[0m\x1B[?25h\x1B[2J\x1B[H"
      // cursor home
    );
  }
  /**
   * Resume Ink after an external TUI handoff with a full repaint.
   * In non-fullscreen mode this exits the alt screen back to main;
   * in fullscreen mode we re-enter alt and clear + repaint.
   *
   * The re-enter matters: terminal editors (vim, nano, less) write
   * smcup/rmcup (?1049h/?1049l), so even though we started in alt,
   * the editor's rmcup on exit drops us to main screen. Without
   * re-entering, the 2J below wipes the user's main-screen scrollback
   * and subsequent renders land in main — native terminal scroll
   * returns, fullscreen scroll is dead.
   */
  exitAlternateScreen() {
    this.options.stdout.write(
      (this.altScreenActive ? ENTER_ALT_SCREEN : "") + // re-enter alt — vim's rmcup dropped us to main
      "\x1B[2J\x1B[H" + // cursor home
      (this.altScreenMouseTracking ? ENABLE_MOUSE_TRACKING : "") + (this.altScreenActive ? "" : "\x1B[?1049l") + // exit alt (non-fullscreen only)
      "\x1B[?25l"
      // hide cursor (Ink manages)
    );
    this.resumeStdin();
    if (this.altScreenActive) {
      this.resetFramesForAltScreen();
    } else {
      this.repaint();
    }
    this.resume();
    this.options.stdout.write(
      "\x1B[?1004h" + (supportsExtendedKeys() ? DISABLE_KITTY_KEYBOARD + ENABLE_KITTY_KEYBOARD + ENABLE_MODIFY_OTHER_KEYS : "")
    );
  }
  onRender() {
    if (this.isUnmounted || this.isPaused) {
      return;
    }
    if (this.isRendering) {
      this.immediateRerenderRequested = true;
      return;
    }
    this.isRendering = true;
    if (this.drainTimer !== null) {
      clearTimeout(this.drainTimer);
      this.drainTimer = null;
    }
    flushInteractionTime();
    const renderStart = performance.now();
    const terminalWidth = this.options.stdout.columns || 80;
    const terminalRows = this.options.stdout.rows || 24;
    const frame = this.renderer({
      frontFrame: this.frontFrame,
      backFrame: this.backFrame,
      isTTY: this.options.stdout.isTTY,
      terminalWidth,
      terminalRows,
      altScreen: this.altScreenActive,
      prevFrameContaminated: this.prevFrameContaminated
    });
    const rendererMs = performance.now() - renderStart;
    const follow = consumeFollowScroll();
    if (follow && this.selection.anchor && // Only translate if the selection is ON scrollbox content. Selections
    // in the footer/prompt/StickyPromptHeader are on static text — the
    // scroll doesn't move what's under them. Without this guard, a
    // footer selection would be shifted by -delta then clamped to
    // viewportBottom, teleporting it into the scrollbox. Mirror the
    // bounds check the deleted check() in ScrollKeybindingHandler had.
    this.selection.anchor.row >= follow.viewportTop && this.selection.anchor.row <= follow.viewportBottom) {
      const { delta, viewportTop, viewportBottom } = follow;
      if (this.selection.isDragging) {
        if (hasSelection(this.selection)) {
          captureScrolledRows(this.selection, this.frontFrame.screen, viewportTop, viewportTop + delta - 1, "above");
        }
        shiftAnchor(this.selection, -delta, viewportTop, viewportBottom);
      } else if (
        // Flag-3 guard: the anchor check above only proves ONE endpoint is
        // on scrollbox content. A drag from row 3 (scrollbox) into the
        // footer at row 6, then release, leaves focus outside the viewport
        // — shiftSelectionForFollow would clamp it to viewportBottom,
        // teleporting the highlight from static footer into the scrollbox.
        // Symmetric check: require BOTH ends inside to translate. A
        // straddling selection falls through to NEITHER shift NOR capture:
        // the footer endpoint pins the selection, text scrolls away under
        // the highlight, and getSelectedText reads the CURRENT screen
        // contents — no accumulation. Dragging branch doesn't need this:
        // shiftAnchor ignores focus, and the anchor DOES shift (so capture
        // is correct there even when focus is in the footer).
        !this.selection.focus || this.selection.focus.row >= viewportTop && this.selection.focus.row <= viewportBottom
      ) {
        if (hasSelection(this.selection)) {
          captureScrolledRows(this.selection, this.frontFrame.screen, viewportTop, viewportTop + delta - 1, "above");
        }
        const cleared = shiftSelectionForFollow(this.selection, -delta, viewportTop, viewportBottom);
        if (cleared) {
          for (const cb of this.selectionListeners) {
            cb();
          }
        }
      }
    }
    let selActive = false;
    let hlActive = false;
    if (this.altScreenActive) {
      selActive = hasSelection(this.selection);
      if (selActive) {
        applySelectionOverlay(frame.screen, this.selection, this.stylePool);
      }
      hlActive = applySearchHighlight(frame.screen, this.searchHighlightQuery, this.stylePool);
      if (this.searchPositions) {
        const sp = this.searchPositions;
        const posApplied = applyPositionedHighlight(
          frame.screen,
          this.stylePool,
          sp.positions,
          sp.rowOffset,
          sp.currentIdx
        );
        hlActive = hlActive || posApplied;
      }
    }
    if (didLayoutShift() || selActive || hlActive || this.prevFrameContaminated) {
      frame.screen.damage = {
        x: 0,
        y: 0,
        width: frame.screen.width,
        height: frame.screen.height
      };
    }
    let prevFrame = this.frontFrame;
    if (this.altScreenActive) {
      prevFrame = {
        ...this.frontFrame,
        cursor: ALT_SCREEN_ANCHOR_CURSOR
      };
    }
    const tDiff = performance.now();
    const diff2 = this.log.render(
      prevFrame,
      frame,
      this.altScreenActive,
      // DECSTBM needs BSU/ESU atomicity — without it the outer terminal
      // renders the scrolled-but-not-yet-repainted intermediate state.
      // tmux is the main case (re-emits DECSTBM with its own timing and
      // doesn't implement DEC 2026, so SYNC_OUTPUT_SUPPORTED is false).
      SYNC_OUTPUT_SUPPORTED
    );
    const diffMs = performance.now() - tDiff;
    this.backFrame = this.frontFrame;
    this.frontFrame = frame;
    if (renderStart - this.lastPoolResetTime > 5 * 60 * 1e3) {
      this.resetPools();
      this.lastPoolResetTime = renderStart;
    }
    const flickers = [];
    for (const patch of diff2) {
      if (patch.type === "clearTerminal") {
        flickers.push({
          desiredHeight: frame.screen.height,
          availableHeight: frame.viewport.height,
          reason: patch.reason
        });
      }
    }
    const tOptimize = performance.now();
    const optimized = optimize(diff2);
    const optimizeMs = performance.now() - tOptimize;
    const hasDiff = optimized.length > 0;
    if (this.altScreenActive && hasDiff) {
      if (this.needsEraseBeforePaint) {
        this.needsEraseBeforePaint = false;
        optimized.unshift(ERASE_THEN_HOME_PATCH);
      } else {
        optimized.unshift(CURSOR_HOME_PATCH);
      }
      optimized.push(this.altScreenParkPatch);
    }
    const decl = this.cursorDeclaration;
    const rect = decl !== null ? nodeCache.get(decl.node) : void 0;
    const target = decl !== null && rect !== void 0 ? {
      x: rect.x + decl.relativeX,
      y: rect.y + decl.relativeY
    } : null;
    const parked = this.displayCursor;
    const targetMoved = target !== null && (parked === null || parked.x !== target.x || parked.y !== target.y);
    if (hasDiff || targetMoved || target === null && parked !== null) {
      if (parked !== null && !this.altScreenActive && hasDiff) {
        const pdx = prevFrame.cursor.x - parked.x;
        const pdy = prevFrame.cursor.y - parked.y;
        if (pdx !== 0 || pdy !== 0) {
          optimized.unshift({
            type: "stdout",
            content: cursorMove(pdx, pdy)
          });
        }
      }
      if (target !== null) {
        if (this.altScreenActive) {
          const row = Math.min(Math.max(target.y + 1, 1), terminalRows);
          const col = Math.min(Math.max(target.x + 1, 1), terminalWidth);
          optimized.push({
            type: "stdout",
            content: cursorPosition(row, col)
          });
        } else {
          const from = !hasDiff && parked !== null ? parked : {
            x: frame.cursor.x,
            y: frame.cursor.y
          };
          const dx = target.x - from.x;
          const dy = target.y - from.y;
          if (dx !== 0 || dy !== 0) {
            optimized.push({
              type: "stdout",
              content: cursorMove(dx, dy)
            });
          }
        }
        this.displayCursor = target;
      } else {
        if (parked !== null && !this.altScreenActive && !hasDiff) {
          const rdx = frame.cursor.x - parked.x;
          const rdy = frame.cursor.y - parked.y;
          if (rdx !== 0 || rdy !== 0) {
            optimized.push({
              type: "stdout",
              content: cursorMove(rdx, rdy)
            });
          }
        }
        this.displayCursor = null;
      }
    }
    const tWrite = performance.now();
    writeDiffToTerminal(this.terminal, optimized, this.altScreenActive && !SYNC_OUTPUT_SUPPORTED);
    const writeMs = performance.now() - tWrite;
    this.prevFrameContaminated = selActive || hlActive || !!frame.absoluteOverlayMoved;
    if (frame.scrollDrainPending) {
      this.drainTimer = setTimeout(() => this.onRender(), FRAME_INTERVAL_MS >> 2);
    }
    const yogaMs = getLastYogaMs();
    const commitMs = getLastCommitMs();
    const yc = this.lastYogaCounters;
    resetProfileCounters();
    this.lastYogaCounters = {
      ms: 0,
      visited: 0,
      measured: 0,
      cacheHits: 0,
      live: 0
    };
    this.options.onFrame?.({
      durationMs: performance.now() - renderStart,
      phases: {
        renderer: rendererMs,
        diff: diffMs,
        optimize: optimizeMs,
        write: writeMs,
        patches: diff2.length,
        yoga: yogaMs,
        commit: commitMs,
        yogaVisited: yc.visited,
        yogaMeasured: yc.measured,
        yogaCacheHits: yc.cacheHits,
        yogaLive: yc.live
      },
      flickers
    });
    this.isRendering = false;
    if (this.immediateRerenderRequested) {
      this.immediateRerenderRequested = false;
      queueMicrotask(() => this.onRender());
    }
  }
  pause() {
    reconciler_default.flushSyncFromReconciler();
    this.onRender();
    this.isPaused = true;
  }
  resume() {
    this.isPaused = false;
    this.onRender();
  }
  /**
   * Reset frame buffers so the next render writes the full screen from scratch.
   * Call this before resume() when the terminal content has been corrupted by
   * an external process (e.g. tmux, shell, full-screen TUI).
   */
  repaint() {
    this.frontFrame = emptyFrame(
      this.frontFrame.viewport.height,
      this.frontFrame.viewport.width,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    );
    this.backFrame = emptyFrame(
      this.backFrame.viewport.height,
      this.backFrame.viewport.width,
      this.stylePool,
      this.charPool,
      this.hyperlinkPool
    );
    this.log.reset();
    this.displayCursor = null;
  }
  /**
   * Clear the physical terminal and force a full redraw.
   *
   * The traditional readline ctrl+l — clears the visible screen and
   * redraws the current content. Also the recovery path when the terminal
   * was cleared externally (macOS Cmd+K) and Ink's diff engine thinks
   * unchanged cells don't need repainting. Scrollback is preserved.
   */
  forceRedraw() {
    if (!this.options.stdout.isTTY || this.isUnmounted || this.isPaused) {
      return;
    }
    this.options.stdout.write(ERASE_SCREEN + CURSOR_HOME);
    if (this.altScreenActive) {
      this.resetFramesForAltScreen();
    } else {
      this.repaint();
      this.prevFrameContaminated = true;
    }
    this.onRender();
  }
  /**
   * Mark the previous frame as untrustworthy for blit, forcing the next
   * render to do a full-damage diff instead of the per-node fast path.
   *
   * Lighter than forceRedraw() — no screen clear, no extra write. Call
   * from a useLayoutEffect cleanup when unmounting a tall overlay: the
   * blit fast path can copy stale cells from the overlay frame into rows
   * the shrunken layout no longer reaches, leaving a ghost title/divider.
   * onRender resets the flag at frame end so it's one-shot.
   */
  invalidatePrevFrame() {
    this.prevFrameContaminated = true;
  }
  /**
   * Called by the <AlternateScreen> component on mount/unmount.
   * Controls cursor.y clamping in the renderer and gates alt-screen-aware
   * behavior in SIGCONT/resize/unmount handlers. Repaints on change so
   * the first alt-screen frame (and first main-screen frame on exit) is
   * a full redraw with no stale diff state.
   */
  setAltScreenActive(active, mouseTracking = false) {
    if (this.altScreenActive === active) {
      return;
    }
    this.altScreenActive = active;
    this.altScreenMouseTracking = active && mouseTracking;
    if (active) {
      this.resetFramesForAltScreen();
    } else {
      this.repaint();
    }
  }
  get isAltScreenActive() {
    return this.altScreenActive;
  }
  /**
   * Re-assert terminal modes after a gap (>5s stdin silence or event-loop
   * stall). Catches tmux detach→attach, ssh reconnect, and laptop
   * sleep/wake — none of which send SIGCONT. The terminal may reset DEC
   * private modes on reconnect; this method restores them.
   *
   * Always re-asserts extended key reporting and mouse tracking. Mouse
   * tracking is idempotent (DEC private mode set-when-set is a no-op). The
   * Kitty keyboard protocol is NOT — CSI >1u is a stack push, so we pop
   * first to keep depth balanced (pop on empty stack is a no-op per spec,
   * so after a terminal reset this still restores depth 0→1). Without the
   * pop, each >5s idle gap adds a stack entry, and the single pop on exit
   * or suspend can't drain them — the shell is left in CSI u mode where
   * Ctrl+C/Ctrl+D leak as escape sequences. The alt-screen
   * re-entry (ERASE_SCREEN + frame reset) is NOT idempotent — it blanks the
   * screen — so it's opt-in via includeAltScreen. The stdin-gap caller fires
   * on ordinary >5s idle + keypress and must not erase; the event-loop stall
   * detector fires on genuine sleep/wake and opts in. tmux attach / ssh
   * reconnect typically send a resize, which already covers alt-screen via
   * handleResize.
   */
  reassertTerminalModes = (includeAltScreen = false) => {
    if (!this.options.stdout.isTTY) {
      return;
    }
    if (this.isPaused) {
      return;
    }
    if (supportsExtendedKeys()) {
      this.options.stdout.write(DISABLE_KITTY_KEYBOARD + ENABLE_KITTY_KEYBOARD + ENABLE_MODIFY_OTHER_KEYS);
    }
    if (!this.altScreenActive) {
      return;
    }
    if (this.altScreenMouseTracking) {
      this.options.stdout.write(ENABLE_MOUSE_TRACKING);
    }
    if (includeAltScreen) {
      this.reenterAltScreen();
    }
  };
  /**
   * Mark this instance as unmounted so future unmount() calls early-return.
   * Called by gracefulShutdown's cleanupTerminalModes() after it has sent
   * EXIT_ALT_SCREEN but before the remaining terminal-reset sequences.
   * Without this, signal-exit's deferred ink.unmount() (triggered by
   * process.exit()) runs the full unmount path: onRender() + writeSync
   * cleanup block + updateContainerSync → AlternateScreen unmount cleanup.
   * The result is 2-3 redundant EXIT_ALT_SCREEN sequences landing on the
   * main screen AFTER printResumeHint(), which tmux (at least) interprets
   * as restoring the saved cursor position — clobbering the resume hint.
   */
  detachForShutdown() {
    this.isUnmounted = true;
    this.scheduleRender.cancel?.();
    const stdin = this.options.stdin;
    this.drainStdin();
    if (stdin.isTTY && stdin.isRaw && stdin.setRawMode) {
      stdin.setRawMode(false);
    }
  }
  /** @see drainStdin */
  drainStdin() {
    drainStdin(this.options.stdin);
  }
  /**
   * Re-enter alt-screen, clear, home, re-enable mouse tracking, and reset
   * frame buffers so the next render repaints from scratch. Self-heal for
   * SIGCONT, resize, and stdin-gap/event-loop-stall (sleep/wake) — any of
   * which can leave the terminal in main-screen mode while altScreenActive
   * stays true. ENTER_ALT_SCREEN is a terminal-side no-op if already in alt.
   */
  reenterAltScreen() {
    this.options.stdout.write(
      ENTER_ALT_SCREEN + ERASE_SCREEN + CURSOR_HOME + (this.altScreenMouseTracking ? ENABLE_MOUSE_TRACKING : "")
    );
    this.resetFramesForAltScreen();
  }
  /**
   * Seed prev/back frames with full-size BLANK screens (rows×cols of empty
   * cells, not 0×0). In alt-screen mode, next.screen.height is always
   * terminalRows; if prev.screen.height is 0 (emptyFrame's default),
   * log-update sees heightDelta > 0 ('growing') and calls renderFrameSlice,
   * whose trailing per-row CR+LF at the last row scrolls the alt screen,
   * permanently desyncing the virtual and physical cursors by 1 row.
   *
   * With a rows×cols blank prev, heightDelta === 0 → standard diffEach
   * → moveCursorTo (CSI cursorMove, no LF, no scroll).
   *
   * viewport.height = rows + 1 matches the renderer's alt-screen output,
   * preventing a spurious resize trigger on the first frame. cursor.y = 0
   * matches the physical cursor after ENTER_ALT_SCREEN + CSI H (home).
   */
  resetFramesForAltScreen() {
    const rows = this.terminalRows;
    const cols = this.terminalColumns;
    const blank = () => ({
      screen: createScreen(cols, rows, this.stylePool, this.charPool, this.hyperlinkPool),
      viewport: {
        width: cols,
        height: rows + 1
      },
      cursor: {
        x: 0,
        y: 0,
        visible: true
      }
    });
    this.frontFrame = blank();
    this.backFrame = blank();
    this.log.reset();
    this.displayCursor = null;
    this.prevFrameContaminated = true;
  }
  /**
   * Copy the current selection to the clipboard without clearing the
   * highlight. Matches iTerm2's copy-on-select behavior where the selected
   * region stays visible after the automatic copy.
   */
  copySelectionNoClear() {
    if (!hasSelection(this.selection)) {
      return "";
    }
    const text = getSelectedText(this.selection, this.frontFrame.screen);
    if (text) {
      void setClipboard(text).then((raw) => {
        if (raw) {
          this.options.stdout.write(raw);
        }
      });
    }
    return text;
  }
  /**
   * Copy the current text selection to the system clipboard via OSC 52
   * and clear the selection. Returns the copied text (empty if no selection).
   */
  copySelection() {
    if (!hasSelection(this.selection)) {
      return "";
    }
    const text = this.copySelectionNoClear();
    clearSelection(this.selection);
    this.notifySelectionChange();
    return text;
  }
  /** Clear the current text selection without copying. */
  clearTextSelection() {
    if (!hasSelection(this.selection)) {
      return;
    }
    clearSelection(this.selection);
    this.notifySelectionChange();
  }
  /**
   * Set the search highlight query. Non-empty → all visible occurrences
   * are inverted (SGR 7) on the next frame; first one also underlined.
   * Empty → clears (prevFrameContaminated handles the frame after). Same
   * damage-tracking machinery as selection — setCellStyleId doesn't track
   * damage, so the overlay forces full-frame damage while active.
   */
  setSearchHighlight(query) {
    if (this.searchHighlightQuery === query) {
      return;
    }
    this.searchHighlightQuery = query;
    this.scheduleRender();
  }
  /** Paint an EXISTING DOM subtree to a fresh Screen at its natural
   *  height, scan for query. Returns positions relative to the element's
   *  bounding box (row 0 = element top).
   *
   *  The element comes from the MAIN tree — built with all real
   *  providers, yoga already computed. We paint it to a fresh buffer
   *  with offsets so it lands at (0,0). Same paint path as the main
   *  render. Zero drift. No second React root, no context bridge.
   *
   *  ~1-2ms (paint only, no reconcile — the DOM is already built). */
  scanElementSubtree(el) {
    if (!this.searchHighlightQuery || !el.yogaNode) {
      return [];
    }
    const width = Math.ceil(el.yogaNode.getComputedWidth());
    const height = Math.ceil(el.yogaNode.getComputedHeight());
    if (width <= 0 || height <= 0) {
      return [];
    }
    const elLeft = el.yogaNode.getComputedLeft();
    const elTop = el.yogaNode.getComputedTop();
    const screen = createScreen(width, height, this.stylePool, this.charPool, this.hyperlinkPool);
    const output = new Output({
      width,
      height,
      stylePool: this.stylePool,
      screen
    });
    render_node_to_output_default(el, output, {
      offsetX: -elLeft,
      offsetY: -elTop,
      prevScreen: void 0
    });
    const rendered = output.get();
    markDirty(el);
    const positions = scanPositions(rendered, this.searchHighlightQuery);
    logForDebugging(
      `scanElementSubtree: q='${this.searchHighlightQuery}' el=${width}x${height}@(${elLeft},${elTop}) n=${positions.length} [${positions.slice(0, 10).map((p) => `${p.row}:${p.col}`).join(",")}${positions.length > 10 ? ",\u2026" : ""}]`
    );
    return positions;
  }
  /** Set the position-based highlight state. Every frame, writes CURRENT
   *  style at positions[currentIdx] + rowOffset. null clears. The scan-
   *  highlight (inverse on all matches) still runs — this overlays yellow
   *  on top. rowOffset changes as the user scrolls (= message's current
   *  screen-top); positions stay stable (message-relative). */
  setSearchPositions(state) {
    this.searchPositions = state;
    this.scheduleRender();
  }
  /**
   * Set the selection highlight background color. Replaces the per-cell
   * SGR-7 inverse with a solid theme-aware bg (matches native terminal
   * selection). Accepts the same color formats as Text backgroundColor
   * (rgb(), ansi:name, #hex, ansi256()) — colorize() routes through
   * chalk so the tmux/xterm.js level clamps in colorize.ts apply and
   * the emitted SGR is correct for the current terminal.
   *
   * Called by React-land once theme is known (ScrollKeybindingHandler's
   * useEffect watching useTheme). Before that call, withSelectionBg
   * falls back to withInverse so selection still renders on the first
   * frame; the effect fires before any mouse input so the fallback is
   * unobservable in practice.
   */
  setSelectionBgColor(color) {
    const wrapped = colorize("\0", color, "background");
    const nul = wrapped.indexOf("\0");
    if (nul <= 0 || nul === wrapped.length - 1) {
      this.stylePool.setSelectionBg(null);
      return;
    }
    this.stylePool.setSelectionBg({
      type: "ansi",
      code: wrapped.slice(0, nul),
      endCode: wrapped.slice(nul + 1)
      // always \x1b[49m for bg
    });
  }
  /**
   * Capture text from rows about to scroll out of the viewport during
   * drag-to-scroll. Must be called BEFORE the ScrollBox scrolls so the
   * screen buffer still holds the outgoing content. Accumulated into
   * the selection state and joined back in by getSelectedText.
   */
  captureScrolledRows(firstRow, lastRow, side) {
    captureScrolledRows(this.selection, this.frontFrame.screen, firstRow, lastRow, side);
  }
  /**
   * Shift anchor AND focus by dRow, clamped to [minRow, maxRow]. Used by
   * keyboard scroll handlers (PgUp/PgDn etc.) so the highlight tracks the
   * content instead of disappearing. Unlike shiftAnchor (drag-to-scroll),
   * this moves BOTH endpoints — the user isn't holding the mouse at one
   * edge. Supplies screen.width for the col-reset-on-clamp boundary.
   */
  shiftSelectionForScroll(dRow, minRow, maxRow) {
    const hadSel = hasSelection(this.selection);
    shiftSelection(this.selection, dRow, minRow, maxRow, this.frontFrame.screen.width);
    if (hadSel && !hasSelection(this.selection)) {
      this.notifySelectionChange();
    }
  }
  /**
   * Keyboard selection extension (shift+arrow/home/end). Moves focus;
   * anchor stays fixed so the highlight grows or shrinks relative to it.
   * Left/right wrap across row boundaries — native macOS text-edit
   * behavior: shift+left at col 0 wraps to end of the previous row.
   * Up/down clamp at viewport edges (no scroll-to-extend yet). Drops to
   * char mode. No-op outside alt-screen or without an active selection.
   */
  moveSelectionFocus(move) {
    if (!this.altScreenActive) {
      return;
    }
    const { focus } = this.selection;
    if (!focus) {
      return;
    }
    const { width, height } = this.frontFrame.screen;
    const maxCol = width - 1;
    const maxRow = height - 1;
    let { col, row } = focus;
    switch (move) {
      case "left":
        if (col > 0) {
          col--;
        } else if (row > 0) {
          col = maxCol;
          row--;
        }
        break;
      case "right":
        if (col < maxCol) {
          col++;
        } else if (row < maxRow) {
          col = 0;
          row++;
        }
        break;
      case "up":
        if (row > 0) {
          row--;
        }
        break;
      case "down":
        if (row < maxRow) {
          row++;
        }
        break;
      case "lineStart":
        col = 0;
        break;
      case "lineEnd":
        col = maxCol;
        break;
    }
    if (col === focus.col && row === focus.row) {
      return;
    }
    moveFocus(this.selection, col, row);
    this.notifySelectionChange();
  }
  /** Whether there is an active text selection. */
  hasTextSelection() {
    return hasSelection(this.selection);
  }
  /**
   * Subscribe to selection state changes. Fires whenever the selection
   * is started, updated, cleared, or copied. Returns an unsubscribe fn.
   */
  subscribeToSelectionChange(cb) {
    this.selectionListeners.add(cb);
    return () => this.selectionListeners.delete(cb);
  }
  notifySelectionChange() {
    this.scheduleRender();
    const active = hasSelection(this.selection);
    if (active !== this.selectionWasActive) {
      this.selectionWasActive = active;
      for (const cb of this.selectionListeners) {
        cb();
      }
    }
  }
  /**
   * Hit-test the rendered DOM tree at (col, row) and bubble a ClickEvent
   * from the deepest hit node up through ancestors with onClick handlers.
   * Returns true if a DOM handler consumed the click. Gated on
   * altScreenActive — clicks only make sense with a fixed viewport where
   * nodeCache rects map 1:1 to terminal cells (no scrollback offset).
   */
  dispatchClick(col, row) {
    if (!this.altScreenActive) {
      return false;
    }
    const blank = isEmptyCellAt(this.frontFrame.screen, col, row);
    return dispatchClick(this.rootNode, col, row, blank);
  }
  dispatchMouseDown(col, row, button) {
    if (!this.altScreenActive) {
      return void 0;
    }
    return dispatchMouse(
      this.rootNode,
      col,
      row,
      "onMouseDown",
      button,
      isEmptyCellAt(this.frontFrame.screen, col, row)
    );
  }
  dispatchMouseUp(target, col, row, button) {
    if (!this.altScreenActive) {
      return;
    }
    dispatchMouse(this.rootNode, col, row, "onMouseUp", button, isEmptyCellAt(this.frontFrame.screen, col, row), target);
  }
  dispatchMouseDrag(target, col, row, button) {
    if (!this.altScreenActive) {
      return;
    }
    dispatchMouse(
      this.rootNode,
      col,
      row,
      "onMouseDrag",
      button,
      isEmptyCellAt(this.frontFrame.screen, col, row),
      target
    );
  }
  dispatchHover(col, row) {
    if (!this.altScreenActive) {
      return;
    }
    dispatchHover(this.rootNode, col, row, this.hoveredNodes);
  }
  dispatchKeyboardEvent(parsedKey) {
    const target = this.focusManager.activeElement ?? this.rootNode;
    const event = new KeyboardEvent(parsedKey);
    dispatcher.dispatchDiscrete(target, event);
    if (!event.defaultPrevented && parsedKey.name === "tab" && !parsedKey.ctrl && !parsedKey.meta) {
      if (parsedKey.shift) {
        this.focusManager.focusPrevious(this.rootNode);
      } else {
        this.focusManager.focusNext(this.rootNode);
      }
    }
  }
  /**
   * Look up the URL at (col, row) in the current front frame. Checks for
   * an OSC 8 hyperlink first, then falls back to scanning the row for a
   * plain-text URL (mouse tracking intercepts the terminal's native
   * Cmd+Click URL detection, so we replicate it). This is a pure lookup
   * with no side effects — call it synchronously at click time so the
   * result reflects the screen the user actually clicked on, then defer
   * the browser-open action via a timer.
   */
  getHyperlinkAt(col, row) {
    if (!this.altScreenActive) {
      return void 0;
    }
    const screen = this.frontFrame.screen;
    const cell = cellAt(screen, col, row);
    let url = cell?.hyperlink;
    if (!url && cell?.width === 2 /* SpacerTail */ && col > 0) {
      url = cellAt(screen, col - 1, row)?.hyperlink;
    }
    return url ?? findPlainTextUrlAt(screen, col, row);
  }
  /**
   * Optional callback fired when clicking an OSC 8 hyperlink in fullscreen
   * mode. Set by FullscreenLayout via useLayoutEffect.
   */
  onHyperlinkClick;
  /**
   * Stable prototype wrapper for onHyperlinkClick. Passed to <App> as
   * onOpenHyperlink so the prop is a bound method (autoBind'd) that reads
   * the mutable field at call time — not the undefined-at-render value.
   */
  openHyperlink(url) {
    this.onHyperlinkClick?.(url);
  }
  /**
   * Handle a double- or triple-click at (col, row): select the word or
   * line under the cursor by reading the current screen buffer. Called on
   * PRESS (not release) so the highlight appears immediately and drag can
   * extend the selection word-by-word / line-by-line. Falls back to
   * char-mode startSelection if the click lands on a noSelect cell.
   */
  handleMultiClick(col, row, count) {
    if (!this.altScreenActive) {
      return;
    }
    const screen = this.frontFrame.screen;
    startSelection(this.selection, col, row);
    if (count === 2) {
      selectWordAt(this.selection, screen, col, row);
    } else {
      selectLineAt(this.selection, screen, row);
    }
    if (!this.selection.focus) {
      this.selection.focus = this.selection.anchor;
    }
    this.notifySelectionChange();
  }
  /**
   * Handle a drag-motion at (col, row). In char mode updates focus to the
   * exact cell. In word/line mode snaps to word/line boundaries so the
   * selection extends by word/line like native macOS. Gated on
   * altScreenActive for the same reason as dispatchClick.
   */
  handleSelectionDrag(col, row) {
    if (!this.altScreenActive) {
      return;
    }
    const sel = this.selection;
    if (sel.anchorSpan) {
      extendSelection(sel, this.frontFrame.screen, col, row);
    } else {
      updateSelection(sel, col, row);
    }
    this.notifySelectionChange();
  }
  // Methods to properly suspend stdin for external editor usage
  // This is needed to prevent Ink from swallowing keystrokes when an external editor is active
  stdinListeners = [];
  wasRawMode = false;
  suspendStdin() {
    const stdin = this.options.stdin;
    if (!stdin.isTTY) {
      return;
    }
    const readableListeners = stdin.listeners("readable");
    logForDebugging(
      `[stdin] suspendStdin: removing ${readableListeners.length} readable listener(s), wasRawMode=${stdin.isRaw ?? false}`
    );
    readableListeners.forEach((listener) => {
      this.stdinListeners.push({
        event: "readable",
        listener
      });
      stdin.removeListener("readable", listener);
    });
    const stdinWithRaw = stdin;
    if (stdinWithRaw.isRaw && stdinWithRaw.setRawMode) {
      stdinWithRaw.setRawMode(false);
      this.wasRawMode = true;
    }
  }
  resumeStdin() {
    const stdin = this.options.stdin;
    if (!stdin.isTTY) {
      return;
    }
    if (this.stdinListeners.length === 0 && !this.wasRawMode) {
      logForDebugging("[stdin] resumeStdin: called with no stored listeners and wasRawMode=false (possible desync)", {
        level: "warn"
      });
    }
    logForDebugging(
      `[stdin] resumeStdin: re-attaching ${this.stdinListeners.length} listener(s), wasRawMode=${this.wasRawMode}`
    );
    this.stdinListeners.forEach(({ event, listener }) => {
      stdin.addListener(event, listener);
    });
    this.stdinListeners = [];
    if (this.wasRawMode) {
      const stdinWithRaw = stdin;
      if (stdinWithRaw.setRawMode) {
        stdinWithRaw.setRawMode(true);
      }
      this.wasRawMode = false;
    }
  }
  // Stable identity for TerminalWriteContext. An inline arrow here would
  // change on every render() call (initial mount + each resize), which
  // cascades through useContext → <AlternateScreen>'s useLayoutEffect dep
  // array → spurious exit+re-enter of the alt screen on every SIGWINCH.
  writeRaw(data) {
    this.options.stdout.write(data);
  }
  setCursorDeclaration = (decl, clearIfNode) => {
    if (decl === null && clearIfNode !== void 0 && this.cursorDeclaration?.node !== clearIfNode) {
      return;
    }
    this.cursorDeclaration = decl;
  };
  render(node) {
    this.currentNode = node;
    const tree = /* @__PURE__ */ jsx15(
      App,
      {
        dispatchKeyboardEvent: this.dispatchKeyboardEvent,
        exitOnCtrlC: this.options.exitOnCtrlC,
        getHyperlinkAt: this.getHyperlinkAt,
        onClickAt: this.dispatchClick,
        onCursorDeclaration: this.setCursorDeclaration,
        onExit: this.unmount,
        onHoverAt: this.dispatchHover,
        onMouseDownAt: this.dispatchMouseDown,
        onMouseDragAt: this.dispatchMouseDrag,
        onMouseUpAt: this.dispatchMouseUp,
        onMultiClick: this.handleMultiClick,
        onOpenHyperlink: this.openHyperlink,
        onSelectionChange: this.notifySelectionChange,
        onSelectionDrag: this.handleSelectionDrag,
        onStdinResume: this.reassertTerminalModes,
        selection: this.selection,
        stderr: this.options.stderr,
        stdin: this.options.stdin,
        stdout: this.options.stdout,
        terminalColumns: this.terminalColumns,
        terminalRows: this.terminalRows,
        children: /* @__PURE__ */ jsx15(TerminalWriteProvider, { value: this.writeRaw, children: node })
      }
    );
    reconciler_default.updateContainerSync(tree, this.container, null, noop2);
    reconciler_default.flushSyncWork();
  }
  unmount(error) {
    if (this.isUnmounted) {
      return;
    }
    this.onRender();
    this.unsubscribeExit();
    if (typeof this.restoreConsole === "function") {
      this.restoreConsole();
    }
    this.restoreStderr?.();
    this.unsubscribeTTYHandlers?.();
    const diff2 = this.log.renderPreviousOutput_DEPRECATED(this.frontFrame);
    writeDiffToTerminal(this.terminal, optimize(diff2));
    if (this.options.stdout.isTTY) {
      if (this.altScreenActive) {
        writeSync(1, EXIT_ALT_SCREEN);
      }
      writeSync(1, DISABLE_MOUSE_TRACKING);
      this.drainStdin();
      writeSync(1, DISABLE_MODIFY_OTHER_KEYS);
      writeSync(1, DISABLE_KITTY_KEYBOARD);
      writeSync(1, DFE);
      writeSync(1, DBP);
      writeSync(1, SHOW_CURSOR);
      writeSync(1, CLEAR_ITERM2_PROGRESS);
      if (supportsTabStatus()) {
        writeSync(1, wrapForMultiplexer(CLEAR_TAB_STATUS));
      }
    }
    this.isUnmounted = true;
    this.scheduleRender.cancel?.();
    if (this.drainTimer !== null) {
      clearTimeout(this.drainTimer);
      this.drainTimer = null;
    }
    reconciler_default.updateContainerSync(null, this.container, null, noop2);
    reconciler_default.flushSyncWork();
    instances_default.delete(this.options.stdout);
    this.rootNode.yogaNode?.free();
    this.rootNode.yogaNode = void 0;
    if (error instanceof Error) {
      this.rejectExitPromise(error);
    } else {
      this.resolveExitPromise();
    }
  }
  async waitUntilExit() {
    this.exitPromise ||= new Promise((resolve, reject) => {
      this.resolveExitPromise = resolve;
      this.rejectExitPromise = reject;
    });
    return this.exitPromise;
  }
  resetLineCount() {
    if (this.options.stdout.isTTY) {
      this.backFrame = this.frontFrame;
      this.frontFrame = emptyFrame(
        this.frontFrame.viewport.height,
        this.frontFrame.viewport.width,
        this.stylePool,
        this.charPool,
        this.hyperlinkPool
      );
      this.log.reset();
      this.displayCursor = null;
    }
  }
  /**
   * Replace char/hyperlink pools with fresh instances to prevent unbounded
   * growth during long sessions. Migrates the front frame's screen IDs into
   * the new pools so diffing remains correct. The back frame doesn't need
   * migration — resetScreen zeros it before any reads.
   *
   * Call between conversation turns or periodically.
   */
  resetPools() {
    this.charPool = new CharPool();
    this.hyperlinkPool = new HyperlinkPool();
    migrateScreenPools(this.frontFrame.screen, this.charPool, this.hyperlinkPool);
    this.backFrame.screen.charPool = this.charPool;
    this.backFrame.screen.hyperlinkPool = this.hyperlinkPool;
  }
  patchConsole() {
    const con = console;
    const originals = {};
    const toDebug = (...args) => logForDebugging(`console.log: ${format(...args)}`);
    const toError = (...args) => logError(new Error(`console.error: ${format(...args)}`));
    for (const m of CONSOLE_STDOUT_METHODS) {
      originals[m] = con[m];
      con[m] = toDebug;
    }
    for (const m of CONSOLE_STDERR_METHODS) {
      originals[m] = con[m];
      con[m] = toError;
    }
    originals.assert = con.assert;
    con.assert = (condition, ...args) => {
      if (!condition) {
        toError(...args);
      }
    };
    return () => Object.assign(con, originals);
  }
  /**
   * Intercept process.stderr.write so stray writes (config.ts, hooks.ts,
   * third-party deps) don't corrupt the alt-screen buffer. patchConsole only
   * hooks console.* methods — direct stderr writes bypass it, land at the
   * parked cursor, scroll the alt-screen, and desync frontFrame from the
   * physical terminal. Next diff writes only changed-in-React cells at
   * absolute coords → interleaved garbage.
   *
   * Swallows the write (routes text to the debug log) and, in alt-screen,
   * forces a full-damage repaint as a defensive recovery. Not patching
   * process.stdout — Ink itself writes there.
   */
  patchStderr() {
    const stderr = process.stderr;
    const originalWrite = stderr.write;
    let reentered = false;
    const intercept = (chunk, encodingOrCb, cb) => {
      const callback = typeof encodingOrCb === "function" ? encodingOrCb : cb;
      if (reentered) {
        const encoding = typeof encodingOrCb === "string" ? encodingOrCb : void 0;
        return originalWrite.call(stderr, chunk, encoding, callback);
      }
      reentered = true;
      try {
        const text = typeof chunk === "string" ? chunk : Buffer.from(chunk).toString("utf8");
        logForDebugging(`[stderr] ${text}`, {
          level: "warn"
        });
        if (this.altScreenActive && !this.isUnmounted && !this.isPaused) {
          this.prevFrameContaminated = true;
          this.scheduleRender();
        }
      } finally {
        reentered = false;
        callback?.();
      }
      return true;
    };
    stderr.write = intercept;
    return () => {
      if (stderr.write === intercept) {
        stderr.write = originalWrite;
      }
    };
  }
};
function drainStdin(stdin = process.stdin) {
  if (!stdin.isTTY) {
    return;
  }
  try {
    while (stdin.read() !== null) {
    }
  } catch {
  }
  if (process.platform === "win32") {
    return;
  }
  const tty = stdin;
  const wasRaw = tty.isRaw === true;
  let fd = -1;
  try {
    if (!wasRaw) {
      tty.setRawMode?.(true);
    }
    fd = openSync("/dev/tty", fsConstants.O_RDONLY | fsConstants.O_NONBLOCK);
    const buf = Buffer.alloc(1024);
    for (let i = 0; i < 64; i++) {
      if (readSync(fd, buf, 0, buf.length, null) <= 0) {
        break;
      }
    }
  } catch {
  } finally {
    if (fd >= 0) {
      try {
        closeSync(fd);
      } catch {
      }
    }
    if (!wasRaw) {
      try {
        tty.setRawMode?.(false);
      } catch {
      }
    }
  }
}
var CONSOLE_STDOUT_METHODS = [
  "log",
  "info",
  "debug",
  "dir",
  "dirxml",
  "count",
  "countReset",
  "group",
  "groupCollapsed",
  "groupEnd",
  "table",
  "time",
  "timeEnd",
  "timeLog"
];
var CONSOLE_STDERR_METHODS = ["warn", "error", "trace"];

// src/ink/root.ts
var renderSync = (node, options) => {
  const opts = getOptions(options);
  const inkOptions = {
    stdout: process.stdout,
    stdin: process.stdin,
    stderr: process.stderr,
    exitOnCtrlC: true,
    patchConsole: true,
    ...opts
  };
  const instance = getInstance(inkOptions.stdout, () => new Ink(inkOptions));
  instance.render(node);
  return {
    rerender: instance.render,
    unmount() {
      instance.unmount();
    },
    waitUntilExit: instance.waitUntilExit,
    cleanup: () => instances_default.delete(inkOptions.stdout)
  };
};
var wrappedRender = async (node, options) => {
  await Promise.resolve();
  const instance = renderSync(node, options);
  logForDebugging(`[render] first ink render: ${Math.round(process.uptime() * 1e3)}ms since process start`);
  return instance;
};
var root_default = wrappedRender;
async function createRoot({
  stdout = process.stdout,
  stdin = process.stdin,
  stderr = process.stderr,
  exitOnCtrlC = true,
  patchConsole = true,
  onFrame
} = {}) {
  await Promise.resolve();
  const instance = new Ink({
    stdout,
    stdin,
    stderr,
    exitOnCtrlC,
    patchConsole,
    onFrame
  });
  instances_default.set(stdout, instance);
  return {
    render: (node) => instance.render(node),
    unmount: () => instance.unmount(),
    waitUntilExit: () => instance.waitUntilExit()
  };
}
var getOptions = (stdout = {}) => {
  if (stdout instanceof Stream) {
    return {
      stdout,
      stdin: process.stdin
    };
  }
  return stdout;
};
var getInstance = (stdout, createInstance) => {
  let instance = instances_default.get(stdout);
  if (!instance) {
    instance = createInstance();
    instances_default.set(stdout, instance);
  }
  return instance;
};

// src/entry-exports.ts
import { default as default2, UncontrolledTextInput } from "ink-text-input";
export {
  AlternateScreen,
  Ansi,
  Box_default as Box,
  Link,
  Newline,
  NoSelect,
  RawAnsi,
  ScrollBox_default as ScrollBox,
  Spacer,
  Text,
  default2 as TextInput,
  UncontrolledTextInput,
  createRoot,
  measure_element_default as measureElement,
  root_default as render,
  renderSync,
  stringWidth,
  use_app_default as useApp,
  useDeclaredCursor,
  useExternalProcess,
  useHasSelection,
  use_input_default as useInput,
  useSelection,
  useStderr,
  use_stdin_default as useStdin,
  useStdout,
  useTabStatus,
  useTerminalFocus,
  useTerminalTitle,
  useTerminalViewport,
  withInkSuspended
};
