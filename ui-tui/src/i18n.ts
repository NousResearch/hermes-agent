/** TUI frontend i18n — Chinese translation layer with caching. */
function isZh(): boolean {
  try {
    const env = typeof process !== "undefined" ? process.env : {};
    if (env.HERMES_LANG?.startsWith("zh")) return true;
    if (env.LANG?.startsWith("zh")) return true;
    if (env.LANGUAGE?.startsWith("zh")) return true;
  } catch {}
  try {
    const cfg = require("fs").readFileSync(
      require("os").homedir() + "/.hermes/config.yaml", "utf-8"
    );
    if (cfg.includes("language: zh")) return true;
  } catch {}
  return false;
}

const ZH: Record<string, string> = {
  "Available Tools": "可用工具",
  "Available Skills": "可用技能",
  "MCP Servers": "MCP 服务器",
  "connected": "已连接",
  "failed": "失败",
  "Session: ": "会话：",
  "scanning skills": "正在扫描技能",
  "/help for commands": "/help 查看命令",
  "commit": "个提交",
  "commits": "个提交",
  "behind": "落后",
  "to update": "以更新",
  "session": "会话",
  "sessions": "会话",
  "commits behind": "个提交落后",
  "No system prompt loaded.": "暂无系统提示词。",
  "System Prompt": "系统提示词",
  "ready": "就绪",
  "running": "正在执行",
  "running...": "正在执行…",
  "starting agent...": "启动 agent…",
  "- run": "运行",
  "voice off": "语音关闭",
  "Ctrl+C to interrupt…": "按 Ctrl+C 中断…",
};

export function tr(key: string): string {
  if (isZh()) {
    return ZH[key] ?? key;
  }
  return key;
}
