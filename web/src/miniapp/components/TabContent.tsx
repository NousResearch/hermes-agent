import type { MiniAppTab } from "../context";
import type { DetailView, MiniAppStatusExtra } from "../types";
import { StatusScreen } from "../screens/StatusScreen";
import { SkillsScreen } from "../screens/SkillsScreen";
import { SkillDetailPane } from "../screens/SkillDetailPane";
import { CronScreen } from "../screens/CronScreen";
import { SessionsScreen } from "../screens/SessionsScreen";
import { SessionDetailPane } from "../screens/SessionDetailPane";
import { SessionNotFoundPane } from "../screens/SessionNotFoundPane";
import { UsersScreen } from "../screens/UsersScreen";
import { LogDetailPane } from "../screens/LogDetailPane";

export interface TabContentProps {
  tab: MiniAppTab;
  detail: DetailView;
  isAdmin: boolean;
  statusExtra: MiniAppStatusExtra;
  onOpenLog: (key: string, label: string) => void;
  onOpenSkill: (name: string) => void;
  onOpenSession: (id: string) => void;
  onSessionNotFound: () => void;
  onBack: () => void;
  onShowLog: (title: string, text: string) => void;
}

/** The active tab's screen, or the detail pane open on top of it. Admin-only tabs render nothing otherwise. */
export function TabContent(props: TabContentProps) {
  const { tab, detail, isAdmin } = props;
  switch (tab) {
    case "status":
      if (detail.kind === "log") return <LogDetailPane fileKey={detail.key} />;
      return detail.kind === "none" ? <StatusScreen statusExtra={props.statusExtra} onOpenLog={props.onOpenLog} /> : null;
    case "skills":
      if (detail.kind === "skill") return <SkillDetailPane name={detail.name} />;
      return detail.kind === "none" ? <SkillsScreen onOpen={props.onOpenSkill} /> : null;
    case "cron":
      return isAdmin ? <CronScreen onShowLog={props.onShowLog} /> : null;
    case "sessions":
      return <SessionsContent {...props} />;
    case "users":
      return isAdmin ? <UsersScreen statusExtra={props.statusExtra} /> : null;
    default:
      return null;
  }
}

function SessionsContent({ detail, onOpenSession, onSessionNotFound, onBack }: TabContentProps) {
  if (detail.kind === "session") {
    return <SessionDetailPane key={detail.id} id={detail.id} onNotFound={onSessionNotFound} onBack={onBack} />;
  }
  if (detail.kind === "session-not-found") return <SessionNotFoundPane onBack={onBack} />;
  return detail.kind === "none" ? <SessionsScreen onOpen={onOpenSession} /> : null;
}
