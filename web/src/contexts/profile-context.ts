import { createContext } from "react";

export interface ProfileContextValue {
  /** Profile every management surface reads/writes ("" = the dashboard
   *  process's own profile). */
  profile: string;
  /** The profile the dashboard process itself runs under. */
  currentProfile: string;
  /** Known profile names (includes "default"). */
  profiles: string[];
  /** Display label per profile id — `display_name (id)` when a display name
   *  is set, else the bare id (mirrors `hermes profile list`). */
  profileLabels: Record<string, string>;
  setProfile: (name: string) => void;
}

export const ProfileContext = createContext<ProfileContextValue>({
  profile: "",
  currentProfile: "default",
  profiles: [],
  profileLabels: {},
  setProfile: () => {},
});
