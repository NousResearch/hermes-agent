import { createContext } from "react";
import { profileVisibleName } from "@hermes/shared";
import type { ProfileInfo } from "@/lib/api";

export interface ProfileContextValue {
  /** Profile every management surface reads/writes ("" = the dashboard
   *  process's own profile). */
  profile: string;
  /** The profile the dashboard process itself runs under. */
  currentProfile: string;
  /** Known profile names (includes "default"). */
  profiles: string[];
  /** Metadata stays keyed by the canonical name, never by its visible label. */
  profileDetails: ProfileInfo[];
  profileLabel: (name: string) => string;
  setProfile: (name: string) => void;
}

export const ProfileContext = createContext<ProfileContextValue>({
  profile: "",
  currentProfile: "default",
  profiles: [],
  profileDetails: [],
  profileLabel: (name) => profileVisibleName({ name }),
  setProfile: () => {},
});
