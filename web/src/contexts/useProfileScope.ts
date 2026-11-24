import { useContext } from "react";
import { ProfileContext } from "@/contexts/profile-context";

export function useProfileScope() {
  return useContext(ProfileContext);
}
