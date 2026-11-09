import { useEffect, useState } from "react";

export function isPickerSearch(search: string): boolean {
  return new URLSearchParams(search).get("embed") === "picker";
}

/** True only for the explicit marketplace iframe contract, ?embed=picker. */
export default function usePickerMode(): boolean {
  const [picker, setPicker] = useState(
    () => typeof window !== "undefined" && isPickerSearch(window.location.search),
  );
  useEffect(() => {
    setPicker(isPickerSearch(window.location.search));
  }, []);
  return picker;
}
