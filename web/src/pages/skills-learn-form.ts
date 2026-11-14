import { useCallback, useState } from "react";
import { useNavigate } from "react-router";

/** "Learn a skill" panel state. Open-ended: dir + URL + free-text inputs are composed into a
 *  single-line /learn command and handed to the chat. /learn resolves to a normal agent turn
 *  (command.dispatch → send), so the live agent gathers the sources with its own tools and
 *  authors the skill via skill_manage. No backend distill endpoint — one code path with the
 *  CLI/TUI/gateway /learn. */
export function useLearnSkillForm() {
  const navigate = useNavigate();
  const [learnOpen, setLearnOpen] = useState(false);
  const [learnDir, setLearnDir] = useState("");
  const [learnUrl, setLearnUrl] = useState("");
  const [learnText, setLearnText] = useState("");
  const openLearn = useCallback(() => {
    setLearnDir("");
    setLearnUrl("");
    setLearnText("");
    setLearnOpen(true);
  }, []);
  const submitLearn = useCallback(() => {
    const segs: string[] = [];
    const dir = learnDir.trim();
    const url = learnUrl.trim();
    const text = learnText.trim();
    if (dir) segs.push(`local source: ${dir}`);
    if (url) segs.push(`URL: ${url}`);
    if (text) segs.push(text);
    // Flatten to a single line — the chat composer submits on the first Enter.
    const composed = segs.join("; ").replace(/\s*\n\s*/g, " ").trim();
    if (!composed) return;
    setLearnOpen(false);
    navigate(`/chat?learn=${encodeURIComponent(composed)}`);
  }, [learnDir, learnUrl, learnText, navigate]);

  return {
    learnOpen, setLearnOpen, learnDir, setLearnDir, learnUrl, setLearnUrl, learnText, setLearnText,
    openLearn, submitLearn,
  };
}
