import { Check, Search } from "lucide-react";
import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { Input } from "@nous-research/ui/ui/components/input";
import { cn } from "@/lib/utils";
import {
  filterProfileModelChoices,
  profileModelKey,
  type ProfileModelChoice,
} from "@/lib/profile-model-picker";

interface ProfileModelPickerProps {
  choices: ProfileModelChoice[] | null;
  emptyLabel: string;
  inheritLabel?: string;
  loadingLabel: string;
  onSelect(value: string): void;
  searchInputId?: string;
  selected: string;
}

export function ProfileModelPicker({
  choices,
  emptyLabel,
  inheritLabel,
  loadingLabel,
  onSelect,
  searchInputId,
  selected,
}: ProfileModelPickerProps) {
  const [query, setQuery] = useState("");
  const listboxId = useId();
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const selectedOptionRef = useRef<HTMLButtonElement | null>(null);
  const filteredChoices = useMemo(
    () => filterProfileModelChoices(choices ?? [], query),
    [choices, query],
  );
  const modelOptionOffset = inheritLabel && choices !== null ? 1 : 0;
  const optionCount = modelOptionOffset + filteredChoices.length;

  useEffect(() => {
    selectedOptionRef.current?.scrollIntoView({ block: "nearest" });
  }, [choices, selected]);

  const focusOption = (index: number) => {
    optionRefs.current[index]?.focus();
  };

  const handleOptionKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      focusOption(Math.min(index + 1, optionCount - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      focusOption(Math.max(index - 1, 0));
    }
  };

  const rowClass = (active: boolean) =>
    cn(
      "flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-mono transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring",
      active
        ? "bg-primary/10 text-primary"
        : "text-foreground hover:bg-muted/50",
    );

  return (
    <div className="grid gap-2">
      <div className="relative">
        <Search
          aria-hidden
          className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          id={searchInputId}
          aria-label="Search models"
          aria-controls={listboxId}
          className="h-8 pl-8 text-sm"
          disabled={choices === null}
          placeholder="Filter models…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown" && optionCount > 0) {
              event.preventDefault();
              focusOption(
                filteredChoices.length > 0 ? modelOptionOffset : 0,
              );
            } else if (event.key === "Enter" && filteredChoices.length > 0) {
              event.preventDefault();
              onSelect(profileModelKey(filteredChoices[0]));
            }
          }}
        />
      </div>

      <div
        id={listboxId}
        role="listbox"
        aria-label="Models"
        className="max-h-80 overflow-y-auto rounded border border-border"
      >
        {inheritLabel && choices !== null && (
          <button
            type="button"
            role="option"
            aria-selected={selected === ""}
            ref={(element) => {
              optionRefs.current[0] = element;
              if (selected === "") selectedOptionRef.current = element;
            }}
            className={rowClass(selected === "")}
            onClick={() => onSelect("")}
            onKeyDown={(event) => handleOptionKeyDown(event, 0)}
          >
            <Check
              aria-hidden
              className={cn(
                "h-3 w-3 shrink-0",
                selected === "" ? "text-primary" : "text-transparent",
              )}
            />
            <span className="truncate">{inheritLabel}</span>
          </button>
        )}

        {choices === null ? (
          <p className="p-3 text-xs text-muted-foreground">{loadingLabel}</p>
        ) : choices.length === 0 ? (
          <p className="p-3 text-xs text-muted-foreground">{emptyLabel}</p>
        ) : filteredChoices.length === 0 ? (
          <p className="p-3 text-xs text-muted-foreground">
            No models match your search.
          </p>
        ) : (
          filteredChoices.map((choice, choiceIndex) => {
            const key = profileModelKey(choice);
            const active = selected === key;
            const optionIndex = choiceIndex + modelOptionOffset;
            return (
              <button
                key={key}
                type="button"
                role="option"
                aria-selected={active}
                ref={(element) => {
                  optionRefs.current[optionIndex] = element;
                  if (active) selectedOptionRef.current = element;
                }}
                className={rowClass(active)}
                onClick={() => onSelect(key)}
                onKeyDown={(event) =>
                  handleOptionKeyDown(event, optionIndex)
                }
              >
                <Check
                  aria-hidden
                  className={cn(
                    "h-3 w-3 shrink-0",
                    active ? "text-primary" : "text-transparent",
                  )}
                />
                <span className="truncate">{choice.label}</span>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}
