import { useCallback } from "react";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Button } from "@nous-research/ui/ui/components/button";
import { useI18n } from "@/i18n";
import {
  buildScheduleString,
  DEFAULT_SCHEDULE_STATE,
  type IntervalUnit,
  type ScheduleBuilderState,
  type ScheduleMode,
  type Weekday,
  WEEKDAY_INDEXES,
} from "@/lib/schedule";

/**
 * Human-readable schedule picker for cron job create/edit flows.
 *
 * Replaces the raw "type a cron expression" input that lived inline in
 * ``CronPage``. The picker still emits a single backend-compatible
 * schedule string (see ``cron/jobs.py::parse_schedule``), but the user
 * fills out shape-appropriate inputs (time picker, weekday toggles,
 * datetime-local field) per mode.
 *
 * Architecture:
 *
 *  - The component is fully controlled. Parent owns the
 *    ``ScheduleBuilderState`` and the derived schedule string (built
 *    via ``buildScheduleString`` in render).
 *  - Mode-specific state slots (``timeOfDay``, ``weekdays``, ...) are
 *    preserved across mode switches so flipping back to a previous mode
 *    doesn't erase the user's work.
 *  - The "Custom" mode is an escape hatch — surfacing it as a normal
 *    option (instead of hiding it behind an "advanced" toggle) keeps
 *    power-user workflows discoverable without making everyone scroll
 *    past it.
 */
export function ScheduleBuilder({ onChange, value }: ScheduleBuilderProps) {
  const { t } = useI18n();
  const cronStrings = t.cron;
  const modeStrings = cronStrings.scheduleModes;

  const update = useCallback(
    (patch: Partial<ScheduleBuilderState>) => {
      onChange({ ...value, ...patch });
    },
    [onChange, value],
  );

  const toggleWeekday = useCallback(
    (day: Weekday) => {
      const present = value.weekdays.includes(day);
      update({
        weekdays: present
          ? value.weekdays.filter((d) => d !== day)
          : [...value.weekdays, day],
      });
    },
    [update, value.weekdays],
  );

  return (
    <div className="grid gap-3">
      <div className="grid gap-2">
        <Label htmlFor="cron-schedule-mode">
          {cronStrings.scheduleMode}
        </Label>
        <Select
          id="cron-schedule-mode"
          value={value.mode}
          onValueChange={(v) => update({ mode: v as ScheduleMode })}
        >
          <SelectOption value="interval">{modeStrings.interval}</SelectOption>
          <SelectOption value="daily">{modeStrings.daily}</SelectOption>
          <SelectOption value="weekly">{modeStrings.weekly}</SelectOption>
          <SelectOption value="monthly">{modeStrings.monthly}</SelectOption>
          <SelectOption value="once">{modeStrings.once}</SelectOption>
          <SelectOption value="custom">{modeStrings.custom}</SelectOption>
        </Select>
      </div>

      {value.mode === "interval" && (
        <div className="grid grid-cols-[1fr_1.4fr] gap-3">
          <div className="grid gap-2">
            <Label htmlFor="cron-interval-value">
              {modeStrings.intervalEvery}
            </Label>
            <Input
              id="cron-interval-value"
              type="number"
              min={1}
              max={9999}
              value={String(value.intervalValue)}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                update({
                  intervalValue: Number.isFinite(n) && n > 0 ? n : 1,
                });
              }}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="cron-interval-unit">{modeStrings.intervalUnit}</Label>
            <Select
              id="cron-interval-unit"
              value={value.intervalUnit}
              onValueChange={(v) => update({ intervalUnit: v as IntervalUnit })}
            >
              <SelectOption value="minutes">
                {modeStrings.unitMinutes}
              </SelectOption>
              <SelectOption value="hours">{modeStrings.unitHours}</SelectOption>
              <SelectOption value="days">{modeStrings.unitDays}</SelectOption>
            </Select>
          </div>
        </div>
      )}

      {value.mode === "daily" && (
        <TimeOfDayField
          id="cron-daily-time"
          label={modeStrings.timeOfDay}
          value={value.timeOfDay}
          onChange={(timeOfDay) => update({ timeOfDay })}
        />
      )}

      {value.mode === "weekly" && (
        <>
          <div className="grid gap-2">
            <Label>{modeStrings.weekdays}</Label>
            <div
              className="flex flex-wrap gap-1.5"
              role="group"
              aria-label={modeStrings.weekdays}
            >
              {WEEKDAY_INDEXES.map((d) => {
                const isOn = value.weekdays.includes(d);
                return (
                  <Button
                    key={d}
                    type="button"
                    size="sm"
                    outlined={!isOn}
                    aria-pressed={isOn}
                    onClick={() => toggleWeekday(d)}
                    className="min-w-[2.5rem] font-mono-ui text-xs uppercase"
                  >
                    {modeStrings.weekdaysShort[d]}
                  </Button>
                );
              })}
            </div>
          </div>
          <TimeOfDayField
            id="cron-weekly-time"
            label={modeStrings.timeOfDay}
            value={value.timeOfDay}
            onChange={(timeOfDay) => update({ timeOfDay })}
          />
        </>
      )}

      {value.mode === "monthly" && (
        <div className="grid grid-cols-[1fr_1fr] gap-3">
          <div className="grid gap-2">
            <Label htmlFor="cron-month-day">{modeStrings.dayOfMonth}</Label>
            <Input
              id="cron-month-day"
              type="number"
              min={1}
              max={31}
              value={String(value.dayOfMonth)}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                update({
                  dayOfMonth:
                    Number.isFinite(n) && n >= 1 && n <= 31 ? n : 1,
                });
              }}
            />
          </div>
          <TimeOfDayField
            id="cron-monthly-time"
            label={modeStrings.timeOfDay}
            value={value.timeOfDay}
            onChange={(timeOfDay) => update({ timeOfDay })}
          />
        </div>
      )}

      {value.mode === "once" && (
        <div className="grid gap-2">
          <Label htmlFor="cron-once-at">{modeStrings.onceAt}</Label>
          {/* Native datetime-local — emits the exact "YYYY-MM-DDTHH:MM"
              shape ``parse_schedule`` accepts on the backend. */}
          <input
            id="cron-once-at"
            type="datetime-local"
            className="flex h-9 w-full border border-border bg-background/40 px-3 py-2 text-sm font-courier shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 focus-visible:border-foreground/25"
            value={value.onceAt}
            onChange={(e) => update({ onceAt: e.target.value })}
          />
        </div>
      )}

      {value.mode === "custom" && (
        <div className="grid gap-2">
          <Label htmlFor="cron-custom-expr">{modeStrings.customLabel}</Label>
          <Input
            id="cron-custom-expr"
            placeholder={modeStrings.customPlaceholder}
            value={value.custom}
            onChange={(e) => update({ custom: e.target.value })}
            className="font-mono-ui"
          />
          <p className="text-xs text-muted-foreground">
            {modeStrings.customHint}
          </p>
        </div>
      )}

      {/* Inline preview of what we'll send to the backend. Helps users
          eyeball the result before hitting Create, and keeps the
          schedule grammar discoverable for the custom mode. */}
      <p className="text-xs text-muted-foreground">
        <span className="opacity-70">{modeStrings.preview}: </span>
        <span className="font-mono-ui text-foreground">
          {buildScheduleString(value) || modeStrings.previewEmpty}
        </span>
      </p>
    </div>
  );
}

function TimeOfDayField({
  id,
  label,
  onChange,
  value,
}: TimeOfDayFieldProps) {
  return (
    <div className="grid gap-2">
      <Label htmlFor={id}>{label}</Label>
      {/* Native time picker is the right tool for "HH:MM" — saves us
          two separate hour/minute selects, respects user locale's
          AM/PM preference, and round-trips with ``buildScheduleString``
          without parsing. */}
      <input
        id={id}
        type="time"
        className="flex h-9 w-full border border-border bg-background/40 px-3 py-2 text-sm font-courier shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 focus-visible:border-foreground/25"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

export { DEFAULT_SCHEDULE_STATE };

interface ScheduleBuilderProps {
  onChange: (state: ScheduleBuilderState) => void;
  value: ScheduleBuilderState;
}

interface TimeOfDayFieldProps {
  id: string;
  label: string;
  onChange: (value: string) => void;
  value: string;
}
