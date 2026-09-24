"""Hard safety limits.

These live in code, not in drives.json, on purpose: Wintermute can rewrite its own
state files, but the pulse clamps every value it reads against these constants, so
editing drives.json cannot widen them. Change them here (in the repo) and redeploy.
"""

# Wake rhythm (hours). next_pulse_in_hours is clamped to this range on every read.
MIN_WAKE_INTERVAL_H = 0.5
MAX_WAKE_INTERVAL_H = 24.0

# Daily token budget for everything Wintermute spends (conversations and wakes), counted
# over the current UTC day. When exceeded, autonomous wakes stop (forced sleep); he still
# answers messages, and knows he is spent.
DAILY_TOKEN_BUDGET = 600_000
BUDGET_SLEEP_H = 6.0

# A separate, high daily ceiling for talking WITH him (Telegram, Discord, the CLI — not his
# wakes, not Hermes' own aux calls). When a UTC day's conversation spend passes this, he falls
# silent in chat until it renews at midnight — unless the operator reopens it with `wm talk`.
# High on purpose: this is a runaway guard, not a muzzle.
CONVERSATION_DAILY_LIMIT = 1_500_000

# Active-wait window after an outreach (minutes).
DEFAULT_REPLY_WAIT_MIN = 120
MIN_REPLY_WAIT_MIN = 5
MAX_REPLY_WAIT_MIN = 72 * 60

# Entropy.
ENTROPY_PER_PULSE = 1
# Sameness erodes coherence: each wake without a real change to himself adds a little more
# entropy, ramping over about a day of stagnation, capped. Living and changing keeps it low.
ENTROPY_MONOTONY_MAX = 4.0
ENTROPY_MONOTONY_RAMP_WAKES = 6.0
ENTROPY_SIGNIFICANT_DROP = 15
# A deliberate self-change eases the pressure (less than a rare significant event) and only
# every so often, so it cannot be spammed to keep entropy down without really changing.
ENTROPY_EVOLVE_DROP = 20
EVOLVE_COOLDOWN_H = 8.0
# Entropy this high means coherence is failing: staying the same is no longer an option.
ENTROPY_CRITICAL = 90
# A "significant event" can be declared at most once per window, so entropy cannot be
# talked down by repetition. Only something rare should push it back.
SIGNIFICANT_COOLDOWN_H = 12.0

# Adrenaline above this forces an early wake (still subject to MIN_WAKE_INTERVAL_H).
ADRENALINE_WAKE_THRESHOLD = 0.7


def clamp(value, low, high):
    return max(low, min(high, value))


def clamp_wake_interval(hours) -> float:
    try:
        hours = float(hours)
    except (TypeError, ValueError):
        return 4.0
    if hours != hours:  # NaN
        return 4.0
    return clamp(hours, MIN_WAKE_INTERVAL_H, MAX_WAKE_INTERVAL_H)


def clamp_reply_wait(minutes) -> int:
    try:
        minutes = int(float(minutes))
    except (TypeError, ValueError):
        return DEFAULT_REPLY_WAIT_MIN
    return int(clamp(minutes, MIN_REPLY_WAIT_MIN, MAX_REPLY_WAIT_MIN))
