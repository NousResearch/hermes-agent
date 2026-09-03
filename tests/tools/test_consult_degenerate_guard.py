"""Guard against degenerate consult answers (leaked markup / request echo).

Regression case: a local/open-weight aux model answered a consult with the
consult request itself wrapped in raw tool-call template markup; the main
agent then presented a fabricated "reference model's take" to the user. The
tool must classify such answers as unavailable, never as a usable answer.
"""

from tools.consult_tool import _degenerate_answer_reason


QUESTION = "Should unoccupied rooms above setpoint get 100% airflow?"
CONTEXT = (
    "## Logic change request\n"
    "Unoccupied rooms (like the Game Room at 75.92F) are currently getting "
    "100% airflow because they are far above the setpoint, and the Priority "
    "Pass skips them for beneficiaries, but it doesn't throttle them. Every "
    "watt of cooling wasted on an empty room is CFM stolen from the occupied "
    "Living Room. Fix the vent opening calculation: when a room is "
    "UNOCCUPIED cap its base tendency to open at 60-70% regardless of how "
    "hot it is, or multiply its effective temperature deficit by a penalty."
)


def test_template_wrapped_echo_is_degenerate():
    # The observed failure shape: chat-template markup wrapping the request
    # text (the sentinel is the fullwidth vertical bar U+FF5C, as used in
    # some open-weight models' raw tool-call templates).
    answer = (
        "\uff5ctool_calls>\n\uff5cinvoke name=\"consult\">\n"
        "\uff5cparameter name=\"context\" string=\"true\">" + CONTEXT
        + "\n</tool_code_code_logic>"
    )
    assert _degenerate_answer_reason(answer, QUESTION, CONTEXT) is not None


def test_plain_echo_is_degenerate():
    assert _degenerate_answer_reason(CONTEXT, QUESTION, CONTEXT) is not None


def test_leading_control_token_is_degenerate():
    assert (
        _degenerate_answer_reason(
            "<|im_start|>assistant here is my view", QUESTION, CONTEXT
        )
        is not None
    )


def test_real_answer_passes():
    answer = (
        "The plan is mostly sound but capping unoccupied rooms at a fixed "
        "60-70% is fragile with only 0/50/100 vent positions available -- "
        "use 50% as the cap and add a hysteresis band so occupancy flapping "
        "(e.g. someone sleeping) doesn't oscillate the vents. Also make the "
        "brake conditional on whole-system delivery handicap, not per-room "
        "temperature alone."
    )
    assert _degenerate_answer_reason(answer, QUESTION, CONTEXT) is None


def test_short_answer_never_flagged_as_echo():
    # Short answers legitimately reuse the question's own words.
    answer = "Yes -- cap unoccupied rooms at 50%, it is the right call."
    assert _degenerate_answer_reason(answer, QUESTION, CONTEXT) is None


def test_answer_quoting_a_template_snippet_without_structure_passes():
    # Mentioning the sentinel alone (e.g. discussing a parser bug) is fine
    # as long as no tool-call structure is present and it is not the prefix.
    answer = (
        "Your parser fails because the model emitted a stray \uff5c "
        "sentinel mid-string; strip it before json parsing. Otherwise the "
        "approach is fine and the retry loop is correct as written."
    )
    assert _degenerate_answer_reason(answer, QUESTION, CONTEXT) is None
