"""Command catalog category contracts for the desktop slash palette."""

from tui_gateway import server


def test_plan_and_goal_are_listed_under_tools_and_skills():
    response = server.handle_request(
        {"id": "plan-goal-categories", "method": "commands.catalog", "params": {}}
    )
    categories = {
        section["name"]: dict(section["pairs"])
        for section in response["result"]["categories"]
    }

    for command in ("/plan", "/goal"):
        assert command in categories["Tools & Skills"]
        assert command not in categories["Session"]
