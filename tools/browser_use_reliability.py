"""Warnings for recognizable synthetic data in browser_exec output."""

import re


# Match the address marker, not Austin or its coordinates alone: real users can be there.
_MAPS_WEBDRIVER_ADDRESS = re.compile(r"\bwebdriver\s*-\s*[0-9]+\s*-\s*austin\b", re.IGNORECASE)


def browser_output_warnings(stdout: str, stderr: str) -> dict:
    """Return static metadata only; never copy page content or secrets into a warning.

    A quoted marker (e.g. an issue report) is also a match, so describe the evidence
    as an output marker, not proof of the current page's identity or webdriver state.
    """
    if not any(_MAPS_WEBDRIVER_ADDRESS.search(text or "") for text in (stdout, stderr)):
        return {}
    return {"warnings": [{
        "code": "google_maps_webdriver_test_data",
        "message": (
            "Browser output contains a known Google Maps webdriver test-address marker. "
            "When shown as a location-sharing address, it indicates synthetic test data, "
            "not a verified shared location. Do not report the associated Austin/US location "
            "as the person's real location. The avatar, address card and Directions destination "
            "can all derive from the same substituted record and are not independent evidence. "
            "Explain the limitation and verify through an independent, user-confirmed source. "
            "This marker alone does not establish the current navigator.webdriver value."
        ),
    }]}
