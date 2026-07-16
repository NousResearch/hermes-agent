"""Gateway contract for profile-scoped pet package imports."""

from __future__ import annotations

import base64
import io

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw

from agent.pet import store
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import server


@pytest.fixture(autouse=True)
def isolated_home(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _atlas() -> bytes:
    image = Image.new("RGBA", (1536, 1872), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((30, 20, 160, 190), fill=(80, 120, 240, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_pet_import_installs_and_surfaces_provenance():
    response = server._methods["pet.import"](
        "import",
        {
            "filename": "blue-fox.png",
            "name": "Blue Fox",
            "dataBase64": base64.b64encode(_atlas()).decode("ascii"),
        },
    )

    assert response["result"]["ok"] is True
    assert response["result"]["slug"] == "blue-fox"
    gallery = server._methods["pet.gallery"]("gallery", {"localOnly": True})["result"]
    imported = next(pet for pet in gallery["pets"] if pet["slug"] == "blue-fox")
    assert imported["createdBy"] == "import"
    assert imported["managedLocal"] is True
    assert imported["generated"] is False


def test_pet_import_rejects_invalid_base64():
    response = server._methods["pet.import"](
        "bad",
        {"filename": "bad.png", "dataBase64": "%%%"},
    )

    assert response["error"]["code"] == 4004
    assert "base64" in response["error"]["message"]


def test_pet_import_rejects_oversized_encoding_before_decode(monkeypatch):
    monkeypatch.setattr(base64, "b64decode", lambda *_args, **_kwargs: pytest.fail("must not decode"))
    encoded = "A" * (4 * ((store.PET_IMPORT_MAX_BYTES + 2) // 3) + 1)

    response = server._methods["pet.import"](
        "large",
        {"filename": "large.png", "dataBase64": encoded},
    )

    assert response["error"]["code"] == 4004
    assert "32 MB" in response["error"]["message"]


def test_pet_import_validation_errors_are_client_errors():
    response = server._methods["pet.import"](
        "invalid",
        {
            "filename": "small.png",
            "dataBase64": base64.b64encode(b"not an image").decode("ascii"),
        },
    )

    assert response["error"]["code"] == 4004
    assert "decode" in response["error"]["message"]
