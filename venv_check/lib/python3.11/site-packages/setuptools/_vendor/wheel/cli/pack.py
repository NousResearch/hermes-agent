from __future__ import annotations

import email.policy
import os.path
import re
from email.generator import BytesGenerator
from email.parser import BytesParser

from wheel.cli import WheelError
from wheel.wheelfile import WheelFile

DIST_INFO_RE = re.compile(r"^(?P<namever>(?P<name>.+?)-(?P<ver>\d.*?))\.dist-info$")


def pack(directory: str, dest_dir: str, build_number: str | None) -> None:
    """Repack a previously unpacked wheel directory into a new wheel file.

    The .dist-info/WHEEL file must contain one or more tags so that the target
    wheel file name can be determined.

    :param directory: The unpacked wheel directory
    :param dest_dir: Destination directory (defaults to the current directory)
    """
    # Find the .dist-info directory
    dist_info_dirs = [
        fn
        for fn in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, fn)) and DIST_INFO_RE.match(fn)
    ]
    if len(dist_info_dirs) > 1:
        raise WheelError(f"Multiple .dist-info directories found in {directory}")
    elif not dist_info_dirs:
        raise WheelError(f"No .dist-info directories found in {directory}")

    # Determine the target wheel filename
    dist_info_dir = dist_info_dirs[0]
    name_version = DIST_INFO_RE.match(dist_info_dir).group("namever")

    # Read the tags and the existing build number from .dist-info/WHEEL
    wheel_file_path = os.path.join(directory, dist_info_dir, "WHEEL")
    with open(wheel_file_path, "rb") as f:
        info = BytesParser(policy=email.policy.compat32).parse(f)
        tags: list[str] = info.get_all("Tag", [])
        existing_build_number = info.get("Build")

        if not tags:
            raise WheelError(
                f"No tags present in {dist_info_dir}/WHEEL; cannot determine target "
                f"wheel filename"
            )

    # Set the wheel file name and add/replace/remove the Build tag in .dist-info/WHEEL
    build_number = build_number if build_number is not None else existing_build_number
    if build_number is not None:
        del info["Build"]
        if build_number:
            info["Build"] = build_number
            name_version += "-" + build_number

        if build_number != existing_build_number:
            with open(wheel_file_path, "wb") as f:
                BytesGenerator(f, maxheaderlen=0).flatten(info)

    # Reassemble the tags for the wheel file
    tagline = compute_tagline(tags)

    # Repack the wheel
    wheel_path = os.path.join(dest_dir, f"{name_version}-{tagline}.whl")
    with WheelFile(wheel_path, "w") as wf:
        print(f"Repacking wheel as {wheel_path}...", end="", flush=True)
        wf.write_files(directory)

    print("OK")


def compute_tagline(tags: list[str]) -> str:
    """Compute a tagline from a list of tags.

    :param tags: A list of tags
    :return: A tagline
    """
    impls = sorted({tag.split("-")[0] for tag in tags})
    abivers = sorted({tag.split("-")[1] for tag in tags})
    platforms = sorted({tag.split("-")[2] for tag in tags})
    return "-".join([".".join(impls), ".".join(abivers), ".".join(platforms)])
