"""Regression invariants for the himalaya skill docs.

The himalaya skill is consumed by agents loading ``SKILL.md`` + the two
``references/*.md`` files plus the website mirror. Each review round on
PR #75212 caught a v2/source-contract defect that survived because no
automated check existed for the underlying claim. These tests pin the
contracts the agents rely on so future edits can't silently regress
them.

Add a test here whenever a review comment identifies a class of bug
("this example would silently break") — the test should fail on the
buggy form and pass on the corrected form, so the next PR that
re-introduces it trips the same gate this PR cleared.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills" / "email" / "himalaya"


def _read(name: str) -> str:
    path = SKILLS_DIR / name
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Round-3 invariants — `--json` and `--page-size` forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    ["SKILL.md", "references/configuration.md"],
)
def test_json_flag_documented_as_global_not_must_precede(filename: str) -> None:
    """`--json` must NOT be documented as "must come before the subcommand".

    Reviewer @Enough1122 caught the contradiction where SKILL.md said
    `--json` is a global flag but ``references/configuration.md`` said
    it "must come *before* the subcommand". v2.0.0 parses both pre- and
    post-subcommand placement; docs must reflect that.
    """
    text = _read(filename)
    assert "must come *before* the subcommand" not in text, (
        f"{filename} still says `--json` must come before the subcommand. "
        "v2.0.0 accepts both placements; update the doc."
    )


def test_no_cat_message_eml_into_compose_in_examples() -> None:
    """Piping a pre-written RFC 822 message into `message compose` rebuilds
    headers and discards From/To/Subject. The v2-correct path is
    `message send < message.eml`. This test fails if that pattern shows up
    inside any ```bash fenced example block.

    Round-3 + 4 from @hendrixfreire + @andrexibiza.
    """
    for md in ["SKILL.md", "references/configuration.md", "references/message-composition.md"]:
        path = SKILLS_DIR / md
        text = path.read_text(encoding="utf-8")
        in_block = False
        block_lines: list[str] = []
        for line in text.splitlines():
            if line.strip().startswith("```"):
                if not in_block:
                    in_block = True
                    block_lines = []
                    continue
                block_text = "\n".join(block_lines)
                assert "cat message.eml | himalaya message compose" not in block_text, (
                    f"{md} shows `cat message.eml | himalaya message compose` "
                    "inside a bash example. v2 `compose` consumes stdin as the "
                    "BODY and rebuilds headers — use `message send < message.eml`."
                )
                in_block = False
                block_lines = []
                continue
            if in_block:
                block_lines.append(line)


# ---------------------------------------------------------------------------
# Round-3 invariants — v2 command surface that no longer exists
# ---------------------------------------------------------------------------


def test_message_send_save_drafts_not_in_executable_examples() -> None:
    """`message send --save drafts` routes via SMTP and only copies the
    message *after* delivery. It must NOT appear inside an executable
    ```bash example block (it would silently deliver the message). It is
    allowed inside warning prose and reference docs that explain the
    footgun, as long as those warnings don't sit inside a code fence.

    Round-4 P1 from @andrexibiza.
    """
    for md in ["SKILL.md", "references/configuration.md", "references/message-composition.md"]:
        path = SKILLS_DIR / md
        text = path.read_text(encoding="utf-8")
        in_block = False
        block_lines: list[str] = []
        for line in text.splitlines():
            if line.strip().startswith("```"):
                if not in_block:
                    in_block = True
                    block_lines = []
                    continue
                # Closing fence — check if the forbidden command is in this block
                block_text = "\n".join(block_lines)
                if "message send --save drafts" in block_text:
                    # Find the offending line(s)
                    for i, l in enumerate(block_lines, 1):
                        if "message send --save drafts" in l:
                            pytest.fail(
                                f"{md} has `message send --save drafts` inside a "
                                f"bash code block (line {i} of the block): "
                                f"`{l.strip()}`. v2.0.0 routes via SMTP and "
                                "appends a copy *after* delivery — copy-pasting "
                                "this recipe would deliver an unfinished message. "
                                "Replace with `message add --mailbox drafts "
                                "--flag draft < message.eml` or remove the line."
                            )
                in_block = False
                block_lines = []
                continue
            if in_block:
                block_lines.append(line)


# ---------------------------------------------------------------------------
# Round-4 invariants — v2 contract specifics
# ---------------------------------------------------------------------------


def test_message_add_recommended_for_drafts_not_send_save() -> None:
    """The "Save a draft without sending" guidance must teach `message add`,
    not `message send --save drafts` (which routes via SMTP and only
    appends a copy after delivery).

    Reviewer @andrexibiza flagged this as P1 — copy-pasting the recipe
    could deliver an unfinished message.
    """
    text = _read("references/message-composition.md")
    assert "himalaya message add --mailbox drafts --flag draft" in text, (
        "references/message-composition.md should teach `message add --mailbox "
        "drafts --flag draft` as the safe 'save without sending' path."
    )


def test_posting_style_does_not_advertise_inline() -> None:
    """`--posting-style` accepts `top | bottom` in v2.0.0. Documenting
    `inline` as valid will fail at parse time.

    Reviewer @andrexibiza caught this in SKILL.md round 4.
    """
    skill_text = _read("SKILL.md")
    ref_text = _read("references/message-composition.md")
    for label, text in [("SKILL.md", skill_text), ("references/message-composition.md", ref_text)]:
        # Reject "top | bottom | inline" or "top | bottom, inline" style listings.
        match = re.search(r"posting[- ]style\s*\([^)]*\binline\b", text, re.IGNORECASE)
        assert not match, (
            f"{label} advertises `inline` as a valid --posting-style value. "
            "v2.0.0's PostingStyle enum is top | bottom; remove inline."
        )


def test_quote_headline_default_is_empty_not_template() -> None:
    """v2.0.0's `--quote-headline` defaults to the empty string and performs
    no placeholder substitution. The skill must NOT advertise the
    v1.x default `"On {date}, {from} wrote:"`.

    Reviewer @andrexibiza caught this in round 4.
    """
    for md in ["SKILL.md", "references/message-composition.md"]:
        text = _read(md)
        assert "On {date}, {from} wrote:" not in text, (
            f"{md} still advertises the v1.x `On {{date}}, {{from}} wrote:` default "
            "for --quote-headline. v2.0.0 defaults to empty and performs no "
            "placeholder substitution; update the doc."
        )


def test_mml_install_uses_cargo_crate_name_not_binary() -> None:
    """The MML install line must use the Cargo crate name
    `mime-meta-language`, not the binary name `mml`.

    Reviewer @andrexibiza flagged this in round 4.
    """
    text = _read("references/message-composition.md")
    # Reject `cargo install mml` (binary name) when it's the only install form;
    # require `mime-meta-language` to appear in the install instructions.
    install_lines = [
        line for line in text.splitlines()
        if line.strip().startswith("cargo install") and "mml" in line.lower()
    ]
    for line in install_lines:
        # Acceptable if line mentions the crate name OR uses the cargo git form
        # (which doesn't need the crate name — it pulls the whole repo)
        if "mime-meta-language" in line:
            continue
        if "git" in line and "pimalaya/mml" in line:
            continue
        # Otherwise fail
        pytest.fail(
            f"MML install line `{line.strip()}` doesn't name the Cargo crate "
            "`mime-meta-language` and isn't a git install. Use one of:\n"
            "  cargo install mime-meta-language --version 1.1.1 --locked --features cli\n"
            "  cargo install --locked --git https://github.com/pimalaya/mml.git --rev ad50fd97786be9c94a9d758fc1f7792a03d6d378"
        )


def test_mml_master_install_pins_rev() -> None:
    """The git-route install of `mml` must pin a specific `--rev` so the
    install doesn't drift with `master`.

    Reviewer @andrexibiza flagged this in round 6: the master install
    `cargo install --git https://github.com/pimalaya/mml.git` resolves
    whatever `master` points to AT INSTALL TIME, which can select a
    different parser and reintroduce the source/CLI drift this skill
    exists to prevent. Pin to a specific commit hash.
    """
    text = _read("references/message-composition.md")
    # The install line may be commented (since it lives in a ```bash
    # example) so match `cargo install` anywhere in the line, not just
    # at the start.
    install_lines = [
        line for line in text.splitlines()
        if "cargo install" in line
        and "pimalaya/mml" in line
        and "--git" in line
    ]
    assert install_lines, (
        "No `cargo install --git https://github.com/pimalaya/mml.git` line "
        "found in references/message-composition.md. The master install "
        "must be present and pinned (or removed entirely, in which case "
        "the v1.1.1 released path becomes the only one)."
    )
    for line in install_lines:
        # Reject the unpinned form: `cargo install --git ... --locked` is the
        # bad form. Acceptable is `cargo install --git ... --rev <sha>`.
        assert "--rev" in line, (
            f"Master install line `{line.strip()}` does not pin `--rev`. "
            "Without `--rev`, the install resolves whatever `master` points "
            "to at install time, which can re-introduce the source/CLI drift "
            "this section exists to prevent. Add "
            "`--rev ad50fd97786be9c94a9d758fc1f7792a03d6d378` (or remove "
            "the master route entirely and keep only the v1.1.1 path)."
        )
        # The pinned rev must be the specific one @andrexibiza verified
        # the `--output` behavior against.
        assert "ad50fd97786be9c94a9d758fc1f7792a03d6d378" in line, (
            f"Master install line `{line.strip()}` is pinned to a rev, "
            "but not the one @andrexibiza verified the `--output` "
            "behavior against. Pin to "
            "`ad50fd97786be9c94a9d758fc1f7792a03d6d378`."
        )


def test_mml_master_route_pairs_rev_with_output_flag() -> None:
    """If the master route is present, the documented `mml compose`
    invocation in that route must use the `--output` flag (matching the
    pinned rev's CLI contract).

    Reviewer @andrexibiza's round-6 closing: the `--rev` + `--output`
    pair must be tested together. An `--rev` pin without a matching
    `--output` invocation, or vice versa, re-introduces the contract
    drift.
    """
    text = _read("references/message-composition.md")
    # Only enforce this test if the master route is present.
    has_master_route = any(
        "pimalaya/mml" in line and "--git" in line
        for line in text.splitlines()
    )
    if not has_master_route:
        pytest.skip("Master route not present; only the v1.1.1 contract is documented.")
    # Locate the master install line and check that the FIRST
    # `mml compose` invocation after it uses the `--output` flag.
    # We must look at the actual `mml compose` command line, not just
    # any mention of `--output` in nearby comments (which always
    # mention the flag for explanatory reasons).
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if (
            "cargo install" in line
            and "pimalaya/mml" in line
            and "--git" in line
        ):
            # Found a master install line; find the next `mml compose`
            # line and verify it uses `--output`.
            for j in range(i+1, min(i+10, len(lines))):
                next_line = lines[j]
                if "mml compose" in next_line:
                    # Check this is the master route's `mml compose`
                    # (not the v1.1.1 route's, which uses positional
                    # output). The master route's `mml compose` line
                    # should contain `--output` (not just a comment
                    # about it).
                    stripped = next_line.lstrip("# ").strip()
                    if not stripped.startswith("mml compose"):
                        # This is a comment line containing the
                        # phrase, not the executable — skip.
                        continue
                    assert "--output" in stripped, (
                        f"Master route's `mml compose` invocation at "
                        f"line {j+1} is `{next_line.strip()}`. The "
                        "pinned rev's CLI requires the global `--output` "
                        "flag. Use `mml compose --from you@example.com "
                        "--output /tmp/draft.eml`."
                    )
                    # Found and validated the master invocation.
                    break
            else:
                pytest.fail(
                    f"Master install at line {i+1} is present, but no "
                    "`mml compose` invocation found in the next 10 lines. "
                    "Add `mml compose --from you@example.com --output "
                    "/tmp/draft.eml` after the install line."
                )


def test_mml_contract_pins_one_version_end_to_end() -> None:
    """MML install source and invocation shape must agree.

    Reviewer @andrexibiza flagged this in round 5: the same example used
    `--output` (master contract) on a v1.1.1 install (positional contract).
    The doc must pin either v1.1.1 (positional) or master (--output)
    end-to-end within each example, and must NOT leave a bottom executable
    line that re-mixes the contracts.
    """
    text = _read("references/message-composition.md")
    # Detect any executable `mml compose ...` line outside the contract blocks.
    # The contract blocks are commentary; the executable line is the
    # one an agent would actually execute.
    executable = [
        line for line in text.splitlines()
        if re.match(r"\s*mml compose", line)
    ]
    assert not executable, (
        "Found executable `mml compose ...` lines outside the contract "
        "commentary. They re-mix the v1.1.1 / master contracts and must be "
        "removed:\n"
        + "\n".join(f"  {line.rstrip()}" for line in executable)
    )


# ---------------------------------------------------------------------------
# Shared invariants — file hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "SKILL.md",
        "references/configuration.md",
        "references/message-composition.md",
    ],
)
def test_markdown_files_end_with_newline(filename: str) -> None:
    """Round-3 review: missing trailing newlines trip CI whitespace
    checks and produce diff noise. Pin the convention.
    """
    raw = (SKILLS_DIR / filename).read_bytes()
    assert raw.endswith(b"\n"), (
        f"{filename} does not end with a newline. Add one to suppress "
        "CI whitespace noise."
    )


def test_no_v1_imap_or_smtp_backend_table() -> None:
    """Round-1 + 2 review: the v1.x `backend = { type = "imap", host = ...,
    auth = { cmd = "..." } }` table was the biggest source of doc drift.
    v2 uses flat per-backend keys (`imap.server`, `imap.sasl.plain.username`,
    etc.). The skill must not still document the nested table form.
    """
    for md in ["SKILL.md", "references/configuration.md", "references/message-composition.md"]:
        text = _read(md)
        # The specific v1.x nested table signature
        assert "backend = { type = \"imap\"" not in text, (
            f"{md} still documents the v1.x nested backend table. v2 uses "
            "flat keys like `imap.server` / `imap.sasl.plain.username`."
        )


def test_no_message_export_full_in_examples() -> None:
    """`himalaya message export --full` does not exist in v2.0.0's
    subcommand set (`unrecognized subcommand 'export'`). The v2-correct
    way to get raw RFC 5322 bytes is `himalaya message read <id> --raw`.

    Round-3 + 4 from @hendrixfreire.
    """
    for md in ["SKILL.md", "references/configuration.md", "references/message-composition.md"]:
        path = SKILLS_DIR / md
        text = path.read_text(encoding="utf-8")
        in_block = False
        block_lines: list[str] = []
        for line in text.splitlines():
            if line.strip().startswith("```"):
                if not in_block:
                    in_block = True
                    block_lines = []
                    continue
                block_text = "\n".join(block_lines)
                assert "message export --full" not in block_text, (
                    f"{md} shows `message export --full` inside a bash example. "
                    "v2 has no `message export` subcommand; use "
                    "`message read <id> --raw` instead."
                )
                in_block = False
                block_lines = []
                continue
            if in_block:
                block_lines.append(line)


def test_no_equals_form_note_for_page_size() -> None:
    """The search example must not annotate `--page-size=20` as the only
    valid form ("NOTE: (equals form!)"). v2.0.0 parses both forms.

    Reviewer @Enough1122 caught this in round 3.
    """
    text = _read("SKILL.md")
    assert "equals form!" not in text, (
        "SKILL.md still has the 'equals form!' annotation on --page-size. "
        "v2.0.0 accepts both --page-size 20 and --page-size=20; drop the note."
    )


@pytest.mark.parametrize(
    "filename",
    ["SKILL.md", "references/configuration.md", "references/message-composition.md"],
)
def test_no_v1_imap_or_smtp_backend_table_in_examples(filename: str) -> None:
    """Round-1 + 2 review: the v1.x `backend = { type = "imap", host = ...,
    auth = { cmd = "..." } }` table was the biggest source of doc drift.
    v2 uses flat per-backend keys (`imap.server`, `imap.sasl.plain.username`,
    etc.). The skill must not still *show* the nested table form as a
    working example.

    Narrative mentions that *describe* the v1.x form to contrast with v2
    (e.g. "Common mistakes #4") are allowed — this test only flags the
    form when it appears inside a ```bash fenced example block, which is
    what an agent would actually execute.
    """
    path = SKILLS_DIR / filename
    text = path.read_text(encoding="utf-8")
    in_block = False
    block_lines: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if not in_block:
                in_block = True
                block_lines = []
                continue
            # Closing fence — check the block we just closed.
            block_text = "\n".join(block_lines)
            assert 'backend = { type = "imap"' not in block_text, (
                f"{filename} documents the v1.x nested backend table inside a "
                "bash example. v2 uses flat keys like `imap.server` / "
                "`imap.sasl.plain.username`."
            )
            in_block = False
            block_lines = []
            continue
        if in_block:
            block_lines.append(line)


@pytest.mark.parametrize(
    "filename",
    ["SKILL.md", "references/configuration.md", "references/message-composition.md"],
)
def test_no_v1_folder_alias_singular_in_examples(filename: str) -> None:
    """`folder.aliases.X` (singular, sub-table) is v1.x. v2 uses
    `mailbox.alias.X` (plural, dotted key under the account block).

    Narrative mentions in the v1.x→v2.x comparison table or in "Common
    mistakes" sections are fine — this test only flags the form when it
    appears inside a ```bash fenced example block.
    """
    path = SKILLS_DIR / filename
    text = path.read_text(encoding="utf-8")
    in_block = False
    block_lines: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if not in_block:
                in_block = True
                block_lines = []
                continue
            block_text = "\n".join(block_lines)
            assert "folder.aliases." not in block_text, (
                f"{filename} shows the v1.x `folder.aliases.X` form inside a "
                "bash example. v2 uses `mailbox.alias.X`."
            )
            in_block = False
            block_lines = []
            continue
        if in_block:
            block_lines.append(line)


@pytest.mark.parametrize(
    "filename",
    ["SKILL.md", "references/configuration.md", "references/message-composition.md"],
)
def test_no_v1_folder_list_subcommand_in_examples(filename: str) -> None:
    """`folder list` is v1.x; v2 renamed the subcommand to `mailbox list`.

    Narrative mentions in the warning text or v1.x→v2.x table are fine —
    this test only flags the form when it appears inside a ```bash
    fenced example block.
    """
    path = SKILLS_DIR / filename
    text = path.read_text(encoding="utf-8")
    in_block = False
    block_lines: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if not in_block:
                in_block = True
                block_lines = []
                continue
            block_text = "\n".join(block_lines)
            assert "himalaya folder list" not in block_text, (
                f"{filename} shows `himalaya folder list` inside a bash "
                "example. v2 renamed it to `himalaya mailbox list`."
            )
            in_block = False
            block_lines = []
            continue
        if in_block:
            block_lines.append(line)


def test_no_v1_template_subcommand() -> None:
    """`himalaya template send` (v1.x) does not exist in v2.0.0's
    MessageCommand enum."""
    for md in ["SKILL.md", "references/configuration.md", "references/message-composition.md"]:
        text = _read(md)
        assert "himalaya template" not in text, (
            f"{md} still references `himalaya template ...`. v2.0.0 has no "
            "`template` subcommand; use `message send` or `message add`."
        )


def test_no_v1_message_write_ed_opens() -> None:
    """`message write` is a `visible_alias` of `message compose` in v2; it
    does NOT open an editor (pre-v1.x behavior)."""
    for md in ["SKILL.md", "references/configuration.md", "references/message-composition.md"]:
        text = _read(md)
        # The phrase "message write opens an editor" (or any claim that
        # write triggers $EDITOR) is the v1.x claim that was wrong.
        for forbidden in [
            "message write opens",
            "message write...editor",
            "message write (opens",
            "write opens an editor",
            "write to open an editor",
        ]:
            assert forbidden not in text, (
                f"{md} still claims `message write` opens an editor. In v2 "
                "it is a `visible_alias` of `message compose` (flag-based, "
                "no $EDITOR). Update the doc."
            )


def test_no_v1_message_reply_all_or_quote() -> None:
    """`message reply --all` and `--quote` flags do not exist in v2.0.0's
    reply parser. Reply-all is via `--cc`/`--to`; quoting is via
    `--posting-style` and `--quote-headline`."""
    for md in ["SKILL.md", "references/configuration.md", "references/message-composition.md"]:
        text = _read(md)
        for forbidden in [
            "message reply --all",
            "message reply --quote",
            "--all flag",
            "--quote flag",
        ]:
            assert forbidden not in text, (
                f"{md} references `{forbidden}`. v2.0.0 has no `--all` or "
                "`--quote` flag on `message reply`; reply-all is via `--cc`/`--to` "
                "and quoting is via `--posting-style` / `--quote-headline`."
            )