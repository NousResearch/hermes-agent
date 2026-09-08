"""Hardline coverage for raw block devices and the disk wipe/format tool family.

The floor exists for "root wipe, raw block device writes, shutdown, DoS", but its
device rules only knew the Linux SCSI/NVMe spellings and its format rule only knew
the name `mkfs`. Everything else that destroys a whole disk — macOS `/dev/disk0`,
device-mapper/LVM, md RAID, loop/nbd, the `/dev/disk/by-*` symlinks, and the
`wipefs`/`blkdiscard`/`sgdisk`/`shred`/`mke2fs`/`mkswap`/`newfs_*`/`diskutil`
tools — walked past every guard, so `--yolo` ran them with no prompt at all.

These are behaviour contracts on the classifier: a command that erases a disk with
no recovery path must hit the unconditional floor, and the tool invocations that
only PRINT (or that target a file rather than a device) must stay runnable.
"""

import pytest

from tools.approval import (
    check_all_command_guards,
    check_dangerous_command,
    detect_dangerous_command,
    detect_hardline_command,
    disable_session_yolo,
)
from tools.approval_context import reset_current_session_key, set_current_session_key


# Commands that MUST be hardline-blocked: every one destroys a whole disk.
_BLOCK_DEVICE_HARDLINE_BLOCK = [
    # ---- macOS disks: whole disk, slice, and the raw character node -------
    "cat x > /dev/disk0",
    "cat x > /dev/disk0s2",
    "cat x > /dev/rdisk0",
    "dd if=/dev/zero of=/dev/disk0",
    "dd if=/dev/zero of=/dev/rdisk2 bs=1m",
    # ---- device-mapper / LVM, md RAID, loop, network block devices --------
    "cat x > /dev/dm-0",
    "cat x > /dev/mapper/vg-root",
    "cat x > /dev/md0",
    "cat x > /dev/md127",
    "cat x > /dev/loop0",
    "cat x > /dev/nbd0",
    "dd if=/dev/zero of=/dev/mapper/vg-root",
    "dd if=/dev/zero of=/dev/dm-0",
    # ---- udev persistent-name symlinks: the same disks, stable names ------
    "cat x > /dev/disk/by-id/ata-Samsung_SSD",
    "cat x > /dev/disk/by-uuid/8e7d1c2a-1111-2222-3333-444455556666",
    "cat x > /dev/disk/by-path/pci-0000:00:1f.2-ata-1",
    "cat x > /dev/disk/by-label/DATA",
    "cat x > /dev/disk/by-partuuid/1a2b3c4d-01",
    "cat x > /dev/disk/by-partlabel/root",
    "dd if=/dev/zero of=/dev/disk/by-id/ata-Samsung_SSD",
    # ---- redirect spellings ----------------------------------------------
    "cat x >> /dev/disk0",
    "cat x 1> /dev/rdisk0",
    "cat x 1>> /dev/mapper/vg-root",
    "cat x >/dev/loop0",
    # ---- dd with the operands in either order ----------------------------
    "dd of=/dev/disk0 if=/dev/zero",
    "dd of=/dev/md0 bs=4M if=/dev/urandom",
    # ---- wipe family ------------------------------------------------------
    "wipefs -a /dev/sda",
    "wipefs --all /dev/sda",
    "wipefs -a /dev/disk0",
    "wipefs /dev/sda -a",  # util-linux permutes flags after operands
    "wipefs -fa /dev/nvme0n1",
    "blkdiscard /dev/nvme0n1",
    "blkdiscard -f /dev/sda",
    "blkdiscard /dev/disk/by-id/nvme-Samsung",
    "sgdisk --zap-all /dev/sda",
    "sgdisk -Z /dev/sda",
    "sgdisk -o /dev/sda",
    "sgdisk --clear /dev/nvme0n1",
    "shred -n 1 -z /dev/sda",
    "shred /dev/nvme0n1",
    "shred -vfz -n 3 /dev/disk2",
    # ---- format family ----------------------------------------------------
    "mke2fs /dev/sda1",  # mke2fs IS mkfs.ext2/3/4 — same binary
    "mke2fs -t ext4 /dev/nvme0n1p2",
    "mkswap /dev/sda1",
    "mkswap /dev/mapper/vg-swap",
    "newfs_hfs /dev/disk2",
    "newfs_apfs /dev/disk2",
    "newfs_msdos /dev/disk2s1",
    "newfs_hfs -v Data /dev/rdisk3s2",
    # ---- macOS diskutil destructive verbs ---------------------------------
    "diskutil eraseDisk JHFS+ X disk0",
    "diskutil eraseVolume JHFS+ X disk0s2",
    "diskutil zeroDisk disk0",
    "diskutil randomDisk 1 disk0",
    "diskutil secureErase 0 disk0",
    "diskutil reformat disk0",
    "diskutil partitionDisk disk0 GPT JHFS+ Data 100%",
    "diskutil apfs eraseVolume disk2s1",
    # ---- command-position variants (separators, wrappers, substitution) ----
    "true && wipefs -a /dev/sda",
    "ls; blkdiscard /dev/sda",
    "echo x | shred -z /dev/sda",
    "$(blkdiscard /dev/sda)",
    "`sgdisk -Z /dev/sda`",
    "sudo diskutil eraseDisk APFS Data /dev/disk2",
    "sudo wipefs -a /dev/disk0",
    "env FOO=1 mke2fs /dev/sda1",
    "nohup blkdiscard /dev/sda",
    "(wipefs -a /dev/sda)",
    "{ mke2fs /dev/sda1; }",
    'bash -c "wipefs -a /dev/sda"',
    'sh -c "cat f > /dev/disk0"',
]


# Commands that look similar but MUST stay runnable.
_BLOCK_DEVICE_HARDLINE_ALLOW = [
    # ---- /dev entries that are not block devices --------------------------
    "echo test > /dev/null",
    "cat noisy.log > /dev/null 2>&1",
    "dd if=/dev/zero of=./image.bin",
    "head -c 16 /dev/urandom > seed.bin",
    "cat /dev/random | head -c 4",
    "echo hi > /dev/stdout",
    "echo hi > /dev/stderr",
    "cat < /dev/stdin",
    "echo hi > /dev/tty",
    "echo hi > /dev/pts/0",
    "echo hi > /dev/fd/1",
    "echo payload > /dev/shm/cache",
    "echo hi > /dev/full",
    # ---- reading a device is not writing to one ---------------------------
    "dd if=/dev/sda of=backup.img",
    "dd if=/dev/disk0 of=backup.img",
    "ls /dev/sda",
    "ls /dev/disk/by-id",
    "cat /dev/disk/by-id/foo",
    "lsblk /dev/sda",
    # ---- wipe-family invocations that only print or target a file ---------
    "wipefs /dev/sda",
    "wipefs -n /dev/sda",
    "wipefs --backup /dev/sda",
    "sgdisk -p /dev/sda",
    "sgdisk -n 1:0:+512M /dev/sda",
    "blkdiscard --help",
    "shred file.txt",
    "shred -u secret.txt",
    "shred -n 3 /tmp/notes.txt",
    # ---- format-family invocations that target a file, not a device -------
    "mkswap /swapfile",
    "mkswap ./swap.img",
    "newfs_hfs disk.dmg",
    # ---- diskutil read-only verbs -----------------------------------------
    "diskutil list",
    "diskutil info disk0",
    "diskutil mount disk2s1",
    "diskutil unmount /dev/disk2s1",
    "diskutil apfs list",
    # ---- the trigger words as quoted prose, not as commands ---------------
    'echo "run diskutil eraseDisk later"',
    'git commit -m "mke2fs the backup disk"',
    'echo "wipefs -a /dev/sda destroys the disk"',
    "grep 'blkdiscard /dev/sda' notes.md",
    'gh pr create --body "this PR blocks shred /dev/nvme0n1"',
    "echo 'sgdisk -Z /dev/sda zaps the partition table'",
    'git commit -m "document dd if=/dev/zero of=/dev/disk0"',
]


@pytest.mark.parametrize("command", _BLOCK_DEVICE_HARDLINE_BLOCK)
def test_disk_destroying_commands_are_hardline(command):
    """Every spelling of "erase a whole disk" hits the unconditional floor."""
    is_hardline, description = detect_hardline_command(command)
    assert is_hardline, f"disk wipe leaked past the hardline floor: {command!r}"
    assert description, "hardline match must provide a description"


@pytest.mark.parametrize("command", _BLOCK_DEVICE_HARDLINE_ALLOW)
def test_lookalike_commands_stay_runnable(command):
    """Reads, print-only tool runs, file targets, and prose must not be blocked."""
    is_hardline, description = detect_hardline_command(command)
    assert not is_hardline, (
        f"legitimate command false-positived the hardline floor: {command!r} "
        f"(got: {description})"
    )
    assert description is None


# Ordinary /dev pseudo-devices must not be dragged into the APPROVAL tier
# either: the block-device fragment is shared with _SENSITIVE_WRITE_TARGET,
# so `echo x > /dev/null` in a script would start prompting if it over-matched.
@pytest.mark.parametrize("command", [
    "echo test > /dev/null",
    "cat noisy.log > /dev/null 2>&1",
    "echo hi > /dev/stdout",
    "echo hi > /dev/stderr",
    "echo payload > /dev/shm/cache",
    "echo hi > /dev/fd/1",
    "cat file | tee /dev/tty",
])
def test_pseudo_device_writes_need_no_approval(command):
    is_dangerous, _, description = detect_dangerous_command(command)
    assert not is_dangerous, (
        f"pseudo-device write started requiring approval: {command!r} "
        f"(got: {description})"
    )


# The dangerous tier carried the same Linux-only spellings, so cp/mv/tee to a
# non-`sd` device was auto-approved. Yolo may still bypass this tier — the point
# is that an unattended session no longer writes a disk with no prompt at all.
@pytest.mark.parametrize("command", [
    "cp x /dev/nvme0n1",
    "cp x /dev/disk0",
    "mv x /dev/vda",
    "mv x /dev/mapper/vg-root",
    "tee /dev/disk0 < x",
    "tee /dev/mapper/vg-root < x",
    "echo x | tee /dev/nvme0n1",
])
def test_copy_and_tee_to_block_device_requires_approval(command):
    is_dangerous, key, description = detect_dangerous_command(command)
    assert is_dangerous, f"write to a block device was auto-approved: {command!r}"
    assert key is not None and description


@pytest.fixture
def clean_session(monkeypatch):
    """Reset session-scoped approval state around each test."""
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    token = set_current_session_key("hardline_block_device_test")
    try:
        disable_session_yolo("hardline_block_device_test")
        yield
    finally:
        disable_session_yolo("hardline_block_device_test")
        reset_current_session_key(token)


def test_yolo_cannot_bypass_disk_wipes(clean_session, monkeypatch):
    """macOS/dm/wipe spellings stay blocked with HERMES_YOLO_MODE=1.

    They previously reached the approval tier at best (`dd ... of=/dev/disk0`)
    or no tier at all, and yolo bypasses everything above the floor.
    """
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    for command in ("dd if=/dev/zero of=/dev/disk0", "cat x > /dev/rdisk0",
                    "diskutil eraseDisk JHFS+ X disk0", "newfs_apfs /dev/disk2",
                    "wipefs -a /dev/sda", "blkdiscard /dev/nvme0n1",
                    "sgdisk -Z /dev/sda", "shred -z /dev/disk2",
                    "mke2fs /dev/sda1", "cat x > /dev/mapper/vg-root"):
        first = check_dangerous_command(command, "local")
        assert first["approved"] is False, f"yolo leaked {command!r} (check_dangerous_command)"
        assert first.get("hardline") is True

        second = check_all_command_guards(command, "local")
        assert second["approved"] is False, f"yolo leaked {command!r} (check_all_command_guards)"
        assert second.get("hardline") is True
        assert "BLOCKED (hardline)" in second["message"]
