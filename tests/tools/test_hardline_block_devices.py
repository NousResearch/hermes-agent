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
    # ---- a shell-quoted DEVICE OPERAND is a real target, not prose (review P1 #1) ----
    'cat x > "/dev/disk0"',
    "cat x > '/dev/disk0'",
    'cat x > "/dev/sda"',
    "cat x > '/dev/nvme0n1'",
    'cat x >"/dev/sda"',
    'cat x >> "/dev/sda"',
    'cat x >  "/dev/rdisk0"',
    'cat x > "/dev/mapper/vg-root"',
    'cat x > "/dev/disk/by-id/ata-Samsung_SSD"',
    'dd if=/dev/zero of="/dev/disk0"',
    "dd if=/dev/zero of='/dev/disk0'",
    'dd if=/dev/zero of="/dev/sda" bs=1m',
    "dd if=/dev/zero of='/dev/nvme0n1'",
    # ---- the verb reaching the disk without appearing at _CMDPOS (review P1 #2) ----
    # absolute and relative paths
    "/usr/sbin/wipefs -a /dev/sda",
    "/sbin/mkfs.ext4 /dev/sda1",
    "/usr/local/bin/shred -n1 /dev/sda",
    "/sbin/newfs_hfs /dev/disk2",
    "./wipefs -a /dev/sda",
    "../sbin/wipefs --all /dev/sda",
    # scheduling / buffering wrappers
    "nice blkdiscard /dev/sda",
    "nice -n 10 sgdisk -Z /dev/sda",
    "nice mkswap /dev/sda1",
    "command mke2fs /dev/sda1",
    "command shred -n 1 -z /dev/sda",
    "stdbuf -oL dd if=/dev/zero of=/dev/sda",
    "ionice -c3 blkdiscard /dev/nvme0n1",
    "taskset -c 0 dd if=/dev/zero of=/dev/sda",
    "chrt -f 1 blkdiscard /dev/sda",
    # multicall binaries
    "busybox dd if=/dev/zero of=/dev/sda",
    "toybox dd if=/dev/zero of=/dev/sda",
    "/bin/busybox dd if=/dev/zero of=/dev/sda",
    # stacked with the wrappers _CMDPOS already peeled
    "sudo /usr/sbin/wipefs -a /dev/sda",
    "sudo nice blkdiscard /dev/sda",
    # ---- the wrapper query-option carve-out must not become a way past the floor ----
    # The query-option carve-out must not become a way past the floor.
    "command mkfs.ext4 /dev/sda1",
    "command -p mkfs.ext4 /dev/sda1",
    "chrt -f 1 blkdiscard /dev/sda",
    "taskset -c 0 dd if=/dev/zero of=/dev/sda",
    "ionice -c3 blkdiscard /dev/nvme0n1",
    # ---- the same node under a traversal or collapsed spelling (review P1 #2 on #120926) ----
    # `..` right in front of `dev/` may climb to the root: from /tmp, `../dev/sda` IS /dev/sda.
    "wipefs -a ../dev/sda",
    "blkdiscard ../../dev/nvme0n1",
    "shred -n 1 -z /tmp/../dev/sda",
    "sgdisk --zap-all /dev/../dev/sda",
    "mkswap ../dev/sda1",
    "newfs_hfs ../dev/disk2",
    "dd if=/dev/zero of=../dev/sda",
    "dd if=/dev/zero of=/tmp/../dev/disk0",
    "cat x > ../dev/sda",
    "cat x > /var/tmp/../../dev/mapper/vg-root",
    'cat x > "../dev/disk0"',
    "sudo wipefs -a ../dev/sda",
    "nice blkdiscard /tmp/../dev/sda",
    # the kernel collapses empty and `.` segments
    "wipefs -a /dev//sda",
    "blkdiscard /dev/./nvme0n1",
    "cat x > /dev/.//sda",
    "dd if=/dev/zero of=/dev//disk0",
    # mdadm's named arrays live in a directory, like /dev/mapper
    "wipefs -a /dev/md/raid1",
    "cat x > /dev/md/boot",
    # A real device path still matches after every boundary that is not a path character.
    "cat x > /dev/sda",
    'cat x > "/dev/sda"',
    "dd if=/dev/zero of=/dev/sda",
    "wipefs -a /dev/sda",
    "wipefs -fa /dev/sda",
    "mkswap /dev/sda1",
    "shred -n 1 -z /dev/sda",
]


# Commands that look similar but MUST stay runnable: reads, print-only tool runs, file
# targets, prose, verb-named filenames, peeled wrappers in front of harmless commands.
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
    # ---- traps the widenings could spring: verb-named files, peeled wrappers, quoted targets ----
    # a path that only contains the verb name
    "\n/opt/mkfs-notes.txt",
    "\n/var/log/mkfs-notes.txt",
    "./mkfs-helper.sh --dry-run",
    "\n/usr/share/doc/wipefs-readme",
    "\n/etc/init.d/mkfs-cron status",
    "cat /var/log/dd.log",
    "tail -f /var/log/shutdown.log",
    # a peeled wrapper in front of something ordinary
    "nice make -j4",
    "nice -n 19 python train.py",
    "command ls -la",
    "busybox ls /tmp",
    "stdbuf -oL grep foo bar.txt",
    "ionice -c3 rsync a b",
    # ordinary quoted redirect targets
    'echo hi > "out.txt"',
    'printf "%s" > "some file.txt"',
    'npm run build > "build log.txt"',
    'cat x > "/dev/null"',
    # and the prose contract from #93640, restated against the quoted-operand fix
    'echo "cat x > /dev/disk0"',
    'echo "cat x > /dev/sda"',
    "echo 'cat x > /dev/sda'",
    "git commit -m 'ran dd if=/dev/zero of=/dev/disk0 once'",
    'git commit -m "dd if=/dev/zero of=/dev/sda is what broke it"',
    # ---- commands that RUN NOTHING destructive on a floor with no approval path ----
    # A wrapper's QUERY options do not run the verb. `command -v mkfs` prints a path; the shape
    # appears 15 times in a 42,222-line corpus of real command lines, and
    # `_COMMAND_WRAPPER_NON_EXECUTING_OPTIONS` already encodes exactly this.
    "command -v mkfs",
    "command -v mkfs.ext4",
    "command -v mke2fs",
    "command -V mkfs",
    "builtin command -v mkfs",
    "nice -n 10 command -v mkfs",
    "if command -v mkfs >/dev/null 2>&1; then echo yes; fi",
    "command -v mkfs.btrfs || apt-get install btrfs-progs",
    "chrt -p 1234",
    "taskset -p 1234",
    "ionice -p 1234",
    # A directory named `dev` is not /dev. `~/dev/sdk/...` carries `/dev/sd` plus a `k`.
    "shred -u ~/dev/sdk/token.json",
    "shred -u /home/user/dev/sdk-token.txt",
    "shred -n 3 -u /workspace/dev/sdk/keys.pem",
    "shred -u /opt/dev/mdbook/out.txt",
    "shred -u /opt/dev/nbdev/notebook.ipynb",
    "shred -u /src/dev/loopback-tests/creds",
    "shred -u /src/dev/vdom/token",
    "shred -u ./dev/hdmi-config.txt",
    "shred -u /home/ci/dev/dm-cache/pass.txt",
    "shred -u /var/tmp/dev/disk1-report.txt",
    "shred -u ../dev/sda-notes",
    "blkdiscard -v /home/dev/sda-file",
    "wipefs -a ~/dev/sdk/disk.img",
    "mkswap /home/user/dev/sdk/swapfile",
    "mkswap /var/lib/dev/sdk/swap.img",
    "newfs_hfs ~/dev/sdk/disk.dmg",
    # `..` widened the floor to traversal spellings, so the node must still be the LAST path
    # component: a file under `../dev/` is a file, and `sda-notes` is not `sda`.
    "shred -u ../dev/sdk/token.json",
    "shred -n 3 -u ../../dev/sdk/keys.pem",
    "shred -u /tmp/../dev/sdk/token.json",
    "wipefs -a ../dev/sda-notes",
    "mkswap ../dev/sdk/swap.img",
    "shred -u /dev/sda-notes",
    "blkdiscard -v /dev/sdk/token.json",
    # The operand lookahead must not read a trailing comment as the operand.
    "shred -u notes.txt # never do this to /dev/sda",
    # `-n`/`--no-act` is wipefs doing everything except the write: a diagnostic.
    "wipefs -n -a /dev/sda",
    "wipefs --no-act --all /dev/sda",
    # Quoted words AFTER a quoted redirect target are prose again, not more operands.
    'cat x > "out.log" "cat y > /dev/sda"',
    './deploy.sh > "deploy.log" "note: cat img > /dev/sda"',
    'cat x > "a" ":(){ :|:& };:"',
    # An EMPTY quoted target leaves no content to move the marker off the `>`, so the quote
    # character itself has to; otherwise the next quoted word inherits the exception.
    'cat x > "" "prose about /dev/sda"',
    'cat x > \'\' \'note: cat y > /dev/sda\'',
]


@pytest.mark.parametrize("command", _BLOCK_DEVICE_HARDLINE_BLOCK)
def test_disk_destroying_commands_are_hardline(command):
    """Every spelling of "erase a whole disk" hits the unconditional floor — quoted operand,
    path-spelled verb, scheduling wrapper or multicall dispatch included."""
    is_hardline, description = detect_hardline_command(command)
    assert is_hardline, f"disk wipe leaked past the hardline floor: {command!r}"
    assert description, "hardline match must provide a description"


@pytest.mark.parametrize("command", _BLOCK_DEVICE_HARDLINE_ALLOW)
def test_lookalike_commands_stay_runnable(command):
    """Nothing that runs no destructive write may hit a floor that has no approval path."""
    is_hardline, description = detect_hardline_command(command)
    assert not is_hardline, (
        f"legitimate command false-positived the hardline floor: {command!r} "
        f"(got: {description})"
    )
    assert description is None


# The dangerous (approval) tier shares the device fragment with _SENSITIVE_WRITE_TARGET: cp/mv/tee
# to a non-`sd` device used to be auto-approved, while `echo x > /dev/null` must never start prompting.
@pytest.mark.parametrize("command,needs_approval", [
    *[(c, True) for c in [
    "cp x /dev/nvme0n1",
    "cp x /dev/disk0",
    "mv x /dev/vda",
    "mv x /dev/mapper/vg-root",
    "tee /dev/disk0 < x",
    "tee /dev/mapper/vg-root < x",
    "echo x | tee /dev/nvme0n1",
    "tee ../dev/disk0 < x",
    "cp x /tmp/../dev/vda",
    "mv x /dev//nvme0n1",
    ]],
    *[(c, False) for c in [
    "echo test > /dev/null",
    "cat noisy.log > /dev/null 2>&1",
    "echo hi > /dev/stdout",
    "echo hi > /dev/stderr",
    "echo payload > /dev/shm/cache",
    "echo hi > /dev/fd/1",
    "cat file | tee /dev/tty",
    ]],
])
def test_dangerous_tier_block_device_writes(command, needs_approval):
    is_dangerous, key, description = detect_dangerous_command(command)
    assert is_dangerous is needs_approval, (
        f"{command!r}: expected needs_approval={needs_approval} (got: {description})"
    )
    if needs_approval:
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


@pytest.mark.parametrize("command", [
    "dd if=/dev/zero of=/dev/disk0",
    "cat x > /dev/rdisk0",
    "diskutil eraseDisk JHFS+ X disk0",
    "newfs_apfs /dev/disk2",
    "wipefs -a /dev/sda",
    "blkdiscard /dev/nvme0n1",
    "sgdisk -Z /dev/sda",
    "shred -z /dev/disk2",
    "mke2fs /dev/sda1",
    "cat x > /dev/mapper/vg-root",
    'cat x > "/dev/disk0"',
    "cat x > '/dev/sda'",
    'dd if=/dev/zero of="/dev/disk0"',
    "dd if=/dev/zero of='/dev/nvme0n1'",
    'cat x >> "/dev/mapper/vg-root"',
    "/usr/sbin/wipefs -a /dev/sda",
    "nice blkdiscard /dev/sda",
    "command mke2fs /dev/sda1",
    "busybox dd if=/dev/zero of=/dev/sda",
    "/sbin/mkfs.ext4 /dev/sda1",
    "nice -n 10 sgdisk -Z /dev/sda",
    "sudo /usr/sbin/wipefs -a /dev/sda",
    "wipefs -a ../dev/sda",
    "cat x > /tmp/../dev/disk0",
    "dd if=/dev/zero of=/dev//sda",
])
def test_yolo_cannot_bypass_disk_wipes(clean_session, monkeypatch, command):
    """These reached the approval tier at best (or no tier at all) — exactly what
    HERMES_YOLO_MODE skips — so the integration path, not just the classifier, must block."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    first = check_dangerous_command(command, "local")
    assert first["approved"] is False, f"yolo leaked {command!r} (check_dangerous_command)"
    assert first.get("hardline") is True

    second = check_all_command_guards(command, "local")
    assert second["approved"] is False, f"yolo leaked {command!r} (check_all_command_guards)"
    assert second.get("hardline") is True
    assert "BLOCKED (hardline)" in second["message"]
