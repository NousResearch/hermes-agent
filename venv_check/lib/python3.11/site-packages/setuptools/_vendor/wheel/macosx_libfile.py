"""
This module contains function to analyse dynamic library
headers to extract system information

Currently only for MacOSX

Library file on macosx system starts with Mach-O or Fat field.
This can be distinguish by first 32 bites and it is called magic number.
Proper value of magic number is with suffix _MAGIC. Suffix _CIGAM means
reversed bytes order.
Both fields can occur in two types: 32 and 64 bytes.

FAT field inform that this library contains few version of library
(typically for different types version). It contains
information where Mach-O headers starts.

Each section started with Mach-O header contains one library
(So if file starts with this field it contains only one version).

After filed Mach-O there are section fields.
Each of them starts with two fields:
cmd - magic number for this command
cmdsize - total size occupied by this section information.

In this case only sections LC_VERSION_MIN_MACOSX (for macosx 10.13 and earlier)
and LC_BUILD_VERSION (for macosx 10.14 and newer) are interesting,
because them contains information about minimal system version.

Important remarks:
- For fat files this implementation looks for maximum number version.
  It not check if it is 32 or 64 and do not compare it with currently built package.
  So it is possible to false report higher version that needed.
- All structures signatures are taken form macosx header files.
- I think that binary format will be more stable than `otool` output.
  and if apple introduce some changes both implementation will need to be updated.
- The system compile will set the deployment target no lower than
  11.0 for arm64 builds. For "Universal 2" builds use the x86_64 deployment
  target when the arm64 target is 11.0.
"""

from __future__ import annotations

import ctypes
import os
import sys
from io import BufferedIOBase
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union

    StrPath = Union[str, os.PathLike[str]]

"""here the needed const and struct from mach-o header files"""

FAT_MAGIC = 0xCAFEBABE
FAT_CIGAM = 0xBEBAFECA
FAT_MAGIC_64 = 0xCAFEBABF
FAT_CIGAM_64 = 0xBFBAFECA
MH_MAGIC = 0xFEEDFACE
MH_CIGAM = 0xCEFAEDFE
MH_MAGIC_64 = 0xFEEDFACF
MH_CIGAM_64 = 0xCFFAEDFE

LC_VERSION_MIN_MACOSX = 0x24
LC_BUILD_VERSION = 0x32

CPU_TYPE_ARM64 = 0x0100000C

mach_header_fields = [
    ("magic", ctypes.c_uint32),
    ("cputype", ctypes.c_int),
    ("cpusubtype", ctypes.c_int),
    ("filetype", ctypes.c_uint32),
    ("ncmds", ctypes.c_uint32),
    ("sizeofcmds", ctypes.c_uint32),
    ("flags", ctypes.c_uint32),
]
"""
struct mach_header {
    uint32_t	magic;		/* mach magic number identifier */
    cpu_type_t	cputype;	/* cpu specifier */
    cpu_subtype_t	cpusubtype;	/* machine specifier */
    uint32_t	filetype;	/* type of file */
    uint32_t	ncmds;		/* number of load commands */
    uint32_t	sizeofcmds;	/* the size of all the load commands */
    uint32_t	flags;		/* flags */
};
typedef integer_t cpu_type_t;
typedef integer_t cpu_subtype_t;
"""

mach_header_fields_64 = mach_header_fields + [("reserved", ctypes.c_uint32)]
"""
struct mach_header_64 {
    uint32_t	magic;		/* mach magic number identifier */
    cpu_type_t	cputype;	/* cpu specifier */
    cpu_subtype_t	cpusubtype;	/* machine specifier */
    uint32_t	filetype;	/* type of file */
    uint32_t	ncmds;		/* number of load commands */
    uint32_t	sizeofcmds;	/* the size of all the load commands */
    uint32_t	flags;		/* flags */
    uint32_t	reserved;	/* reserved */
};
"""

fat_header_fields = [("magic", ctypes.c_uint32), ("nfat_arch", ctypes.c_uint32)]
"""
struct fat_header {
    uint32_t	magic;		/* FAT_MAGIC or FAT_MAGIC_64 */
    uint32_t	nfat_arch;	/* number of structs that follow */
};
"""

fat_arch_fields = [
    ("cputype", ctypes.c_int),
    ("cpusubtype", ctypes.c_int),
    ("offset", ctypes.c_uint32),
    ("size", ctypes.c_uint32),
    ("align", ctypes.c_uint32),
]
"""
struct fat_arch {
    cpu_type_t	cputype;	/* cpu specifier (int) */
    cpu_subtype_t	cpusubtype;	/* machine specifier (int) */
    uint32_t	offset;		/* file offset to this object file */
    uint32_t	size;		/* size of this object file */
    uint32_t	align;		/* alignment as a power of 2 */
};
"""

fat_arch_64_fields = [
    ("cputype", ctypes.c_int),
    ("cpusubtype", ctypes.c_int),
    ("offset", ctypes.c_uint64),
    ("size", ctypes.c_uint64),
    ("align", ctypes.c_uint32),
    ("reserved", ctypes.c_uint32),
]
"""
struct fat_arch_64 {
    cpu_type_t	cputype;	/* cpu specifier (int) */
    cpu_subtype_t	cpusubtype;	/* machine specifier (int) */
    uint64_t	offset;		/* file offset to this object file */
    uint64_t	size;		/* size of this object file */
    uint32_t	align;		/* alignment as a power of 2 */
    uint32_t	reserved;	/* reserved */
};
"""

segment_base_fields = [("cmd", ctypes.c_uint32), ("cmdsize", ctypes.c_uint32)]
"""base for reading segment info"""

segment_command_fields = [
    ("cmd", ctypes.c_uint32),
    ("cmdsize", ctypes.c_uint32),
    ("segname", ctypes.c_char * 16),
    ("vmaddr", ctypes.c_uint32),
    ("vmsize", ctypes.c_uint32),
    ("fileoff", ctypes.c_uint32),
    ("filesize", ctypes.c_uint32),
    ("maxprot", ctypes.c_int),
    ("initprot", ctypes.c_int),
    ("nsects", ctypes.c_uint32),
    ("flags", ctypes.c_uint32),
]
"""
struct segment_command { /* for 32-bit architectures */
    uint32_t	cmd;		/* LC_SEGMENT */
    uint32_t	cmdsize;	/* includes sizeof section structs */
    char		segname[16];	/* segment name */
    uint32_t	vmaddr;		/* memory address of this segment */
    uint32_t	vmsize;		/* memory size of this segment */
    uint32_t	fileoff;	/* file offset of this segment */
    uint32_t	filesize;	/* amount to map from the file */
    vm_prot_t	maxprot;	/* maximum VM protection */
    vm_prot_t	initprot;	/* initial VM protection */
    uint32_t	nsects;		/* number of sections in segment */
    uint32_t	flags;		/* flags */
};
typedef int vm_prot_t;
"""

segment_command_fields_64 = [
    ("cmd", ctypes.c_uint32),
    ("cmdsize", ctypes.c_uint32),
    ("segname", ctypes.c_char * 16),
    ("vmaddr", ctypes.c_uint64),
    ("vmsize", ctypes.c_uint64),
    ("fileoff", ctypes.c_uint64),
    ("filesize", ctypes.c_uint64),
    ("maxprot", ctypes.c_int),
    ("initprot", ctypes.c_int),
    ("nsects", ctypes.c_uint32),
    ("flags", ctypes.c_uint32),
]
"""
struct segment_command_64 { /* for 64-bit architectures */
    uint32_t	cmd;		/* LC_SEGMENT_64 */
    uint32_t	cmdsize;	/* includes sizeof section_64 structs */
    char		segname[16];	/* segment name */
    uint64_t	vmaddr;		/* memory address of this segment */
    uint64_t	vmsize;		/* memory size of this segment */
    uint64_t	fileoff;	/* file offset of this segment */
    uint64_t	filesize;	/* amount to map from the file */
    vm_prot_t	maxprot;	/* maximum VM protection */
    vm_prot_t	initprot;	/* initial VM protection */
    uint32_t	nsects;		/* number of sections in segment */
    uint32_t	flags;		/* flags */
};
"""

version_min_command_fields = segment_base_fields + [
    ("version", ctypes.c_uint32),
    ("sdk", ctypes.c_uint32),
]
"""
struct version_min_command {
    uint32_t	cmd;		/* LC_VERSION_MIN_MACOSX or
                               LC_VERSION_MIN_IPHONEOS or
                               LC_VERSION_MIN_WATCHOS or
                               LC_VERSION_MIN_TVOS */
    uint32_t	cmdsize;	/* sizeof(struct min_version_command) */
    uint32_t	version;	/* X.Y.Z is encoded in nibbles xxxx.yy.zz */
    uint32_t	sdk;		/* X.Y.Z is encoded in nibbles xxxx.yy.zz */
};
"""

build_version_command_fields = segment_base_fields + [
    ("platform", ctypes.c_uint32),
    ("minos", ctypes.c_uint32),
    ("sdk", ctypes.c_uint32),
    ("ntools", ctypes.c_uint32),
]
"""
struct build_version_command {
    uint32_t	cmd;		/* LC_BUILD_VERSION */
    uint32_t	cmdsize;	/* sizeof(struct build_version_command) plus */
                                /* ntools * sizeof(struct build_tool_version) */
    uint32_t	platform;	/* platform */
    uint32_t	minos;		/* X.Y.Z is encoded in nibbles xxxx.yy.zz */
    uint32_t	sdk;		/* X.Y.Z is encoded in nibbles xxxx.yy.zz */
    uint32_t	ntools;		/* number of tool entries following this */
};
"""


def swap32(x: int) -> int:
    return (
        ((x << 24) & 0xFF000000)
        | ((x << 8) & 0x00FF0000)
        | ((x >> 8) & 0x0000FF00)
        | ((x >> 24) & 0x000000FF)
    )


def get_base_class_and_magic_number(
    lib_file: BufferedIOBase,
    seek: int | None = None,
) -> tuple[type[ctypes.Structure], int]:
    if seek is None:
        seek = lib_file.tell()
    else:
        lib_file.seek(seek)
    magic_number = ctypes.c_uint32.from_buffer_copy(
        lib_file.read(ctypes.sizeof(ctypes.c_uint32))
    ).value

    # Handle wrong byte order
    if magic_number in [FAT_CIGAM, FAT_CIGAM_64, MH_CIGAM, MH_CIGAM_64]:
        if sys.byteorder == "little":
            BaseClass = ctypes.BigEndianStructure
        else:
            BaseClass = ctypes.LittleEndianStructure

        magic_number = swap32(magic_number)
    else:
        BaseClass = ctypes.Structure

    lib_file.seek(seek)
    return BaseClass, magic_number


def read_data(struct_class: type[ctypes.Structure], lib_file: BufferedIOBase):
    return struct_class.from_buffer_copy(lib_file.read(ctypes.sizeof(struct_class)))


def extract_macosx_min_system_version(path_to_lib: str):
    with open(path_to_lib, "rb") as lib_file:
        BaseClass, magic_number = get_base_class_and_magic_number(lib_file, 0)
        if magic_number not in [FAT_MAGIC, FAT_MAGIC_64, MH_MAGIC, MH_MAGIC_64]:
            return

        if magic_number in [FAT_MAGIC, FAT_CIGAM_64]:

            class FatHeader(BaseClass):
                _fields_ = fat_header_fields

            fat_header = read_data(FatHeader, lib_file)
            if magic_number == FAT_MAGIC:

                class FatArch(BaseClass):
                    _fields_ = fat_arch_fields

            else:

                class FatArch(BaseClass):
                    _fields_ = fat_arch_64_fields

            fat_arch_list = [
                read_data(FatArch, lib_file) for _ in range(fat_header.nfat_arch)
            ]

            versions_list: list[tuple[int, int, int]] = []
            for el in fat_arch_list:
                try:
                    version = read_mach_header(lib_file, el.offset)
                    if version is not None:
                        if el.cputype == CPU_TYPE_ARM64 and len(fat_arch_list) != 1:
                            # Xcode will not set the deployment target below 11.0.0
                            # for the arm64 architecture. Ignore the arm64 deployment
                            # in fat binaries when the target is 11.0.0, that way
                            # the other architectures can select a lower deployment
                            # target.
                            # This is safe because there is no arm64 variant for
                            # macOS 10.15 or earlier.
                            if version == (11, 0, 0):
                                continue
                        versions_list.append(version)
                except ValueError:
                    pass

            if len(versions_list) > 0:
                return max(versions_list)
            else:
                return None

        else:
            try:
                return read_mach_header(lib_file, 0)
            except ValueError:
                """when some error during read library files"""
                return None


def read_mach_header(
    lib_file: BufferedIOBase,
    seek: int | None = None,
) -> tuple[int, int, int] | None:
    """
    This function parses a Mach-O header and extracts
    information about the minimal macOS version.

    :param lib_file: reference to opened library file with pointer
    """
    base_class, magic_number = get_base_class_and_magic_number(lib_file, seek)
    arch = "32" if magic_number == MH_MAGIC else "64"

    class SegmentBase(base_class):
        _fields_ = segment_base_fields

    if arch == "32":

        class MachHeader(base_class):
            _fields_ = mach_header_fields

    else:

        class MachHeader(base_class):
            _fields_ = mach_header_fields_64

    mach_header = read_data(MachHeader, lib_file)
    for _i in range(mach_header.ncmds):
        pos = lib_file.tell()
        segment_base = read_data(SegmentBase, lib_file)
        lib_file.seek(pos)
        if segment_base.cmd == LC_VERSION_MIN_MACOSX:

            class VersionMinCommand(base_class):
                _fields_ = version_min_command_fields

            version_info = read_data(VersionMinCommand, lib_file)
            return parse_version(version_info.version)
        elif segment_base.cmd == LC_BUILD_VERSION:

            class VersionBuild(base_class):
                _fields_ = build_version_command_fields

            version_info = read_data(VersionBuild, lib_file)
            return parse_version(version_info.minos)
        else:
            lib_file.seek(pos + segment_base.cmdsize)
            continue


def parse_version(version: int) -> tuple[int, int, int]:
    x = (version & 0xFFFF0000) >> 16
    y = (version & 0x0000FF00) >> 8
    z = version & 0x000000FF
    return x, y, z


def calculate_macosx_platform_tag(archive_root: StrPath, platform_tag: str) -> str:
    """
    Calculate proper macosx platform tag basing on files which are included to wheel

    Example platform tag `macosx-10.14-x86_64`
    """
    prefix, base_version, suffix = platform_tag.split("-")
    base_version = tuple(int(x) for x in base_version.split("."))
    base_version = base_version[:2]
    if base_version[0] > 10:
        base_version = (base_version[0], 0)
    assert len(base_version) == 2
    if "MACOSX_DEPLOYMENT_TARGET" in os.environ:
        deploy_target = tuple(
            int(x) for x in os.environ["MACOSX_DEPLOYMENT_TARGET"].split(".")
        )
        deploy_target = deploy_target[:2]
        if deploy_target[0] > 10:
            deploy_target = (deploy_target[0], 0)
        if deploy_target < base_version:
            sys.stderr.write(
                "[WARNING] MACOSX_DEPLOYMENT_TARGET is set to a lower value ({}) than "
                "the version on which the Python interpreter was compiled ({}), and "
                "will be ignored.\n".format(
                    ".".join(str(x) for x in deploy_target),
                    ".".join(str(x) for x in base_version),
                )
            )
        else:
            base_version = deploy_target

    assert len(base_version) == 2
    start_version = base_version
    versions_dict: dict[str, tuple[int, int]] = {}
    for dirpath, _dirnames, filenames in os.walk(archive_root):
        for filename in filenames:
            if filename.endswith(".dylib") or filename.endswith(".so"):
                lib_path = os.path.join(dirpath, filename)
                min_ver = extract_macosx_min_system_version(lib_path)
                if min_ver is not None:
                    min_ver = min_ver[0:2]
                    if min_ver[0] > 10:
                        min_ver = (min_ver[0], 0)
                    versions_dict[lib_path] = min_ver

    if len(versions_dict) > 0:
        base_version = max(base_version, max(versions_dict.values()))

    # macosx platform tag do not support minor bugfix release
    fin_base_version = "_".join([str(x) for x in base_version])
    if start_version < base_version:
        problematic_files = [k for k, v in versions_dict.items() if v > start_version]
        problematic_files = "\n".join(problematic_files)
        if len(problematic_files) == 1:
            files_form = "this file"
        else:
            files_form = "these files"
        error_message = (
            "[WARNING] This wheel needs a higher macOS version than {}  "
            "To silence this warning, set MACOSX_DEPLOYMENT_TARGET to at least "
            + fin_base_version
            + " or recreate "
            + files_form
            + " with lower "
            "MACOSX_DEPLOYMENT_TARGET:  \n" + problematic_files
        )

        if "MACOSX_DEPLOYMENT_TARGET" in os.environ:
            error_message = error_message.format(
                "is set in MACOSX_DEPLOYMENT_TARGET variable."
            )
        else:
            error_message = error_message.format(
                "the version your Python interpreter is compiled against."
            )

        sys.stderr.write(error_message)

    platform_tag = prefix + "_" + fin_base_version + "_" + suffix
    return platform_tag
