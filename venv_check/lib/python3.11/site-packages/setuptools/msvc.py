"""
Environment info about Microsoft Compilers.

>>> getfixture('windows_only')
>>> ei = EnvironmentInfo('amd64')
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import os.path
import platform
from typing import TYPE_CHECKING, TypedDict

from more_itertools import unique_everseen

import distutils.errors

if TYPE_CHECKING:
    from typing_extensions import LiteralString, NotRequired

# https://github.com/python/mypy/issues/8166
if not TYPE_CHECKING and platform.system() == 'Windows':
    import winreg
    from os import environ
else:
    # Mock winreg and environ so the module can be imported on this platform.

    class winreg:
        HKEY_USERS = None
        HKEY_CURRENT_USER = None
        HKEY_LOCAL_MACHINE = None
        HKEY_CLASSES_ROOT = None

    environ: dict[str, str] = dict()


class PlatformInfo:
    """
    Current and Target Architectures information.

    Parameters
    ----------
    arch: str
        Target architecture.
    """

    current_cpu = environ.get('processor_architecture', '').lower()

    def __init__(self, arch) -> None:
        self.arch = arch.lower().replace('x64', 'amd64')

    @property
    def target_cpu(self):
        """
        Return Target CPU architecture.

        Return
        ------
        str
            Target CPU
        """
        return self.arch[self.arch.find('_') + 1 :]

    def target_is_x86(self):
        """
        Return True if target CPU is x86 32 bits..

        Return
        ------
        bool
            CPU is x86 32 bits
        """
        return self.target_cpu == 'x86'

    def current_is_x86(self):
        """
        Return True if current CPU is x86 32 bits..

        Return
        ------
        bool
            CPU is x86 32 bits
        """
        return self.current_cpu == 'x86'

    def current_dir(self, hidex86=False, x64=False) -> str:
        """
        Current platform specific subfolder.

        Parameters
        ----------
        hidex86: bool
            return '' and not '\x86' if architecture is x86.
        x64: bool
            return '\x64' and not '\amd64' if architecture is amd64.

        Return
        ------
        str
            subfolder: '\target', or '' (see hidex86 parameter)
        """
        return (
            ''
            if (self.current_cpu == 'x86' and hidex86)
            else r'\x64'
            if (self.current_cpu == 'amd64' and x64)
            else rf'\{self.current_cpu}'
        )

    def target_dir(self, hidex86=False, x64=False) -> str:
        r"""
        Target platform specific subfolder.

        Parameters
        ----------
        hidex86: bool
            return '' and not '\x86' if architecture is x86.
        x64: bool
            return '\x64' and not '\amd64' if architecture is amd64.

        Return
        ------
        str
            subfolder: '\current', or '' (see hidex86 parameter)
        """
        return (
            ''
            if (self.target_cpu == 'x86' and hidex86)
            else r'\x64'
            if (self.target_cpu == 'amd64' and x64)
            else rf'\{self.target_cpu}'
        )

    def cross_dir(self, forcex86=False):
        r"""
        Cross platform specific subfolder.

        Parameters
        ----------
        forcex86: bool
            Use 'x86' as current architecture even if current architecture is
            not x86.

        Return
        ------
        str
            subfolder: '' if target architecture is current architecture,
            '\current_target' if not.
        """
        current = 'x86' if forcex86 else self.current_cpu
        return (
            ''
            if self.target_cpu == current
            else self.target_dir().replace('\\', f'\\{current}_')
        )


class RegistryInfo:
    """
    Microsoft Visual Studio related registry information.

    Parameters
    ----------
    platform_info: PlatformInfo
        "PlatformInfo" instance.
    """

    HKEYS = (
        winreg.HKEY_USERS,
        winreg.HKEY_CURRENT_USER,
        winreg.HKEY_LOCAL_MACHINE,
        winreg.HKEY_CLASSES_ROOT,
    )

    def __init__(self, platform_info) -> None:
        self.pi = platform_info

    @property
    def visualstudio(self) -> str:
        """
        Microsoft Visual Studio root registry key.

        Return
        ------
        str
            Registry key
        """
        return 'VisualStudio'

    @property
    def sxs(self):
        """
        Microsoft Visual Studio SxS registry key.

        Return
        ------
        str
            Registry key
        """
        return os.path.join(self.visualstudio, 'SxS')

    @property
    def vc(self):
        """
        Microsoft Visual C++ VC7 registry key.

        Return
        ------
        str
            Registry key
        """
        return os.path.join(self.sxs, 'VC7')

    @property
    def vs(self):
        """
        Microsoft Visual Studio VS7 registry key.

        Return
        ------
        str
            Registry key
        """
        return os.path.join(self.sxs, 'VS7')

    @property
    def vc_for_python(self) -> str:
        """
        Microsoft Visual C++ for Python registry key.

        Return
        ------
        str
            Registry key
        """
        return r'DevDiv\VCForPython'

    @property
    def microsoft_sdk(self) -> str:
        """
        Microsoft SDK registry key.

        Return
        ------
        str
            Registry key
        """
        return 'Microsoft SDKs'

    @property
    def windows_sdk(self):
        """
        Microsoft Windows/Platform SDK registry key.

        Return
        ------
        str
            Registry key
        """
        return os.path.join(self.microsoft_sdk, 'Windows')

    @property
    def netfx_sdk(self):
        """
        Microsoft .NET Framework SDK registry key.

        Return
        ------
        str
            Registry key
        """
        return os.path.join(self.microsoft_sdk, 'NETFXSDK')

    @property
    def windows_kits_roots(self) -> str:
        """
        Microsoft Windows Kits Roots registry key.

        Return
        ------
        str
            Registry key
        """
        return r'Windows Kits\Installed Roots'

    def microsoft(self, key, x86=False):
        """
        Return key in Microsoft software registry.

        Parameters
        ----------
        key: str
            Registry key path where look.
        x86: str
            Force x86 software registry.

        Return
        ------
        str
            Registry key
        """
        node64 = '' if self.pi.current_is_x86() or x86 else 'Wow6432Node'
        return os.path.join('Software', node64, 'Microsoft', key)

    def lookup(self, key, name):
        """
        Look for values in registry in Microsoft software registry.

        Parameters
        ----------
        key: str
            Registry key path where look.
        name: str
            Value name to find.

        Return
        ------
        str
            value
        """
        key_read = winreg.KEY_READ
        openkey = winreg.OpenKey
        closekey = winreg.CloseKey
        ms = self.microsoft
        for hkey in self.HKEYS:
            bkey = None
            try:
                bkey = openkey(hkey, ms(key), 0, key_read)
            except OSError:
                if not self.pi.current_is_x86():
                    try:
                        bkey = openkey(hkey, ms(key, True), 0, key_read)
                    except OSError:
                        continue
                else:
                    continue
            try:
                return winreg.QueryValueEx(bkey, name)[0]
            except OSError:
                pass
            finally:
                if bkey:
                    closekey(bkey)
        return None


class SystemInfo:
    """
    Microsoft Windows and Visual Studio related system information.

    Parameters
    ----------
    registry_info: RegistryInfo
        "RegistryInfo" instance.
    vc_ver: float
        Required Microsoft Visual C++ version.
    """

    # Variables and properties in this class use originals CamelCase variables
    # names from Microsoft source files for more easy comparison.
    WinDir = environ.get('WinDir', '')
    ProgramFiles = environ.get('ProgramFiles', '')
    ProgramFilesx86 = environ.get('ProgramFiles(x86)', ProgramFiles)

    def __init__(self, registry_info, vc_ver=None) -> None:
        self.ri = registry_info
        self.pi = self.ri.pi

        self.known_vs_paths = self.find_programdata_vs_vers()

        # Except for VS15+, VC version is aligned with VS version
        self.vs_ver = self.vc_ver = vc_ver or self._find_latest_available_vs_ver()

    def _find_latest_available_vs_ver(self):
        """
        Find the latest VC version

        Return
        ------
        float
            version
        """
        reg_vc_vers = self.find_reg_vs_vers()

        if not (reg_vc_vers or self.known_vs_paths):
            raise distutils.errors.DistutilsPlatformError(
                'No Microsoft Visual C++ version found'
            )

        vc_vers = set(reg_vc_vers)
        vc_vers.update(self.known_vs_paths)
        return sorted(vc_vers)[-1]

    def find_reg_vs_vers(self):
        """
        Find Microsoft Visual Studio versions available in registry.

        Return
        ------
        list of float
            Versions
        """
        ms = self.ri.microsoft
        vckeys = (self.ri.vc, self.ri.vc_for_python, self.ri.vs)
        vs_vers = []
        for hkey, key in itertools.product(self.ri.HKEYS, vckeys):
            try:
                bkey = winreg.OpenKey(hkey, ms(key), 0, winreg.KEY_READ)
            except OSError:
                continue
            with bkey:
                subkeys, values, _ = winreg.QueryInfoKey(bkey)
                for i in range(values):
                    with contextlib.suppress(ValueError):
                        ver = float(winreg.EnumValue(bkey, i)[0])
                        if ver not in vs_vers:
                            vs_vers.append(ver)
                for i in range(subkeys):
                    with contextlib.suppress(ValueError):
                        ver = float(winreg.EnumKey(bkey, i))
                        if ver not in vs_vers:
                            vs_vers.append(ver)
        return sorted(vs_vers)

    def find_programdata_vs_vers(self) -> dict[float, str]:
        r"""
        Find Visual studio 2017+ versions from information in
        "C:\ProgramData\Microsoft\VisualStudio\Packages\_Instances".

        Return
        ------
        dict
            float version as key, path as value.
        """
        vs_versions: dict[float, str] = {}
        instances_dir = r'C:\ProgramData\Microsoft\VisualStudio\Packages\_Instances'

        try:
            hashed_names = os.listdir(instances_dir)

        except OSError:
            # Directory not exists with all Visual Studio versions
            return vs_versions

        for name in hashed_names:
            try:
                # Get VS installation path from "state.json" file
                state_path = os.path.join(instances_dir, name, 'state.json')
                with open(state_path, 'rt', encoding='utf-8') as state_file:
                    state = json.load(state_file)
                vs_path = state['installationPath']

                # Raises OSError if this VS installation does not contain VC
                os.listdir(os.path.join(vs_path, r'VC\Tools\MSVC'))

                # Store version and path
                vs_versions[self._as_float_version(state['installationVersion'])] = (
                    vs_path
                )

            except (OSError, KeyError):
                # Skip if "state.json" file is missing or bad format
                continue

        return vs_versions

    @staticmethod
    def _as_float_version(version):
        """
        Return a string version as a simplified float version (major.minor)

        Parameters
        ----------
        version: str
            Version.

        Return
        ------
        float
            version
        """
        return float('.'.join(version.split('.')[:2]))

    @property
    def VSInstallDir(self):
        """
        Microsoft Visual Studio directory.

        Return
        ------
        str
            path
        """
        # Default path
        default = os.path.join(
            self.ProgramFilesx86, f'Microsoft Visual Studio {self.vs_ver:0.1f}'
        )

        # Try to get path from registry, if fail use default path
        return self.ri.lookup(self.ri.vs, f'{self.vs_ver:0.1f}') or default

    @property
    def VCInstallDir(self):
        """
        Microsoft Visual C++ directory.

        Return
        ------
        str
            path
        """
        path = self._guess_vc() or self._guess_vc_legacy()

        if not os.path.isdir(path):
            msg = 'Microsoft Visual C++ directory not found'
            raise distutils.errors.DistutilsPlatformError(msg)

        return path

    def _guess_vc(self):
        """
        Locate Visual C++ for VS2017+.

        Return
        ------
        str
            path
        """
        if self.vs_ver <= 14.0:
            return ''

        try:
            # First search in known VS paths
            vs_dir = self.known_vs_paths[self.vs_ver]
        except KeyError:
            # Else, search with path from registry
            vs_dir = self.VSInstallDir

        guess_vc = os.path.join(vs_dir, r'VC\Tools\MSVC')

        # Subdir with VC exact version as name
        try:
            # Update the VC version with real one instead of VS version
            vc_ver = os.listdir(guess_vc)[-1]
            self.vc_ver = self._as_float_version(vc_ver)
            return os.path.join(guess_vc, vc_ver)
        except (OSError, IndexError):
            return ''

    def _guess_vc_legacy(self):
        """
        Locate Visual C++ for versions prior to 2017.

        Return
        ------
        str
            path
        """
        default = os.path.join(
            self.ProgramFilesx86,
            rf'Microsoft Visual Studio {self.vs_ver:0.1f}\VC',
        )

        # Try to get "VC++ for Python" path from registry as default path
        reg_path = os.path.join(self.ri.vc_for_python, f'{self.vs_ver:0.1f}')
        python_vc = self.ri.lookup(reg_path, 'installdir')
        default_vc = os.path.join(python_vc, 'VC') if python_vc else default

        # Try to get path from registry, if fail use default path
        return self.ri.lookup(self.ri.vc, f'{self.vs_ver:0.1f}') or default_vc

    @property
    def WindowsSdkVersion(self) -> tuple[LiteralString, ...]:
        """
        Microsoft Windows SDK versions for specified MSVC++ version.

        Return
        ------
        tuple of str
            versions
        """
        if self.vs_ver <= 9.0:
            return '7.0', '6.1', '6.0a'
        elif self.vs_ver == 10.0:
            return '7.1', '7.0a'
        elif self.vs_ver == 11.0:
            return '8.0', '8.0a'
        elif self.vs_ver == 12.0:
            return '8.1', '8.1a'
        elif self.vs_ver >= 14.0:
            return '10.0', '8.1'
        return ()

    @property
    def WindowsSdkLastVersion(self):
        """
        Microsoft Windows SDK last version.

        Return
        ------
        str
            version
        """
        return self._use_last_dir_name(os.path.join(self.WindowsSdkDir, 'lib'))

    @property
    def WindowsSdkDir(self) -> str | None:  # noqa: C901  # is too complex (12)  # FIXME
        """
        Microsoft Windows SDK directory.

        Return
        ------
        str
            path
        """
        sdkdir: str | None = ''
        for ver in self.WindowsSdkVersion:
            # Try to get it from registry
            loc = os.path.join(self.ri.windows_sdk, f'v{ver}')
            sdkdir = self.ri.lookup(loc, 'installationfolder')
            if sdkdir:
                break
        if not sdkdir or not os.path.isdir(sdkdir):
            # Try to get "VC++ for Python" version from registry
            path = os.path.join(self.ri.vc_for_python, f'{self.vc_ver:0.1f}')
            install_base = self.ri.lookup(path, 'installdir')
            if install_base:
                sdkdir = os.path.join(install_base, 'WinSDK')
        if not sdkdir or not os.path.isdir(sdkdir):
            # If fail, use default new path
            for ver in self.WindowsSdkVersion:
                intver = ver[: ver.rfind('.')]
                path = rf'Microsoft SDKs\Windows Kits\{intver}'
                d = os.path.join(self.ProgramFiles, path)
                if os.path.isdir(d):
                    sdkdir = d
        if not sdkdir or not os.path.isdir(sdkdir):
            # If fail, use default old path
            for ver in self.WindowsSdkVersion:
                path = rf'Microsoft SDKs\Windows\v{ver}'
                d = os.path.join(self.ProgramFiles, path)
                if os.path.isdir(d):
                    sdkdir = d
        if not sdkdir:
            # If fail, use Platform SDK
            sdkdir = os.path.join(self.VCInstallDir, 'PlatformSDK')
        return sdkdir

    @property
    def WindowsSDKExecutablePath(self):
        """
        Microsoft Windows SDK executable directory.

        Return
        ------
        str
            path
        """
        # Find WinSDK NetFx Tools registry dir name
        if self.vs_ver <= 11.0:
            netfxver = 35
            arch = ''
        else:
            netfxver = 40
            hidex86 = True if self.vs_ver <= 12.0 else False
            arch = self.pi.current_dir(x64=True, hidex86=hidex86).replace('\\', '-')
        fx = f'WinSDK-NetFx{netfxver}Tools{arch}'

        # list all possibles registry paths
        regpaths = []
        if self.vs_ver >= 14.0:
            for ver in self.NetFxSdkVersion:
                regpaths += [os.path.join(self.ri.netfx_sdk, ver, fx)]

        for ver in self.WindowsSdkVersion:
            regpaths += [os.path.join(self.ri.windows_sdk, f'v{ver}A', fx)]

        # Return installation folder from the more recent path
        for path in regpaths:
            execpath = self.ri.lookup(path, 'installationfolder')
            if execpath:
                return execpath

        return None

    @property
    def FSharpInstallDir(self):
        """
        Microsoft Visual F# directory.

        Return
        ------
        str
            path
        """
        path = os.path.join(self.ri.visualstudio, rf'{self.vs_ver:0.1f}\Setup\F#')
        return self.ri.lookup(path, 'productdir') or ''

    @property
    def UniversalCRTSdkDir(self):
        """
        Microsoft Universal CRT SDK directory.

        Return
        ------
        str
            path
        """
        # Set Kit Roots versions for specified MSVC++ version
        vers = ('10', '81') if self.vs_ver >= 14.0 else ()

        # Find path of the more recent Kit
        for ver in vers:
            sdkdir = self.ri.lookup(self.ri.windows_kits_roots, f'kitsroot{ver}')
            if sdkdir:
                return sdkdir or ''

        return None

    @property
    def UniversalCRTSdkLastVersion(self):
        """
        Microsoft Universal C Runtime SDK last version.

        Return
        ------
        str
            version
        """
        return self._use_last_dir_name(os.path.join(self.UniversalCRTSdkDir, 'lib'))

    @property
    def NetFxSdkVersion(self):
        """
        Microsoft .NET Framework SDK versions.

        Return
        ------
        tuple of str
            versions
        """
        # Set FxSdk versions for specified VS version
        return (
            ('4.7.2', '4.7.1', '4.7', '4.6.2', '4.6.1', '4.6', '4.5.2', '4.5.1', '4.5')
            if self.vs_ver >= 14.0
            else ()
        )

    @property
    def NetFxSdkDir(self):
        """
        Microsoft .NET Framework SDK directory.

        Return
        ------
        str
            path
        """
        sdkdir = ''
        for ver in self.NetFxSdkVersion:
            loc = os.path.join(self.ri.netfx_sdk, ver)
            sdkdir = self.ri.lookup(loc, 'kitsinstallationfolder')
            if sdkdir:
                break
        return sdkdir

    @property
    def FrameworkDir32(self):
        """
        Microsoft .NET Framework 32bit directory.

        Return
        ------
        str
            path
        """
        # Default path
        guess_fw = os.path.join(self.WinDir, r'Microsoft.NET\Framework')

        # Try to get path from registry, if fail use default path
        return self.ri.lookup(self.ri.vc, 'frameworkdir32') or guess_fw

    @property
    def FrameworkDir64(self):
        """
        Microsoft .NET Framework 64bit directory.

        Return
        ------
        str
            path
        """
        # Default path
        guess_fw = os.path.join(self.WinDir, r'Microsoft.NET\Framework64')

        # Try to get path from registry, if fail use default path
        return self.ri.lookup(self.ri.vc, 'frameworkdir64') or guess_fw

    @property
    def FrameworkVersion32(self) -> tuple[str, ...]:
        """
        Microsoft .NET Framework 32bit versions.

        Return
        ------
        tuple of str
            versions
        """
        return self._find_dot_net_versions(32)

    @property
    def FrameworkVersion64(self) -> tuple[str, ...]:
        """
        Microsoft .NET Framework 64bit versions.

        Return
        ------
        tuple of str
            versions
        """
        return self._find_dot_net_versions(64)

    def _find_dot_net_versions(self, bits) -> tuple[str, ...]:
        """
        Find Microsoft .NET Framework versions.

        Parameters
        ----------
        bits: int
            Platform number of bits: 32 or 64.

        Return
        ------
        tuple of str
            versions
        """
        # Find actual .NET version in registry
        reg_ver = self.ri.lookup(self.ri.vc, f'frameworkver{bits}')
        dot_net_dir = getattr(self, f'FrameworkDir{bits}')
        ver = reg_ver or self._use_last_dir_name(dot_net_dir, 'v') or ''

        # Set .NET versions for specified MSVC++ version
        if self.vs_ver >= 12.0:
            return ver, 'v4.0'
        elif self.vs_ver >= 10.0:
            return 'v4.0.30319' if ver.lower()[:2] != 'v4' else ver, 'v3.5'
        elif self.vs_ver == 9.0:
            return 'v3.5', 'v2.0.50727'
        elif self.vs_ver == 8.0:
            return 'v3.0', 'v2.0.50727'
        return ()

    @staticmethod
    def _use_last_dir_name(path, prefix=''):
        """
        Return name of the last dir in path or '' if no dir found.

        Parameters
        ----------
        path: str
            Use dirs in this path
        prefix: str
            Use only dirs starting by this prefix

        Return
        ------
        str
            name
        """
        matching_dirs = (
            dir_name
            for dir_name in reversed(os.listdir(path))
            if os.path.isdir(os.path.join(path, dir_name))
            and dir_name.startswith(prefix)
        )
        return next(matching_dirs, None) or ''


class _EnvironmentDict(TypedDict):
    include: str
    lib: str
    libpath: str
    path: str
    py_vcruntime_redist: NotRequired[str | None]


class EnvironmentInfo:
    """
    Return environment variables for specified Microsoft Visual C++ version
    and platform : Lib, Include, Path and libpath.

    This function is compatible with Microsoft Visual C++ 9.0 to 14.X.

    Script created by analysing Microsoft environment configuration files like
    "vcvars[...].bat", "SetEnv.Cmd", "vcbuildtools.bat", ...

    Parameters
    ----------
    arch: str
        Target architecture.
    vc_ver: float
        Required Microsoft Visual C++ version. If not set, autodetect the last
        version.
    vc_min_ver: float
        Minimum Microsoft Visual C++ version.
    """

    # Variables and properties in this class use originals CamelCase variables
    # names from Microsoft source files for more easy comparison.

    def __init__(self, arch, vc_ver=None, vc_min_ver=0) -> None:
        self.pi = PlatformInfo(arch)
        self.ri = RegistryInfo(self.pi)
        self.si = SystemInfo(self.ri, vc_ver)

        if self.vc_ver < vc_min_ver:
            err = 'No suitable Microsoft Visual C++ version found'
            raise distutils.errors.DistutilsPlatformError(err)

    @property
    def vs_ver(self):
        """
        Microsoft Visual Studio.

        Return
        ------
        float
            version
        """
        return self.si.vs_ver

    @property
    def vc_ver(self):
        """
        Microsoft Visual C++ version.

        Return
        ------
        float
            version
        """
        return self.si.vc_ver

    @property
    def VSTools(self):
        """
        Microsoft Visual Studio Tools.

        Return
        ------
        list of str
            paths
        """
        paths = [r'Common7\IDE', r'Common7\Tools']

        if self.vs_ver >= 14.0:
            arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
            paths += [r'Common7\IDE\CommonExtensions\Microsoft\TestWindow']
            paths += [r'Team Tools\Performance Tools']
            paths += [rf'Team Tools\Performance Tools{arch_subdir}']

        return [os.path.join(self.si.VSInstallDir, path) for path in paths]

    @property
    def VCIncludes(self):
        """
        Microsoft Visual C++ & Microsoft Foundation Class Includes.

        Return
        ------
        list of str
            paths
        """
        return [
            os.path.join(self.si.VCInstallDir, 'Include'),
            os.path.join(self.si.VCInstallDir, r'ATLMFC\Include'),
        ]

    @property
    def VCLibraries(self):
        """
        Microsoft Visual C++ & Microsoft Foundation Class Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver >= 15.0:
            arch_subdir = self.pi.target_dir(x64=True)
        else:
            arch_subdir = self.pi.target_dir(hidex86=True)
        paths = [f'Lib{arch_subdir}', rf'ATLMFC\Lib{arch_subdir}']

        if self.vs_ver >= 14.0:
            paths += [rf'Lib\store{arch_subdir}']

        return [os.path.join(self.si.VCInstallDir, path) for path in paths]

    @property
    def VCStoreRefs(self):
        """
        Microsoft Visual C++ store references Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0:
            return []
        return [os.path.join(self.si.VCInstallDir, r'Lib\store\references')]

    @property
    def VCTools(self):
        """
        Microsoft Visual C++ Tools.

        Return
        ------
        list of str
            paths

        When host CPU is ARM, the tools should be found for ARM.

        >>> getfixture('windows_only')
        >>> mp = getfixture('monkeypatch')
        >>> mp.setattr(PlatformInfo, 'current_cpu', 'arm64')
        >>> ei = EnvironmentInfo(arch='irrelevant')
        >>> paths = ei.VCTools
        >>> any('HostARM64' in path for path in paths)
        True
        """
        si = self.si
        tools = [os.path.join(si.VCInstallDir, 'VCPackages')]

        forcex86 = True if self.vs_ver <= 10.0 else False
        arch_subdir = self.pi.cross_dir(forcex86)
        if arch_subdir:
            tools += [os.path.join(si.VCInstallDir, f'Bin{arch_subdir}')]

        if self.vs_ver == 14.0:
            path = f'Bin{self.pi.current_dir(hidex86=True)}'
            tools += [os.path.join(si.VCInstallDir, path)]

        elif self.vs_ver >= 15.0:
            host_id = self.pi.current_cpu.replace('amd64', 'x64').upper()
            host_dir = os.path.join('bin', f'Host{host_id}%s')
            tools += [
                os.path.join(si.VCInstallDir, host_dir % self.pi.target_dir(x64=True))
            ]

            if self.pi.current_cpu != self.pi.target_cpu:
                tools += [
                    os.path.join(
                        si.VCInstallDir, host_dir % self.pi.current_dir(x64=True)
                    )
                ]

        else:
            tools += [os.path.join(si.VCInstallDir, 'Bin')]

        return tools

    @property
    def OSLibraries(self):
        """
        Microsoft Windows SDK Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver <= 10.0:
            arch_subdir = self.pi.target_dir(hidex86=True, x64=True)
            return [os.path.join(self.si.WindowsSdkDir, f'Lib{arch_subdir}')]

        else:
            arch_subdir = self.pi.target_dir(x64=True)
            lib = os.path.join(self.si.WindowsSdkDir, 'lib')
            libver = self._sdk_subdir
            return [os.path.join(lib, f'{libver}um{arch_subdir}')]

    @property
    def OSIncludes(self):
        """
        Microsoft Windows SDK Include.

        Return
        ------
        list of str
            paths
        """
        include = os.path.join(self.si.WindowsSdkDir, 'include')

        if self.vs_ver <= 10.0:
            return [include, os.path.join(include, 'gl')]

        else:
            if self.vs_ver >= 14.0:
                sdkver = self._sdk_subdir
            else:
                sdkver = ''
            return [
                os.path.join(include, f'{sdkver}shared'),
                os.path.join(include, f'{sdkver}um'),
                os.path.join(include, f'{sdkver}winrt'),
            ]

    @property
    def OSLibpath(self):
        """
        Microsoft Windows SDK Libraries Paths.

        Return
        ------
        list of str
            paths
        """
        ref = os.path.join(self.si.WindowsSdkDir, 'References')
        libpath = []

        if self.vs_ver <= 9.0:
            libpath += self.OSLibraries

        if self.vs_ver >= 11.0:
            libpath += [os.path.join(ref, r'CommonConfiguration\Neutral')]

        if self.vs_ver >= 14.0:
            libpath += [
                ref,
                os.path.join(self.si.WindowsSdkDir, 'UnionMetadata'),
                os.path.join(ref, 'Windows.Foundation.UniversalApiContract', '1.0.0.0'),
                os.path.join(ref, 'Windows.Foundation.FoundationContract', '1.0.0.0'),
                os.path.join(
                    ref, 'Windows.Networking.Connectivity.WwanContract', '1.0.0.0'
                ),
                os.path.join(
                    self.si.WindowsSdkDir,
                    'ExtensionSDKs',
                    'Microsoft.VCLibs',
                    f'{self.vs_ver:0.1f}',
                    'References',
                    'CommonConfiguration',
                    'neutral',
                ),
            ]
        return libpath

    @property
    def SdkTools(self):
        """
        Microsoft Windows SDK Tools.

        Return
        ------
        list of str
            paths
        """
        return list(self._sdk_tools())

    def _sdk_tools(self):
        """
        Microsoft Windows SDK Tools paths generator.

        Return
        ------
        generator of str
            paths
        """
        if self.vs_ver < 15.0:
            bin_dir = 'Bin' if self.vs_ver <= 11.0 else r'Bin\x86'
            yield os.path.join(self.si.WindowsSdkDir, bin_dir)

        if not self.pi.current_is_x86():
            arch_subdir = self.pi.current_dir(x64=True)
            path = f'Bin{arch_subdir}'
            yield os.path.join(self.si.WindowsSdkDir, path)

        if self.vs_ver in (10.0, 11.0):
            if self.pi.target_is_x86():
                arch_subdir = ''
            else:
                arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
            path = rf'Bin\NETFX 4.0 Tools{arch_subdir}'
            yield os.path.join(self.si.WindowsSdkDir, path)

        elif self.vs_ver >= 15.0:
            path = os.path.join(self.si.WindowsSdkDir, 'Bin')
            arch_subdir = self.pi.current_dir(x64=True)
            sdkver = self.si.WindowsSdkLastVersion
            yield os.path.join(path, f'{sdkver}{arch_subdir}')

        if self.si.WindowsSDKExecutablePath:
            yield self.si.WindowsSDKExecutablePath

    @property
    def _sdk_subdir(self):
        """
        Microsoft Windows SDK version subdir.

        Return
        ------
        str
            subdir
        """
        ucrtver = self.si.WindowsSdkLastVersion
        return (f'{ucrtver}\\') if ucrtver else ''

    @property
    def SdkSetup(self):
        """
        Microsoft Windows SDK Setup.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver > 9.0:
            return []

        return [os.path.join(self.si.WindowsSdkDir, 'Setup')]

    @property
    def FxTools(self):
        """
        Microsoft .NET Framework Tools.

        Return
        ------
        list of str
            paths
        """
        pi = self.pi
        si = self.si

        if self.vs_ver <= 10.0:
            include32 = True
            include64 = not pi.target_is_x86() and not pi.current_is_x86()
        else:
            include32 = pi.target_is_x86() or pi.current_is_x86()
            include64 = pi.current_cpu == 'amd64' or pi.target_cpu == 'amd64'

        tools = []
        if include32:
            tools += [
                os.path.join(si.FrameworkDir32, ver) for ver in si.FrameworkVersion32
            ]
        if include64:
            tools += [
                os.path.join(si.FrameworkDir64, ver) for ver in si.FrameworkVersion64
            ]
        return tools

    @property
    def NetFxSDKLibraries(self):
        """
        Microsoft .Net Framework SDK Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0 or not self.si.NetFxSdkDir:
            return []

        arch_subdir = self.pi.target_dir(x64=True)
        return [os.path.join(self.si.NetFxSdkDir, rf'lib\um{arch_subdir}')]

    @property
    def NetFxSDKIncludes(self):
        """
        Microsoft .Net Framework SDK Includes.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0 or not self.si.NetFxSdkDir:
            return []

        return [os.path.join(self.si.NetFxSdkDir, r'include\um')]

    @property
    def VsTDb(self):
        """
        Microsoft Visual Studio Team System Database.

        Return
        ------
        list of str
            paths
        """
        return [os.path.join(self.si.VSInstallDir, r'VSTSDB\Deploy')]

    @property
    def MSBuild(self):
        """
        Microsoft Build Engine.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 12.0:
            return []
        elif self.vs_ver < 15.0:
            base_path = self.si.ProgramFilesx86
            arch_subdir = self.pi.current_dir(hidex86=True)
        else:
            base_path = self.si.VSInstallDir
            arch_subdir = ''

        path = rf'MSBuild\{self.vs_ver:0.1f}\bin{arch_subdir}'
        build = [os.path.join(base_path, path)]

        if self.vs_ver >= 15.0:
            # Add Roslyn C# & Visual Basic Compiler
            build += [os.path.join(base_path, path, 'Roslyn')]

        return build

    @property
    def HTMLHelpWorkshop(self):
        """
        Microsoft HTML Help Workshop.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 11.0:
            return []

        return [os.path.join(self.si.ProgramFilesx86, 'HTML Help Workshop')]

    @property
    def UCRTLibraries(self):
        """
        Microsoft Universal C Runtime SDK Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0:
            return []

        arch_subdir = self.pi.target_dir(x64=True)
        lib = os.path.join(self.si.UniversalCRTSdkDir, 'lib')
        ucrtver = self._ucrt_subdir
        return [os.path.join(lib, f'{ucrtver}ucrt{arch_subdir}')]

    @property
    def UCRTIncludes(self):
        """
        Microsoft Universal C Runtime SDK Include.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0:
            return []

        include = os.path.join(self.si.UniversalCRTSdkDir, 'include')
        return [os.path.join(include, f'{self._ucrt_subdir}ucrt')]

    @property
    def _ucrt_subdir(self):
        """
        Microsoft Universal C Runtime SDK version subdir.

        Return
        ------
        str
            subdir
        """
        ucrtver = self.si.UniversalCRTSdkLastVersion
        return (f'{ucrtver}\\') if ucrtver else ''

    @property
    def FSharp(self):
        """
        Microsoft Visual F#.

        Return
        ------
        list of str
            paths
        """
        if 11.0 > self.vs_ver > 12.0:
            return []

        return [self.si.FSharpInstallDir]

    @property
    def VCRuntimeRedist(self) -> str | None:
        """
        Microsoft Visual C++ runtime redistributable dll.

        Returns the first suitable path found or None.
        """
        vcruntime = f'vcruntime{self.vc_ver}0.dll'
        arch_subdir = self.pi.target_dir(x64=True).strip('\\')

        # Installation prefixes candidates
        prefixes = []
        tools_path = self.si.VCInstallDir
        redist_path = os.path.dirname(tools_path.replace(r'\Tools', r'\Redist'))
        if os.path.isdir(redist_path):
            # Redist version may not be exactly the same as tools
            redist_path = os.path.join(redist_path, os.listdir(redist_path)[-1])
            prefixes += [redist_path, os.path.join(redist_path, 'onecore')]

        prefixes += [os.path.join(tools_path, 'redist')]  # VS14 legacy path

        # CRT directory
        crt_dirs = (
            f'Microsoft.VC{self.vc_ver * 10}.CRT',
            # Sometime store in directory with VS version instead of VC
            f'Microsoft.VC{int(self.vs_ver) * 10}.CRT',
        )

        # vcruntime path
        candidate_paths = (
            os.path.join(prefix, arch_subdir, crt_dir, vcruntime)
            for (prefix, crt_dir) in itertools.product(prefixes, crt_dirs)
        )
        return next(filter(os.path.isfile, candidate_paths), None)  # type: ignore[arg-type] #python/mypy#12682

    def return_env(self, exists: bool = True) -> _EnvironmentDict:
        """
        Return environment dict.

        Parameters
        ----------
        exists: bool
            It True, only return existing paths.

        Return
        ------
        dict
            environment
        """
        env = _EnvironmentDict(
            include=self._build_paths(
                'include',
                [
                    self.VCIncludes,
                    self.OSIncludes,
                    self.UCRTIncludes,
                    self.NetFxSDKIncludes,
                ],
                exists,
            ),
            lib=self._build_paths(
                'lib',
                [
                    self.VCLibraries,
                    self.OSLibraries,
                    self.FxTools,
                    self.UCRTLibraries,
                    self.NetFxSDKLibraries,
                ],
                exists,
            ),
            libpath=self._build_paths(
                'libpath',
                [self.VCLibraries, self.FxTools, self.VCStoreRefs, self.OSLibpath],
                exists,
            ),
            path=self._build_paths(
                'path',
                [
                    self.VCTools,
                    self.VSTools,
                    self.VsTDb,
                    self.SdkTools,
                    self.SdkSetup,
                    self.FxTools,
                    self.MSBuild,
                    self.HTMLHelpWorkshop,
                    self.FSharp,
                ],
                exists,
            ),
        )
        if self.vs_ver >= 14 and self.VCRuntimeRedist:
            env['py_vcruntime_redist'] = self.VCRuntimeRedist
        return env

    def _build_paths(self, name, spec_path_lists, exists):
        """
        Given an environment variable name and specified paths,
        return a pathsep-separated string of paths containing
        unique, extant, directories from those paths and from
        the environment variable. Raise an error if no paths
        are resolved.

        Parameters
        ----------
        name: str
            Environment variable name
        spec_path_lists: list of str
            Paths
        exists: bool
            It True, only return existing paths.

        Return
        ------
        str
            Pathsep-separated paths
        """
        # flatten spec_path_lists
        spec_paths = itertools.chain.from_iterable(spec_path_lists)
        env_paths = environ.get(name, '').split(os.pathsep)
        paths = itertools.chain(spec_paths, env_paths)
        extant_paths = list(filter(os.path.isdir, paths)) if exists else paths
        if not extant_paths:
            msg = f"{name.upper()} environment variable is empty"
            raise distutils.errors.DistutilsPlatformError(msg)
        unique_paths = unique_everseen(extant_paths)
        return os.pathsep.join(unique_paths)
