from tools.native_cpu_compat import x86_64_local_voice_native_unsupported_reason


def test_x86_64_local_voice_native_unsupported_reason_flags_old_linux_cpu():
    reason = x86_64_local_voice_native_unsupported_reason(
        system="Linux",
        machine="x86_64",
        cpuinfo_text="flags\t: fpu vme de pse tsc msr sse sse2 sse3 ssse3\n",
    )

    assert reason is not None
    assert "sse4_1" in reason
    assert "sse4_2" in reason


def test_x86_64_local_voice_native_unsupported_reason_allows_modern_linux_cpu():
    assert (
        x86_64_local_voice_native_unsupported_reason(
            system="Linux",
            machine="x86_64",
            cpuinfo_text="flags\t: fpu sse sse2 sse4_1 sse4_2 avx\n",
        )
        is None
    )


def test_x86_64_local_voice_native_unsupported_reason_fails_open_without_flags():
    assert (
        x86_64_local_voice_native_unsupported_reason(
            system="Linux",
            machine="x86_64",
            cpuinfo_text="processor\t: 0\n",
        )
        is None
    )


def test_x86_64_local_voice_native_unsupported_reason_ignores_non_linux_hosts():
    assert (
        x86_64_local_voice_native_unsupported_reason(
            system="Darwin",
            machine="x86_64",
            cpuinfo_text="flags\t: fpu sse sse2\n",
        )
        is None
    )
