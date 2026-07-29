"""skills.sh sitemap/detail fetches must stay on skills.sh (SSRF)."""

from tools.skills_hub import SkillsShSource


def test_is_skills_sh_url_accepts_www_and_apex():
    assert SkillsShSource._is_skills_sh_url("https://skills.sh/sitemap-skills-1.xml")
    assert SkillsShSource._is_skills_sh_url("https://www.skills.sh/sitemap-skills-2.xml")


def test_is_skills_sh_url_rejects_foreign_hosts():
    assert not SkillsShSource._is_skills_sh_url(
        "http://169.254.169.254/sitemap-skills.xml"
    )
    assert not SkillsShSource._is_skills_sh_url(
        "https://evil.example/sitemap-skills.xml"
    )
    assert not SkillsShSource._is_skills_sh_url("ftp://skills.sh/sitemap-skills.xml")
