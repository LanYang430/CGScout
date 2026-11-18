"""Release metadata: one version everywhere, the project names its author and its former name, and publishes no contact detail."""
import re
import tomllib
from pathlib import Path

from packaging.version import Version

import cgscout

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
CFF = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_version_is_the_same_in_pyproject_init_and_citation_and_is_the_pep440_form_of_the_release_tag():
    assert PYPROJECT["version"] == cgscout.__version__ == "0.1.0b0"
    assert Version("v0.1.0-beta") == Version(cgscout.__version__)
    assert re.search(r'^version: "0\.1\.0b0"$', CFF, re.M)


def test_the_author_and_the_project_links_are_the_confirmed_ones():
    assert PYPROJECT["authors"] == [{"name": "Lan Yang"}]
    assert PYPROJECT["urls"]["Homepage"] == "https://github.com/LanYang430/CGScout"
    assert PYPROJECT["urls"]["Issues"] == "https://github.com/LanYang430/CGScout/issues"
    assert 'repository-code: "https://github.com/LanYang430/CGScout"' in CFF


def test_citation_file_lists_only_the_confirmed_author_and_affiliation():
    assert CFF.count("family-names:") == 1 and 'family-names: "Yang"' in CFF and 'given-names: "Lan"' in CFF
    assert 'affiliation: "Georgia Institute of Technology"' in CFF
    assert "orcid" not in CFF.lower()


def test_no_contact_detail_is_published_in_the_metadata():
    for text in (CFF, (ROOT / "pyproject.toml").read_text(encoding="utf-8")):
        assert "@" not in text.replace("git@", "")


def test_readme_gives_the_former_name_and_says_what_the_legacy_tag_keeps():
    assert "CGScout was developed under the working name BEAM." in README
    assert "benchmark-construction and regression scripts" in README and "Python files only" in README and "cannot reproduce them" in README
