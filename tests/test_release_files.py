"""Release files: SECURITY.md describes the real classifier file, and the CI workflow runs with the least permission it needs."""
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src" / "cgscout" / "screening" / "sim_classifier_model.joblib"


def test_security_md_names_the_real_classifier_file_and_the_scikit_learn_pin():
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() in text, "SECURITY.md must carry the SHA-256 of the shipped classifier (update it when the model is rebuilt)"
    assert f"{MODEL.stat().st_size} bytes" in text
    pin = re.search(r'"(scikit-learn[^"]*)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8")).group(1)
    assert pin in text, pin
    for needle in ("pickle", "arbitrary code", "Report a vulnerability", "OPENAI_API_KEY", "OPENALEX_API_KEY"):
        assert needle in text, needle


def test_security_md_publishes_no_personal_contact_detail():
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "@" not in text and "/Users/" not in text


def test_ci_has_read_only_permissions_and_no_secrets():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r"^permissions:\n  contents: read\n", ci, re.M), "top-level least-privilege permissions are required"
    assert not re.search(r":\s*write\b", ci), "no write permission anywhere"
    assert "secrets." not in ci and "pull_request_target" not in ci
