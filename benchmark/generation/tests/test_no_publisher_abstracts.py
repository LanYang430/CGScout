"""The benchmark-construction data keeps identifiers and a checksum of each abstract, never the publishers' text."""
import csv
import json
import re
from pathlib import Path

BB = Path(__file__).resolve().parents[1]            # benchmark/generation
ROOT = BB.parents[1]
DATA = BB / "data"
KEYS = ("abstract", "seed_abstract")
SHA = re.compile(r"[0-9a-f]{64}")
JSON_FILES = ["seed_pool.json", "pilot_seeds.json", "pilot_seeds_final.json", "seed_proposal_14.json", "l3_more_candidates.json", "l3_more_candidates_r3.json"]
JSONL_FILES = ["pilot_v0.jsonl", "pilot_v1.jsonl"]
CSV_FILES = ["pilot_v0.csv", "pilot_v1.csv"]
REPLACED = 248                                       # abstracts replaced by a checksum (168 in the json files, 40 in the jsonl files, 40 in the csv files)


def dicts(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from dicts(v)
    elif isinstance(o, list):
        for v in o:
            yield from dicts(v)


def all_dicts():
    for f in JSON_FILES:
        yield f, from_json(DATA / f)
    for f in JSONL_FILES:
        for line in (DATA / f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield f, json.loads(line)


def from_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_no_abstract_text_remains_in_any_data_file_or_fixture():
    for f, obj in all_dicts():
        for d in dicts(obj):
            for k in KEYS:
                assert not d.get(k), f"{f} still has abstract text under {k!r}"
    for f in CSV_FILES:
        header = next(csv.reader((DATA / f).open(encoding="utf-8", newline="")))
        assert not set(header) & set(KEYS), f"{f} still has an abstract column"
    for f in sorted(DATA.glob("*.json*")) + sorted((ROOT / "tests" / "fixtures").rglob("*.json")):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r'"(abstract|seed_abstract)"\s*:\s*"([^"]*)"', text):
            assert len(m.group(2)) == 0, f"{f.name} has a non-empty {m.group(1)!r} string"


def test_every_replaced_abstract_left_a_checksum_and_a_length():
    n = 0
    for f, obj in all_dicts():
        for d in dicts(obj):
            for k in KEYS:
                if k + "_sha256" in d:
                    n += 1
                    assert SHA.fullmatch(d[k + "_sha256"]), (f, d[k + "_sha256"])
                    assert isinstance(d[k + "_chars"], int) and d[k + "_chars"] > 0
    for f in CSV_FILES:
        for row in csv.DictReader((DATA / f).open(encoding="utf-8", newline="")):
            if row["seed_abstract_sha256"]:
                n += 1
                assert SHA.fullmatch(row["seed_abstract_sha256"]) and int(row["seed_abstract_chars"]) > 0
    assert n == REPLACED, n


def test_the_identifiers_of_every_paper_are_kept():
    pool = from_json(DATA / "seed_pool.json")
    assert len(pool) == 100 and all(r["doi"] and r["title"] and r["authors"] for r in pool)
    for f in ("pilot_seeds.json", "pilot_seeds_final.json"):
        seeds = from_json(DATA / f)
        assert len(seeds) == 20 and all(r["doi"] and r["title"] for r in seeds), f
    assert len(from_json(DATA / "seed_proposal_14.json")["proposals"]) == 14
    for f in JSONL_FILES:
        rows = [json.loads(line) for line in (DATA / f).read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 20 and all(r["seed_doi"] and r["seed_title"] for r in rows), f
    for f in CSV_FILES:
        rows = list(csv.DictReader((DATA / f).open(encoding="utf-8", newline="")))
        assert len(rows) == 20 and all(r["seed_doi"] and r["seed_title"] for r in rows), f


def test_reproduce_md_explains_how_to_check_a_re_fetched_abstract():
    text = (ROOT / "docs" / "reproduce.md").read_text(encoding="utf-8")
    for needle in ("Publisher abstracts are not distributed", "abstract_sha256", "abstract_chars", "248 abstracts"):
        assert needle in text, needle
