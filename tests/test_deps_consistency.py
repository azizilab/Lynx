"""Guard against dependency drift across the three install tracks.

``requirements.txt`` (pip), ``pyproject.toml`` (poetry-core / ``pip install -e .``)
and ``environment.yml`` (conda) each declare the runtime stack independently — none
is generated from another. (Poetry's ``export`` is deliberately not used: the PyG
find-links wheels and the ``scFates --no-deps`` step don't round-trip through a lock
file — see the dep-file headers for the rationale.)

These tests fail loudly if a critical pin is bumped in one file but not the others,
or if the ``scFates`` special-casing is accidentally undone. They use only the stdlib
(no TOML/YAML deps), so they run anywhere and stay cheap inside the existing tests job.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = (ROOT / "requirements.txt").read_text()
PYP = (ROOT / "pyproject.toml").read_text()
ENV = (ROOT / "environment.yml").read_text()

# Packages whose version spec must be IDENTICAL across all three files.
CRITICAL = {
    "pandas": ">=2.1.0",
    "numpy": ">=1.23.5,<2.0",
    "squidpy": "==1.6.1",
    "scanpy": ">=1.10.1",
}

# scFates is installed out of the resolver (``--no-deps``); these are its genuine
# runtime deps, which must be present (so --no-deps is safe) in every track.
SCFATES_RUNTIME = [
    "elpigraph-python",
    "simpleppt",
    "plotly",
    "adjustText",
    "scikit-misc",
    "python-igraph",
]


def _norm_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _norm_spec(spec: str) -> str:
    """Normalize a version spec across pip / poetry / conda syntaxes."""
    spec = spec.strip().replace(" ", "")
    if not spec or spec == "*":
        return spec
    if re.match(r"^[0-9]", spec):       # poetry bare version "1.6.1" -> ==1.6.1
        return "==" + spec
    spec = re.sub(r"^=(?!=)", "==", spec)  # conda single = -> ==
    return spec


def _parse_requirements(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):   # skip blanks, -f/-r/-e flags
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", line)
        if m:
            out[_norm_name(m.group(1))] = _norm_spec(m.group(2))
    return out


def _parse_pyproject(text: str) -> dict:
    sec = re.search(r"\[tool\.poetry\.dependencies\](.*?)(\n\[|\Z)", text, re.S)
    body = sec.group(1) if sec else ""
    out = {}
    for line in body.splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r'^([A-Za-z0-9_.\-]+)\s*=\s*"([^"]+)"', line)
        if m:
            out[_norm_name(m.group(1))] = _norm_spec(m.group(2))
    return out


def _parse_environment(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        m = re.match(r"^\s*-\s*([A-Za-z0-9_.\-]+)\s*([<>=!].*)?$", line)
        if m:
            out[_norm_name(m.group(1))] = _norm_spec(m.group(2) or "")
    return out


def _all_parsed():
    return {
        "requirements.txt": _parse_requirements(REQ),
        "pyproject.toml": _parse_pyproject(PYP),
        "environment.yml": _parse_environment(ENV),
    }


def test_critical_pins_agree():
    parsed = _all_parsed()
    for pkg, expected in CRITICAL.items():
        specs = {f: p.get(pkg) for f, p in parsed.items()}
        missing = [f for f, s in specs.items() if s is None]
        assert not missing, f"{pkg!r} is missing from: {missing}"
        assert len(set(specs.values())) == 1, f"{pkg!r} pins disagree across files: {specs}"
        assert next(iter(specs.values())) == _norm_spec(expected), (
            f"{pkg!r} expected {expected!r} but files say {specs}"
        )


def test_scfates_out_of_resolver():
    # scFates must NOT be a resolver-managed dependency anywhere — it conflicts
    # with squidpy and is installed separately with --no-deps.
    for fname, parsed in _all_parsed().items():
        assert "scfates" not in parsed, (
            f"scFates is a resolved dependency in {fname}; it must be installed "
            f"out of the resolver with `pip install scFates==1.0.8 --no-deps`"
        )


def test_scfates_runtime_deps_present():
    parsed = _all_parsed()
    for dep in SCFATES_RUNTIME:
        d = _norm_name(dep)
        missing = [f for f, p in parsed.items() if d not in p]
        assert not missing, (
            f"scFates runtime dep {dep!r} is missing from {missing} — without it "
            f"the `--no-deps` install would leave scFates unimportable"
        )


def test_scfates_nodeps_step_documented():
    for fname, text in (("requirements.txt", REQ), ("pyproject.toml", PYP), ("environment.yml", ENV)):
        assert "scFates==1.0.8 --no-deps" in text, (
            f"{fname} lost the documented `scFates==1.0.8 --no-deps` install step"
        )
