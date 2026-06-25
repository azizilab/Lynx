# LYNX — Docs Release Automation & CI Plan

> **Status:** plan only. No implementation yet. This file specifies the work to
> wire up (1) a ReadTheDocs hosted-docs connection, (2) a minimal test suite
> (starting with an `import lynx` smoke test), and (3) a GitHub Actions CI loop
> that validates docs + tests on every push and conditionally re-executes the
> tutorial notebooks.

## Goal (the loop you asked for)

When the `lynx` package changes, you want this loop to be as hands-off as possible:

1. **Reinstall** the package (`pip install -e .`).
2. **Update the tutorials** *only if needed* — i.e. only when you manually change
   something, **or** a unit test fails (today the relevant failure mode is a
   broken `import lynx`). Otherwise the committed notebook outputs are left as-is.
3. **Push** to GitHub.
4. **Sync with ReadTheDocs** for a version bump.

### How this actually maps to tooling (important design facts)

- **ReadTheDocs is pull-based.** Once the GitHub webhook is installed, **every
  push to a tracked branch auto-rebuilds the docs.** There is no separate "sync"
  step — `git push` *is* the trigger. (Step 4 needs no manual action.)
- A new **docs version** (e.g. `v0.2.0` alongside `latest`) appears on RTD only
  when you push a **git tag** and activate it once in the RTD *Versions* UI.
  So "version bump" = bump `pyproject.toml` version → `git tag vX.Y.Z` → push tag.
- **CI's job** is therefore *validation before RTD builds*, not triggering RTD:
  build the docs and run tests on every push/PR so a broken build is caught in
  GitHub (red check) rather than silently failing on RTD. CI can *optionally*
  also poke the RTD API to force a build, but with the webhook that's redundant.

---

## Current state (verified 2026-06-24)

- **No test infrastructure.** No `test_*.py` unit tests (the two `test_assoc.py`
  files are CCI-association source modules, not tests), no pytest config in
  `pyproject.toml`, no `tests/` dir.
- **No CI.** No `.github/workflows/`.
- **`import lynx` works** under `env/bin/python` (reports `0.1.0`). It emits a
  benign numba `RuntimeWarning` on import — a smoke test must assert on the
  exit code, **not** on empty stderr.
- **Headless notebook execution is viable in `env`:** `nbconvert 7.11.0` +
  `nbclient` present; the full tutorial stack (`scanpy, squidpy, scFates, torch`)
  imports cleanly.
- **RTD not yet connected.** GitHub remote is `azizilab/liver3d`, but every
  README docs URL/badge uses the **placeholder slug `lynx`**
  (`lynx.readthedocs.io`, `readthedocs.org/projects/lynx/badge`). The real slug
  is decided at RTD project-import time and the README's 9 URLs (across 8 lines)
  must be finalized to match.

---

## Part A — One-time ReadTheDocs connection (manual, browser)

These steps are inherently manual (they need your RTD/GitHub auth) and only
happen once.

1. Sign in at **readthedocs.org** with GitHub; authorize the RTD GitHub App so it
   can install the push webhook on `azizilab/liver3d`.
2. **Import a Project** → select `azizilab/liver3d`. RTD auto-detects the existing
   `.readthedocs.yaml`.
3. **Record the real slug.** The slug is derived from the project *name* you give
   at import, not the repo name:
   - If you name it `lynx` and it's free → slug `lynx` → README already matches,
     nothing to change.
   - If `lynx` is taken (likely) → RTD assigns e.g. `lynx-spatial`. Then finalize
     the README (Part B).
4. Trigger the first build; confirm green; the badge then resolves.

## Part B — Finalize the README slug (one command, after Part A)

Once the real slug is known, run from repo root:

```bash
SLUG=<real-slug-from-rtd>     # e.g. lynx-spatial
sed -i "s|lynx\.readthedocs\.io|${SLUG}.readthedocs.io|g; s|projects/lynx/|projects/${SLUG}/|g" README.md
```

Touches: the badge line + the 8 doc/tutorial/API/installation URLs in `README.md`.
Verify the rendered badge URL loads before committing.

> Optional hardening (decide later): pin RTD to build only `main` + tags, set the
> default version to `stable` (latest tag) instead of `latest` (HEAD of main) so
> the badge tracks releases, not every commit.

---

## Part C — Minimal test suite (new)

Start with the cheapest, highest-value guard: **the import smoke test you flagged
we don't have.** Grow only as needed.

### Files to add
```
tests/
├── __init__.py
└── test_imports.py
```
- `[tool.poetry.group.dev.dependencies]` in `pyproject.toml`: add `pytest = "*"`.
- (Optional) `[tool.pytest.ini_options]` with `testpaths = ["tests"]`.

### `tests/test_imports.py` — what it asserts
1. **`import lynx` succeeds** and `lynx.__version__` is a non-empty string.
2. **Every advertised submodule imports**: `lynx.{model, dataset, config, io,
   plot, utils, trajectory, test_assoc}` — these are exactly the names in
   `lynx.__all__`, so the test doubles as a guard that the shim re-exports stay
   wired to the underlying `models/` + `util/` modules.
3. **Key public symbols resolve** (cheap attribute checks, no GPU/data):
   e.g. `lynx.model.HeteroAttnVGAE`, `lynx.dataset.HeteroDataset`,
   `lynx.trajectory.get_curve`, `lynx.plot.disp_trajectory`,
   `lynx.io.load_xenium`, `lynx.utils.get_zonations`,
   `lynx.test_assoc.test_cci`. Catches a renamed/moved internal symbol that
   `import lynx` alone wouldn't surface.

**Caveats baked into the test design:**
- Don't assert clean stderr — the numba `RuntimeWarning` on import is expected.
- These tests need the **full ML stack installed** (they import the real
  `torch`/`squidpy` chain), so they run in the CI job that installs the package,
  *not* the lightweight docs job. (The docs build mocks those imports instead.)

### Why this satisfies your loop's step 2
"Update the tutorials only if a unit test fails" → the import test is the
tripwire. If a `lynx` change breaks an import the notebooks rely on, CI goes red,
which is the signal to re-run/fix the notebooks. While tests pass, notebooks are
left untouched.

---

## Part D — GitHub Actions CI (new)

`/.github/workflows/ci.yml` with **two jobs**:

### Job 1 — `docs` (always; the RTD pre-flight)
- Mirrors the RTD environment: Python 3.9, `apt-get install -y pandoc`,
  `pip install -r docs/requirements-docs.txt` (the heavy stack is mocked via
  `autodoc_mock_imports`, so **no torch/GPU needed** — fast).
- Build strict: `python -m sphinx -b html -n --keep-going docs docs/_build/html`.
- Fails the check on broken autosummary/toctree refs **before** RTD ever builds.
- `nbsphinx_execute = "never"` is already set, so committed notebook outputs are
  rendered, not executed — this job stays light.

### Job 2 — `tests` (always; the import tripwire)
- Python 3.9. Install the package + stack. Two sub-options for cost (pick at
  implementation time):
  - **(a) Full install** (`pip install -e .` + PyG find-links) — most faithful,
    slowest; CPU-only torch wheel is fine since tests don't train.
  - **(b) Import-light** — if (a) is too heavy for CI minutes, install only what
    the import chain needs. Start with (a); fall back to (b) only if CI is slow.
- Run `pytest -q tests/`.
- **This red/green is the canonical "did a `lynx` change break things" signal.**

### Conditional Job 3 — `notebooks` (opt-in, NOT on every push)
Per your decision, notebooks are normally left as-is. Re-execute them only when
explicitly requested. Trigger options (choose at implementation):
- `workflow_dispatch` (manual button), and/or
- a commit-message/path trigger (e.g. only when `lynx/**` or the notebooks
  themselves change), and/or
- gated on Job 2 having **passed** (re-running notebooks against a broken import
  is pointless).
- **Blocker to flag:** real execution needs the **gitignored `results/`
  snapshots** and the full GPU/CPU stack. CI runners don't have the 52 GB inputs,
  but the notebooks only load the small committed-workflow `results/*.h5ad`/`.npy`
  snapshots for plotting — *if* those snapshots are accessible to CI. They are
  **gitignored**, so Job 3 on a hosted runner can't see them. ⇒ Job 3 is most
  realistic as a **local** step, not hosted CI. See Part E.

---

## Part E — Local release helper (ties the loop together)

A `scripts/release.sh` (run locally with `env/bin/python`) for the times you
*do* change pipeline behavior and want everything refreshed before pushing:

```text
1. env/bin/python -m pip install -e . --no-deps          # reinstall lynx
2. env/bin/python -m pytest -q tests/                     # import tripwire
3. (optional, --run-notebooks) re-execute the 3 tutorials headless against
   the local results/ snapshots:
     env/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.kernel_name=lynx-env \
       docs/tutorials/{liver,breast,thymus}.ipynb
   (uses nbconvert 7.11.0 already in env; needs the lynx-env kernel + results/)
4. env/bin/python -m sphinx -b html -n --keep-going docs docs/_build/html
5. git add -A && git commit && git push          # push → RTD auto-rebuilds
6. on version bump: bump pyproject version, git tag vX.Y.Z, git push --tags,
   then activate the new version once in the RTD Versions UI
```

This is the only place notebook re-execution lives, and it's **opt-in
(`--run-notebooks`)** so the default fast path leaves notebooks untouched —
matching your "leave them as-is unless I change something" rule. Hosted CI
(Part D) handles validation; this local script handles the heavyweight refresh
that needs `results/` + a GPU/kernel.

---

## Files this plan will create/modify (when implemented)

| Path | Change |
|------|--------|
| `tests/__init__.py`, `tests/test_imports.py` | **new** — import smoke + symbol-resolution tests |
| `pyproject.toml` | add `pytest` dev dep; optional `[tool.pytest.ini_options]` |
| `.github/workflows/ci.yml` | **new** — `docs` + `tests` jobs (+ opt-in `notebooks`) |
| `scripts/release.sh` | **new** — local reinstall→test→(opt notebooks)→build→push helper |
| `README.md` | finalize RTD slug (Part B), once project is imported |

## Open decisions to confirm before implementing

1. **CI install strategy** for the `tests` job: full PyG install (faithful, slow)
   vs import-light (fast, less faithful). Recommend starting full, CPU-torch.
2. **Notebook re-exec trigger**: manual `workflow_dispatch` only, vs also a
   path/commit-message trigger. Recommend manual-only first (and really, local
   via `release.sh`, given the `results/` snapshots are gitignored).
3. Whether to flip RTD's **default/badge version** from `latest` (main HEAD) to
   `stable` (latest tag) for a release-tracking badge.

## Verification (when implemented)
- `env/bin/python -m pytest -q tests/` → green locally.
- Push a no-op commit → GitHub `docs` + `tests` checks go green; RTD shows an
  auto-triggered build.
- Manually run the `notebooks` workflow (or `scripts/release.sh --run-notebooks`)
  → notebooks re-execute against `results/` and figures match current `lynx`.
- Badge on the README resolves to the real RTD project.
