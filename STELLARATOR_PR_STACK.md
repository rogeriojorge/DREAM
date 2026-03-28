# Stellarator Geometry PR Stack

This branch has been split into a local stacked PR series using branch pointers.
The intended base is `origin/stellarator` at commit `202c09b9`.

## Branch Stack

1. `codex/pr1-geometry-foundation`
   Tip: `0ae09c78`
   Parent: `origin/stellarator`
   Scope:
   - geometry packages,
   - provider routing,
   - package-backed smoke coverage,
   - initial flux-tube evaluator plumbing.

2. `codex/pr2-boozer-kernel-parity`
   Tip: `f6657571`
   Parent: `codex/pr1-geometry-foundation`
   Scope:
   - improved Boozer loading,
   - VMEC Boozer path cleanup,
   - package-vs-legacy kernel parity checks.

3. `codex/pr3-desc-boozer-flux-tube`
   Tip: `bff9a345`
   Parent: `codex/pr2-boozer-kernel-parity`
   Scope:
   - DESC-native Boozer block,
   - DESC flux-tube evaluation support,
   - DESC-vs-VMEC flux-tube comparison tests.

4. `codex/pr4-provider-api-cleanup`
   Tip: `c4c64321`
   Parent: `codex/pr3-desc-boozer-flux-tube`
   Included commits:
   - `4faed300` Tighten stellarator geometry resolution handling
   - `c4c64321` Clarify stellarator provider API
   Scope:
   - automatic DESC grid up-resolution,
   - provider-first `setStellarator()` API cleanup,
   - deprecation warnings for legacy arguments,
   - updated package/provider examples and docs.

5. `codex/pr5-maintained-workflow`
   Tip: `d0ef793c`
   Parent: `codex/pr4-provider-api-cleanup`
   Scope:
   - maintained end-to-end workflow example,
   - `provider=auto` smoke-path cleanup,
   - workflow documentation and smoke coverage.

6. `codex/pr6-legacy-example-quarantine`
   Tip: `9ddb5549`
   Parent: `codex/pr5-maintained-workflow`
   Scope:
   - examples directory index,
   - legacy script labeling,
   - `__main__` guards on exploratory scripts.

7. `codex/pr7-package-parity-gates`
   Tip: `56bc1d0f`
   Parent: `codex/pr6-legacy-example-quarantine`
   Scope:
   - generated-package parity gates for DESC and VMEC provider/package frontend loads.

8. `codex/pr8-geometry-compare-utility`
   Tip: `4ba06a0a`
   Parent: `codex/pr7-package-parity-gates`
   Scope:
   - user-facing geometry comparison utility,
   - parity-report example coverage,
   - README updates for the comparison workflow.

## Suggested Push / PR Order

Once a writable remote is available, push and open PRs in this order:

```bash
git push <remote> codex/pr1-geometry-foundation
git push <remote> codex/pr2-boozer-kernel-parity
git push <remote> codex/pr3-desc-boozer-flux-tube
git push <remote> codex/pr4-provider-api-cleanup
git push <remote> codex/pr5-maintained-workflow
git push <remote> codex/pr6-legacy-example-quarantine
git push <remote> codex/pr7-package-parity-gates
git push <remote> codex/pr8-geometry-compare-utility
```

Each PR should target the previous branch in the stack, except PR1 which should
target `stellarator`.

## Current Limitation

The configured `origin` remote is the upstream DREAM repository and is not
writable from this environment, so the branches currently exist only locally.
