# Your first experiment with Strata

Strata can run an existing script, keep its settings and outputs, compare a sweep,
and prepare a research packet for your agent. You do not need to write a Python
adapter. This guide assumes an installed Strata environment and Linux with
bubblewrap/user namespaces available. It runs offline; it does not submit results
to a competition.

## Try an included pipeline

From a source checkout, the reference fit has a separate verifier and a declared
cost formula:

```bash
.venv/bin/strata experiment run examples/workflow/numeric.toml --seed 0 --store work/first-run
.venv/bin/strata experiment-report --store work/first-run --format markdown
```

The result distinguishes `execution_finished`, `artifact_validated`, and
`accepted_by_verifier`. Metrics without a verifier remain worker-reported
measurements. An accepted result includes the pinned verifier's receipt; only
resources independently checked by it enter a verified cost comparison.

This is a small reference task, not an official contest scorer. The circuit and
non-contest examples exercise the same API:

```bash
.venv/bin/strata experiment run examples/workflow/circuit.json --seed 0 --store work/circuit-example
.venv/bin/strata experiment run examples/workflow/words.json --seed 0 --store work/words-example
```

## Wrap your script

For an existing `fit.py` that accepts a step count and writes `result.json` with
a numeric `loss` field:

```bash
.venv/bin/strata wrap --out fit-run.json \
  --output result=result.json:json --metric loss=result:loss:squared-error \
  --setting steps=100 --timeout 120 --threads 1 \
  -- .venv/bin/python fit.py '{setting:steps}'
.venv/bin/strata experiment run fit-run.json --store work/my-fit
```

`wrap` pins existing file arguments as dependencies and writes a readable
manifest. Declare additional data inputs/imported source directories in it before
running. Placeholders occupy an entire argument; commands never pass through a
shell. The run starts in a fresh directory. JSON, JSONL and CSV outputs are
supported. A script without machine-readable metrics still gets logs and execution
records, but Strata reports numeric analysis as unavailable.

The wrapper defaults to a 64 MiB cooperative temporary-storage reservation and
64 MiB per-file limit. Adjust `--storage-mb`, `--max-file-mb`, `--memory-mb`,
`--threads` and `--timeout` for your script. Reservations coordinate participating
workers; they are not hard aggregate host quotas. A request for unsupported hard
aggregate enforcement fails before launch.

## Declare the sweep once

The included grid declares twenty settings. Run it with two workers:

```bash
.venv/bin/strata sweep create examples/workflow/numeric.toml \
  --parameters examples/workflow/twenty-settings.json \
  --seeds 0 --replicates 1 --max-jobs 20 --workers 2 --store work/sweep
```

Copy the returned `sweep` handle into:

```bash
.venv/bin/strata sweep run SWEEP_HANDLE --store work/sweep
.venv/bin/strata sweep status SWEEP_HANDLE --store work/sweep
.venv/bin/strata board --store work/sweep
.venv/bin/strata experiment-report --store work/sweep --format markdown
```

Use the same handle to resume. Successful verified artifacts are reused only
while their frozen dependencies/runtime/reader match. Seeds and explicit
replicates identify intentional repeated trials; retry attempts retain their own
logs and charged budgets.

## Give the next agent the result

```bash
.venv/bin/strata agent-packet --store work/sweep --out work/next-agent.json
.venv/bin/strata board --store work/sweep --events-after 0
```

The packet contains the verified baseline, attempted settings, failures,
uncertainties, active jobs and recorded hypotheses. Completion events have a
cursor, so a supervising agent can work elsewhere and read only new events.
Strata does not call a remote AI provider on its own. Your connected agent reads
the packet, proposes a bounded decisive experiment and records its decision.

See [the workflow guide](V0.4-WORKFLOW.md) for verification, checkpoint/resume,
calibrated screening, continuation policies, cancellation and the complete
Strata–agent loop.
