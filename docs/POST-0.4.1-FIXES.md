# Reported workflow fixes after v0.4.1

This batch addresses reports #60–85 and makes small-instance-first research the
default guidance for connected agents. Historical semantic-discovery milestones
remain separate; these fixes do not establish a competition optimization gain.

| Issues | Result and scope |
| --- | --- |
| #60 | Completion validates the staged dependency bytes actually used. Live source edits still block new attempts/resume; they no longer discard completed work. Live executable/reader checks remain enforced. |
| #61 | Disk-backed canonical tapes align unchanged prefixes and suffixes before bounded local proof recovery. Clean restoration is a separate obligation and may remain unknown. The regression uses 240,003 lines; the original 51M-line artifact was not rerun. |
| #62 | Stage and verifier CPU limits derive from their own wall timeout unless explicitly inherited. Diagnosed process-pool crashes and invalid verifier protocols are execution failures, not scientific rejections. |
| #63, #70, #74, #78, #81 | Explicit idle-only board capacity, FIFO admission, abandoned queue cleanup, shared sweep deadlines, and filtered events with advancing cursors. Reservations remain cooperative, not hard aggregate OS quotas. |
| #64 | No-verdict attempts remain visible but do not permanently poison a matching rerun. Audited historical infrastructure corrections require retained machine evidence; genuine scientific rejection cannot be overridden. |
| #65 | Reports and agent packets select a comparison identity across experiments. Display filters cannot hide other observations in the same scientific group. |
| #66 | Verifier-only metrics declare their selector and unit and retain independently verified provenance. |
| #67, #69 | Worker-only successes retain their success state and reuse checked artifacts. Budget exhaustion is separately reported; a new replicate explicitly requests new work. |
| #68 | Onboarding documents how to adapt self-sandboxing wrappers without bypassing Strata's outer sandbox. |
| #71 | Relative output arguments do not become inputs merely because a neighboring file exists. `wrap` excludes declared output paths from automatic dependency pinning. |
| #72 | Screening can pair a proxy with an independently verified cost or complete draw aggregate in different units, bound to identical artifact bytes. Calibration and decisions revalidate retained evidence and reject leakage. Insufficient data still abstains; a single baseline does not justify pruning. |
| #73 | File discovery ranges default to native operation offsets; explicit `text-line` mode preserves the former convention. Receipts label coordinates. |
| #75 | Pinned interpreter-plus-script arrays are supported; command strings, command-evaluation flags and unsafe shell startup forms remain rejected. |
| #76 | Outcome-conditioned `NEG` participates in the proven affine HMR reduction. |
| #77 | Integral float resource values normalize safely; invalid values identify the offending field. |
| #79 | `python -m strata` invokes the same CLI. |
| #80 | Explicit independent-map checking can compare different HMR instruments by proving outcome independence on each side. It preserves exact phase, all quantum outputs and public classical outputs; private outcome bits must be declared. |
| #82 | Classical registers have an explicit namespace in exhaustive lab contracts. Inputs are enumerated and through/clean obligations checked separately from qubits. Unsupported quantum-only inference rejects mixed namespaces explicitly. |
| #83 | The lab documents and diagnoses true non-affine outcome branching. Affine measured cleanup, including conditioned `NEG`, is supported; arbitrary nonlinear outcome dependence is not silently labelled exact after sampling. |
| #84 | Wire-list CLI options accept inclusive ranges and reject malformed/oversized ranges with explicit usage errors. |
| #85 | Boolean gate-floor documentation explicitly excludes measured/classically controlled components and explains why dropping HMR repayment would not establish a whole-component bound. No measurement-aware floor is claimed. |

## Scheduling and report commands

```bash
python -m strata board --store work/research --init-capacity capacity.json
python -m strata board --store work/research --events-after 0 --exclude heartbeat
python -m strata experiment-report --store work/research --comparison-identity ID
python -m strata agent-packet --store work/research --comparison-identity ID
```

Use the returned event `cursor` even when filtering leaves no displayed events.
Capacity can change only while the board has no queued or active work; repeated
identical declarations are safe. FIFO may leave small resources idle while an
older large request waits, preventing starvation of precommitted verification.

An infrastructure correction is append-only:

```bash
python -m strata experiment invalidate RESULT_HANDLE --store work/research \
  --actor researcher --reason 'retained child process was killed by its CPU limit'
```

The original receipt stays in the Store. A correction does not supply a passing
verdict: the same artifact and draw still need a fresh valid check. If no
machine-backed infrastructure evidence exists, the correction is refused.

## Independent screening objectives

Pass `artifact` when recording a screening candidate whose cheap receipt has
multiple outputs. Pair it with a complete independently accepted finish or a
retained report containing a complete draw aggregate, using `metric: "cost"`.
For a report with multiple artifacts, also provide `comparison_identity` and
`artifact_identity`. The full unit comes from the verified objective; it need
not equal the proxy unit. Policies can pin `full_unit` and
`full_comparison_identity` explicitly.

Changing candidate names cannot remove training/held-out artifact leakage.
Subsequent contradictory verdicts invalidate the evidence used by calibration,
decisions and audits, including historical verified-metric pairs. Thresholds
with inadequate independent calibration remain advisory.

## Default research sequence

Read [small-first research](SMALL-FIRST.md) before launching a full build. The
default agent packet includes a structured research ladder. The agent skill
requires an explicit stage and explains exceptions for meaningful tiny cases
that cannot be produced. Ordinary wrapped scripts remain backward-compatible;
Strata does not guess mathematical scale from an opaque command name.

Tiny-instance correctness is evidence about that instance. Full-size resource
predictions are models. Full scientific verification remains a separate gate.
