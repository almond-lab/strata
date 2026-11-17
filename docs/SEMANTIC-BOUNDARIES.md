# Circuit coordinates and bounded equivalence

`discover` and `inspect-file --range START:STOP` use zero-based, half-open
**native operation indices** by default. `REGISTER` and `APPEND_TO_REGISTER`
are serialization metadata and do not consume native indices. `INFO_TAG`,
classical controls, and measurements do consume indices. This convention matches
`len(kickmix.Circuit)` for both `.kmx` and `.kmb` sources.

Use `--range-unit text-line` for offsets obtained from canonical `.kmx` text,
including register declarations. Scan receipts record `range_unit`,
`operation_range`, and `text_line_range`. Kernel groups and the SQLite
`scan_metadata.coordinate_unit` entry identify their occurrence coordinates.
Small-file inferred regions and `evaluate-file` requested regions remain
canonical text-line ranges, explicitly identified in their receipts.

## Local edits in large files

Candidate evaluation spools canonical text to disk and indexes one byte offset
per 1,024 lines. Common prefixes and suffixes are compared exactly before a
bounded middle alignment. A deleted cancelling pair in a long repetitive file
therefore enters the semantic checker as a two-line edit. An unmatched middle
larger than 4,096 lines stays an explicit replacement interval; it is not fed
into an unbounded repetitive sequence matcher. Interval growth and its stopping
reason appear in `recovery.attempts` and `recovery.stop_reason`.

Declared clean inputs no longer force every large-file edit into a
whole-artifact polynomial. The checker first attempts arbitrary-boundary local
substitutions, then separately transports bounded constant facts across the
baseline to prove the declared zero outputs and reset legality. The verified
local substitution preserves those outputs in the candidate. If those facts
cannot prove cleanup, the overall result stays **unknown**, even when the local
edit is equivalent. A local mismatch under arbitrary inputs is not reported as
a clean-domain counterexample.

The parser still holds both native circuits, and canonical spools and retained
artifacts require disk space. This is bounded Python line storage, not a
constant-memory native decoder. Regression coverage includes a 240,003-line
interior deletion with a 12 MiB Python allocation ceiling; it does not constitute
a benchmark on the issue's 51-million-line artifact.

## Different measurement instruments

The default finite-domain checker preserves coupled HMR events and all classical
outputs. To compare different implementations of an outcome-independent map:

```sh
strata evaluate-file base.kmb candidate.kmb --out run/check \
  --domain promised-inputs.json --measurement-mode independent-map \
  --internal-bits 0,1 --max-window 10000
```

`--internal-bits` explicitly declares classical outputs private. Omit it to keep
all classical outputs observable. Random HMR records generally prevent an
outcome-independent map unless they are private or reset. Every physical quantum
output, including dirty workspace, and the exact sign phase remain obligations.
`--clean-inputs` additionally requires those output wires to be zero. The promise,
measurement mode, and private output list are bound into the contract hash.

Each circuit must implement the same exact map for every HMR outcome on every
promised input. A proven affine phase permits zero and unit outcome witnesses
(`m + 1` executions for `m` measurements); other controls require complete branch
enumeration. A conditioned `NEG` contributes its affine activation expression to
the phase and is supported, including the cleanup emitted by kickmix `iadd`.
Products of independent outcomes are nonlinear and cannot use that reduction.

The API `equivalent_on_domain` bounds executed branches per construction with
`max_outcomes` (default 1,024), total executions with `max_trajectories` (default
1,000,000), and each circuit with `max_operations`. Exceeding a bound returns
unknown. A measurement-dependent output or phase also returns unknown with an
independence witness; it is never promoted to equivalence. Once independence is
proved, differing boundary maps produce a concrete counterexample. Fractional
rotations remain unsupported by this exact sign checker. These receipts prove
finite promised basis/sign maps, not general quantum-channel equivalence or
host-domain acceptance.
