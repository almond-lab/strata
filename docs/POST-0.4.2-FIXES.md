# Reports addressed during v0.5 development

Version 0.5.0 includes the following fixes.
Passing a scoped regression is not grounds to mark a wider research milestone
complete. The [discovery guide](V0.5-DISCOVERY.md) describes the new algorithms.

| Report | Implemented behavior | Remaining limit |
|---|---|---|
| #86 | A live supervised experiment reports `running` with operation/budget information | Unknown/stale supervision still needs reconciliation |
| #87 | Synthesis and equivalence share prefix-transported zero evidence; replay reconstructs it independently | No guessed interior clean promises |
| #88 | `restructure`/`search` accept explicit ranges and automatically try bounded episodes in a long file | Fractional endpoints have a separate bounded constructor; complete host architecture lifting remains open |
| #89 | `--width` declares a larger parent layout so untouched declared helpers can be used | Enlarging the actual parent qubit budget is not allowed |
| #90 | Supported local lookup/phase segments can be rebuilt without requiring the whole episode's phase to be zero | Exact repairs and measurement provenance remain obligations; local changes are not classified as deep wins |
| #91 | Exact conditional expected counts with correlated HMR expressions and optional explicit external-bit joint distribution | No guessed probability for an unknown external control; nonlinear cost-of-mean differs from mean cost |
| #92 | `optimize --clean-inputs` remains windowed and transports facts instead of silently switching to a whole-file cap | Large unsupported boundaries remain explicit unknowns |
| #93 | `floors --outputs` selects wanted Boolean outputs | Relaxing unwanted garbage does not prove free cleanup |
| #94 | Value-only domain queries can omit phase-only classical bits after proving output independence | Equivalence and outcome-dependent values keep stricter obligations |
| #95 | CLI bounded exact AND search and finite care-set search return witnessed minima or unknown | Only the named Boolean model, not unrestricted quantum cost |
| #96 | Factored final values feed queries/local constructors; algebraic factorization avoids small truth-table enumeration on supported boundaries | Term/decision explosion, fractional phase and arbitrary word-level recovery remain bounded |
| #97 | `observe-boundary` produces source/cut-bound replayed reachable observations usable by value/slack/floor queries | Samples stay observational; no sampled support becomes an all-input promise |
| #98 | Proven empty abandoned legacy FIFO entries are cancelled conservatively | Active or ambiguous process/ownership rows are not silently removed |
| #99 | Query CLI `--range-unit operation|text-line` matches discovery and emits explicit coordinate labels plus canonical text cuts | SDK low-level queries retain their documented text-line convention |
| #100 | `function-of` reports the size of the requested monomial family. `hmr-debt` reconstructs the pre-measurement value at later cuts, with a least-degree witness on the supplied support | The cubic count prices that witness. A global minimum CCZ count needs a separate optimization. |
| #101 | `evaluate-file` accepts `--max-proof-operations`, private `bN` syntax, independent measurement maps, compact address intervals and expected-cost mode | Outcome dependence, exposed outcomes and proof-budget exhaustion remain explicit. Tested on reproducible retiming fixtures. |
| #102 | A complete binary slice space with a proved minimum rank supplies a Griesmer tensor-rank lower bound; the 12 × 24 × 24 double field product reaches 54 from 35 after enumerating 4,095 nonzero left slices | Partial slice enumeration supplies no Griesmer certificate. |
| #103 | `wrap --dependency FILE_OR_DIR` pins sibling modules or source trees. Wrapping a standalone Python script surfaces a dependency notice | External imports and installed packages must still be supplied by the caller. |
| #104 | Default per-process CPU time scales with reserved threads and stage wall time. Explicit CPU limits take precedence. A shell-style exit 152 records a possible SIGXCPU and the configured limit | Process CPU time and whole-job wall time remain separate quantities. |

## Earlier measurement limitations

#83 has a substantive new backend, but its original inverse fixture is still
bounded on nonzero inputs. The small four-syndrome GF(4) BM fixture passes all
256 inputs and all2^72 formal outcomes. The enlarged BM fixture fails scratch
restoration with a native counterexample. Keep the original unresolved coverage
visible; documentation and a different fixture are not a fix for that input.

#85 now supports scoped Boolean value-map floors through repaired measured
episodes. It still does not prove generic quantum-resource floors for arbitrary
outcome-dependent instruments. The reported floor explicitly says which model
and outputs it covers.

#9 (complete untagged contract trees), #10 (historical unhinted discovery),
#48–50 (fractional-phase/host-domain/host-verifier integration), and #19 (semantic
binding of real negative-result history) retain their unmet acceptance work.

## Evidence

Tests cover transported-zero mutation, factored cancellation, selected outputs,
phase-only domains, observed versus exhaustive support, nonlinear all-outcome
phase checking, correlated conditional costs, private classical-bit collisions,
native range coordinates, live status and legacy queue ownership. Structural
search has a separate [frozen generated comparison](../benchmarks/v05-structural/README.md).
Neither source-level equivalence nor generated discovery fixtures establish a
new official competition score.
