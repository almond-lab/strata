# Lookups with measurement-based cleanup

The `measured_lookup_resynthesis` constructor is included in the default
`strata restructure` registry. It factors the exact Boolean output functions
of arithmetic segments between measurement, phase and control boundaries.
It keeps every HMR event and its conditional phase-correction sequence in
order. A measured lookup no longer has to pass through a permutation-only
constructor.

```python
import kickmix as km
from strata.lookup import resynthesize_measured_lookup

loader = km.Circuit('''R q4
CCX q0 q1 q4
CCX q0 q2 q4
CCX q1 q2 q4
CX q4 q3
HMR q4 b0
CZ q0 q1 if b0
CZ q0 q2 if b0
CZ q1 q2 if b0''')
result = resynthesize_measured_lookup(loader, clean_inputs=[4])
assert result['status'] == 'constructed'
assert result['check']['equivalent'] is True
```

Here `q0..q2` are arbitrary address bits, `q3` is an arbitrary dirty output,
and `q4` is promised zero and restored. Quadratic-form factorization implements
the same three-input table with one CCX instead of three, borrowing no extra
wire. The original measured value and every phase correction are preserved.
The complete gate count includes basis changes, output fanout and cleanup.
This is a synthetic fixture gain, not a host-circuit or competition gain.

The constructor can share a factored result through an output known to start
zero and fan it into dirty targets in a checked dependency order. A fact from
an unconditional HMR can establish zero for the next segment. Conditional
writes invalidate that fact. Reset syntax alone never grants a clean lease.
Physical width is retained even when a trailing identity region disappears;
no new named host register is required.

Each proposal requires exact whole-boundary Boolean/sign semantics and
discharged phase debts. The independent construction-loop check compares all
physical outputs, classical outputs, the ordered measurement provenance and
the final sign for every symbolic input and outcome. Regression tests also
force every measurement branch through native kickmix for small examples.
Changing a phase correction is rejected even if the table output is unchanged.

The implemented families are **quadratic and cubic XOR-accumulator segments**.
The cubic planner shares input-pair products and factors their third-input and
output masks, using an existing provably zero helper or suitable clean output
holder. All setup, fanout and uncomputation are charged; coefficients are
checked before emission and the complete proposal is checked afterward.
Plans that lose on cost, lack a legal helper, or exceed the operation budget
retain the original segment and expose the priced obstruction. A tested cubic
fixture reduces six CCX to three while preserving all measurement branches.

On the real loader described in [kernel recovery](KERNEL-RECOVERY.md), all four
cubic segments now produce coefficient-verified plans, but none beats the
existing implementation. Two plans exceed the emission budget. No real-loader
gain is claimed. Reproduce this scope with a suitable construction budget,
including `--max-product-terms`; the default remains bounded.

Higher-degree neighboring segments are retained; unresolved phase debts,
fractional rotations or exhausted symbolic budgets remain explicit unknowns
or unsupported constructions. This is not a general Boolean-table compiler,
not a proof of arbitrary measurement-instrument equivalence, and not a promise
that every loader has a cheaper implementation. Host verification and its
full objective still apply before promoting a real candidate.

Tests: `tests/test_measured_lookup.py`, `tests/test_lookup_review.py`, and the
reset/HMR cases in `tests/test_streaming_budget_issues.py`.

## Word cleanup before reversible synthesis

The search constructors that require a reversible boundary can normalize a word erased by private HMR outcomes. Declare the measurement bits with `--internal-bits` and identify any promised zero input wires with `--clean-inputs`. A bit can control several supported phase corrections, including corrections interleaved with those for another word bit. The constructor places the corresponding reversible XOR at each original correction position. This preserves changes of basis on its controls between terms. An unconditional overwrite starts a new private-bit lifetime.

The measured wire must remain available until its final correction. A public outcome, an intervening read or write of that wire, an unsupported use of the outcome, or failure of the whole-boundary basis/sign proof blocks this normalization. The proof ranges over all public boundary values under the declared clean promise and verifies independence from the private measurement outcomes. It does not certify general quantum channels. Each matched HMR records every phase-repair interval in canonical text-line coordinates under `phase_repairs`.

On the GF(64) decoder inverse cell from the original word-cleanup report, the constructor verifies ten erasures. Two bits each control six linear terms and two quadratic terms. The reversible intermediate has 32 Toffolis versus 20 in the measured source. It enables further construction and does not by itself lower the host cost. The in-place GF(64) cell also verifies, with twelve erasures. The GF(4096) inverse and phase inverse cells each verify 22 erasures and eight reset preconditions on all 4,096 values of the twelve input wires that can affect these obligations. The phase inverse's other twelve public wires remain arbitrary. Both measured originals cost 65 Toffolis; the verified reversible intermediates cost 157 and 143 respectively.

The focused regression in `tests/test_word_phase_repair.py` forces both measured bits through all four outcome combinations. It also checks a wrong phase correction, scratch reuse during repayment, a non-diagonal use of an outcome, a public outcome, and unconditional reuse of a private bit.

For flat private HMRs followed by inline conditional NEG, Z or CZ corrections, the scoped proof checks every preserved native instruction against the candidate and reconstructs each reversible toggle at its original position. Only the unobserved scratch wires may differ before their final repayment. It rejects preserved data or phase operations touching those wires. On each private measurement epoch it verifies the XOR of the measured value and every phase term is identically zero. This identity proves the candidate erases its scratch and that the original measurement outcome contributes zero net sign. A conservative dependency pass excludes unrelated public inputs from the subsequent exhaustive Boolean evaluation. Unconditional ordinary resets have their own zero proofs. The receipt names the checked wires, assignment count, epoch and reset obligations.

The scoped proof caps truth rows at the smaller of 65,536 and the normal term budget, and caps estimated packed Boolean operations using the product-term budget. When a shape or budget falls outside that proof, Strata tries the existing whole-boundary decision-diagram verifier. The mechanism also preserves separate promised-domain verifiers when the caller supplies one.
