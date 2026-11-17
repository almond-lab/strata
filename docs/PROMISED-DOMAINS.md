# Discovery on a declared input domain

Use `strata restructure loader.kmx --domain allowed.json --out checked` when the caller guarantees which physical input states can reach the entry to the supplied artifact. The domain file is part of the contract for `checked/best.kmx`. The receipt records its normalized content, its hash, the canonical source and candidate hashes, the native resource bill, and the scope of the comparison. The caller establishes reachability in the enclosing circuit.

An explicit finite promise lists every allowed input. Every row supplies each physical quantum wire, including dirty work wires, and every classical bit read before an unconditional write. Quantum-only rows can use little-endian bit vectors. Object rows accept `qN`, `bN`, and declared little-endian quantum register names. For example, this promise permits four input states.

```json
{
  "version": 1,
  "type": "finite",
  "registers": {"address": [0, 1], "word": [2]},
  "assignments": [
    {"address": 0, "word": 0}, {"address": 0, "word": 1},
    {"address": 3, "word": 0}, {"address": 3, "word": 1}
  ]
}
```

A compact address promise combines a set of legal addresses with every incoming dirty word and every external classical input. Supply all three quantum-wire lists. They must be disjoint and cover the whole circuit width. The address register is least-significant bit first. An inclusive interval represents an arbitrarily large contiguous set without listing its rows. A sparse set uses `"addresses": [0, 3, 7]` in place of `interval`.

```json
{
  "version": 1,
  "type": "address_promise",
  "address_wires": [0, 1, 2],
  "interval": [1, 6],
  "arbitrary_wires": [3, 4],
  "clean_wires": [5]
}
```

For this second example pass `--clean-inputs 5`. A three-bit address from 1 through 6 and two arbitrary bits give 24 physical quantum assignments. Each clean wire must start and finish at zero. A captured observation receipt has no authority to change this contract.

## What the search builds and checks

The constructor replays the native circuit to recover the output XOR updates and sign on the declared rows. For a compact promise it replays each allowed address with the arbitrary wires and classical inputs initially zero to propose a design. It fits joint Boolean functions using low-degree GF(2) interpolation and by merging compatible partial truth tables. It lowers the resulting terms to native gates, shares product controls across outputs when profitable, and restores temporary products using available clean wires.

For explicit rows, an independent basis/sign interpreter compares every promised assignment. For compact domains, a canonical Boolean decision graph checks whether any legal address, dirty word, classical input, or formal measurement outcome can distinguish the two artifacts. The checker compares each physical quantum output, public classical output, ordered named register, and sign. Only a cheaper implementation that passes the full comparison is returned. A `promise_specialization` witness also supplies a concrete input where unrestricted equality fails.

The compact comparison also accepts `measurement_mode="independent-map"` through the Python API. It checks outcome independence separately for each circuit on the allowed address set. Public classical outputs remain in the comparison. Use `internal_bits` to identify private classical outputs. A measurement-dependent public result produces an unknown result. A case with 3,133,452 legal addresses and eight arbitrary bits exercises moved, privately consumed measurements without enumerating that input cube.

## Budgets and scope

Construction currently handles deterministic reversible Boolean updates and sign gates. It can preserve deterministic classical control and updates. Synthesis tries functions of unchanged input wires, so a general in-place permutation or a map requiring a dirty-dependent correction can exhaust the available proposals. The same bounded search may find several completions with equal cost. It makes no optimality claim.

The defaults allow 65,536 source rows, 2,097,152 native table cells and 128 physical quantum plus classical wires for construction. Low-degree interpolation tries degrees zero through three. Cofactor completion handles up to 12 remaining independent input columns and charges the completed table against the cell budget. A compact checker can verify a much larger interval symbolically, subject to decision-variable, node, operation and apply-step budgets. Fractional rotations and measurement-based construction require other constructors. The independent-map check covers exact native basis/sign semantics.

A regression fixture uses 1,220 legal 11-bit addresses, 40 arbitrary dirty bits, one clean work bit and a public classical output. Its source spends 42 Toffoli-like gates on a multi-output lookup with sign. Completion uses one. The checker covers all `1220 * 2^40` promised quantum inputs and all external classical inputs. This is a synthetic circuit result.
