# Acceptance ledger

This matrix is intentionally stricter than passing unit tests. “Implemented”
means the API has executable evidence; it does not imply the Kaggle acceptance
case is complete.

| Specification area | State | Evidence / remaining work |
|---|---|---|
| Format-neutral contracts, debts, patches | Implemented core | Dataclasses, fingerprints, mutation and ownership tests |
| Real kickmix ingest | Implemented for individual circuits | SDK integration tests and original fixture |
| Directory/bundle hashing | Implemented | Every file participates; symlinks rejected |
| Full bundle project orchestration | Pending | Chemistry loader + factorization adapter needed |
| Official baseline cost | Implemented adapter gate | Demo verifier only ships; contest adapters supplied explicitly |
| Flame chart / diagram / reaction depth | Implemented | Actual kickmix outputs; diagram bounded to small window |
| Liveness | Conservative + empirical | Exact nonzero proofs pending |
| Allocation ownership | Partial | Declared register mapping; no inferred dynamic object ownership |
| Executed costs / conditions / fingerprints | Bounded empirical runner | Explicit input batches and seed; no sampled result called proof |
| Information slack | Unknown, reported explicitly | Need rank/function proofs |
| Untagged semantic lift | Research pending | Primitive nodes and opaque fallback; structural coverage reported 0 |
| Block registry | Implemented extension point | Checked clean-product schema; broad arithmetic recognizers pending |
| Build-derived prices | Implemented | Constrained demo exact 72→27; no-paper-price policy tested |
| Floors and weighted gaps | Implemented evidence model | User-supplied bounds only; automatic SAT search pending |
| Generated exhaustive lab | Implemented supported subset | All basis inputs, dirty states, all HMR branches; budget rejects partial passes |
| Parameterized full-scale build | Implemented adapter model | Same generator tested at bits=1,2,3 |
| L3/L4 structural example | Implemented | Materialized-product boundary removed; certified replay |
| Move enumeration | Partial | Typed catalog, adjacency proposals and obligations; no general synthesizer |
| Lineage + replay | Implemented core | Artifact tamper, stale patch and replay tests |
| Composition | Compatibility + gated rebuild | Disjoint footprints required; no blind concatenation |
| Negative-result memory | Implemented | Exact-small size scope; estimates priced; synthetic322-row regression |
| CLI / Python / compact agent queries | Implemented subset | CLI integration; unknown evidence returned explicitly |
| Offline execution | Implemented runner | Bubblewrap required, no fallback; local network-namespace/environment smoke passed; not a security audit |
| R1 value/derivative decoder | NOT RUN | Historical source/patch adapter and independent contract recovery needed |
| R2 four-pass carry pairs | NOT RUN | Same |
| R3 64-row inverse table + digit | NOT RUN | Same |
| R4 rank176→173 block | NOT RUN | Tensor synthesis adapter needed |
| R5 hosts' complete contract trees | NOT MET | Automatic semantic lift is not implemented |
| R6 actual322 historical attempts | NOT RUN | Synthetic regression passes; actual ledger needs contract binding |
| ≥90% structural typing | NOT MET | Primitive coverage never substituted |
| ≥95% boundary/tag agreement | NOT MEASURED | Golden tagged/untagged datasets needed |
| <30min full contest gap table | NOT MEASURED | No full contest benchmark run |
| 16 recorded decoder lab draws | NOT RUN | Dataset/adapter needed |
| 3/4 restructures without answer hints | NOT RUN | Search engine and historical adapters needed |
| New verified competitive design | NOT ACHIEVED | Example is a demonstration, not a research discovery |
| ecdsa.fail front end | Pending | Core is format-neutral; only kickmix front end exists |

Next engineering work follows the requested phase order: bind real tagged
stage tables and execution distributions, implement semantic lifting with
independent block certificates, then bind scalable models and historical
replay adapters. Do not start a broad proposal engine ahead of those layers.

Real-input smoke: `REAL-INPUT-SMOKE.json` records analysis of the pinned cand1
inverse file (44Q, 22,143 operations). It contains derived metrics only and
explicitly reports zero structural typing. `DEMO-RECEIPT.json` records the
original fixture restructuring. Neither is a historical R1–R6 result.
