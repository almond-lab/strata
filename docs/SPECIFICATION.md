# Strata: a restructuring library on kickmix

Product document · Nov 16, 2026 · @Sid

## Summary

Strata is a library on kickmix that finds deep restructures of a circuit against a cost function you supply. It models a circuit as a tree of contracts between stages, not as a list of gates.

You give it a contest: the base circuits, the verifiers and the cost formula. It returns a graph of verified designs that all descend from the base circuit. Each design carries its cost, its proof and the named change that produced it.

It works in five steps:

1. **Analyse** the base circuit with every kickmix view, plus the measurements kickmix lacks.
2. **Lift** the flat gate list to a contract tree: the stages, what each receives, what it must return, which wires it borrows.
3. **Price** every stage exactly and compute a floor for each contract, so the gap ranks where to work.
4. **Restructure** by editing contracts. Each candidate is tested on every input at small size before it is built at full size.
5. **Prove and keep** every result as a patch with a certificate, in a lineage that never loses a line of work.

Three parts of it exist nowhere today: lifting of untagged circuit files to contracts, a floor next to every cost, and small-size exhaustive tests generated from the same contract as the full-size build.

&#91;embedded content: architecture · 6 layers and the store\]

The contract model is the new layer. The engine, the evidence and the store all read and write it; only the analysis pack and the front ends touch gates.

## Problem

Today's tools work at the wrong level: they see gates or tasks, and deep gains come from changing what a stage promises to its neighbours.

| Tool | What it sees | What it cannot see |
| --- | --- | --- |
| kickmix (simulator, flame chart, HTML diagram, reaction depth) | gates, wires, time | what a stage is for |
| Gate-level rewriters, ours and public | pairs and windows of gates | anything across a stage boundary |
| lethe | agents and their teams | the circuit |
| kickmix-arena | one component task and its score | how components trade cost with each other |
| Our workflow (notes, price lists, attempt log) | whatever an agent wrote down | whether a written estimate is true |

What that cost us on the Kaggle contest, measured:

- **Gate-level work is exhausted.** A whole-file census of the decoder found 252 removable Toffolis in about 398,000 (0.064%). An exhaustive census of the inverse sqrt found none.
- **The large gains were contract changes.** Inverse sqrt went from cost 94,538 to 80,163 when the stage list changed: table and polynomial became root, table and divider.
- **Paper estimates were wrong in both directions.** Carry pairs in two decoder passes were ruled out on paper; built, the stage lost 5,536 Toffolis. One divider row was priced at +12 Toffolis and measured at +1.
- **Every contract was recovered by hand.** The decoder's stage contract took one agent several hours. The best public inverse sqrt file has no source and no tags.
- **Every small-size model was written by hand.** The decoder's failure classes were understood only after we wrote a GF(16) model of it from scratch.
- **Knowledge went stale.** Our attempt log has 322 entries keyed to file hashes, and the base file changed several times in one week.

## Goals, non-goals and users

Strata succeeds if a team can start from a contest's base circuits and reach several different deep restructures without writing tooling per circuit.

Goals:

1. **One project file per contest.** Base circuits, verifiers, cost formula, input domain, limits.
2. **Analysis with no manual step.** The full analysis pack for a base circuit in under 10 minutes.
3. **Lifting that works without source.** At least 90% of Toffolis assigned to a typed region on untagged files.
4. **Exact prices.** Every price is measured by a build under a constraint, never estimated.
5. **Many designs from one base.** Lines of work branch from any design, run in isolation, and merge when they touch different regions.
6. **Beyond the known designs.** Floors and the move catalogue point at regions where no known design is close to the floor.
7. **One command reproduces any result.** Design, patch chain, certificate.
8. **Format-neutral core.** kickmix is the first front end. The ecdsa.fail operation stream is the second.

Non-goals:

- Not a gate-level optimiser. Strata calls existing ones as a last pass inside a region.
- Not a simulator. It wraps kickmix and a fast exact oracle.
- Not an agent harness. It gives any harness a typed interface.
- No hardware layout, routing or error-correction modelling.
- No automatic submission to any contest.

| User | What they do with it |
| --- | --- |
| Team lead | Opens lines, reads the gap table, decides what to build |
| Agent | Queries prices and floors, proposes moves, runs the gates |
| Circuit author | Publishes a design with its contract tree and certificate |
| Contest host | Ships the project file so every team starts from the same analysis |

## Core concepts

Eleven terms carry the whole design.

| Term | Meaning | Example from our circuits |
| --- | --- | --- |
| Region | A span of operations whose boundary holds few live wires | The decoder's extract stage |
| Contract | What a region promises: registers in and out with their meaning, helper wires borrowed and returned clean, inputs on which it may fail | Divider: product word in, 26 quotient signs and a remainder out, 3 working wires |
| Debt | A phase correction owed because a wire was measured in the X basis. It names the region that must repay it | One decoder's extract stage measures about 200 words; its reverse repays with 7,465 conditional phase gates |
| Ledger | Measured cost of each region: executed Toffolis, peak wires and who holds them, failure probability | Divider and its reverse are 41% of the inverse sqrt Toffolis |
| Marginal price | Change in total cost per unit of one resource in one region | One decoder wire = 818 Toffolis. A clean wire in the divider cleanup = 13.5 |
| Floor | The least any implementation of a contract can cost, tagged by how it is known: proof, counting bound or exact search | Scan zero test: 11 per position (proof). Chemistry loaders: 3,656 per pass (counting bound) against 4,017 today |
| Scale parameter | A knob that shrinks the instance and keeps its structure | Field GF(4096) down to GF(16); 35 slots down to 5 |
| Move | A typed edit of the contract tree | Change the representation that crosses between two stages |
| Patch | A change to a generator or a file, keyed to a region, with a certificate | The carry-pair patch on the convert stage |
| Certificate | The evidence for a patch | All inputs pass at small size, full domain passes at real size, modelled failure rate |
| Design | A contract tree plus the generator and parameters that build it | Root, 64-row table, divider, 46 wires |

A **lineage** is the graph of designs descended from one base circuit. Each edge carries a depth class.

| Depth | What changes | Example | Gain |
| --- | --- | --- | --- |
| L0 rewrite | Gates inside a region | Exact pair and fan-out merges on the decoder | 0.2% |
| L1 tune | Parameters of a design | Divider truncation schedule | 0.35% |
| L2 region redesign | One region's implementation, same contract | Fused field block at rank 173 instead of 176 | 0.2% |
| L3 boundary change | Two neighbouring contracts | 64-row table with a sixth interpolation digit | 2.4% |
| L4 architecture | The stage list | Table and polynomial replaced by root, table and divider | 15% |

Strata exists for L3 and L4. The lower classes are tools it calls.

## How a project runs

A project starts at the contest's base circuits and grows a lineage of designs. Nothing is overwritten.

&#91;embedded content: inverse sqrt lineage · 8 designs, 3 depth classes\]

This is the lineage Strata must be able to grow from the hosts' base circuit: three sibling designs under one parent, two of them kept although they do not win today.

1. **Init.** `strata init` reads the project file: one task per base circuit, with its verifier, cost formula and input domain.
2. **Analyse.** `strata analyze` produces the analysis pack for each base circuit.
3. **Lift.** `strata lift` proposes a contract tree. A person or an agent names the regions and confirms. The result is frozen as design D0.
4. **Rank.** `strata gaps` lists every region with its cost, its floor and the gap, weighted by the task's cost formula.
5. **Open lines.** A line is one hypothesis about one contract change. It branches from any design, not only the newest.
6. **Pass the gates.** Each line passes an exhaustive test at small size, then a price measured by a build, then the official verifier at full size.
7. **Merge or keep apart.** Two lines that touch different regions merge automatically. Two lines that change the same contract stay as sibling designs.
8. **Portfolio.** `strata best` picks the cheapest verified design per task and computes the contest score.

Three rules keep the search wide:

- **A parked line stays, with its reopen condition.** The 12-bit-root inverse sqrt costs 97,142 today. It stays in the lineage with "reopens if the table word loses two wires".
- **A design is not a file.** Parameter changes give new files of the same design. Only a contract change gives a new design.
- **Every new base is ingested again.** When a rival publishes a better circuit, Strata lifts it, compares its contract tree with ours and lists which of our patches still apply.

## Pipeline in detail

Six stages turn a flat file into a ranked list of places to work.

### 1. Ingest

Reads `.kmb` and `.kmx` through kickmix and records the hash, the gate census and the registers. It runs the official verifier once and stores the receipt as the reference cost.

### 2. Analysis pack

Everything a person would look at first, produced in one run and stored as data, not only as pictures.

| View | Source | Answers |
| --- | --- | --- |
| Flame chart | kickmix `flame_chart_svg` | Where the Toffolis sit in time |
| Wire diagram | kickmix `html_diagram` | What a small window does |
| Reaction depth | kickmix `reaction_depth` | How measurement outcomes chain |
| Liveness profile | Strata | Which wires are non-zero at each operation, and the peaks |
| Allocation trace | Strata | Which object holds each wire at each peak |
| Executed-cost profile | Strata | Mean executed Toffolis per region over the real input distribution |
| Value fingerprints | Strata | Which wires hold the same value, a linear function or a product of others |
| Conditional profile | Strata | Which measurement bits control which later gates |
| Slack profile | Strata | Bits of information on the live wires at each peak, against the wire count |

### 3. Lift

Finds region boundaries where the live state is smallest, then types each region against a block library: adder, comparator, controlled shift, lookup, field product, field inverse, zero test, measured uncompute. Generator tags are used when present and checked, never trusted. A region that matches no block stays "opaque", with its measured boundary.

### 4. Ledger

Builds the cost table per region and answers price queries by running the generator under a constraint: one wire fewer in this region, one helper more, one failed shot allowed. A price is the difference of two builds, so it is exact.

### 5. Floors

Attaches up to three numbers to each contract: a counting bound, the best construction in the block library, and the exact minimum at small size where SAT can decide it. Each number carries its kind, so a proof is never confused with an estimate.

### 6. Lab

Generates the small-size instance of any region, or of the whole design, from the same contract. The lab runs every input, compares with the contract's reference function and sorts failures into classes. It then scales failure rates to full size by counting sites.

## Restructure engine

The engine searches over contract edits, not gate edits, and every edit is one of a fixed set of typed moves.

| Move | What it changes | Example from our work |
| --- | --- | --- |
| Replace the stage list | The algorithm | Table and polynomial replaced by root, table and divider: 15% |
| Re-represent a boundary | The form in which data crosses between two regions | Decoder nodes as value plus derivative: about 2,100 Toffolis net |
| Move a debt | Which region repays a measurement | Slope wires measured when idle and repaid in the unload |
| Fuse regions | Two contracts become one synthesis problem | Four chain products of 99 fused into blocks of 176 and 186 |
| Split a region | One contract becomes two with a cheaper middle | Two-step payload in the convert stage: 1,928 Toffolis |
| Trade wires for products | Helper count against Toffolis, at measured prices | Decoder with 12 more helpers: 3,461 Toffolis back, a net loss of 1.74% |
| Host a guest | An idle wire carries another value for a span | Lowest remainder bit packed into a spare exponent code |
| Relax exactness | The allowed failure set, priced by the cost formula | Extractor cutoff 15 to 17: 258 Toffolis for 0.012 failed shots per run |
| Co-design numerics | Table, rounding rules and angles solved together | Merged low rows with a refitted table: 1.2% |
| Re-synthesise a block | A new minimum for one typed block | Field block rank 176 to 173 by an elliptic-curve scheme |

One search round:

1. Take the regions with the largest weighted gap between cost and floor.
2. For each, list the moves whose preconditions hold. A precondition is a query, such as "is this wire a function of the live wires here" or "which region sets the peak here".
3. Check the negative-results store for the same move on the same contract.
4. Run the lab on the edited contract at small size. A wrong answer on any input stops the line with a counterexample.
5. Build at full size and measure the price. An estimate cannot close a line.
6. Run the official verifier. On a pass, write the patch and its certificate and add the design to the lineage.

Proposals come from three sources: the engine enumerating moves, agents through the typed interface, and people. All three pass the same gates.

## Verification, patches and memory

Nothing enters the lineage without a certificate, and no line is closed without evidence of a stated kind.

**Certificates.** Each holds four items:

- the lab result at small size: all inputs pass, or the counterexample;
- the full-size oracle result on the contest's input domain or draw;
- the official verifier's receipt, with the verifier's hashes pinned;
- the modelled failure rate and the draws it was checked against.

**Patches.** A patch is keyed to a region and an occurrence, not to byte offsets. It survives a rebuild of the file and a change in another region. Patches on different regions compose. Strata refuses to compose two patches on one contract and keeps them as sibling designs.

**Comparing candidates.** Circuits with random failures are compared on expected cost or on fixed seeds, never on one draw. Choosing a variant by its own test draw is blocked.

**Negative results.** Each closed line stores the contract it was tried on, the move, the measured result, the kind of evidence and the condition under which it reopens.

| Evidence kind | May close a line |
| --- | --- |
| Proof | Yes |
| Exact search at small size | Yes for that size; marked "open at full size" |
| Measured build | Yes |
| Ledger or paper estimate | No; the line stays "priced" |

Results attach to contracts, so they survive a change of base file. When a new base is lifted, Strata matches its contracts to the stored ones and reports which closed lines still apply and which reopen.

## Configuration

One file, `strata.toml`, defines a contest: its tasks, base circuits, verifiers, cost formulas and limits. The example below is the Kaggle contest we ran.

```toml
# strata.toml : one project = one contest

[project]
name     = "vqcc"
engine   = "kickmix"              # front end for reading and simulating circuits
store    = "/data/strata/vqcc"    # lineage, patches, certificates, negative results
sandbox  = "no-network"           # third-party generators only run here

[score]
# contest score from the task costs; higher is better
formula  = "geomean(task.base_cost / task.cost)"

# ---------------------------------------------------------------- task 1
[[task]]
id        = "invsqrt"
base      = "data/sample_submission/invsqrt.kmb"
verifier  = "data/verifiers/invsqrt"
cost      = "Q * (toffoli + t / 4)"   # the verifier charges rotations as T gates
resources = ["Q", "toffoli", "t", "rotations", "margin"]

[task.domain]
kind      = "exhaustive"              # every input is checked
register  = "x"
range     = [1, 3133452]

[task.accept]
kind      = "phase"
tolerance = "2^-24 relative"

[task.scale]                          # knobs the lab may shrink
x_bits    = { full = 22, small = [8, 10, 12] }

# ---------------------------------------------------------------- task 2
[[task]]
id        = "dqi"
base      = "data/sample_submission/dqi.kmb"
verifier  = "data/verifiers/dqi"
cost      = "R(fail_rate) * Q^2 * toffoli"
resources = ["Q", "toffoli", "fail_rate"]
forbid    = ["Z_POW"]                 # no rotations in this task

[task.domain]
kind      = "sampled"
shots     = 102400
seeds     = [101, 202, 303]           # fixed draws for comparing candidates
compare   = "expected"                # never pick a variant by one draw

[task.accept]
kind      = "decoder"
max_fail  = "1/16"

[task.scale]
field_bits = { full = 12, small = [3, 4, 5] }
slots      = { full = 35, small = [3, 5, 7] }

# ---------------------------------------------------------------- task 3
[[task]]
id        = "dfthcsos"
base      = "data/sample_submission/dfthcsos"   # circuits + factorisation file
verifier  = "data/verifiers/dfthcsos"
cost      = "lambda_eff * T_total * Q_total"
resources = ["Q_total", "T_total", "lambda_eff", "error_1", "error_2", "E_gap"]

[task.accept]
kind      = "chemistry"
error_1   = "<= 0.020"                # Hartree
error_2   = "<= 0.020"
E_gap     = ">= 3.914898"
circuit_qubits = "<= 128"

[task.scale]
orbitals  = { full = 76, small = [8, 12] }
word_bits = { full = 8,  small = [3, 4] }

# ---------------------------------------------------------------- shared
[analysis]
views = ["flame", "diagram", "reaction_depth", "liveness", "allocation",
         "executed_cost", "fingerprints", "conditionals", "slack"]
fingerprint_inputs = 4096

[lift]
trust_tags         = false            # tags are hints, checked against measurement
min_region_toffoli = 50
block_library      = ["adder", "compare", "controlled_shift", "lookup",
                      "gf_product", "gf_inverse", "zero_test", "measured_uncompute"]

[floors]
kinds         = ["counting", "library", "exact_small"]
sat_timeout_s = 3600

[engine]
moves          = "all"
rank_by        = "weighted_gap"
max_open_lines = 12
close_requires = ["proof", "exact_small", "build"]   # an estimate cannot close a line

[gates]
order = ["lab", "build_price", "oracle_full", "official_verifier"]

[budget]
cores               = 48
memory_gb           = 300
wall_hours_per_line = 6

[publish]
submit = "manual"                     # Strata never submits by itself
```

| Section | What it fixes |
| --- | --- |
| `[[task]]` | One base circuit, its verifier, its cost formula and the resources that formula reads |
| `task.domain` | Whether inputs are checked exhaustively or drawn, and how candidates are compared |
| `task.accept` | The pass rule the verifier applies |
| `task.scale` | Which sizes the lab may shrink, and to what |
| `[lift]`, `[floors]` | How regions are found and typed, and which kinds of floor are computed |
| `[engine]`, `[gates]` | Which moves are searched, how lines are ranked and what may close one |
| `[budget]`, `[publish]` | Machine limits and the rule that submission stays manual |

The cost formula is the only place a contest's objective appears. Marginal prices, the gap ranking and the failure budget are all derived from it.

## Interfaces

Three interfaces sit on one core: a command line for people, a Python API for scripts, and typed queries with small answers for agents.

| Command | Does |
| --- | --- |
| `strata init` | Reads `strata.toml`, ingests the base circuits, stores reference costs |
| `strata analyze TASK` | Writes the analysis pack |
| `strata lift TASK` | Proposes the contract tree; `--confirm` freezes it as a design |
| `strata gaps TASK` | Regions ranked by weighted gap between cost and floor |
| `strata price TASK --region R --resource wire --delta -1` | Exact marginal price, by build |
| `strata line open / run / park / status` | Manages lines of work |
| `strata lab LINE --scale field_bits=4,slots=5` | Exhaustive small-size test of a line |
| `strata verify LINE` | Full-size oracle, then the official verifier |
| `strata diff DESIGN_A DESIGN_B` | Contract-level difference between two designs |
| `strata best` | Cheapest verified design per task and the contest score |
| `strata replay DESIGN` | Rebuilds a design from the base circuit and checks its certificate |

```python
import strata

p  = strata.open("strata.toml")
d0 = p.task("dqi").base_design()

for g in d0.gaps(top=5):                  # region, cost, floor, kind of floor
    print(g.region, g.cost, g.floor, g.floor_kind, g.weighted_gap)

d0.price(region="unextract", resource="helper_wire", delta=+12)

line = p.open_line(parent=d0, move="re-represent",
                   regions=["extract", "unextract"],
                   note="value + derivative nodes")
line.lab(field_bits=4, slots=5)           # every input
line.build()
line.verify()                             # writes the patch and certificate
```

Queries for agents return a few lines and a handle to the full data, so an agent's context is not spent on dumps.

| Query | Answer |
| --- | --- |
| `holders(task, peak)` | Which objects hold each wire at a peak |
| `is_function_of(wire, at, support)` | Yes with the form found, or no with the search that was run |
| `debts(region)` | Measurements made here and where each is repaid |
| `price(region, resource, delta)` | Exact cost change |
| `floor(contract)` | The numbers and their kinds |
| `tried(move, contract)` | Earlier lines, results, evidence kind, reopen condition |
| `applies(patch, design)` | Whether a stored patch still fits a newly lifted design |

## Acceptance tests and success metrics

Strata is accepted when it replays the restructures we found by hand, from the files they started from, and then finds one we did not.

| Test | Start | Must reach | Depth |
| --- | --- | --- | --- |
| R1 | Decoder `7cbe8ecf` (889 wires, 383,156 Toffolis) | The value-plus-derivative extractor | L3 |
| R2 | The same decoder | Carry pairs in all four convert passes (5,536 Toffolis) | L3 |
| R3 | Inverse sqrt `321aff83` (46 wires, cost 80,163) | 64-row table with a sixth digit (cost 76,940) | L3 |
| R4 | Fused chain block at rank 176 | Rank 173 | L2 |
| R5 | The hosts' three base circuits | Complete contract trees with no manual step except naming regions | Analysis |
| R6 | Our 322 logged attempts | Every one closed only by an estimate is reported as "priced", not "closed" | Memory |

| Metric | Target |
| --- | --- |
| Toffolis assigned to a typed region on untagged files | 90% or more |
| Lifted stage boundaries against the generator's own tags | 95% agreement |
| Price query against a hand-measured build | Exact match |
| Time from base file to gap table, 48 cores | Under 30 minutes |
| Lab agreement with the full-size circuit on flagged inputs | 100% (our hand-built decoder lab reached this on 2,612 lanes) |
| Replay tests R1 to R4 found without a hint that names the answer | 3 of 4 |
| New verified design outside the replay set, on any task | At least 1 |

## Build plan, risks and open questions

Build the model before the search. Without the lift, the ledger and the lab, a proposal engine only repeats what agents do today.

| Phase | Builds | Gate to pass |
| --- | --- | --- |
| 1 | Ingest, analysis pack, ledger on tagged files | Reproduces our measured stage tables for the three Kaggle circuits |
| 2 | Lift on untagged files, block library | 90% of Toffolis typed on the public 44-wire inverse sqrt and on the untagged decoder `7cbe8ecf` |
| 3 | Scale parameters and the lab | Matches our hand-built decoder lab on all 16 recorded draws |
| 4 | Patch format, composition, lineage store, negative results | Replays our decoder patch chain byte for byte |
| 5 | Floors and the gap table | Ranks the regions where we found gains above the regions where we found none |
| 6 | Move catalogue and search loop | Replay tests R1 to R4 |
| 7 | Second front end: the ecdsa.fail operation stream | Lifts one 634-qubit row |

Risks:

- **Lifting may not generalise.** Typing regions against a block library works for arithmetic and field circuits. A circuit made of unfamiliar blocks lifts to "opaque" regions until the library grows.
- **Scale-down needs the generator.** A file without source can be lifted but not shrunk. For those, the lab works per typed block only.
- **Floors can mislead.** A loose counting bound makes a region look worse than it is. That is why every floor carries its kind.
- **Prices by build are slow on large circuits.** A decoder build takes minutes, so price queries must be cached by region and constraint.
- **Numeric co-design is specific to each task.** Table fits and rounding need a model per task. Strata can give them a common interface, not a common solver.

Open questions:

- [ ] Scope at the start: kickmix only, or the format-neutral core from the first day?
- [ ] Is the library open source, and under which licence?
- [ ] Do contest hosts ship the project file, or does each team write its own?
- [ ] The name. "Strata" is a working title.
