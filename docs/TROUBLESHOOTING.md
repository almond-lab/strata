# Troubleshooting and interpreting a stopped line

| Message or observation | Meaning | Appropriate response |
|---|---|---|
| `task needs adapter` | There is no trusted builder/reference/verifier connection | Register the task adapter; analysis alone cannot supply it |
| `generator does not reproduce the supplied base` | Source/parameters and base bytes differ | Recover and pin the correct generator; do not weaken the comparison |
| `external specification changed` | Candidate changed the reference, promise, phase or failure rule | Redesign the internal boundary or deliberately extend the framework's acceptance model |
| `patch ... outside declared regions` | The builder changed undeclared operations | Declare the actual producer–consumer region; include its full cost |
| `L2 requires the same region contract` | The change is deeper than a same-contract block replacement | Classify and declare the appropriate L3/L4 move |
| `unknown region` after fusion | Replacement region names were not declared | Set `replacement_regions` to the new region IDs |
| `exhaustive domain exceeds budget` | The requested lab is too large | Choose a declared smaller size or a suitable exact proof backend; a sample is not exhaustive |
| Lab counterexample has phase only | Values and scratch can be correct while HMR debt or rotation phase is wrong | Inspect the input, branch and exact rational phase; preserve the reference |
| `constrained build did not achieve ... delta` | Generator did not meet the requested wire change | Fix the allocation schedule or report the constraint infeasible for this implementation |
| `project cost formula changed` | Old prices no longer describe the active objective | Start a newly priced certification; do not silently reuse old costs |
| `artifact integrity failure` | Stored bytes changed | Restore the pinned artifact or regenerate and recertify it |
| `sandbox failed` | Bubblewrap or its isolation setup failed | Fix host capability/mounts; do not silently run the generator outside isolation |
| High primitive coverage, zero structural coverage | Native gates were recognized, not arithmetic stage contracts | Add certified semantic block recognition; do not report the 90% lifting target as achieved |
| Correct circuit, larger cost | The attempt is useful evidence but is not an improvement | Retain it as a sibling with its measured price and possible reopen condition |

Useful records live under the configured project store. `line status` returns
the event history; `best` considers certified costs; `replay` rebuilds and
rechecks the chosen design. Large artifacts remain behind hashes, while agent
queries can return small previews and handles.
