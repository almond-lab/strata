# Composing changes from two native circuits

Use `patch-plan` when you know which operation intervals a donor changed relative to a particular parent. It pins the parent and donor files and checks every proposed interval against the actual native instructions. Use `patch-compose` to apply plans from that parent in its original operation order. The result is a candidate for separate semantic verification.

## Try two independent cancellations

From a new scratch directory with the Strata CLI on your path, run the following commands. The first donor removes two X gates from the start, and the second removes two Z gates from the end. Their native operation counts happen to agree, but each eliminates different gates.

```sh
printf 'X q0\nX q0\nCX q0 q1\nZ q1\nZ q1\n' > parent.kmx
printf 'CX q0 q1\nZ q1\nZ q1\n' > left.kmx
printf 'X q0\nX q0\nCX q0 q1\n' > right.kmx

strata patch-plan parent.kmx left.kmx --edit 0 2 0 0 > left.json
strata patch-plan parent.kmx right.kmx --edit 3 5 3 3 > right.json
strata patch-compose parent.kmx right.json left.json --output combined.kmx > receipt.json
cat combined.kmx
```

The combined file contains one `CX q0 q1`. Reversing plan order reproduces the same artifact bytes. Existing output paths are refused. Use `.kmb` as the output suffix for native binary.

## Operation cuts and verification

Each `--edit P_START P_STOP D_START D_STOP` gives a zero-based, half-open interval in the immutable parent and a corresponding interval in that donor. A donor with several edits supplies the flag once per interval, in order. An empty parent interval inserts. An empty donor interval deletes. For a length-changing edit, each subsequent donor coordinate refers to the actual donor file. All parent coordinates continue to refer to the original file.

Coordinates count Kickmix native operations, including classical operations, condition operations and tags. Register declarations and register membership are checked separately and must agree. Native serialization determines the spelling of an instruction. Comments, source formatting and text line numbers are outside this coordinate system. Equal instructions at different offsets can be distinguished by supplying explicit cuts. The command does not infer historical intent or discover an alignment.

The plan binds both complete artifact files by SHA-256 and pins the original operation fragment for each edit. The fragment digest uses the canonical UTF-8 instruction lines with one LF per line, including the last one. Its empty digest is SHA-256 of zero bytes. Relative donor paths in plans resolve against the plan's directory. Generated plans record the resolved donor path. Rebinding a path to a byte-identical artifact is supported and changes the plan fingerprint when the plan file changes.

Composition rechecks artifact pins, fragment pins, the complete unchanged gaps and tails, and identical serialized register declarations. It accepts adjacent nonempty parent cuts and rejects intersections, repeated insertion cuts, insertion at either endpoint of another cut, and empty-to-empty or identical replacements. Exactness applies to the edits as declared. A broader interval can include unchanged context and can conservatively collide with another donor. Choose the intervals whose composition you intend to test.

The receipt identifies the parent, donor bytes, plan bytes, original and replacement fragments, and resulting artifact. `geometry: verified` records completion of the structural checks. `semantic_equivalence: unverified` and `accepted: false` leave acceptance to the verifier for the intended contract. Disjoint intervals may interact through a live scratch wire, measurement branch, or phase. The regression test includes two donors which each satisfy the identity contract on every input with clean scratch. Their composition changes the relative phase of the input state when its data bit is one. Replay both values and phases on the fresh compound artifact.

## Resources and publication

Native input files are limited to Kickmix `.kmx` and `.kmb`. The implementation snapshots each input, canonicalizes with the installed Kickmix SDK, and streams spans from disk-backed instruction tapes. The Python index stores one offset per 1,024 instructions per tape. The SDK requires memory for one complete circuit at a time. Temporary space scales with all retained operation tapes plus the canonical and output staging files. Binary input also incurs a canonical text reparse to check native widths and register count. Multiple full passes over large files can take time. For large circuits, set `TMPDIR` to a filesystem with enough capacity.

The final file is staged in the output directory and published using a hard link. The output filesystem must support hard links. A conflicting plan, mismatched file, native parse failure or existing destination does not replace an input or publish a partially written result.
