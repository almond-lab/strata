# Transfer on twenty production circuit episodes

The benchmark runs ordinary Strata search on consecutive native operation spans taken directly from the supplied candidate archive. The parent of each span is bound to the archive's companion manifest. The archive SHA-256 is `0212e251262a4d3d40ab5a36af664e8e11f24c14305db9c2105ceaf22253e443`. The decoder member contains 50,934,412 operations at 889 qubits, the inverse member contains 22,143 at 44 qubits, and the five sampled chemistry loaders contain 38,954–42,446 operations at 125–128 qubits.

## Results

**Neither release run found a cheaper episode.** In each run, Kickmix confirmed exact native identity between the input and the retained incumbent for all 20 cases. Every original physical span and every declared clean input matched between runs. The universal run took 52.33 seconds and the run with additional proved promises took 49.69 seconds. Both used the installed v0.5.0 wheel with SHA-256 `2cdc93d7ad590130c722f7ab3cd1a1b3fee6fe0e51082901bc15799151f6c74a`. [Universal receipt](release-universal.json) · [Promise receipt](release-promises.json)

| Family | Episodes | Static paid gates across selected spans, before → retained | Cheaper, universal | Cheaper, with derived promises |
|---|---:|---:|---:|---:|
| Inverse square root | 7 | 16 → 16 | 0/7 | 0/7 |
| Decoder | 6 | 31 → 31 | 0/6 | 0/6 |
| Chemistry loaders | 7 | 28 → 28 | 0/7 | 0/7 |

Five inverse cuts had exact recovered maps but no applicable constructor trial. Two of these cuts gained a proved-private bit in the second run. The decoder had three unresolved reset preconditions. Its measured repayment cut already cost zero. The universal run timed out one constructor on the decoder bilinear cut and one on a loader address bank. All 20 workers completed in each run. Bounded negatives give no optimality conclusion for positive-cost cells.

Four loader cuts had a strictly smaller admissible set. The loader-0 word cut admitted 20 patterns on five determined coordinates with 13 arbitrary wires. No cheaper completion passed. The loader-1 address-products cut admitted 40 patterns on 14 coordinates but its resets exceeded the completion constructor’s supported instruction set. The loader-1 and loader-7 word cuts exceeded the source-table budget. Two other projections admitted the original cube, and one prefix had an unproved reset. The latter case used only its universal boundary facts.

### Secondary inverse endpoint pricing

The 13-operation endpoint slice contains one CCX, one fine rotation and zero native fixed T gates. The primary static paid-gate objective assigns no price to the fine rotation. A separate replay pinned Q to the parent width of 44. Under `Q*(toffoli+t/4)` the incumbent cost stayed at **44** and the verified fractional proposal tied it. To test explicit rotation pricing, the second replay used **29.287992121699897 synthesized T per rotation**, from the archived cand1 inverse precision epsilon. Its incumbent cost stayed at **366.1679133386989** and its verified proposal cost 88 more. Both retained native bodies were identical to the supplied span. The rotation rate was held fixed for this sensitivity check. A changed whole circuit would require repricing its actual rotation budget and precision margin. [Endpoint receipt](fractional-endpoint.json)

The objective is the number of statically emitted CCX, CCZ and CSWAP gates in each supplied span. This fixed-width proxy makes a retained local saving easy to price. An official inverse score also prices rotations and executed gates, the decoder prices reliability and executed gates at squared width, and the chemistry score includes host work and its lambda factor. Each result is for its own selected cut. No official verifier, leaderboard submission or portfolio score is part of this benchmark.

## Inputs and boundary evidence

The inverse cuts include normalization, a small root cell, carry updates and a fractional endpoint. Decoder cuts include a measured flag, a bilinear update, a complete guarded scan, its conjunction and repayment stages, and a guarded phase mark. Chemistry cuts cover address products and word updates in loaders 0, 1, 3, 7 and 14. The locations are in zero-based, half-open native operation coordinates. Earlier retained-circuit profiles identified likely families. The range selection and all hashes use the supplied archive. This is a curated transfer benchmark on exposed cases.

A preliminary development-wheel pilot used arbitrary live words and classical outputs. One initial candidate cut crossed a condition pop and was corrected before completing the pilot. Both release-wheel runs use the same final twenty ranges, native bytes, static objective and search budgets. The stronger boundary promises were added after seeing the pilot. They come from the original parent program and caller interface. They should be read as a separate experiment.

* The universal control permits arbitrary values on every active input except transported clean facts. For inverse and chemistry, the audited initialization sets unregistered workspace to zero. All input and word registers remain arbitrary. An unconditional HMR establishes a new zero fact. Register numbers and apparent reset idioms confer no zero promise. The decoder starts each 1,024-operation lookback without clean promises. The scanner checks the true enclosing conditional depth.
* The promise replay also hides an existing classical output only if its first use in the next 1,024 parent operations is an unconditional overwrite. The recorded overwrite position can be checked against the native parent. The guard remains unchanged around every selected region.
* Each loader accepts an 11-bit selection in 0–1,219 and five incoming eight-bit words. A three-valued evaluation of the actual prefix tracks, separately at each address, which bits are determined for every incoming dirty word and measurement history. The projection admits every remaining active wire independently. This remains valid for both loader directions. For each completed prefix analysis, a seeded independent native simulation checks asserted bits at all 1,220 addresses. The receipt records whether the derived set restricts the ordinary input cube.

The runner copies one compressed member at a time into a private temporary file. For the decoder, bounded opcode histograms locate operand-stream offsets and Kickmix decodes only the requested KMB spans. For every small parent, a separate native decode hashes the same spans and checks the extraction. A reversible renaming packs active wires for search. The receipt binds the physical wires and original native bytes. If a changed winner fits the replay budget, the runner compares every allowed quantum and classical input, every visible output and its phase. The runner also reconstructs and statically reprices changed small parents after fresh bit allocation.

## Replay

Install the frozen release wheel in an isolated environment with its declared Kickmix and NumPy dependencies. Pass the original archive together with the adjacent `MANIFEST.txt`. The private scratch directory should be outside the checkout and have space for the 437 MB decoder. Use new output filenames.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TMPDIR="$PWD/build/verification-tmp"
mkdir -p "$TMPDIR"
python benchmarks/v05-transfer/run.py --wheel "$WHEEL" \
  --archive "$ARCHIVE" --scratch "$PRIVATE_SCRATCH" --universal --out "$RESULTS/control.json"
python benchmarks/v05-transfer/run.py --wheel "$WHEEL" \
  --archive "$ARCHIVE" --scratch "$PRIVATE_SCRATCH" --out "$RESULTS/promises.json"
python benchmarks/v05-transfer/check_fraction.py \
  --archive "$ARCHIVE" --scratch "$PRIVATE_SCRATCH" --out "$RESULTS/endpoint.json"
```

The CLI gets one round, up to 24 candidate attempts, 128 architectural proposals, 320 operations, 96 variables, 8,192 terms and product terms, 12 seconds per case to start constructor trials and 2 seconds per trial. A separate worker limit is 30 seconds. The native changed-candidate check has a 65,536-assignment cap. Wall time and scheduling-dependent trial counts can vary on replay. Receipt hashes bind the search wheel, installed Python sources, native SDK and benchmark code.
