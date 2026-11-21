# Redesign one generator stage

Pick a source function whose implementation could change the task score. For a decoder this might be syndrome-to-locator reconstruction. For a chemistry loader it might be the joint choice of angle words and normalization. Name the routine that consumes its result. Together they define which internal choices can change and which outputs the caller needs.

## Try two algorithms against one contract

From an environment with Strata 0.6.1 installed, run the [field-product example](../src/strata/examples/field_product.py).

```sh
python -m strata.examples.field_product
```

The stage adds a field product to an arbitrary two-bit accumulator and restores both inputs. Its independent reference multiplies polynomials and reduces modulo x² + x + 1. Each generator receives the same three registers. Schoolbook multiplication forms four coefficient products. Karatsuba forms three by also evaluating both inputs at one, then restores the input basis. Kickmix executes what each generator emits. Strata checks all 64 input/accumulator combinations and the output phase. The two result rows give 4 and 3 Toffolis respectively at 6 qubits. These are measured costs for one GF(4) operation.

To try a third algorithm, add a function with the same signature to a working copy and include it in the loop. Keep the reference fixed. Try deleting the final input restoration in the Karatsuba function in that copy. The checker prints a failing input even though the field product reaches the accumulator.

## Use the real generator

1. Identify the source entry point and pin the parent implementation. Record the few stages that dominate cost, how often each runs, its forward and return work, and where its live registers contribute to peak width. Select one stage whose potential saving matters under the task's score formula.
2. State its external contract in terms a caller can test. Include legal inputs, output encoding, retained values, scratch restoration and phase. If a proposed representation changes the interface, extend the experiment through the consumer and the return path. For an approximate chemistry stage, use the task's physical error and normalization checks as well.
3. Implement an alternative in the generator. Run its smallest meaningful instance against an independent oracle. Preserve counterexamples. A passing reduced instance establishes that instance's behavior. Use the [small-first workflow](SMALL-FIRST.md) for evidence linking a scaled build to the same source.
4. Translate the target score into a budget for the modified task with the other tasks fixed. Compare both designs on the same input distribution. Include conversions, reconstruction, cleanup, invocation counts, and any resources displaced into another stage. Compute width from simultaneous live registers. Record which inputs to the forecast are measured, modeled, or unknown.
5. Build a full candidate when the complete estimate supports the target. Regenerate from source, run the independent task verifier, and retain the source revision and replay evidence. If combining changes from different parents, port the source changes to a common parent and verify the combination.

For example, a decoder recurrence can save multiplications at every interpolation node while leaving the locator in a form the mask scan cannot use. Test the locator or the observable phase from the public syndrome before crediting the loop saving. If four coefficient registers make a particular reversible implementation too wide, record that schedule and the cost of those registers. Another schedule can reuse them and requires its own check.

## Use the existing tools

- [Project adapters](ADAPTER-GUIDE.md) connect the real builder, reference and task verifier. [Structural research rounds](STRUCTURAL-RESEARCH.md) run alternatives with their consumers and cleanup, retaining verified outcomes and reopen conditions.
- [Experiment runs and sweeps](TEN-MINUTES.md) wrap an existing generator or numeric fitter. [Numeric studies](V0.3-NUMERIC-STUDIES.md) handle modeled errors, resource prices and candidate comparisons.
- [File analysis](FILE-FIRST.md) helps when only compiled bytes are available or when a source change needs a native diagnostic. [Patch composition](patch-composition.md) covers existing compiled edits with a known common parent.
