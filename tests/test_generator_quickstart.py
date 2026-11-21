from strata.examples.field_product import check_stage, karatsuba, schoolbook


def test_field_product_examples():
    for emit, expected in ((schoolbook, 4), (karatsuba, 3)):
        circuit, receipt = check_stage(emit)
        assert receipt.passed and receipt.exhaustive and receipt.cases == 64
        assert circuit.num_qubits == 6
        assert circuit.operation_statistics().num_toffoli_like == expected


def test_changed_input_is_rejected():
    def unrestored(builder, a, b, z):
        karatsuba(builder, a, b, z)
        builder.cx(a[0], a[1])

    _, receipt = check_stage(unrestored)
    assert not receipt.passed and receipt.counterexample
