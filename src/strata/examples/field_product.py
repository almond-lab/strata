"""Compare two source implementations of z ^= a*b over GF(4)."""
import json

import kickmix

from strata.lab import exhaustive
from strata.model import Contract, Register


def schoolbook(builder, a, b, z):
    """Four coefficient products, reduced modulo x² + x + 1."""
    builder.ccx(a[0], b[0], z[0])
    builder.ccx(a[0], b[1], z[1])
    builder.ccx(a[1], b[0], z[1])
    # Add the high coefficient to both output lanes, preserving incoming z.
    builder.cx(z[0], z[1])
    builder.ccx(a[1], b[1], z[0])
    builder.cx(z[0], z[1])


def karatsuba(builder, a, b, z):
    """Reuse a0*b0 in both outputs and compute the cross term by evaluation."""
    builder.cx(z[0], z[1])
    builder.ccx(a[0], b[0], z[0])
    builder.cx(z[0], z[1])
    builder.ccx(a[1], b[1], z[0])
    builder.cx(a[0], a[1])
    builder.cx(b[0], b[1])
    builder.ccx(a[1], b[1], z[1])
    builder.cx(b[0], b[1])
    builder.cx(a[0], a[1])


def reference(values, _scale):
    a, b, z = (values[name] for name in ('a', 'b', 'z'))
    product = (a if b & 1 else 0) ^ ((a << 1) if b & 2 else 0)
    if product & 4:
        product ^= 0b111
    return dict(a=a, b=b, z=z ^ product), 0


def check_stage(emit):
    builder = kickmix.CircuitBuilder()
    wires = builder.alloc_qubits(6)
    emit(builder, wires[:2], wires[2:4], wires[4:])
    circuit = builder.finish_circuit()
    contract = Contract('GF(4) multiply into accumulator',
                        (Register('a', (0, 1)), Register('b', (2, 3)),
                         Register('z', (4, 5), role='output')),
                        reference='polynomial product modulo x^2+x+1')
    return circuit, exhaustive(circuit, contract, reference)


def main():
    for emit in (schoolbook, karatsuba):
        circuit, receipt = check_stage(emit)
        if not (receipt.passed and receipt.exhaustive):
            raise SystemExit(f'{emit.__name__} failed: {receipt.counterexample}')
        print(json.dumps(dict(generator=emit.__name__, exact_cases=receipt.cases,
                              qubits=circuit.num_qubits,
                              toffolis=circuit.operation_statistics().num_toffoli_like)))


if __name__ == '__main__':
    main()
