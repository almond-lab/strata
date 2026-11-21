from itertools import combinations_with_replacement, product, repeat

import kickmix as km

from strata.quadratic_rank_floor import quadratic_rank_floor


def truth_rows(n):
    return list(product((0, 1), repeat=n))


def test_matches_all_three_input_circuits_through_two_ands():
    rows = truth_rows(3)
    wires = [sum(bits[i] << j for j, bits in enumerate(rows)) for i in range(3)]
    affine = {0, (1 << len(rows)) - 1}
    for q in wires:
        affine |= {a ^ q for a in affine}

    def coset(table):
        return min(table ^ a for a in affine)

    def bound(tables):
        outputs = [[(value >> j) & 1 for value in tables] for j in range(len(rows))]
        return quadratic_rank_floor(rows, outputs)

    first = {coset(a & b) for a, b in combinations_with_replacement(affine, 2)}
    seen = set()
    for g in first:
        assert bound([g])["value"] == int(bool(g))
        next_inputs = affine | {a ^ g for a in affine}
        for a, b in combinations_with_replacement(next_inputs, 2):
            h = coset(a & b)
            pair = (g, h)
            if pair not in seen:
                report = bound(pair)
                if report is not None:
                    assert report["value"] <= 2
                seen.add(pair)
    assert any(g and h and g != h for g, h in seen)


def test_scalar_rank_and_linearly_dependent_outputs():
    rows = truth_rows(4)[::-1]
    out = []
    for a, b, c, d in rows:
        q = (a & b) ^ (c & d)
        out.append([q, q ^ a, a ^ c])
    certificate = quadratic_rank_floor(rows, out)
    assert certificate["value"] == 2
    assert certificate["independent_output_indices"] == [0]
    assert certificate["polar_rank_histogram"] == {4: 1}
    assert quadratic_rank_floor(rows, [[a ^ b] for a, b, c, d in rows])["value"] == 0


def test_unsupported_domains_and_degrees():
    rows = truth_rows(3)
    assert quadratic_rank_floor(rows[:-1], [[0]] * 7) is None
    assert quadratic_rank_floor(rows, [[a & b & c] for a, b, c in rows]) is None
    assert quadratic_rank_floor(truth_rows(9), [[0]] * 512) is None
    assert quadratic_rank_floor([(), ()], [[], []])["value"] == 0
    assert quadratic_rank_floor(repeat([0]), repeat([0])) is None
    assert quadratic_rank_floor([repeat(0)], [[0]]) is None


def test_gf8_cubic_modulus_with_six_product_kickmix_upper_witness():
    # F_2[T]/(T^3+T+1), polynomial-basis operands q0..2 and q3..5.
    # Intermediates are retained, as allowed in the XOR/AND value-map model.
    rows = truth_rows(6)
    expected = []
    for row in rows:
        a = sum(row[i] << i for i in range(3))
        b = sum(row[i + 3] << i for i in range(3))
        raw = 0
        for i in range(3):
            if (b >> i) & 1:
                raw ^= a << i
        for degree in (4, 3):
            if (raw >> degree) & 1:
                raw ^= 0b1011 << (degree - 3)
        expected.append([(raw >> i) & 1 for i in range(3)])

    lines = [f"CCX q{i} q{i+3} q{i+6}" for i in range(3)]
    for destination, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 9):
        mix = [f"CX q{i} q{j}", f"CX q{i+3} q{j+3}"]
        lines.extend([*mix, f"CCX q{j} q{j+3} q{destination}", *reversed(mix)])
    for output, terms in enumerate(((6, 7, 8, 11), (6, 9, 11), (6, 7, 10)), 12):
        lines.extend(f"CX q{term} q{output}" for term in terms)

    assert sum(line.startswith("CCX ") for line in lines) == 6
    sim = km.Simulator(64)
    for i in range(6):
        sim.write_across_shots(km.q(i), sum(row[i] << j for j, row in enumerate(rows)))
    sim.do(km.Circuit("\n".join(lines)))
    for i in range(3):
        assert sim.read_across_shots(km.q(12 + i)).tolist() == [bool(row[i]) for row in expected]
    assert not sim.read_phase_flipped_across_shots().any()

    proof = quadratic_rank_floor(rows, expected)
    assert proof["independent_output_indices"] == [0, 1, 2]
    assert proof["polar_rank_histogram"] == {6: 7}
    assert proof["value"] == 5
    assert proof["affine_operand_only_bound"]["value"] == 6

