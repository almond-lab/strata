"""Comprehensive tests for Strata v0.7.0 C++20 native latency engine and Layer A discovery modules."""
from __future__ import annotations

import json
from pathlib import Path
import time

import kickmix
import numpy as np
import pytest

import strata
from strata import cli
from strata.patch_composition import compose_patches, make_patch_plan


def _build_sample_circuit(path: Path, *, num_blocks: int = 8) -> kickmix.Circuit:
    lines = [
        "APPEND_TO_REGISTER q0 r0",
        "APPEND_TO_REGISTER q1 r0",
        "APPEND_TO_REGISTER q2 r0",
        "APPEND_TO_REGISTER q3 r0",
        "APPEND_TO_REGISTER q4 r0",
        "APPEND_TO_REGISTER q5 r0",
        'REGISTER r0 "data"',
        # op 0..1: Set up q7 = q0 ^ q1 as a persistent idle affine wire across inner blocks!
        "CX q0 q7",
        "CX q1 q7",
    ]
    # Each block has 6 ops:
    #   +0: Z_POW q2 <angle>
    #   +1: CCX q0 q1 q8
    #   +2: CCX q2 q8 q9
    #   +3: CX q9 q{3 + blk%3}
    #   +4: CCX q2 q8 q9   (uncompute q9)
    #   +5: CCX q0 q1 q8   (uncompute q8)
    for blk in range(num_blocks):
        ang = 0.25 * ((blk % 3) + 1)
        lines.extend([
            f"Z_POW q2 {ang}",
            "CCX q0 q1 q8",
            "CCX q2 q8 q9",
            f"CX q9 q{3 + (blk % 3)}",
            "CCX q2 q8 q9",
            "CCX q0 q1 q8",
        ])
    lines.extend([
        "HMR q6 b0",
        "CZ q0 q1 if b0",
        "BIT_STORE0 b3",
        'INFO_TAG "sample_v07"',
    ])
    circ = kickmix.Circuit("\n".join(lines))
    circ.to_file(str(path), format="kmb" if path.suffix == ".kmb" else "kmx")
    return circ


def test_native_extension_and_latency_sla():
    assert strata.__version__ == "0.7.0"
    assert strata.NATIVE_AVAILABLE is True
    bench = strata.benchmark_latency(num_ops=50_000, num_qubits=32, num_cones=8)
    assert bench["native_available"] is True
    assert bench["kmb_open_index_ms"] < 50.0
    assert bench["single_cone_256_ms"] < 50.0
    assert bench["batch_per_candidate_ms"] < 25.0


def test_kmb_tape_roundtrip_and_fast_compose(tmp_path: Path):
    kmb = tmp_path / "parent.kmb"
    kmx = tmp_path / "parent.kmx"
    circ = _build_sample_circuit(kmb, num_blocks=4)
    circ.to_file(str(kmx), format="kmx")

    tape = strata.KmbTape.open(kmb)
    assert tape.num_ops == len(circ)
    assert tape.num_qubits == circ.num_qubits
    assert tape.num_bits == circ.num_bits

    expected_ops = [
        ln for ln in kmx.read_text().splitlines()
        if ln and not ln.startswith(("REGISTER", "APPEND_TO_REGISTER"))
    ]
    assert tape.decode_span(0, tape.num_ops) == expected_ops
    assert tape.same_span(2, 15, tape, 2, 15)

    # Splice swapped control order on CCX at op 3 (`CCX q0 q1 q8` -> `CCX q1 q0 q8`) and verify cone!
    assert expected_ops[3] == "CCX q0 q1 q8"
    out_kmb = tmp_path / "spliced.kmb"
    receipt = strata.compose_kmb_fast(
        kmb,
        [(3, 4, "CCX q1 q0 q8\n")],
        out_kmb,
        verify_cone=True,
    )
    assert receipt["verify_cone"] is True
    assert len(receipt["cone_checks"]) == 1 and receipt["cone_checks"][0]["passed"]
    reloaded = kickmix.Circuit.from_file(str(out_kmb))
    assert len(reloaded) == len(circ)

    # Semantic bug must fail verify_cone immediately
    bad_kmb = tmp_path / "bad.kmb"
    with pytest.raises(strata.StrataError, match="causal-cone equivalence failed"):
        strata.compose_kmb_fast(kmb, [(3, 4, "CX q0 q8\n")], bad_kmb, verify_cone=True)
    assert not bad_kmb.exists()


def test_session_funnel_and_patch_compose_verify_cone(tmp_path: Path):
    parent = tmp_path / "parent.kmb"
    donor_good = tmp_path / "donor_good.kmb"
    donor_bad = tmp_path / "donor_bad.kmb"
    circ = _build_sample_circuit(parent, num_blocks=3)

    # Create donor_good (commute controls on CCX at op 3) and donor_bad (drop CCX at op 3)
    kmx_lines = (tmp_path / "tmp.kmx")
    circ.to_file(str(kmx_lines), format="kmx")
    raw_lines = kmx_lines.read_text().splitlines()
    # Find the first CCX q0 q1 q8 line
    idx_ccx = raw_lines.index("CCX q0 q1 q8")
    good_lines = list(raw_lines)
    good_lines[idx_ccx] = "CCX q1 q0 q8"
    bad_lines = list(raw_lines)
    bad_lines[idx_ccx] = "CX q0 q8"

    (tmp_path / "good.kmx").write_text("\n".join(good_lines) + "\n")
    (tmp_path / "bad.kmx").write_text("\n".join(bad_lines) + "\n")
    kickmix.Circuit.from_file(str(tmp_path / "good.kmx")).to_file(str(donor_good), format="kmb")
    kickmix.Circuit.from_file(str(tmp_path / "bad.kmx")).to_file(str(donor_bad), format="kmb")

    plan_good = tmp_path / "plan_good.json"
    plan_good.write_text(json.dumps(make_patch_plan(parent, donor_good, [(3, 4, 3, 4)])))
    out_good = tmp_path / "composed_good.kmb"
    rec = compose_patches(parent, [plan_good], out_good, verify_cone=True)
    assert rec["semantic_equivalence"] == "cone_sampled_256_verified"
    assert rec["cone_checks"][0]["passed"] is True

    plan_bad = tmp_path / "plan_bad.json"
    plan_bad.write_text(json.dumps(make_patch_plan(parent, donor_bad, [(3, 4, 3, 4)])))
    with pytest.raises(strata.StrataError, match="causal-cone equivalence failed"):
        compose_patches(parent, [plan_bad], tmp_path / "composed_bad.kmb", verify_cone=True)

    # Test 4-tier Session.funnel
    with strata.Session.open(parent) as sess:
        funnel_res = sess.funnel(
            [
                strata.FunnelCandidate("good_commute", 3, 4, "CCX q1 q0 q8\n", parent_toffolis=1),
                strata.FunnelCandidate("bad_drop", 3, 4, "CX q0 q8\n", parent_toffolis=1),
                strata.FunnelCandidate("too_expensive", 3, 4, "CCX q1 q0 q8\nCCX q1 q0 q8\nCCX q1 q0 q8\n", parent_toffolis=1),
            ],
            max_toffoli_delta=0,
        )
        assert funnel_res["tier0_count"] == 2
        assert funnel_res["tier1_count"] == 1
        assert [s["name"] for s in funnel_res["survivors"]] == ["good_commute"]


def test_cofit_slider_gauge_deadslots_rounder_and_truncator():
    # 1. BoundarySlider + CircuitAwareRounder (columns 4 and 5 form a nonsingular 2x2 restoration basis)
    C = np.array([[1.0, 1.0, 0.5, -0.5, 0.25, 0.50], [0.0, 1.0, -1.0, 0.5, 1.0, -1.0]])
    x0 = np.array([1.4, -0.3, 0.9, 0.2, -0.8, 0.7])
    rhs = C @ x0
    slider = strata.BoundarySlider(
        objective_fn=lambda x: float(np.sum(np.abs(x))),
        constraint_fn=lambda x: (C @ x) - rhs,
        jac_constraint=lambda _: C,
    )
    slid = slider.slide_multistart(x0, n_starts=4, seed=3)
    assert slid.constraint_violation < 1e-8
    assert slid.objective < slid.initial_objective
    assert slid.nullspace_dim == 4

    rounder = strata.CircuitAwareRounder(slider, step_size=1.0 / 32.0, group_size=2)
    qres = rounder.round(slid.x, quantize_mask=[True, True, True, True, False, False], n_seeds=2)
    assert qres["constraint_violation"] < 1e-8
    assert qres["naive_violation"] > 1e-3

    # 2. GaugeSweep preserves Gram matrix A @ A.T while shrinking L1 norm and planting zeros
    rng = np.random.default_rng(11)
    A = rng.standard_normal((24, 5))
    gs = strata.GaugeSweep(metric="l1").sweep(A)
    assert np.allclose(gs.matrix @ gs.matrix.T, A @ A.T, atol=1e-10)
    assert gs.final_norm < gs.initial_norm
    assert gs.zeros_planted >= 1

    gs_sq = strata.GaugeSweep(metric="row_l1_sq").sweep(A)
    assert np.allclose(gs_sq.matrix @ gs_sq.matrix.T, A @ A.T, atol=1e-10)
    assert gs_sq.final_norm <= gs_sq.initial_norm

    # 3. Givens-tree dead slot detection + plant_bit_relations + QuotientChainDP
    # Root angle = 0 -> cos(0)=1 (left child 1 live), sin(0)=0 (right child 2 and its children 5,6 dead!)
    angles = np.array([0.0, 0.4, 0.9, 0.2, 0.3, 1.1, 0.7])
    dead = strata.DeadSlotPlanter.detect_givens_tree_dead_slots(angles)
    assert dead.tolist() == [False, False, True, False, False, True, True]

    planted = strata.DeadSlotPlanter.plant_bit_relations(
        [0b1011, 0b1110, 0b1011, 0b0101],
        [False, True, False, True],
        bit_width=4,
    )
    assert planted["after_xor_weight"] == 0
    assert planted["after_gf2_rank"] <= planted["before_gf2_rank"]

    dp = strata.QuotientChainDP.optimize_chain(
        [0b1111, 0b0000, 0b1111, 0b0000, 0b1111],
        [0b1111, 0b0000, 0b1111, 0b0000, 0b1111],
        bit_width=4,
        boundary_zero=False,
    )
    assert dp["xor_weight"] == 0 and dp["naive_xor_weight"] == 16

    # 4. CompensatedTruncator minimax refitting (pure-NumPy & optional scipy)
    grid = np.linspace(-1.0, 1.0, 128)
    B = np.column_stack([np.ones_like(grid), grid, grid**2, grid**3])
    drift_ok = -(0.02 * B[:, 1] - 0.015 * B[:, 3]) + 0.0003 * np.cos(13 * grid)
    drift_bad = -(0.02 * B[:, 1]) + 0.004 * np.cos(13 * grid)
    trunc = strata.CompensatedTruncator(B, budget=1e-3)
    best = trunc.select_best_truncation([
        strata.TruncationChoice("aggressive_ok", toffolis_saved=300, residual_error=drift_ok),
        strata.TruncationChoice("too_coarse", toffolis_saved=450, residual_error=drift_bad),
    ])
    assert best["selected"]["name"] == "aggressive_ok"
    assert best["selected"]["uncompensated_max_abs_error"] > 0.004
    assert best["selected"]["compensated_max_abs_error"] < 1e-3


def test_mbu_demolish_phase_fit_and_swap_hoist(tmp_path: Path):
    kmb = tmp_path / "mbu_target.kmb"
    _build_sample_circuit(kmb, num_blocks=2)

    with strata.Session.open(kmb) as sess:
        # Block 0 spans ops 2..8; ops 6..8 are the 2-CCX reversible uncompute cascade:
        # op 6: CCX q2 q8 q9; op 7: CCX q0 q1 q8
        res = strata.demolish_and_piggyback(
            sess, 6, 8, clean_wires=(8, 9),
        )
        assert res["toffolis_removed"] == 2
        assert res["verified"] is True
        assert "HMR q9" in res["replacement_kmx"] and "HMR q8" in res["replacement_kmx"]

        # Fit multi-cut phase modulo 8 across cuts [2, 8, 14]
        fit = strata.fit_multi_cut_phase(sess, [2, 8, 14], modulus=8, clean_wires=(8, 9))
        assert fit["consistent"] is True
        assert fit["residual_zero"] is True

    hoisted = strata.hoist_hmr_across_swaps("SWAP q2 q3\nSWAP q3 q4\nHMR q4 b0\nHMR q3 b1\nSWAP q3 q4\n")
    assert hoisted["hoisted_steps"] >= 2
    assert hoisted["swaps_eliminated"] >= 1
    assert hoisted["lines"][0].startswith("HMR q2 b0")


def test_shapes_pebbling_and_affine_wire_loans(tmp_path: Path):
    model = strata.ScoreModel("invsqrt")
    ex = model.exchange_rate(qubits=38, toffolis=1500)
    assert ex["toffolis_per_qubit"] > 100.0

    explorer = strata.ShapeExplorer(model)
    ranked = explorer.rank_shapes([
        {"name": "wide_fast", "qubits": 40, "toffolis": 1400},
        {"name": "narrow_loan", "qubits": 37, "toffolis": 1650},
    ])
    assert ranked["best"]["name"] == "narrow_loan"
    assert len(ranked["pareto_frontier"]) == 2

    # 1. MBU DAG pebbling breaks the 3-qubit classical Bennett barrier on 0->1->2->3->4
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    rev3 = strata.solve_mbu_pebble_schedule(
        edges, 5, input_nodes=[0], output_nodes=[4], max_qubits=3, allow_hmr=False, compute_toffolis=[10]*5,
    )
    mbu3 = strata.solve_mbu_pebble_schedule(
        edges, 5, input_nodes=[0], output_nodes=[4], max_qubits=3, allow_hmr=True, compute_toffolis=[10]*5,
    )
    assert not rev3.feasible and mbu3.feasible and mbu3.hmr_erasures > 0

    # 2. At max_qubits=4 on 0->1->2->3, both succeed and MBU strictly beats reversible Bennett on Toffolis
    edges4 = [(0, 1), (1, 2), (2, 3)]
    rev4 = strata.solve_mbu_pebble_schedule(
        edges4, 4, input_nodes=[0], output_nodes=[3], max_qubits=4, allow_hmr=False, compute_toffolis=[10]*4,
    )
    mbu4 = strata.solve_mbu_pebble_schedule(
        edges4, 4, input_nodes=[0], output_nodes=[3], max_qubits=4, allow_hmr=True, compute_toffolis=[10]*4,
    )
    assert rev4.feasible and mbu4.feasible
    assert mbu4.total_toffolis < rev4.total_toffolis

    # Affine wire loan discovery & verification on sample circuit (q7 == q0 ^ q1 across [2, 8))
    kmb = tmp_path / "loan.kmb"
    _build_sample_circuit(kmb, num_blocks=2)
    with strata.Session.open(kmb) as sess:
        loans = strata.find_affine_wire_loans(sess, 2, 8, clean_wires=(7, 8, 9))
        q7_loans = [ln for ln in loans if ln["loaned_wire"] == 7]
        assert q7_loans and q7_loans[0]["support_wires"] == [0, 1]
        applied = strata.apply_affine_wire_loan(sess, 2, 8, q7_loans[0], clean_wires=(7, 8, 9))
        assert applied["verified"] is True


def test_boundary_classical_program_oom_guard_and_projective_gf2k():
    prog = strata.ClassicalProgram(max_ops=50_000, max_kmb_bytes=500_000)
    in_b = prog.alloc_bits(6)
    M = np.array([
        [1, 1, 0, 1, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1, 0],
    ], dtype=np.uint8)
    out_b = prog.xor_linear_map(M, in_b)
    wt_b = prog.popcount_bits(out_b)
    k0, k1, k2 = prog.alloc_bits(3), prog.alloc_bits(3), prog.alloc_bits(3)
    prog.odd_even_sort([k0, k1, k2], descending=True)

    # Test simulation of linear map + popcount + descending sort (values 2, 6, 5 -> 6, 5, 2)
    init = {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0}
    # k0 = 2 (0,1,0), k1 = 6 (0,1,1), k2 = 5 (1,0,1)
    for idx, val in zip((k0, k1, k2), (2, 6, 5)):
        for bit in range(3):
            init[idx[bit]] = (val >> bit) & 1
    st = prog.simulate(init)
    expected_y = (M @ np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)) & 1
    actual_y = np.array([st[b] for b in out_b], dtype=np.uint8)
    assert np.array_equal(actual_y, expected_y)
    actual_wt = sum(int(st[wt_b[i]]) << i for i in range(len(wt_b)))
    assert actual_wt == int(np.sum(expected_y))
    sorted_vals = [sum(int(st[g[i]]) << i for i in range(3)) for g in (k0, k1, k2)]
    assert sorted_vals == [6, 5, 2]
    # Verify Kickmix parses the emitted 0-Toffoli classical program directly!
    km_prog = kickmix.Circuit(prog.emit_kmx())
    assert km_prog.operation_statistics().num_toffoli_like == 0

    # Verify OOM guard triggers cleanly
    tiny_prog = strata.ClassicalProgram(max_ops=5)
    tiny_bits = tiny_prog.alloc_bits(4)
    with pytest.raises(strata.StrataError, match="max_ops"):
        for _ in range(10):
            tiny_prog.bit_not(tiny_bits[0])

    # Hoist classical prefix
    hoisted = strata.hoist_classical_prefix(
        "X q0\nCX q0 q1\nCCX q0 q1 q2\nHMR q2 b5\nCX q2 q3\n",
        classical_wires=[0, 1, 2],
    )
    assert hoisted["toffolis_hoisted"] == 1
    assert "CCX" not in hoisted["kmx"]

    # Projective GF(2^8) exact equivalence and Toffoli reduction
    gf = strata.ProjectiveGF2k(8)
    nums = [11, 203, 74, 19, 155, 88]
    dens = [37, 91, 14, 240, 63, 129]
    aff = gf.eval_affine_sum(nums, dens)
    _, _, proj = gf.eval_projective_sum(nums, dens)
    assert aff == proj
    est = gf.price_projective_closing(len(nums))
    assert est.toffolis_saved > 500


def test_tensors_and_invariants_and_cli(tmp_path: Path, capsys):
    # 1. Lovasz matroid parity rank on disjoint symplectic pairs x0*x1 ^ x2*x3 -> 2 pair ANDs
    S = np.zeros((1, 4, 4), dtype=np.uint8)
    S[0, 0, 1] = 1
    S[0, 2, 3] = 1
    rk = strata.lovasz_matroid_parity_rank(S)
    assert rk["min_pair_ands"] == 2

    # 2. Relation hyperplanes: bit 2 == bit 0 ^ bit 1 ^ x0 (purely affine combination!)
    words = []
    for x in range(16):
        x0, x1, x2, x3 = (x & 1), ((x >> 1) & 1), ((x >> 2) & 1), ((x >> 3) & 1)
        b0 = (x0 & x1) ^ (x2 & x3)
        b1 = (x0 & x2) ^ x3
        b2 = b0 ^ b1 ^ x0
        words.append(b0 | (b1 << 1) | (b2 << 2))
    rel = strata.search_relation_hyperplanes(words, bit_width=3)
    assert rel["free_affine_relations_exist"] is True
    assert any(r["mask"] == 0b111 and r["pair_ands"] == 0 for r in rel["relations"])

    # 3. Multi-pass QROAM planner
    qroam = strata.plan_multipass_qroam(1024, 24, max_qubits=64)
    assert qroam["best"] is not None and qroam["best"]["qubits"] <= 64

    # 4. Flip-graph bilinear rank reduction
    U = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.uint8)
    V = np.array([[1, 1], [1, 1], [1, 0]], dtype=np.uint8)
    W = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    fg = strata.flip_graph_reduce(U, V, W, steps=200)
    assert fg["reduced_rank"] == 2 and fg["toffolis_saved"] == 1

    # 5. Cross-cell invariants & failure budget pricing + shot-paired dominance
    kmb = tmp_path / "cells.kmb"
    _build_sample_circuit(kmb, num_blocks=4)
    with strata.Session.open(kmb) as sess:
        inv = strata.scan_cross_cell_invariants(sess, [2, 8, 14, 20], wires=[0, 1, 7], clean_wires=(7, 8, 9))
        # q0 ^ q1 ^ q7 is conserved across all cells!
        assert [0, 1, 7] in inv["conserved_parities"] or inv["conserved_dimension"] >= 1

    fb = strata.price_failure_budget(0.05, max_failure_budget=0.003, attempt_toffolis=100, attempt_qubits=20)
    assert fb["repetitions"] == 2 and fb["achieved_fail_prob"] <= 0.003

    dom = strata.audit_shot_paired_dominance([True, True, False, True], [True, True, False, True], max_failure_budget=0.3)
    assert dom["shot_monotone"] is True and dom["accepted"] is True
    dom_bad = strata.audit_shot_paired_dominance([True, True, False, True], [True, False, False, True], max_failure_budget=0.5)
    assert dom_bad["child_only_failures"] == 1 and dom_bad["accepted"] is False

    # 6. CLI subcommands smoke checks
    for argv in (
        ["benchmark-latency", "--ops", "10000", "--qubits", "24", "--cones", "4"],
        ["profile-fast", str(kmb)],
        ["trace-cuts", str(kmb), "--cuts", "2,8,14"],
        ["phase-fit", str(kmb), "--cuts", "2,8", "--modulus", "8"],
        ["wire-loans", str(kmb), "--range", "2:8", "--clean-inputs", "7,8,9"],
    ):
        rc = cli.main(argv)
        out = json.loads(capsys.readouterr().out)
        assert rc == 0 and "error" not in out
