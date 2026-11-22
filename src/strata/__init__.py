"""Strata: algorithmic co-design and low-latency restructuring on Kickmix."""

__version__ = "0.7.0"

from .project import Project, Design, Line
from .model import Contract, Region, Register, Debt, Floor, StrataError
from .engine import Patch
from .queries import Queries
from .research import Hypothesis, Dependency, ResearchPolicy, ResearchRound
from .file_workflow import inspect_file, optimize_file
from .discovery import discovery_brief, evaluate_candidate, record_research_note
from .bundle import Bundle, BundleFrontend, Component, DataFrontend
from .callbacks import (CallbackAdapter, PinnedCallback, BuildRequest, BuildReceipt,
                        CheckRequest, CheckReceipt, FullArtifactLab,
                        StructuralRequest, StructuralReceipt)
from .component_lab import ComponentLab, BundleLab
from .probability import ProbabilityPolicy
from .frontends import OperationStream
from .region_evidence import replay_region_evidence
from .restructure import restructure_file, ConstructorRegistry
from .numeric import NumericStudy, EvidenceValue, ResourceReceipt, error_budget, design_space, break_even_scale
from .numeric_candidates import NumericCandidate, NumericScorer, NumericStage, NumericPipeline
from .numeric_jobs import NumericJob, score_bets, certify_numeric_bet
from .experiment_board import ExperimentBoard
from .experiments import (load_manifest, create_experiment, run_experiment,
                          experiment_status, create_sweep, run_sweep, sweep_status)
from .experiment_policy import ScreeningPolicy, ContinuationPolicy
from .experiment_reports import experiment_report, agent_packet, record_agent_decision

# v0.7.0 Low-Latency Native Engine & Session Funnel
from .native import (NATIVE_AVAILABLE, KmbTape, BitSlicedSim,
                     gf2_rank_fast, solve_zmod_fast, lovasz_rank_fast, eval_phase_margin_fast)
from .session import Session, FunnelCandidate, benchmark_latency

# v0.7.0 Layer A Optimization-Discovery Modules
from .cofit import (BoundarySlider, SlideResult, GaugeSweep, GaugeResult,
                    DeadSlotPlanter, CircuitAwareRounder, QuotientChainDP,
                    CompensatedTruncator, TruncationChoice)
from .mbu import fit_multi_cut_phase, demolish_and_piggyback, hoist_hmr_across_swaps
from .shapes import (ScoreModel, PebbleSchedule, solve_mbu_pebble_schedule,
                     find_affine_wire_loans, apply_affine_wire_loan, ShapeExplorer)
from .boundary import ClassicalProgram, hoist_classical_prefix, ProjectiveGF2k, ProjectiveCostEstimate
from .tensors import (lovasz_matroid_parity_rank, search_relation_hyperplanes,
                      plan_multipass_qroam, flip_graph_reduce)
from .invariants import scan_cross_cell_invariants, price_failure_budget, audit_shot_paired_dominance
from .patch_composition import compose_kmb_fast


def open(path='strata.toml', adapters=None):
    if adapters is None:
        from .demo import AndFusion
        adapters = {'and_fusion': AndFusion()}
    return Project(path, adapters)
