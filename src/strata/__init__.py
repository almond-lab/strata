"""Strata: contract-driven restructuring on kickmix."""
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


def open(path='strata.toml', adapters=None):
    if adapters is None:
        from .demo import AndFusion
        adapters = {'and_fusion': AndFusion()}
    return Project(path, adapters)
