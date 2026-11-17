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


def open(path='strata.toml', adapters=None):
    if adapters is None:
        from .demo import AndFusion
        adapters = {'and_fusion': AndFusion()}
    return Project(path, adapters)
