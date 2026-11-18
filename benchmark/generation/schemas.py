"""Pydantic schemas for the benchmark-generation LLM calls (benchmark code only; not part of CGScout)."""
from typing import Literal

from pydantic import BaseModel, Field


class NeedCard(BaseModel):
    system_class: str
    scientific_process: str
    phenomenon_or_goal: str
    important_constraints: list[str] = Field(default_factory=list)
    required_model_capability: str
    transferable_features: list[str] = Field(default_factory=list)


class NeedCards(BaseModel):
    """Stage A output. `specific_need` keeps the concrete system (-> L1); `abstracted_need` generalises it (-> L2/L3)."""
    specific_need: NeedCard
    abstracted_need: NeedCard


class QueryCandidates(BaseModel):
    queries: list[str]


class SisterProposal(BaseModel):
    system_name: str            # a specific, searchable system (protein / complex / material / nucleic-acid motif ...)
    description: str            # one sentence: what it is and why a researcher would want to simulate it
    shared_feature: str         # the transferable property it shares with the seed's system
    key_difference: str         # how it differs from the seed's system


class SisterProposals(BaseModel):
    proposals: list[SisterProposal]


class DirectCheckItem(BaseModel):
    index: int
    is_direct_cg_for_target: bool
    reason: str


class DirectCheck(BaseModel):
    items: list[DirectCheckItem]


class SeedTag(BaseModel):
    system_specific: bool       # studies a specific named biomolecular / chemical system, not a generic tool or method
    named_system: str           # that system in a few words ("" if generic)
    paper_type: Literal["application", "method_or_forcefield", "benchmark_or_comparison", "tool_or_software", "other"]


class TransferRationale(BaseModel):
    """Why a seed paper X (a CG study of system X) should inform a researcher studying a different system Y."""
    target_objective: str                 # the scientific objective a researcher would have for Y
    transfer_axis: Literal["family_function", "close_structural_homology", "physical_process", "chemistry_representation", "environment_context", "other"]
    transferable_feature: str             # the shared property, stated generically
    reason_seed_is_useful_for_Y: str      # 1-2 sentences: what the seed's CG approach would give a researcher on Y
    confidence: Literal["low", "medium", "high"]
    caveat: str                           # the strongest reason the transfer might NOT hold ("" if none)
