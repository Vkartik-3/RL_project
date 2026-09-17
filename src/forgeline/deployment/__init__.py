"""Model lifecycle: candidate registry, promotion gates, feature flags, staged rollout, rollback."""

from forgeline.deployment.candidates import ALLOWED_TRANSITIONS, CandidateState, ModelCandidate
from forgeline.deployment.feature_flags import FeatureFlag, FeatureFlags, stable_bucket
from forgeline.deployment.promotion import GateRule, PromotionGate, PromotionReport
from forgeline.deployment.registry import CandidateRegistry
from forgeline.deployment.rollback import disable_candidate, rollback
from forgeline.deployment.rollout import RoutingDecision, RoutingPolicy, offline_replay, simulate_traffic

__all__ = [
    "ALLOWED_TRANSITIONS", "CandidateState", "ModelCandidate", "FeatureFlag", "FeatureFlags", "stable_bucket", "GateRule",
    "PromotionGate", "PromotionReport", "CandidateRegistry", "disable_candidate", "rollback", "RoutingDecision",
    "RoutingPolicy", "offline_replay", "simulate_traffic",
]
