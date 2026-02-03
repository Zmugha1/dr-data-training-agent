"""Agent 08: Generate 70:20:10 intervention plans from risk profile."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import uuid
from typing import Optional, Dict, List, Any

from core.schema import (
    InterventionCategory,
    InterventionPlan,
)
from human_loop.decision_interface import DecisionManager


class InterventionPlannerAgent:
    """
    Generate 70:20:10 plans based on risk_profile (incident_risk, skill_gap).
    - If skill_gap > 0.5 → OTJ_70 activities
    - If incident_risk > 0.6 → Social_20 peer coaching
    - If difficulty >= 3 → Formal_10 eLearning
    """

    def __init__(self, decision_manager: DecisionManager) -> None:
        self._dm = decision_manager

    def generate_plan(
        self,
        analyst_id: str = "",
        aop_id: str = "",
        risk_profile: Optional[Dict[str, float]] = None,
        difficulty: Optional[int] = None,
        job_task_refs: Optional[List[str]] = None,
    ) -> InterventionPlan:
        """
        Build 70:20:10 plan from risk_profile and optional difficulty.
        risk_profile should include incident_risk and skill_gap in [0, 1].
        """
        risk_profile = risk_profile or {}
        incident_risk = float(risk_profile.get("incident_risk", 0.0))
        skill_gap = float(risk_profile.get("skill_gap", 0.0))
        diff = difficulty if difficulty is not None else 2

        otj_70: List[Dict[str, Any]] = []
        social_20: List[Dict[str, Any]] = []
        formal_10: List[Dict[str, Any]] = []
        evidence_strs: List[str] = []

        # If skill_gap > 0.5 → OTJ_70 activities
        if skill_gap > 0.5:
            otj_70 = [
                {"activity": "Structured on-the-job practice with checklist"},
                {"activity": "Shadowing with debrief"},
                {"activity": "Gradual handoff with feedback loops"},
            ]
            evidence_strs.append("OTJ checklist – on-the-job 70% for skill gap")

        # If incident_risk > 0.6 → Social_20 peer coaching
        if incident_risk > 0.6:
            social_20 = [
                {"activity": "Peer coaching sessions"},
                {"activity": "Community of practice discussions"},
                {"activity": "Mentor check-ins"},
            ]
            evidence_strs.append("Peer coaching plan – Social 20% for incident risk")

        # If difficulty >= 3 → Formal_10 eLearning
        if diff >= 3:
            formal_10 = [
                {"activity": "eLearning module (difficulty-aligned)"},
                {"activity": "Formal assessment before sign-off"},
                {"activity": "Certification or badge path"},
            ]
            evidence_strs.append("Formal 10% eLearning – high-difficulty tasks")

        if not otj_70:
            otj_70 = [{"activity": "Standard on-the-job practice"}]
        if not social_20:
            social_20 = [{"activity": "Optional peer review"}]
        if not formal_10:
            formal_10 = [{"activity": "Optional reference material"}]

        # Use enum value lookup so code works regardless of member name (SOCIAL_20 vs Social_20)
        interventions = {
            InterventionCategory.OTJ_70: otj_70,
            InterventionCategory("Social_20"): social_20,
            InterventionCategory("Formal_10"): formal_10,
        }

        plan = InterventionPlan(
            plan_id=str(uuid.uuid4()),
            analyst_id=analyst_id or "system",
            aop_id=aop_id or "default",
            risk_profile={"incident_risk": incident_risk, "skill_gap": skill_gap},
            interventions=interventions,
            evidence_artifacts=evidence_strs,
            success_criteria={"risk_reduction": True},
        )
        return plan

    def generate_and_submit(
        self,
        agent_id: int,
        analyst_id: str = "",
        aop_id: str = "",
        risk_profile: Optional[Dict[str, float]] = None,
        difficulty: Optional[int] = None,
        job_task_refs: Optional[List[str]] = None,
    ) -> tuple[InterventionPlan, str]:
        """Generate plan and submit to DecisionManager. Returns (plan, decision_id)."""
        plan = self.generate_plan(
            analyst_id=analyst_id,
            aop_id=aop_id,
            risk_profile=risk_profile,
            difficulty=difficulty,
            job_task_refs=job_task_refs,
        )
        payload = plan.model_dump(mode="json")
        decision_id = self._dm.propose_decision(
            agent_id=agent_id,
            decision_type="intervention_plan",
            data=payload,
            auto_approve=False,
        )
        return plan, decision_id
