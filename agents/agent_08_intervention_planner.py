"""Agent 08: Generate 70:20:10 intervention plans from risk profile."""

import uuid
from typing import Optional

from core.schema import (
    EvidenceArtifact,
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
    Returns InterventionPlan with evidence artifacts.
    """

    def __init__(self, decision_manager: DecisionManager) -> None:
        self._dm = decision_manager

    def generate_plan(
        self,
        analyst_id: Optional[str] = None,
        risk_profile: Optional[dict[str, float]] = None,
        difficulty: Optional[int] = None,
        job_task_refs: Optional[list[str]] = None,
    ) -> InterventionPlan:
        """
        Build 70:20:10 plan from risk_profile and optional difficulty.
        risk_profile should include incident_risk and skill_gap in [0, 1].
        """
        risk_profile = risk_profile or {}
        incident_risk = float(risk_profile.get("incident_risk", 0.0))
        skill_gap = float(risk_profile.get("skill_gap", 0.0))
        diff = difficulty if difficulty is not None else 2

        otj_70: list[str] = []
        social_20: list[str] = []
        formal_10: list[str] = []
        artifacts: list[EvidenceArtifact] = []

        # If skill_gap > 0.5 → OTJ_70 activities
        if skill_gap > 0.5:
            otj_70 = [
                "Structured on-the-job practice with checklist",
                "Shadowing with debrief",
                "Gradual handoff with feedback loops",
            ]
            artifacts.append(
                EvidenceArtifact(
                    artifact_id=str(uuid.uuid4()),
                    label="OTJ checklist",
                    category=InterventionCategory.OTJ_70,
                    description="On-the-job 70% activities for skill gap",
                )
            )

        # If incident_risk > 0.6 → Social_20 peer coaching
        if incident_risk > 0.6:
            social_20 = [
                "Peer coaching sessions",
                "Community of practice discussions",
                "Mentor check-ins",
            ]
            artifacts.append(
                EvidenceArtifact(
                    artifact_id=str(uuid.uuid4()),
                    label="Peer coaching plan",
                    category=InterventionCategory.Social_20,
                    description="Social 20% for incident risk mitigation",
                )
            )

        # If difficulty >= 3 → Formal_10 eLearning
        if diff >= 3:
            formal_10 = [
                "eLearning module (difficulty-aligned)",
                "Formal assessment before sign-off",
                "Certification or badge path",
            ]
            artifacts.append(
                EvidenceArtifact(
                    artifact_id=str(uuid.uuid4()),
                    label="Formal 10% eLearning",
                    category=InterventionCategory.Formal_10,
                    description="Formal 10% for high-difficulty tasks",
                )
            )

        # Defaults if nothing triggered
        if not otj_70:
            otj_70 = ["Standard on-the-job practice"]
        if not social_20:
            social_20 = ["Optional peer review"]
        if not formal_10:
            formal_10 = ["Optional reference material"]

        plan = InterventionPlan(
            plan_id=str(uuid.uuid4()),
            analyst_id=analyst_id,
            risk_profile={"incident_risk": incident_risk, "skill_gap": skill_gap},
            otj_70_activities=otj_70,
            social_20_activities=social_20,
            formal_10_activities=formal_10,
            evidence_artifacts=artifacts,
            job_task_refs=job_task_refs or [],
        )
        return plan

    def generate_and_submit(
        self,
        analyst_id: Optional[str] = None,
        risk_profile: Optional[dict[str, float]] = None,
        difficulty: Optional[int] = None,
        job_task_refs: Optional[list[str]] = None,
    ) -> tuple[InterventionPlan, str]:
        """
        Generate plan and submit to DecisionManager for human approval.
        Never auto-approves. Returns (plan, decision_id).
        """
        plan = self.generate_plan(
            analyst_id=analyst_id,
            risk_profile=risk_profile,
            difficulty=difficulty,
            job_task_refs=job_task_refs,
        )
        payload = plan.model_dump(mode="json")
        decision = self._dm.propose_decision(
            proposal_type="intervention_plan",
            proposal_payload=payload,
        )
        return plan, decision.decision_id
