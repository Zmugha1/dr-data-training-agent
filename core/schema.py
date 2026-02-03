from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class DifficultyLevel(int, Enum):
    EASY = 1
    COMPLEX = 2
    DIFFICULT = 3
    VERY_DIFFICULT = 4

class SkillType(str, Enum):
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    TECH_STACK = "tech_stack"
    SOFT_SKILL = "soft_skill"

class ProficiencyLevel(int, Enum):
    BEGINNER = 1
    ADVANCED = 2
    PRACTITIONER = 3
    EXPERT = 4

class InterventionCategory(str, Enum):
    OTJ_70 = "OTJ_70"
    SOCIAL_20 = "Social_20"
    FORMAL_10 = "Formal_10"

class DecisionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

class JobTask(BaseModel):
    task_id: str
    aop_id: str
    verb: str
    task: str
    product: str
    statement: str = Field(..., description="Verb – Task – Product")
    critical_incident: bool = False
    difficulty: DifficultyLevel
    difficulty_reason: Optional[str] = None
    common_errors: List[str] = []
    cues_strategies: List[str] = []
    skills: List[str] = []
    kpis: List[str] = []

class HumanDecision(BaseModel):
    decision_id: str
    agent_id: int
    decision_type: str
    proposed_data: Dict[str, Any]
    human_override: Optional[Dict[str, Any]] = None
    status: DecisionStatus = DecisionStatus.PENDING
    timestamp_proposed: datetime = Field(default_factory=datetime.now)
    timestamp_decided: Optional[datetime] = None
    human_expert_id: Optional[str] = None
    rationale: Optional[str] = None
    audit_trail: List[str] = []

class InterventionPlan(BaseModel):
    plan_id: str
    analyst_id: str
    aop_id: str
    generated_at: datetime = Field(default_factory=datetime.now)
    risk_profile: Dict[str, float]
    interventions: Dict[InterventionCategory, List[Dict]]
    evidence_artifacts: List[str]
    success_criteria: Dict[str, Any]
    human_approval_status: DecisionStatus = DecisionStatus.PENDING
    human_notes: Optional[str] = None
