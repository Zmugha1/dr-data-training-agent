"""Tests for agents and human-in-the-loop components."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from core.schema import JobTask, DifficultyLevel, InterventionPlan, InterventionCategory
from human_loop.decision_interface import DecisionManager
from agents.agent_01_context_ingestion import ContextIngestionAgent
from agents.agent_08_intervention_planner import InterventionPlannerAgent


@pytest.fixture
def temp_pending(tmp_path):
    return tmp_path / "pending_decisions"


@pytest.fixture
def dm(temp_pending):
    return DecisionManager(pending_dir=temp_pending)


@pytest.fixture
def ingestion_agent(dm):
    return ContextIngestionAgent(dm)


@pytest.fixture
def planner_agent(dm):
    return InterventionPlannerAgent(dm)


def test_job_task_verb_task_product():
    task = JobTask(verb="Analyze", task="customer feedback", product="weekly report")
    assert task.to_verb_task_product() == "Analyze customer feedback → weekly report"


def test_ingestion_parse_simple(ingestion_agent):
    text = "Analyze customer feedback → weekly report"
    tasks = ingestion_agent.parse_text(text)
    assert len(tasks) >= 1
    assert tasks[0].verb == "Analyze"
    assert "customer feedback" in tasks[0].task
    assert "weekly report" in tasks[0].product


def test_ingestion_confidence(ingestion_agent):
    task = JobTask(verb="Perform", task="data validation", product="validation report")
    score = ingestion_agent.confidence_score(task)
    assert 0 <= score <= 1


def test_planner_generate(planner_agent):
    plan = planner_agent.generate_plan(
        risk_profile={"incident_risk": 0.7, "skill_gap": 0.6},
        difficulty=3,
    )
    assert plan.plan_id
    assert plan.otj_70_activities
    assert plan.social_20_activities
    assert plan.formal_10_activities
    assert plan.risk_profile["incident_risk"] == 0.7
    assert plan.risk_profile["skill_gap"] == 0.6


def test_decision_manager_propose_and_get(dm):
    decision = dm.propose_decision("job_task", {"verb": "Test", "task": "t", "product": "p"})
    assert decision.status == "pending"
    pending = dm.get_pending_decisions()
    assert len(pending) >= 1
    assert pending[0].decision_id == decision.decision_id


def test_decision_manager_make_decision(dm):
    decision = dm.propose_decision("test", {"key": "value"})
    updated = dm.make_decision(
        decision.decision_id,
        "approved",
        human_actor="Tester",
        human_rationale="OK",
    )
    assert updated is not None
    assert updated.status == "approved"
    pending = dm.get_pending_decisions()
    assert not any(d.decision_id == decision.decision_id for d in pending)
