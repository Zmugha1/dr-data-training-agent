"""Application state manager for agents and UI."""
import sys
from pathlib import Path
from typing import Any, Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.schema import (
    HumanDecision,
    InterventionPlan,
    JobTask,
)


class StateManager:
    """Manages in-memory and persisted state for the decision intelligence system."""

    def __init__(self, data_root: Optional[Path] = None) -> None:
        self._data_root = data_root or Path("data")
        self._pending_decisions: list[HumanDecision] = []
        self._job_tasks: list[JobTask] = []
        self._intervention_plans: list[InterventionPlan] = []

    @property
    def data_root(self) -> Path:
        return self._data_root

    @property
    def pending_decisions_path(self) -> Path:
        return self._data_root / "pending_decisions"

    def get_pending_decisions(self) -> list[HumanDecision]:
        """Return list of pending decisions (in-memory cache)."""
        return [d for d in self._pending_decisions if d.status == "pending"]

    def add_pending_decision(self, decision: HumanDecision) -> None:
        """Add a decision to pending list."""
        self._pending_decisions.append(decision)

    def update_decision_status(
        self,
        decision_id: str,
        status: str,
        human_actor: Optional[str] = None,
        rationale: Optional[str] = None,
        modified_payload: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update a decision by ID. Returns True if found and updated."""
        from datetime import datetime

        for d in self._pending_decisions:
            if d.decision_id == decision_id:
                d.status = status
                d.human_actor = human_actor or d.human_actor
                d.human_rationale = rationale or d.human_rationale
                d.decided_at = datetime.utcnow()
                if modified_payload is not None:
                    d.modified_payload = modified_payload
                return True
        return False

    def get_decision_by_id(self, decision_id: str) -> Optional[HumanDecision]:
        """Return decision by ID or None."""
        for d in self._pending_decisions:
            if d.decision_id == decision_id:
                return d
        return None

    def add_job_task(self, task: JobTask) -> None:
        """Register a validated job task."""
        self._job_tasks.append(task)

    def get_job_tasks(self) -> list[JobTask]:
        return list(self._job_tasks)

    def add_intervention_plan(self, plan: InterventionPlan) -> None:
        self._intervention_plans.append(plan)

    def get_intervention_plans(self) -> list[InterventionPlan]:
        return list(self._intervention_plans)
