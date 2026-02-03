"""Audit trail for human decisions: append-only log with timestamps."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
from datetime import datetime
from typing import Any

from core.schema import HumanDecision


class AuditTrail:
    """Append-only audit log for human-in-the-loop decisions."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []

    def append(
        self,
        decision_id: str,
        action: str,
        actor: str | None = None,
        rationale: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append an audit entry and persist to JSON."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision_id": decision_id,
            "action": action,
            "actor": actor,
            "rationale": rationale,
            "payload": payload or {},
        }
        self._entries.append(entry)
        self._persist()

    def append_decision(self, decision: HumanDecision) -> None:
        """Log a full decision record."""
        self.append(
            decision_id=decision.decision_id,
            action=decision.status,
            actor=decision.human_actor,
            rationale=decision.human_rationale,
            payload={
                "proposal_type": decision.proposal_type,
                "proposal_payload": decision.proposal_payload,
                "modified_payload": decision.modified_payload,
                "created_at": decision.created_at.isoformat() if decision.created_at else None,
                "decided_at": decision.decided_at.isoformat() if decision.decided_at else None,
            },
        )

    def _persist(self) -> None:
        """Write current entries to JSON file (append-style: overwrite file with all entries)."""
        try:
            with open(self._log_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2, default=str)
        except (OSError, TypeError) as e:
            raise RuntimeError(f"Audit trail persist failed: {e}") from e

    def load(self) -> list[dict[str, Any]]:
        """Load existing audit entries from disk."""
        if not self._log_path.exists():
            return []
        try:
            with open(self._log_path, encoding="utf-8") as f:
                self._entries = json.load(f)
            return self._entries
        except (OSError, json.JSONDecodeError):
            return []

    def get_entries(self) -> list[dict[str, Any]]:
        """Return in-memory audit entries."""
        return list(self._entries)
