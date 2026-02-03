"""Agent 01: Parse SME Design Documents into JobTask objects (Verb-Task-Product)."""

import re
from pathlib import Path
from typing import Optional

from core.schema import (
    DifficultyLevel,
    JobTask,
    SkillType,
)
from human_loop.decision_interface import DecisionManager


# Verb-Task-Product regex: verb (action), task (what), product (deliverable)
VTP_PATTERN = re.compile(
    r"(?P<verb>\w+)\s+(?P<task>[^→\-]+?)(?:\s*[→\-]\s*|\s+to produce\s+)(?P<product>.+?)",
    re.IGNORECASE | re.DOTALL,
)
# Fallback: "Verb task product" on one line
VTP_SIMPLE = re.compile(
    r"^(?P<verb>\w+)\s+(?P<task>.+?)\s+→\s+(?P<product>.+)$",
    re.IGNORECASE | re.MULTILINE,
)


class ContextIngestionAgent:
    """
    Parse SME Design Documents into JobTask objects.
    Extract Verb-Task-Product format using regex (and optional LLM).
    Confidence scoring. Submit to DecisionManager for human validation.
    """

    def __init__(self, decision_manager: DecisionManager) -> None:
        self._dm = decision_manager

    def parse_text(self, text: str, source_doc: Optional[str] = None) -> list[JobTask]:
        """
        Parse raw text into candidate JobTask objects using regex.
        Returns list of JobTask; does not submit to human queue.
        """
        tasks: list[JobTask] = []
        text = text.strip()
        if not text:
            return tasks

        # Try structured VTP lines first
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = VTP_SIMPLE.match(line)
            if m:
                tasks.append(
                    JobTask(
                        verb=m.group("verb").strip(),
                        task=m.group("task").strip(),
                        product=m.group("product").strip(),
                        critical_incident=False,
                        difficulty=DifficultyLevel.LEVEL_1,
                        source_doc=source_doc,
                    )
                )
                continue
            # Block match
            m = VTP_PATTERN.match(line)
            if m:
                tasks.append(
                    JobTask(
                        verb=m.group("verb").strip(),
                        task=m.group("task").strip(),
                        product=m.group("product").strip(),
                        critical_incident=False,
                        difficulty=DifficultyLevel.LEVEL_1,
                        source_doc=source_doc,
                    )
                )

        # If no structured match, try to infer one task from first line
        if not tasks and text:
            first_line = text.splitlines()[0].strip()[:200]
            tasks.append(
                JobTask(
                    verb="Perform",
                    task=first_line,
                    product="(parsed deliverable)",
                    critical_incident=False,
                    difficulty=DifficultyLevel.LEVEL_1,
                    source_doc=source_doc,
                )
            )
        return tasks

    def confidence_score(self, task: JobTask) -> float:
        """
        Return confidence in [0, 1] for parsed task.
        Higher if verb/task/product are non-generic and present.
        """
        score = 0.0
        if task.verb and len(task.verb) > 1:
            score += 0.25
        if task.task and len(task.task) > 5:
            score += 0.35
        if task.product and "(parsed deliverable)" not in task.product:
            score += 0.35
        if task.cues or task.strategies:
            score += 0.05
        return min(1.0, score)

    def ingest_and_submit(
        self,
        text: str,
        source_doc: Optional[str] = None,
        submit_all: bool = True,
    ) -> list[tuple[JobTask, float, Optional[str]]]:
        """
        Parse text into JobTasks, score confidence, and submit each to DecisionManager
        for human validation. Never auto-approves.
        Returns list of (JobTask, confidence, decision_id).
        """
        tasks = self.parse_text(text, source_doc=source_doc)
        results: list[tuple[JobTask, float, Optional[str]]] = []
        for task in tasks:
            confidence = self.confidence_score(task)
            if not submit_all and confidence < 0.5:
                results.append((task, confidence, None))
                continue
            payload = task.model_dump(mode="json")
            payload["_confidence"] = confidence
            decision = self._dm.propose_decision(
                proposal_type="job_task",
                proposal_payload=payload,
            )
            results.append((task, confidence, decision.decision_id))
        return results

    def ingest_file(
        self,
        path: Path,
        submit_all: bool = True,
    ) -> list[tuple[JobTask, float, Optional[str]]]:
        """Read file and run ingest_and_submit."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Context file not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_and_submit(text, source_doc=str(path), submit_all=submit_all)
