"""Agent 01: Parse SME Design Documents into JobTask objects (Verb-Task-Product)."""
import sys
import uuid
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import re

from core.schema import DifficultyLevel, JobTask
from human_loop.decision_interface import DecisionManager


# Verb-Task-Product regex
VTP_PATTERN = re.compile(
    r"(?P<verb>\w+)\s+(?P<task>[^→\-]+?)(?:\s*[→\-]\s*|\s+to produce\s+)(?P<product>.+?)",
    re.IGNORECASE | re.DOTALL,
)
VTP_SIMPLE = re.compile(
    r"^(?P<verb>\w+)\s+(?P<task>.+?)\s+→\s+(?P<product>.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _make_job_task(verb: str, task: str, product: str, aop_id: str = "default") -> JobTask:
    statement = f"{verb} {task} → {product}"
    return JobTask(
        task_id=str(uuid.uuid4()),
        aop_id=aop_id,
        verb=verb,
        task=task,
        product=product,
        statement=statement,
        critical_incident=False,
        difficulty=DifficultyLevel.EASY,
    )


class ContextIngestionAgent:
    """
    Parse SME Design Documents into JobTask objects.
    Extract Verb-Task-Product format. Confidence scoring. Submit to DecisionManager.
    """

    def __init__(self, decision_manager: DecisionManager) -> None:
        self._dm = decision_manager

    def parse_text(
        self,
        text: str,
        source_doc: Optional[str] = None,
        aop_id: str = "default",
    ) -> list[JobTask]:
        """Parse raw text into candidate JobTask objects. Returns list of JobTask."""
        tasks: list[JobTask] = []
        text = text.strip()
        if not text:
            return tasks

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = VTP_SIMPLE.match(line)
            if m:
                tasks.append(
                    _make_job_task(
                        m.group("verb").strip(),
                        m.group("task").strip(),
                        m.group("product").strip(),
                        aop_id=aop_id,
                    )
                )
                continue
            m = VTP_PATTERN.match(line)
            if m:
                tasks.append(
                    _make_job_task(
                        m.group("verb").strip(),
                        m.group("task").strip(),
                        m.group("product").strip(),
                        aop_id=aop_id,
                    )
                )

        if not tasks and text:
            first_line = text.splitlines()[0].strip()[:200]
            tasks.append(
                _make_job_task("Perform", first_line, "(parsed deliverable)", aop_id=aop_id)
            )
        return tasks

    def confidence_score(self, task: JobTask) -> float:
        """Return confidence in [0, 1] for parsed task."""
        score = 0.0
        if task.verb and len(task.verb) > 1:
            score += 0.25
        if task.task and len(task.task) > 5:
            score += 0.35
        if task.product and "(parsed deliverable)" not in task.product:
            score += 0.35
        if task.cues_strategies:
            score += 0.05
        return min(1.0, score)

    def ingest_and_submit(
        self,
        text: str,
        source_doc: Optional[str] = None,
        aop_id: str = "default",
        submit_all: bool = True,
    ) -> list[tuple[JobTask, float, Optional[str]]]:
        """Parse, score, and submit each JobTask to DecisionManager. Returns (JobTask, confidence, decision_id)."""
        tasks = self.parse_text(text, source_doc=source_doc, aop_id=aop_id)
        results: list[tuple[JobTask, float, Optional[str]]] = []
        for task in tasks:
            confidence = self.confidence_score(task)
            if not submit_all and confidence < 0.5:
                results.append((task, confidence, None))
                continue
            payload = task.model_dump(mode="json")
            payload["_confidence"] = confidence
            decision_id = self._dm.propose_decision(
                agent_id=1,
                decision_type="job_task",
                data=payload,
                auto_approve=False,
            )
            results.append((task, confidence, decision_id))
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
