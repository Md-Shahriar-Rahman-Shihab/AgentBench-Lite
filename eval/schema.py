import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Category(str, Enum):
    easy = "easy"
    multi_step = "multi_step"
    adversarial = "adversarial"


class CheckType(str, Enum):
    exact = "exact"        # normalized text must match
    contains = "contains"  # expected text must appear in the answer
    numeric = "numeric"    # last number in the answer must match
    judge = "judge"        # open-ended: graded by LLM judge with a rubric


class FailureLabel(str, Enum):
    none = "none"
    wrong_answer = "wrong_answer"
    wrong_tool = "wrong_tool"
    hallucination = "hallucination"
    loop = "loop"
    timeout = "timeout"
    bad_format = "bad_format"
    unsupported_claim = "unsupported_claim"


class Task(BaseModel):
    id: str
    category: Category
    input: str
    check: CheckType
    expected: str | None = None   # required unless check == judge
    rubric: str | None = None     # required when check == judge
    tolerance: float = 1e-6       # for numeric checks
    notes: str = ""

    @model_validator(mode="after")
    def _validate_check_fields(self):
        if self.check == CheckType.judge:
            if not self.rubric:
                raise ValueError(f"{self.id}: judge tasks need a rubric")
        elif self.expected is None:
            raise ValueError(f"{self.id}: non-judge tasks need an expected answer")
        if self.check == CheckType.numeric:
            float(self.expected)  # fails early if the label isn't a number
        return self


class RunResult(BaseModel):
    task_id: str
    agent_version: str
    repeat: int
    success: bool
    answer: str | None = None
    failure_label: FailureLabel = FailureLabel.none
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def load_tasks(path: str | Path = "data/tasks.jsonl") -> list[Task]:
    """Read one JSON task per line; fail loudly with the line number."""
    tasks, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = Task(**json.loads(line))
        except Exception as e:
            raise ValueError(f"{path} line {n}: {e}") from e
        if task.id in seen:
            raise ValueError(f"{path} line {n}: duplicate id {task.id}")
        seen.add(task.id)
        tasks.append(task)
    return tasks