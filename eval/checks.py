import re

from eval.schema import CheckType, Task

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s).strip().lower()


def _last_number(text: str) -> float | None:
    matches = _NUM.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace(",", "").rstrip("."))


def check_answer(task: Task, answer: str | None) -> bool | None:
    """True/False for programmatic checks, None if an LLM judge is needed."""
    if task.check == CheckType.judge:
        return None
    if answer is None:
        return False
    if task.check == CheckType.exact:
        return _normalize(answer) == _normalize(task.expected)
    if task.check == CheckType.contains:
        return _normalize(task.expected) in _normalize(answer)
    if task.check == CheckType.numeric:
        got = _last_number(answer)
        return got is not None and abs(got - float(task.expected)) <= task.tolerance
    return False