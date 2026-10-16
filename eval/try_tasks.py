from eval.checks import check_answer
from eval.schema import load_tasks

tasks = load_tasks()
print(f"Loaded {len(tasks)} tasks\n")

for t in tasks:
    if t.expected is None:
        print(f"{t.id:10} [{t.category.value}] judge task -> check_answer = {check_answer(t, 'x')}")
        continue
    good = check_answer(t, f"The answer is {t.expected}.")
    bad = check_answer(t, "I have no idea 7")
    print(f"{t.id:10} [{t.category.value}] right answer -> {good} | wrong answer -> {bad}")