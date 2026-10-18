from agent.agent import run_agent
from agent.llm import LLMClient
from agent.versions import VERSIONS
from eval.checks import check_answer
from eval.schema import load_tasks

llm = LLMClient()
tasks = load_tasks()

for name, config in VERSIONS.items():
    passed = total = 0
    tokens = tools = 0
    latency = 0.0
    print(f"\n=== {name} ({config.model}) ===")
    for t in tasks:
        run = run_agent(t.input, config, llm)
        verdict = check_answer(t, run.answer)
        tokens += run.prompt_tokens + run.completion_tokens
        tools += run.n_tool_calls
        latency += run.latency_s
        if verdict is None:
            label = "JUDGE"
        else:
            total += 1
            passed += verdict
            label = "PASS" if verdict else "FAIL"
        shown = (run.answer or run.error or "")[:70]
        print(f"{t.id:10} {label:6} tools={run.n_tool_calls} | {shown!r}")
    print(f"-> {passed}/{total} programmatic tasks passed | tokens={tokens} "
          f"| tool calls={tools} | latency={latency:.1f}s")