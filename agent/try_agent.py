from agent.agent import run_agent
from agent.llm import LLMClient
from agent.versions import VERSIONS
from eval.checks import check_answer
from eval.schema import load_tasks

llm = LLMClient()
config = VERSIONS["v1"]

for t in load_tasks():
    run = run_agent(t.input, config, llm)
    verdict = check_answer(t, run.answer)
    label = "NEEDS JUDGE" if verdict is None else ("PASS" if verdict else "FAIL")
    tokens = run.prompt_tokens + run.completion_tokens
    shown = (run.answer or run.error or "")[:80]
    print(f"{t.id:10} {label:11} tools={run.n_tool_calls} calls={run.n_llm_calls} tokens={tokens} | {shown!r}")