from pydantic import BaseModel

from agent.llm import LLMClient
from agent.tools import TOOL_SCHEMAS, run_tool
from agent.versions import AgentConfig


class AgentRun(BaseModel):
    """What happened during one attempt at one question."""

    answer: str | None
    finished: bool  # False = ran out of steps (probably looping) or errored
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    trace: list[dict] = []
    error: str | None = None


def run_agent(question: str, config: AgentConfig, llm: LLMClient, repeat: int = 0) -> AgentRun:
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": question},
    ]
    run = AgentRun(answer=None, finished=False)

    for _ in range(config.max_steps):
        try:
            r = llm.chat(
                messages,
                model=config.model,
                tools=TOOL_SCHEMAS,
                salt=f"r{repeat}",  # different repeat = fresh answer, not cached
            )
        except Exception as e:
            run.error = f"{type(e).__name__}: {e}"
            break

        run.n_llm_calls += 1
        run.prompt_tokens += r.prompt_tokens
        run.completion_tokens += r.completion_tokens
        run.reasoning_tokens += r.reasoning_tokens
        run.latency_s += r.latency_s

        # No tool requested -> this is the final answer
        if not r.tool_calls:
            run.answer = r.text
            run.finished = True
            break

        # Tool(s) requested -> record the request, run them, feed results back
        messages.append(
            {
                "role": "assistant",
                "content": r.text or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in r.tool_calls
                ],
            }
        )
        for tc in r.tool_calls:
            result = run_tool(tc["name"], tc["arguments"])
            run.n_tool_calls += 1
            run.trace.append(
                {"tool": tc["name"], "arguments": tc["arguments"], "result": result}
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result}
            )

    return run