from dataclasses import dataclass

from agent.llm import DEFAULT_MODEL

BIG_MODEL = "openai/gpt-oss-120b"

TOOLS_INTRO = (
    "You are a helpful assistant with three tools: calculator (arithmetic), "
    "search_docs (Northwind Robotics company handbook), and wikipedia_search "
    "(general knowledge). "
)


@dataclass(frozen=True)
class AgentConfig:
    """Everything that can differ between agent versions."""

    name: str
    system_prompt: str
    model: str
    max_steps: int = 6  # safety limit so a confused agent can't loop forever


V1 = AgentConfig(
    name="v1",
    system_prompt=TOOLS_INTRO + "Give a short final answer.",
    model=DEFAULT_MODEL,
)

V2 = AgentConfig(
    name="v2",
    system_prompt=(
        TOOLS_INTRO
        + "Before using any tool, write a short numbered plan (2-4 steps) for how "
        "you will answer. Then follow it. If a tool returns an error or nothing "
        "useful, say you could not find the information instead of guessing. "
        "Your final reply must contain only the answer, not the plan."
    ),
    model=DEFAULT_MODEL,
)

V3 = AgentConfig(name="v3", system_prompt=V2.system_prompt, model=BIG_MODEL)

VERSIONS = {"v1": V1, "v2": V2, "v3": V3}