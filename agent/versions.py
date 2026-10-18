from dataclasses import dataclass

from agent.llm import DEFAULT_MODEL


@dataclass(frozen=True)
class AgentConfig:
    """Everything that can differ between agent versions."""

    name: str
    system_prompt: str
    model: str
    max_steps: int = 6  # safety limit so a confused agent can't loop forever


V1 = AgentConfig(
    name="v1",
    system_prompt=(
        "You are a helpful assistant. Use the calculator tool for arithmetic. "
        "Give a short final answer."
    ),
    model=DEFAULT_MODEL,
)

VERSIONS = {"v1": V1}