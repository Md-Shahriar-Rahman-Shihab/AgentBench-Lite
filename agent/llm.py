import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

CACHE_PATH = Path("data/cache.db")
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-oss-20b")


class LLMResult(BaseModel):
    """Everything we want to remember about one LLM call."""

    model: str
    text: str | None
    tool_calls: list[dict] = []  # used from Step 4 (agent tools)
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int = 0
    latency_s: float
    cached: bool = False


class Cache:
    """Tiny key-value store on SQLite: request hash -> saved response."""

    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()


def make_key(model, messages, temperature, tools, salt) -> str:
    """Hash of everything that affects the answer."""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "salt": salt,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class LLMClient:
    def __init__(self, use_cache: bool = True):
        # max_retries=0: we do retries ourselves with tenacity, so we control them
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=0)
        self.cache = Cache() if use_cache else None

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
        ),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call_api(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        salt: str = "",
    ) -> LLMResult:
        model = model or DEFAULT_MODEL
        key = make_key(model, messages, temperature, tools, salt)

        # 1) cache hit? return the saved answer (original tokens/latency kept)
        if self.cache:
            hit = self.cache.get(key)
            if hit:
                data = json.loads(hit)
                data["cached"] = True
                return LLMResult(**data)

        # 2) real API call (with retries)
        kwargs = dict(model=model, messages=messages, temperature=temperature)
        if tools:
            kwargs["tools"] = tools

        start = time.perf_counter()
        resp = self._call_api(**kwargs)
        latency = time.perf_counter() - start

        msg = resp.choices[0].message
        details = getattr(resp.usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
        tool_calls = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            for tc in (msg.tool_calls or [])
        ]

        result = LLMResult(
            model=model,
            text=msg.content,
            tool_calls=tool_calls,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            reasoning_tokens=reasoning,
            latency_s=latency,
        )

        # 3) save for next time
        if self.cache:
            self.cache.put(key, result.model_dump_json())
        return result