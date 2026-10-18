import ast
import json
import operator
import re
from pathlib import Path

import httpx

# ---------------------------------------------------------------- calculator
# Never use eval() on model output: it can run arbitrary code.
# We parse the expression and allow only basic math operators.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return str(_eval(tree.body))


# ------------------------------------------------------------- local doc search
DOCS_DIR = Path("data/docs")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "is", "are", "for", "and", "or",
    "what", "how", "many", "much", "do", "does", "at", "on", "per", "can",
}


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower())) - _STOP


def search_docs(query: str) -> str:
    """Keyword-overlap retrieval over paragraphs of data/docs/*.txt."""
    q = _tokens(query)
    scored = []
    for f in sorted(DOCS_DIR.glob("*.txt")):
        for para in f.read_text(encoding="utf-8").split("\n\n"):
            para = para.strip()
            if para:
                score = len(q & _tokens(para))
                if score:
                    scored.append((score, f.name, para))
    if not scored:
        return "No matching documents found."
    scored.sort(key=lambda x: -x[0])
    return "\n\n".join(f"[{name}] {para}" for _, name, para in scored[:2])


# ------------------------------------------------------------------ wikipedia
WIKI_HEADERS = {
    "User-Agent": "agentbench-lite/0.1 (student portfolio project; contact: https://github.com/Md-Shahriar-Rahman-Shihab) httpx",
    "Accept": "application/json",
}


def wikipedia_search(query: str) -> str:
    resp = httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 1,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
        },
        headers=WIKI_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    if not pages:
        return "No results found."
    best = min(pages.values(), key=lambda p: p.get("index", 99))
    return f"{best['title']}: {best.get('extract', '')[:1200]}"


# ------------------------------------------------------------------- registry
# name -> python function
TOOL_FUNCS = {
    "calculator": calculator,
    "search_docs": search_docs,
    "wikipedia_search": wikipedia_search,
}


def _schema(name: str, description: str, arg: str, arg_desc: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {arg: {"type": "string", "description": arg_desc}},
                "required": [arg],
            },
        },
    }


# What the LLM sees: tool names, descriptions, and argument format
TOOL_SCHEMAS = [
    _schema(
        "calculator",
        "Evaluate an arithmetic expression, e.g. '(2480*15)/100+37'. "
        "Supports + - * / // % ** and parentheses.",
        "expression",
        "The arithmetic expression to evaluate",
    ),
    _schema(
        "search_docs",
        "Search the internal Northwind Robotics company handbook "
        "(policies, benefits, expenses, equipment, support hours).",
        "query",
        "Keywords to search for",
    ),
    _schema(
        "wikipedia_search",
        "Look up general knowledge on Wikipedia. Returns the intro of the best match.",
        "query",
        "Topic to look up",
    ),
]


def run_tool(name: str, arguments_json: str) -> str:
    """Run a tool; never crash. Errors go back to the LLM as text."""
    try:
        if name not in TOOL_FUNCS:
            return f"ERROR: unknown tool '{name}'"
        args = json.loads(arguments_json or "{}")
        return TOOL_FUNCS[name](**args)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"