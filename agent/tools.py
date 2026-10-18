import ast
import json
import operator

# Never use eval() on model output: it can run arbitrary code.
# Instead we parse the expression and allow only basic math operators.
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


# name -> python function
TOOL_FUNCS = {"calculator": calculator}

# What the LLM sees: tool names, descriptions, and argument format
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression, e.g. '(2480*15)/100+37'. "
            "Supports + - * / // % ** and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    }
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