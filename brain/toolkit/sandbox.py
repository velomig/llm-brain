"""Sandboxed Python execution tool."""
from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import time

import config
from brain.toolkit.basic import safe_eval

_SANDBOX_RUNNER = r"""
import ast, contextlib, io, json, math, sys

mem_mb, out_cap, cpu_s = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

try:
    import resource
    if mem_mb > 0:
        lim = mem_mb * 1024 * 1024
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            rl = getattr(resource, name, None)
            if rl is not None:
                try:
                    resource.setrlimit(rl, (lim, lim))
                except (ValueError, OSError):
                    pass
    if cpu_s > 0:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError):
            pass
except ImportError:
    pass


class CappedOutput(io.StringIO):
    clipped = False

    def write(self, s):
        room = out_cap - self.tell()
        if room <= 0:
            self.clipped = True
            return len(s)
        if len(s) > room:
            self.clipped = True
            s = s[:room]
        return super().write(s)


script = sys.stdin.read()

allowed_nodes = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.If,
    ast.Break, ast.Continue, ast.Pass, ast.Load, ast.Store, ast.Name, ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice, ast.BinOp,
    ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.keyword, ast.comprehension,
    ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Attribute,
)
allowed_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "pow": pow, "print": print,
    "range": range, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}

def validate(tree):
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"unsupported syntax: {type(node).__name__}")
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            raise ValueError(f"unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError("private/dunder attributes are not allowed")
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"):
                raise ValueError("attribute access is limited to math.<name>")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in allowed_builtins:
            raise ValueError(f"call not allowed: {node.func.id}")

try:
    tree = ast.parse(script, mode="exec")
    validate(tree)
    env = {"__builtins__": allowed_builtins, "math": math}
    buf = CappedOutput()
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, "<tool-python>", "exec"), env, env)
    out = {"stdout": buf.getvalue(), "ok": True}
    if buf.clipped:
        out["truncated"] = True
    print(json.dumps(out))
except Exception as exc:
    stdout = buf.getvalue() if "buf" in locals() else ""
    print(json.dumps({"stdout": stdout, "ok": False, "error": f"{type(exc).__name__}: {exc}"}))
"""


def _sandbox_rss_bytes(pid: int) -> int | None:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=1.0)
        return int(out.stdout.strip()) * 1024
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def _static_int(node: ast.AST) -> int | None:
    try:
        value = safe_eval(ast.Expression(node))
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _range_count(call: ast.Call) -> int | None:
    if not isinstance(call.func, ast.Name) or call.func.id != "range":
        return None
    vals = [_static_int(arg) for arg in call.args]
    if any(v is None for v in vals):
        return None
    if len(vals) == 1:
        start, stop, step = 0, vals[0], 1
    elif len(vals) == 2:
        start, stop, step = vals[0], vals[1], 1
    elif len(vals) == 3:
        start, stop, step = vals
    else:
        return None
    if step == 0:
        return None
    if (stop - start) * step <= 0:
        return 0
    return max(0, (abs(stop - start) + abs(step) - 1) // abs(step))


def _static_memory_risk(script: str) -> bool:
    mem_mb = int(getattr(config, "TOOL_PYTHON_MEMORY_MB", 0) or 0)
    if mem_mb <= 0:
        return False
    try:
        tree = ast.parse(script, mode="exec")
    except SyntaxError:
        return False
    max_items = max(1, (mem_mb * 1024 * 1024) // 16)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"list", "tuple", "set", "sorted"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Call):
                    count = _range_count(arg)
                    if count is not None and count > max_items:
                        return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            multiplier = _static_int(node.right)
            seq = node.left
            if multiplier is None:
                multiplier = _static_int(node.left)
                seq = node.right
            if multiplier is None or multiplier <= 0:
                continue
            if isinstance(seq, (ast.List, ast.Tuple, ast.Set)) and len(seq.elts) * multiplier > max_items:
                return True
            if isinstance(seq, ast.Constant) and isinstance(seq.value, (str, bytes)):
                if len(seq.value) * multiplier > mem_mb * 1024 * 1024:
                    return True
    return False


def execute_python_code(args: dict) -> dict:
    script = str((args or {}).get("script", "")).strip()
    if not script:
        return {"error": "empty script"}
    if len(script) > config.MAX_TOOL_PYTHON_CHARS:
        return {"error": "script too long"}
    if _static_memory_risk(script):
        return {"error": f"memory limit exceeded ({config.TOOL_PYTHON_MEMORY_MB} MB)"}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            _SANDBOX_RUNNER,
            str(config.TOOL_PYTHON_MEMORY_MB),
            str(config.MAX_TOOL_PYTHON_OUTPUT_CHARS),
            str(math.ceil(config.TOOL_PYTHON_TIMEOUT) + 1),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(config.BASE_DIR),
    )
    proc.stdin.write(script)
    proc.stdin.close()
    mem_cap = config.TOOL_PYTHON_MEMORY_MB * 1024 * 1024
    deadline = time.monotonic() + config.TOOL_PYTHON_TIMEOUT
    breach: str | None = None
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            breach = f"execution timed out after {config.TOOL_PYTHON_TIMEOUT:g}s"
            break
        rss = _sandbox_rss_bytes(proc.pid) if mem_cap > 0 else None
        if rss is not None and rss > mem_cap:
            breach = f"memory limit exceeded ({config.TOOL_PYTHON_MEMORY_MB} MB)"
            break
        time.sleep(0.02)
    if breach:
        proc.kill()
        proc.communicate()
        return {"error": breach}
    stdout_raw, stderr_raw = proc.communicate()
    stdout = (stdout_raw or "").strip()
    if proc.returncode != 0:
        return {"error": ((stderr_raw or "") or stdout or f"python exited {proc.returncode}")[:2000]}
    try:
        parsed = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": "sandbox returned invalid output", "stdout": stdout[:2000]}
    if "stdout" in parsed and len(parsed["stdout"]) > config.MAX_TOOL_PYTHON_OUTPUT_CHARS:
        parsed["stdout"] = parsed["stdout"][: config.MAX_TOOL_PYTHON_OUTPUT_CHARS]
        parsed["truncated"] = True
    return parsed
