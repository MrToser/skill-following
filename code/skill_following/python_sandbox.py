# LOCKED: false
"""使用 AST import 校验的隔离 Python sandbox adapter。"""

from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .logger import my_logger


LOGGER = my_logger("skill_following_experiment_suite.python_sandbox")
ALLOWED_MODULES = {
    "math",
    "random",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "operator",
    "statistics",
    "decimal",
    "fractions",
}
FORBIDDEN_CALLS = {
    "__import__",
    "eval",
    "exec",
    "open",
    "input",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "type",
    "isinstance",
    "issubclass",
    "super",
    "property",
    "staticmethod",
    "classmethod",
}
PYTHON_TIMEOUT_SECONDS = 120
PYTHON_MEMORY_BYTES = 4 * 1024 * 1024 * 1024
PYTHON_CPU_SECONDS = 120
_SANDBOX_SEMAPHORE = asyncio.Semaphore(32)


# 数据：import 的完整 module 名。算法：只比较顶层包并限制在数学安全白名单。
def validate_module_name(module_name: str) -> None:
    top_level_module = module_name.split(".", maxsplit=1)[0]
    if top_level_module not in ALLOWED_MODULES:
        raise ValueError(f"import of '{module_name}' is not allowed")


# 数据：Python 源码。算法：用 AST 区分 module 与 imported symbol，并拒绝危险调用和双下划线访问。
def validate_python_safety(code: str) -> ast.Module:
    assert isinstance(code, str) and code.strip()
    try:
        module = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"invalid Python syntax: {exc.msg}") from exc

    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                validate_module_name(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                raise ValueError("relative imports are not allowed")
            validate_module_name(node.module)
            if any(alias.name == "*" or alias.name.startswith("_") for alias in node.names):
                raise ValueError("star and private-symbol imports are not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"call to '{node.func.id}' is not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("double-underscore names are not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("double-underscore attributes are not allowed")
    return module


# 数据：用户 Python 源码。算法：增加资源上限 prelude，不改写用户 import 或计算语义。
def build_limited_script(code: str) -> str:
    return (
        "import resource\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({PYTHON_MEMORY_BYTES}, {PYTHON_MEMORY_BYTES}))\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({PYTHON_CPU_SECONDS}, {PYTHON_CPU_SECONDS}))\n"
        + code
        + "\n"
    )


# 数据：已通过 AST 校验的 Python。算法：满足 posix_spawn 条件并跳过 site hook，避免训练多线程进程 fork 死锁。
async def execute_python(code: str) -> tuple[str, bool]:
    try:
        validate_python_safety(code)
    except ValueError as exc:
        return f"\n\n<interpreter>Error: {exc}</interpreter>\n\n", False

    async with _SANDBOX_SEMAPHORE:
        with tempfile.TemporaryDirectory(prefix="unified_python_sandbox_") as temp_dir:
            script_path = Path(temp_dir) / "code.py"
            script_path.write_text(build_limited_script(code), encoding="utf-8")
            environment = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            }
            stdout_path = Path(temp_dir) / "stdout.txt"
            stderr_path = Path(temp_dir) / "stderr.txt"
            with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
                process = subprocess.Popen(
                    [sys.executable, "-I", "-S", str(script_path)],
                    env=environment,
                    close_fds=False,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                event_loop = asyncio.get_running_loop()
                deadline = event_loop.time() + PYTHON_TIMEOUT_SECONDS
                while process.poll() is None and event_loop.time() < deadline:
                    await asyncio.sleep(0.05)
                if process.poll() is None:
                    process.kill()
                    while process.poll() is None:
                        await asyncio.sleep(0.01)
                    return (
                        f"\n\n<interpreter>Error: execution timed out after {PYTHON_TIMEOUT_SECONDS} seconds.</interpreter>\n\n",
                        False,
                    )
            stdout_bytes = stdout_path.read_bytes()
            stderr_bytes = stderr_path.read_bytes()

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    if process.returncode != 0:
        error_text = stderr or f"process exited with code {process.returncode}"
        return f"\n\n<interpreter>Error: {error_text}</interpreter>\n\n", False
    if not stdout:
        return "\n\n<interpreter>Error: Python produced no stdout.</interpreter>\n\n", False
    result = f"Output: {stdout}"
    if stderr:
        result += f"\nWarnings: {stderr}"
    return f"\n\n<interpreter>{result}</interpreter>\n\n", True
